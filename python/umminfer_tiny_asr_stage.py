#!/usr/bin/env python3
"""Deterministic tiny acoustic model and its real ummInfer pipeline stage."""

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import importlib.util

import mlx.core as mx
import numpy as np


def load_pipeline_stage():
    path = Path(__file__).resolve().with_name("umminfer_mlx_stage.py")
    spec = importlib.util.spec_from_file_location("umminfer_mlx_stage", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MLXStage


MLXStage = load_pipeline_stage()


ALPHABET = " abcdefghijklmnopqrstuvwxyz0123456789.,!?"


def create_model(output):
    """Create initialized weights; this is an inference test model, not trained ASR."""
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    n_fft = 400
    sample_rate = 16000
    mel_bins = 12
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    mel = 2595.0 * np.log10(1.0 + frequencies / 700.0)
    edges = np.linspace(2595.0 * np.log10(1.0 + 20.0 / 700.0),
                        2595.0 * np.log10(1.0 + 7600.0 / 700.0), mel_bins + 2)
    filterbank = np.zeros((mel_bins, len(frequencies)), dtype=np.float32)
    for index in range(mel_bins):
        left = (mel[index + 1] - mel[index]) or 1.0
        right = (mel[index + 2] - mel[index + 1]) or 1.0
        filterbank[index] = np.maximum(0.0, np.minimum(
            (mel - mel[index]) / left, (mel[index + 2] - mel) / right))
    filterbank /= np.maximum(filterbank.sum(axis=1, keepdims=True), 1e-6)

    config = {
        "format": "umminfer-tiny-ctc-v1",
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "hop_length": 160,
        "mel_bins": mel_bins,
        "hidden_size": 16,
        "alphabet": ALPHABET,
        "max_audio_seconds": 30,
        "seed": 20261111,
    }
    mx.random.seed(config["seed"])
    weights = {
        "filterbank": mx.array(filterbank),
        "encoder.weight": mx.random.normal((16, mel_bins)) * 0.20,
        "encoder.bias": mx.zeros((16,)),
        "decoder.weight": mx.random.normal((16, 16)) * 0.20,
        "decoder.bias": mx.zeros((16,)),
        "classifier.weight": mx.random.normal((len(ALPHABET) + 1, 16)) * 0.20,
        "classifier.bias": mx.zeros((len(ALPHABET) + 1,)),
    }
    weights["classifier.bias"][0] = -1.0
    mx.save_safetensors(output / "model.safetensors", weights,
                        metadata={"format": config["format"]})
    (output / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


class TinyCTCModel:
    def __init__(self, path):
        self.config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        required = {"format", "sample_rate", "n_fft", "hop_length", "mel_bins",
                    "hidden_size", "alphabet", "max_audio_seconds"}
        if not required <= self.config.keys():
            raise ValueError("tiny model config is incomplete")
        if self.config["format"] != "umminfer-tiny-ctc-v1":
            raise ValueError("unsupported tiny model format")
        if self.config["sample_rate"] != 16000 or self.config["n_fft"] != 400:
            raise ValueError("unsupported tiny audio configuration")
        if self.config["alphabet"] != ALPHABET:
            raise ValueError("unsupported tiny model alphabet")
        self.weights = mx.load(path / "model.safetensors")
        vocab_size = len(ALPHABET) + 1
        expected = {
            "filterbank": (self.config["mel_bins"], self.config["n_fft"] // 2 + 1),
            "encoder.weight": (self.config["hidden_size"], self.config["mel_bins"]),
            "encoder.bias": (self.config["hidden_size"],),
            "decoder.weight": (self.config["hidden_size"], self.config["hidden_size"]),
            "decoder.bias": (self.config["hidden_size"],),
            "classifier.weight": (vocab_size, self.config["hidden_size"]),
            "classifier.bias": (vocab_size,),
        }
        if self.weights.keys() != expected.keys():
            raise ValueError("tiny model weights do not match the config")
        for name, shape in expected.items():
            if self.weights[name].shape != shape:
                raise ValueError(f"invalid shape for {name}")

    def decode_tokens(self, pcm, max_tokens):
        """Return the raw CTC token list and whether the cap was reached.

        The token facts are recorded before any text post-processing, so a
        stripped transcript cannot change the reported count or reason.
        """
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("audio must contain complete PCM16 samples")
        if len(pcm) < self.config["n_fft"] * 2:
            raise ValueError("audio is shorter than one analysis frame")
        limit = self.config["sample_rate"] * self.config["max_audio_seconds"] * 2
        if len(pcm) > limit:
            raise ValueError("audio exceeds the tiny model duration limit")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        hop = self.config["hop_length"]
        frame_count = 1 + (len(samples) - self.config["n_fft"]) // hop
        frames = np.lib.stride_tricks.as_strided(
            samples, shape=(frame_count, self.config["n_fft"]),
            strides=(samples.strides[0] * hop, samples.strides[0])).copy()
        window = np.hanning(self.config["n_fft"] + 1)[:-1].astype(np.float32)
        spectrum = mx.fft.rfft(mx.array(frames * window), axis=1)
        power = mx.square(mx.abs(spectrum)) / self.config["n_fft"]
        features = mx.log(mx.maximum(
            power @ self.weights["filterbank"].T, 1e-7))
        hidden = mx.tanh(features @ self.weights["encoder.weight"].T +
                         self.weights["encoder.bias"])
        hidden = mx.tanh(hidden @ self.weights["decoder.weight"].T +
                         self.weights["decoder.bias"])
        logits = hidden @ self.weights["classifier.weight"].T + self.weights["classifier.bias"]
        token_ids = np.asarray(mx.argmax(logits, axis=1))
        tokens = []
        truncated = False
        previous = 0
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id and token_id != previous:
                # The cap is only a truncation when one more token exists. A run
                # that fills the cap exactly at the end of the audio stopped on
                # its own and reports "stop".
                if len(tokens) >= max_tokens:
                    truncated = True
                    break
                tokens.append(ALPHABET[token_id - 1])
            previous = token_id
        return tokens, truncated

    def transcribe(self, pcm, max_tokens):
        tokens, _ = self.decode_tokens(pcm, max_tokens)
        return "".join(tokens).strip()

    def generate(self, pcm, max_tokens, temperature):
        if temperature != 0:
            raise ValueError("the tiny CTC model supports greedy decoding only")
        tokens, truncated = self.decode_tokens(pcm, max_tokens)
        self.last_generated_tokens = len(tokens)
        self.last_truncated = truncated
        text = "".join(tokens).strip()
        if not text:
            raise ValueError("the tiny model produced an empty transcription")
        for character in text:
            yield SimpleNamespace(text=character)


class TinyASRStage(MLXStage):
    def load(self):
        if self.generate is None:
            model = TinyCTCModel(Path(self.model_path))
            self.model = model
            self.generate = model.generate
            self.tokenizer = None

    def result_metadata(self, parts, max_tokens, last_response):
        # Counts and reasons come from the CTC token sequence, not from the
        # stripped text that is streamed to the caller.
        return (self.model.last_generated_tokens,
                "length" if self.model.last_truncated else "stop")

    def parse_request(self, message):
        data = message["data"]
        request = data["request"]
        inputs = request["inputs"]
        if not isinstance(inputs, dict) or set(inputs) != {"audio"}:
            raise ValueError("tiny ASR inputs must contain only audio")
        params = request.get("params", {})
        if not isinstance(params, dict):
            raise ValueError("params must be a map")
        max_tokens = params.get("max_new_tokens", 128)
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ValueError("max_new_tokens must be an integer")
        if not 1 <= max_tokens <= 4096:
            raise ValueError("max_new_tokens is outside the supported range")
        temperature = params.get("temperature", 0.0)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise ValueError("temperature must be a number")
        if not math.isfinite(temperature) or temperature != 0:
            raise ValueError("the tiny CTC model requires temperature 0")
        return inputs["audio"], max_tokens, float(temperature), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="create a new deterministic model")
    generate.add_argument("--output", required=True, type=Path)
    serve = commands.add_parser("serve", help="run the real pipeline stage")
    serve.add_argument("--name", required=True)
    serve.add_argument("--recv", required=True)
    serve.add_argument("--coordinator", required=True)
    serve.add_argument("--abort", required=True)
    serve.add_argument("--model", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        create_model(args.output)
        return
    stage = TinyASRStage(args.name, args.recv, args.coordinator, args.abort,
                         str(args.model.resolve()))
    stage.start()
    stage.run()


if __name__ == "__main__":
    main()
