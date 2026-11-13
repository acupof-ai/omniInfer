#!/usr/bin/env python3
"""Unit tests for the provenance collection helpers.

Synthetic bytes only; no SSH, no model contact.
"""

import importlib.util
import unittest
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().with_name("model_provenance.py")
    spec = importlib.util.spec_from_file_location("model_provenance", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prov = load_module()


class ParseTests(unittest.TestCase):
    def test_parse_inventory(self):
        text = "config.json\t6193\nsub/x.bin\t1024\n"
        self.assertEqual(prov.parse_inventory(text),
                         {"config.json": 6193, "sub/x.bin": 1024})
        self.assertEqual(prov.parse_inventory(""), {})

    def test_parse_sha256_lines(self):
        text = ("76d3ae46  ./config.json\n"
                "11ad7efa  ./.gitattributes\n")
        self.assertEqual(
            prov.parse_sha256_lines(text),
            {"config.json": "76d3ae46", ".gitattributes": "11ad7efa"})

    def test_server_provenance_marks_missing_revision(self):
        out = prov.server_provenance(
            {"served_model_name": "qwen3-asr",
             "model_path": "/data/asr", "revision": None,
             "model_checksum": None, "version": "0.5.15", "status": "ready"})
        self.assertEqual(out["server_revision"], prov.UNKNOWN)
        self.assertEqual(out["server_model_checksum"], prov.UNKNOWN)
        self.assertEqual(out["served_model_name"], "qwen3-asr")

    def test_server_provenance_keeps_real_revision(self):
        out = prov.server_provenance({"revision": "abc123"})
        self.assertEqual(out["server_revision"], "abc123")


class HfVerificationTests(unittest.TestCase):
    def setUp(self):
        # content: b"abc" -> sha256 ba7816bf..., b"xyz" -> sha256 3608bca1...
        self.remote = {
            "model.safetensors":
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "config.json":
                "3608bca1e44ea6c4d268eb6db02260269892c0b42b86bbf1e77a6fa16c3c9282",
        }
        self.tree = [
            {"path": "model.safetensors", "repo": "o/r",
             "lfs": {"oid": self.remote["model.safetensors"], "size": 3}},
            {"path": "config.json", "repo": "o/r"},
        ]

    def fetcher(self, repo, path):
        return {"config.json": b"xyz"}[path]

    def test_all_identical(self):
        verdict = prov.verify_against_hf(
            self.remote, self.tree, "deadbeef", self.fetcher)
        self.assertTrue(verdict["all_files_byte_identical"])
        self.assertEqual(verdict["mismatched"], [])
        self.assertEqual(verdict["file_count"], 2)
        kinds = {f["path"]: f["kind"] for f in verdict["files"]}
        self.assertEqual(kinds["model.safetensors"], "lfs_oid")
        self.assertEqual(kinds["config.json"], "raw_content")
        self.assertEqual(verdict["hf_main_commit_sha"], "deadbeef")

    def test_lfs_mismatch_detected(self):
        self.tree[0]["lfs"]["oid"] = "0" * 64
        verdict = prov.verify_against_hf(
            self.remote, self.tree, "deadbeef", self.fetcher)
        self.assertFalse(verdict["all_files_byte_identical"])
        self.assertEqual(verdict["mismatched"], ["model.safetensors"])

    def test_missing_deployed_file_detected(self):
        verdict = prov.verify_against_hf(
            {"config.json": self.remote["config.json"]},
            self.tree, "deadbeef", self.fetcher)
        self.assertFalse(verdict["all_files_byte_identical"])
        self.assertEqual(verdict["mismatched"], ["model.safetensors"])
        missing = next(f for f in verdict["files"]
                       if f["path"] == "model.safetensors")
        self.assertEqual(missing["reason"], "missing")


if __name__ == "__main__":
    unittest.main()
