# RL architecture — phase 2: real optimizer-step validation

Phase 1 (`docs/rl.md`) delivered the recorder, reward interface and worker
protocol but ran no training. Phase 2 closes the loop with a small, real
reinforcement-learning run. This is an **architecture validation**, not a
business model-training result: the model is 0.5B parameters, the dataset is a
small held-out slice, and no model-quality claim is made. Every property the
acceptance criteria name is measured from real tensors and real processes.

## What is real, what is not

| Component | Real | Notes |
| --- | --- | --- |
| Policy model | yes | Qwen2.5-0.5B-Instruct, unmodified Hugging Face weights |
| Rollouts | yes | Sampled with `temperature=1.0` over loopback HTTP/SSE through the phase-1 recorder |
| Trajectory export | yes | The same `umminfer_rl_trajectory` JSONL, validated by the worker |
| Reward | yes | Exact integer match against GSM8K gold answers; 1.0/0.0, never fabricated |
| Training worker | yes | Separate OS process, NDJSON protocol, serving never imports optimizer code |
| Optimizer step | yes | AdamW update (`weight_decay=0`) from a backpropagated REINFORCE loss on GPU |
| Gradient and weight change | yes | Measured from real tensor values; the runner fails the run if a step has zero non-zero gradients or a zero weight delta, so a signal-free step cannot masquerade as training |
| Checkpoints | yes | At least two versioned manifests with the phase-1 content hash chain, plus real weight files |
| Business training outcome | no | 8 prompts, 2 steps; no convergence or quality guarantee |

## Components

| File | Responsibility |
| --- | --- |
| `python/umminfer_rl_gsm8k_data.py` | Real GSM8K download/parse, shared prompt template |
| `python/umminfer_rl_gsm8k_parquet.py` | HF parquet -> upstream-shaped JSONL converter (fallback data path) |
| `python/umminfer_rl_gsm8k_reward.py` | Verifiable exact-match reward, loads through `load_reward` |
| `python/umminfer_rl_policy_serve.py` | Small-model OpenAI-compatible SSE policy server (separate process) |
| `python/umminfer_rl_train.py` | REINFORCE trainer: group-relative advantage, AdamW, measured metrics |
| `python/umminfer_rl_worker.py` | Adds the `train_step` verb; phase-1 `train` is unchanged |
| `python/umminfer_rl_eval.py` | Separate-process greedy evaluation, with or without checkpoint weights |
| `python/umminfer_rl_run.py` | End-to-end orchestrator; itself imports no torch |
| `tests/rl_phase2_test.py` | Deterministic tests, no torch/GPU/network |

## Algorithm

REINFORCE with a group-relative baseline (the same advantage normalization
GRPO uses, without a value head or KL term):

```
L_i = mean over completion tokens of log p_theta(token | prompt, earlier tokens)
A_i = (reward_i - mean_group reward) / (std_group reward + 1e-8)
loss = -mean_i A_i * L_i
```

Rollouts sharing one prompt form a group. A group whose rewards have zero
variance gets zero advantage and contributes no gradient signal. That is the
correct REINFORCE semantics, not a masked-out failure. The runner treats a
step whose gradients are all zero as a hard failure: no validation run can
finish on weight decay or noise, because the optimizer uses
`weight_decay=0` and the runner gates on `nonzero_grad_params > 0` and a
positive measured weight delta before accepting a checkpoint.

This is policy-gradient RL, not supervised fine-tuning: the loss contains no
gold-answer token target. The gold answer enters only through the scalar
reward, applied as a weight on the likelihood of what the policy itself
sampled.

### Replay precision limit

The trajectory export stores completion **text**, while the trainer
re-tokenizes that text to recompute log probabilities. Token boundaries at the
prompt/completion join can differ from the sampled token ids by at most one
token. The export format has no token-id field in phase 1; adding one is a
schema change recorded as future work. The limit affects one boundary token
per rollout, not the reward or the optimizer mechanics.

## Worker protocol additions

One new verb, no version bump (unknown fields were already ignored by
contract):

| Request | Response |
| --- | --- |
| `{"type": "train_step", "task_id", "trajectory_path", "model_path", "hyperparameters": {"lr", "seed", "dtype"}}` | `{"type": "train_step_result", "sequence", "checkpoint_path", "weights_path", "metrics", ...}` or `{"type": "error", ...}` |

Rules:

- `train_step` rejects any record without a numeric reward score before
  touching torch, and a rejected request consumes no checkpoint sequence.
- The trainer is created on the first `train_step` and stays resident, so
  optimizer momentum carries across steps. A later request with a different
  model path or hyperparameter set is rejected; a new state directory starts a
  new trainer.
- Each step writes a real `weights-<sequence>.pt` (model and optimizer state)
  and a manifest whose `algorithm`, `weights_ref`, `weights_sha256`,
  `hyperparameters` and `metrics` slots are now filled. The phase-1 hash
  chain (`base_checkpoint_sha256` plus `.sha256` sidecars) is unchanged, and
  reopening the directory still re-verifies every link.
- The phase-1 `train` verb keeps writing null algorithm/weights slots.

## Reproducibility

The run manifest records: seed, all hyperparameters, sampling parameters, the
GSM8K source URL with byte count and SHA-256, train/eval prompt indexes, base
model weights SHA-256, per-step trajectory and checkpoint hashes, and every
measured metric. The policy server seeds its CUDA generator; the trainer seeds
torch and CUDA before model construction.

Determinism limits, stated rather than hidden:

- CUDA kernel determinism is not forced (no
  `torch.use_deterministic_algorithms`); bit-identical reruns are not
  guaranteed, statistical reproduction is.
- Sampling order is reproducible only while the single-threaded policy server
  handles requests in orchestrator order, which the runner enforces.
- GPU name, driver and CUDA versions, Python/torch/transformers versions
  and the visible GPU count are captured from child processes into the run
  manifest (`environment` block); the runner process itself imports no torch.

## Isolation

- Everything runs as the unprivileged container user in a user-created
  venv; no system package is modified.
- `CUDA_VISIBLE_DEVICES=4` pins all model work to the dedicated GPU. GPUs
  0/1 serving other workloads are never opened.
- The policy server and the training worker are separate processes from any
  serving engine, and the runner never starts the umm-infer server.

## How to run

On H20B, inside the `linyu-dev` container, with a cu124-compatible torch in a
user venv (driver 535 cannot use the container's default cu130 torch):

```
# Optional when the GitHub raw JSONL is too slow: fetch the HF parquet and
# convert it; the produced JSONL has the same question/answer shape.
python3 -I python/umminfer_rl_gsm8k_parquet.py 0000.parquet gsm8k-test.jsonl

CUDA_VISIBLE_DEVICES=4 \
python3 -I python/umminfer_rl_run.py \
  --model-path <qwen2.5-0.5b-instruct> \
  --data-path <gsm8k-test.jsonl> \
  --data-source <provenance-url-recorded-in-manifest> \
  --work-dir <evidence-dir>/run \
  --python <venv>/bin/python \
  --train-indexes 1,4,9,11,20,30,35,37 \
  --eval-indexes 2,10,17,40,41,42,43,44,45,46,47,48,49,50,51,52 \
  --group-size 8 --steps 2 --temperature 1.0
```

`--train-indexes`/`--eval-indexes` select explicit dataset rows; without them
a contiguous slice is used. The manifest records every selected index, so the
split is reproducible either way.

Deterministic local tests need no model:

```
python3 -I tests/rl_phase2_test.py
python3 -I tests/rl_architecture_test.py
```

## Measured H20B run (2026-11-14)

One complete run on GPU 4 of H20B, container `linyu-dev`, in a user venv
(torch 2.6.0+cu124, transformers 4.51.3, Python 3.12). The full evidence is
in the container at `/root/rl-validation-20261114/evidence/run`; a
non-weight subset is under `/tmp/umminfer-phase6-rl-h20b-evidence/run`.

- Model: Qwen2.5-0.5B-Instruct, base weights SHA-256
  `fdf756fa…8fb7fe`, bf16.
- Data: the real openai/gsm8k test split (1319 items). The GitHub raw JSONL
  could not be fully downloaded on this host, so the bytes came from the
  hf-mirror HF parquet conversion, converted to question/answer JSONL with
  `umminfer_rl_gsm8k_parquet.py`. The resulting byte SHA-256
  `752adc99…afbac` and provenance URL are in the manifest `data` block.
- Training prompts: 8 indexes `1,4,9,11,20,30,35,37`, each sampled 8 times
  at temperature 1.0. They were chosen by a greedy pre-scan of the first 40
  problems that kept only prompts whose base-policy output ends in a
  parseable `####` marker; without that, this 0.5B model emits no marker on
  most problems and every reward group is homogeneous (zero advantage).
  Provenance split between two places, stated exactly: the greedy pre-scan
  selection rule is recorded in this document; the selected indexes are
  recorded in `run_manifest.json` (`train_prompt_indexes`). The manifest does
  not contain the pre-scan rule or the per-prompt scan results.
- Held-out eval: 16 different indexes, greedy, before training and after each
  step.

Measured optimizer evidence (from the two checkpoints, real tensors):

| Step | loss | grad norm | non-zero grad tensors | max |Δw| | mean reward |
| --- | --- | --- | --- | --- | --- |
| 0 | -0.02543 | 1.9766 | 290 / 290 | 1.53e-05 | 0.0156 |
| 1 | -0.01491 | 2.2656 | 290 / 290 | 1.53e-05 | 0.0469 |

Independent tensor comparison against the untouched base model: 240 of 291
parameter tensors changed after step 0; the two probe tensors moved
cumulatively (base→w0 1.53e-05, base→w1 3.05e-05, w0→w1 1.53e-05), and the
step-1 checkpoint carries non-zero AdamW `exp_avg`/`exp_avg_sq` momentum.
The two weight files have distinct SHA-256 hashes and the checkpoint
manifests form a verified hash chain (a fresh worker reopen re-verified it
and reported next sequence 2).

Held-out greedy exact match: baseline 0.125, after step 1 0.000, after step 2
0.125 (2 correct of 16). This is flat at two steps on 64 rollouts per step
and is **not** a convergence or quality result. The run proves the RL
mechanics end to end; it does not show the model improved.

## What remains NOT_COVERED

- PPO/GRPO with a value head or KL regularization, DPO, and any multimodal
  reward. Only group-relative REINFORCE is exercised.
- Training the shared 30B model or any production serving model.
- Bit-exact deterministic replay across hosts.
- A token-id field in the trajectory export (the boundary-token limit above).
