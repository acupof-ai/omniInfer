# SPDX-License-Identifier: Apache-2.0
"""REINFORCE training with group-relative baseline for the RL validation.

Real optimizer updates on a real Hugging Face causal LM. Rollouts are sampled
by the separate policy server; this module only recomputes token log
probabilities under the current weights and applies the policy-gradient rule:

    L_i = mean over completion tokens of log p_theta(token | prompt, tokens)
    A_i = (reward_i - mean_group reward) / (std_group reward + eps)
    loss = -mean_i A_i * L_i

Groups are rollouts that share one prompt. A group with zero reward variance
has zero advantage, which is the correct REINFORCE behaviour, not a fudge.
Every claim this module makes (non-zero gradients, weight deltas, loss) is
measured from real tensors and returned in the step metrics.

Completion text is re-tokenized for the log-probability pass rather than
reusing sampled token ids: the trajectory export stores text, and the boundary
effect (one token at most) is documented as the replay precision limit in
docs/rl-phase2.md.
"""

import json
from pathlib import Path

ALGORITHM_NAME = "reinforce-group-relative-v1"


class ReinforceTrainer:
    def __init__(self, model_path, *, lr, seed, dtype="bfloat16",
                 grad_clip=1.0, advantage_eps=1e-8):
        import torch
        self.torch = torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=getattr(torch, dtype)
        ).to("cuda").train()
        # weight_decay=0 so every measured weight delta is gradient-driven;
        # no decoupled decay can masquerade as an RL update in the metrics.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.0)
        self.config = {"algorithm": ALGORITHM_NAME, "lr": lr, "seed": seed,
                       "dtype": dtype, "grad_clip": grad_clip,
                       "advantage_eps": advantage_eps,
                       "weight_decay": 0.0}
        self.grad_clip = grad_clip
        self.advantage_eps = advantage_eps
        # Three named tensors (embedding, middle layer, head) probe real
        # weight movement without snapshotting the whole model twice.
        names = [name for name, _ in self.model.named_parameters()]
        self.probe_names = [names[0], names[len(names) // 2], names[-1]]

    def completion_logprob(self, messages, completion_text):
        """Mean per-token log probability of the completion under current weights."""
        torch = self.torch
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        completion_ids = self.tokenizer(
            completion_text, return_tensors="pt",
            add_special_tokens=False).input_ids
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1).to("cuda")
        prompt_len = prompt_ids.shape[1]
        logits = self.model(input_ids).logits
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        targets = input_ids[:, 1:]
        token_log_probs = log_probs.gather(
            2, targets.unsqueeze(-1)).squeeze(-1)
        # Targets are shifted by one: target index j predicts input position
        # j+1. The first completion token sits at input position prompt_len,
        # i.e. target index prompt_len-1, so the mask starts there.
        mask = torch.arange(targets.shape[1], device="cuda") >= prompt_len - 1
        selected = token_log_probs.masked_select(mask)
        if selected.numel() == 0:
            # A loud failure beats a NaN that would poison the loss and be
            # reported as a real metric.
            raise ValueError("empty completion after re-tokenization")
        return selected.mean(), selected.numel()

    def train_step(self, rollouts):
        """One optimizer step over rollout records.

        Each record carries ``messages``, ``output_text``, ``reward.score`` and
        a ``prompt_id`` grouping rollouts of the same prompt.
        """
        torch = self.torch
        groups = {}
        for row in rollouts:
            groups.setdefault(row["prompt_id"], []).append(row)

        weighted = []
        group_stats = []
        for members in groups.values():
            rewards = torch.tensor([m["reward"]["score"] for m in members],
                                   dtype=torch.float32, device="cuda")
            advantages = (rewards - rewards.mean()) / (
                rewards.std(unbiased=False) + self.advantage_eps)
            group_stats.append({
                "size": len(members),
                "reward_mean": rewards.mean().item(),
                "reward_std": rewards.std(unbiased=False).item()})
            for member, advantage in zip(members, advantages):
                weighted.append((member, advantage))

        weight_before = self._probe_weights()
        self.optimizer.zero_grad()
        losses = []
        token_counts = []
        for row, advantage in weighted:
            log_prob, token_count = self.completion_logprob(
                row["messages"], row["output_text"])
            losses.append(-advantage * log_prob)
            token_counts.append(token_count)
        loss = torch.stack(losses).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip)
        nonzero_grads = sum(
            1 for param in self.model.parameters()
            if param.grad is not None and param.grad.abs().sum().item() > 0)
        total_grad_params = sum(
            1 for param in self.model.parameters() if param.grad is not None)
        self.optimizer.step()
        return {
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "nonzero_grad_params": nonzero_grads,
            "grad_params": total_grad_params,
            "weight_delta_max_abs": self._max_abs_delta(weight_before),
            "probe_weights": self.probe_names,
            "rollout_count": len(rollouts),
            "group_count": len(groups),
            "group_stats": group_stats,
            "completion_tokens": int(sum(token_counts)),
            "reward_mean": sum(r["reward"]["score"] for r in rollouts)
            / len(rollouts),
        }

    def _probe_weights(self):
        return {name: param.detach().float().clone()
                for name, param in self.model.named_parameters()
                if name in self.probe_names}

    def _max_abs_delta(self, before):
        deltas = [(param.detach().float() - before[name]).abs().max().item()
                  for name, param in self.model.named_parameters()
                  if name in before]
        return max(deltas)

    def greedy_answer(self, messages, *, max_new_tokens=512):
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        with self.torch.no_grad():
            output = self.model.generate(
                input_ids, do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(output[0, input_ids.shape[1]:],
                                     skip_special_tokens=True)

    def save_weights(self, path):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        self.torch.save({"model": self.model.state_dict(),
                         "optimizer": self.optimizer.state_dict(),
                         "config": self.config}, tmp)
        tmp.replace(path)
        return path


def write_rollout_file(path, rollouts):
    with open(path, "w", encoding="utf-8") as handle:
        for row in rollouts:
            handle.write(json.dumps(row, sort_keys=True,
                                    ensure_ascii=False, allow_nan=False) + "\n")
