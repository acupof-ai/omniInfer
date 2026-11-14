# SPDX-License-Identifier: Apache-2.0
"""Verifiable GSM8K exact-match reward for the RL architecture validation.

The reward still reads only trajectory records at scoring time and has no
access to serving memory. The gold answers are static data loaded once at
construction: a JSONL file mapping ``request_id`` to the integer gold answer.
The path is supplied through UMMINFER_RL_GOLD_PATH so the existing
``load_reward("module:factory")`` entry point needs no new arguments.

A response is scored 1.0 only when its final ``#### <number>`` marker parses
to the gold integer; every other parseable response is 0.0. A record whose
request id has no gold entry scores None, keeping "not scorable" distinct from
"scored zero", matching the phase-1 reward contract.
"""

import json
import os
import re

from umminfer_rl_reward import Reward

_MARKER = re.compile(r"####\s*(-?[\d,]+)\s*$")


def extract_answer(text):
    """Return the integer after the final #### marker, or None."""
    match = _MARKER.search(text.strip())
    if not match:
        return None
    return int(match.group(1).replace(",", ""))


class GSM8KExactMatchReward(Reward):
    name = "gsm8k-exact-match-v1"

    def __init__(self, gold_path=None):
        gold_path = gold_path or os.environ.get("UMMINFER_RL_GOLD_PATH")
        if not gold_path:
            raise ValueError(
                "gold answer path required: pass gold_path or set "
                "UMMINFER_RL_GOLD_PATH")
        self.gold = {}
        with open(gold_path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    self.gold[row["request_id"]] = int(row["gold_answer"])

    def score(self, record):
        gold = self.gold.get(record.get("request_id"))
        if gold is None:
            return None
        answer = extract_answer(record.get("output_text", ""))
        return 1.0 if answer == gold else 0.0


def load_default():
    """Zero-arg factory for ``load_reward("module:load_default")``."""
    return GSM8KExactMatchReward()
