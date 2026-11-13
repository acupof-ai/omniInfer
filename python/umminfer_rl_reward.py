# SPDX-License-Identifier: Apache-2.0
"""Pluggable reward interface. No business reward is implemented here.

A reward reads only a recorded trajectory export (the JSONL records written
by umminfer_rl_trajectory), never serving-process memory. A missing score is a
first-class outcome: NullReward returns None and the record keeps
``score: null`` so downstream consumers distinguish "reward missing" from
"reward zero". The concrete reward function is an undecided user input.
"""

import importlib
import json
from pathlib import Path

REWARD_SCHEMA_VERSION = 1


class Reward:
    """Maps one trajectory record to a float score, or None if not scorable."""

    #: Stable name written into the record so scores stay interpretable.
    name = "abstract"

    def score(self, record):
        raise NotImplementedError


class NullReward(Reward):
    """Default reward: scoring is not configured. Never invents a number."""

    name = "null"

    def score(self, record):
        return None


def apply_reward(trajectory_path, reward, *, output_path=None):
    """Score every record in a trajectory JSONL file.

    Writes records back (or to ``output_path``) with a ``reward`` block. A
    None score is stored as null, not 0.0.
    """
    source = Path(trajectory_path)
    target = Path(output_path) if output_path else source
    scored = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        record = json.loads(line)
        record["reward"] = {
            "schema_version": REWARD_SCHEMA_VERSION,
            "name": reward.name,
            "score": reward.score(record),
        }
        scored.append(json.dumps(record, sort_keys=True,
                                 ensure_ascii=False, allow_nan=False))
    target.write_text("\n".join(scored) + "\n", encoding="utf-8")
    return target


def load_reward(dotted_path):
    """Load a Reward subclass or zero-arg factory from ``module:attr``.

    The future business reward ships in its own module and stays out of the
    serving codebase. A bare class is instantiated with no args.
    """
    module_name, attr = dotted_path.split(":", 1)
    obj = getattr(importlib.import_module(module_name), attr)
    return obj() if isinstance(obj, type) else obj
