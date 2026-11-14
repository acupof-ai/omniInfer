#!/usr/bin/env python3
"""Pin the phase-6 SGLang copy patch without a GPU or a real checkout."""

import importlib.util
from pathlib import Path
import sys
import tempfile

MODULE_PATH = Path(__file__).with_name("sglang_nan_injection.py")
spec = importlib.util.spec_from_file_location("sglang_nan_injection", MODULE_PATH)
injection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(injection)

SOURCE = '''import dataclasses
import logging

logger = logging.getLogger(__name__)


class LogitsProcessor:
    def _get_logits(self, hidden_states, lm_head, logits_metadata,
                    embedding_bias=None):
        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )

        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
        return logits
'''


def main():
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "sglang/srt/layers/logits_processor.py"
        target.parent.mkdir(parents=True)
        target.write_text(SOURCE)

        injection.patch(target)
        text = target.read_text()

        checks = {
            "import_os": "\nimport os\n" in text,
            "hook_defined": "def _phase6_maybe_inject_nan(" in text,
            "trigger_env": 'os.environ.get("PHASE6_NAN_TRIGGER", "")' in text,
            "decode_gate": "if logits_metadata.forward_mode.is_decode():" in text,
            "consumed_once": "os.replace(_PHASE6_TRIGGER" in text,
            "nan_value": 'float("nan")' in text,
            "original_logits_call": (
                "logits = self._compute_lm_head(hidden_states, lm_head, "
                "embedding_bias)" in text),
        }

        # Patching twice must be a no-op, not a second insertion.
        before = target.read_text()
        injection.patch(target)
        checks["idempotent"] = target.read_text() == before

        for name, ok in checks.items():
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
