#!/usr/bin/env python3
"""Patch an independent SGLang copy to inject one NaN decode step.

The edit goes into <copy>/sglang/srt/layers/logits_processor.py:

1. import os
2. add _phase6_maybe_inject_nan(): consume a trigger file once, then multiply
   the final hidden states by NaN on the GPU, so the real LM-head matmul emits
   NaN logits;
3. call it in LogitsProcessor._get_logits() for decode forwards only.

The trigger is a file path in PHASE6_NAN_TRIGGER. The file is renamed to
<path>.fired on first use, so a server lifetime sees at most one injection.
With the file absent, the code path is a no-op.

Run `--check` to confirm an existing copy carries the hook. Runs on the host
that owns the copy; it never edits /sgl-workspace/sglang.
"""

import argparse
from pathlib import Path
import sys

HELPER = '''
# --- phase6 fault injection (independent copy only) -------------------------
# Consume the trigger file once and corrupt the final hidden states on the GPU.
# The real LM-head matmul then produces NaN logits for the sampler's real
# NaN detection. Absent trigger file = exact original behaviour.
_PHASE6_TRIGGER = os.environ.get("PHASE6_NAN_TRIGGER", "")
_phase6_injected = False


def _phase6_maybe_inject_nan(hidden_states):
    global _phase6_injected
    if not _PHASE6_TRIGGER or _phase6_injected:
        return hidden_states
    try:
        if not os.path.exists(_PHASE6_TRIGGER):
            return hidden_states
        os.replace(_PHASE6_TRIGGER, _PHASE6_TRIGGER + ".fired")
    except OSError:
        return hidden_states
    _phase6_injected = True
    logger.warning(
        "PHASE6_NAN_INJECT trigger=%s hidden_shape=%s dtype=%s device=%s",
        _PHASE6_TRIGGER, tuple(hidden_states.shape), hidden_states.dtype,
        hidden_states.device,
    )
    return hidden_states * float("nan")
# --- end phase6 fault injection ---------------------------------------------
'''

CALL = """
        if logits_metadata.forward_mode.is_decode():
            hidden_states = _phase6_maybe_inject_nan(hidden_states)
"""

ANCHOR = """        hidden_states, local_hidden_states = self._gather_dp_attn_hidden_states(
            hidden_states, logits_metadata
        )

        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)
"""


def patch(target: Path) -> None:
    text = target.read_text()
    if "_phase6_maybe_inject_nan" in text:
        print(f"already patched: {target}")
        return
    if "\nimport os\n" not in text:
        if "import dataclasses\n" not in text:
            raise SystemExit("import anchor missing")
        text = text.replace("import dataclasses\n",
                            "import dataclasses\nimport os\n", 1)
    if "logger = logging.getLogger(__name__)\n" not in text:
        raise SystemExit("logger anchor missing")
    text = text.replace("logger = logging.getLogger(__name__)\n",
                        "logger = logging.getLogger(__name__)\n" + HELPER, 1)
    if text.count(ANCHOR) != 1:
        raise SystemExit(f"call anchor count {text.count(ANCHOR)}")
    text = text.replace(ANCHOR, ANCHOR.replace(
        "        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)\n",
        CALL + "\n        logits = self._compute_lm_head(hidden_states, lm_head, embedding_bias)\n"), 1)
    target.write_text(text)
    print(f"patched {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copy", required=True,
                        help="SGLang checkout copy that contains sglang/")
    parser.add_argument("--check", action="store_true",
                        help="only verify the hook is present")
    args = parser.parse_args()
    target = Path(args.copy) / "sglang/srt/layers/logits_processor.py"
    if not target.is_file():
        raise SystemExit(f"not a SGLang copy: {target}")
    if args.check:
        present = "_phase6_maybe_inject_nan" in target.read_text()
        print(f"hook_present={present} file={target}")
        return 0 if present else 1
    patch(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
