# SPDX-License-Identifier: Apache-2.0
"""Real GSM8K data for the RL architecture validation.

Source: the upstream openai/grade-school-math JSONL (real grade-school math
problems with human reference answers). No problem is synthesized. Each item
keeps its question and the numeric gold answer extracted from the ``####``
marker. The SHA-256 of the downloaded bytes is recorded in the run manifest so
a silent upstream change is detectable.
"""

import hashlib
import json
import urllib.request

GSM8K_TEST_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    "master/grade_school_math/data/test.jsonl"
)

PROMPT_TEMPLATE = (
    "Solve the grade-school math problem in at most three short sentences. "
    "Do not write headings or LaTeX. Your last line must have the exact form "
    '"#### <integer>", as in this example:\n'
    "She has 2 times 4 apples, so 8 apples in total.\n"
    "#### 8\n\n"
    "Question: {question}")


def build_messages(question):
    """The exact chat messages every rollout and evaluation sends."""
    return [{"role": "user",
             "content": PROMPT_TEMPLATE.format(question=question)}]


def download_gsm8k(url=GSM8K_TEST_URL, *, timeout=600):
    """Return the raw JSONL bytes of the GSM8K test split.

    The file is ~0.7-1 MB but slow mirrors can take several minutes; a short
    timeout truncates the last JSONL line and makes parsing fail loudly.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def parse_gsm8k(raw_bytes):
    """Parse GSM8K JSONL bytes into question/gold items."""
    items = []
    for line in raw_bytes.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        gold_text = row["answer"].split("####", 1)[1].strip()
        items.append({
            "question": row["question"],
            "gold_text": gold_text,
            "gold_answer": int(gold_text.replace(",", "")),
        })
    return items


def data_fingerprint(raw_bytes, source=GSM8K_TEST_URL):
    return {
        "source": source,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "byte_count": len(raw_bytes),
        "total_items": len(parse_gsm8k(raw_bytes)),
    }
