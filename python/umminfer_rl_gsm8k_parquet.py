# SPDX-License-Identifier: Apache-2.0
"""Convert the hf-mirror openai/gsm8k parquet split to upstream-shaped JSONL.

The GitHub raw JSONL is unreliably slow on some hosts; the HF parquet
conversion branch serves the same split over a fast CDN. This emits only the
question/answer columns so the output matches the upstream file shape consumed
by ``umminfer_rl_gsm8k_data.parse_gsm8k``. pyarrow is imported in ``main`` so
importing the RL modules never needs it.
"""

import argparse
import json


def convert(parquet_path, jsonl_path):
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet_path).to_pylist()
    with open(jsonl_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(
                {"question": row["question"], "answer": row["answer"]},
                ensure_ascii=False) + "\n")
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet_path")
    parser.add_argument("jsonl_path")
    args = parser.parse_args(argv)
    print(f"rows={convert(args.parquet_path, args.jsonl_path)}")


if __name__ == "__main__":
    main()
