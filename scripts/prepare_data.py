"""Download the b-mc2/sql-create-context dataset and write train/eval JSONL files.

Each output line is a chat-format record:
  {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}

The eval split is held out and never seen during training.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

PROMPT_TEMPLATE = """Given this database schema:

{context}

Write a SQL query to answer the following question. Reply with only the SQL query, no explanation or formatting.

Question: {question}"""


def to_chat(row: dict) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(
                    context=row["context"], question=row["question"]
                ),
            },
            {"role": "assistant", "content": row["answer"]},
        ]
    }


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(to_chat(row)) + "\n")
    print(f"wrote {len(rows)} examples -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, default=3000)
    ap.add_argument("--eval-size", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("data"))
    args = ap.parse_args()

    ds = load_dataset("b-mc2/sql-create-context", split="train")
    print(f"loaded {len(ds)} rows")

    ds = ds.shuffle(seed=args.seed)
    eval_rows = ds.select(range(args.eval_size))
    train_rows = ds.select(range(args.eval_size, args.eval_size + args.train_size))

    write_jsonl(args.out_dir / "train.jsonl", train_rows)
    write_jsonl(args.out_dir / "eval.jsonl", eval_rows)


if __name__ == "__main__":
    main()
