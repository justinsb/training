"""Evaluate a model (base or LoRA-tuned) on the held-out SQL eval set.

Scores by normalized exact match: whitespace/case-insensitive comparison of the
generated SQL against the reference, after stripping markdown fences. Crude but
objective — good enough to compare base vs fine-tuned.

Base model:  python scripts/eval.py --limit 100
Fine-tuned:  python scripts/eval.py --adapter outputs/sql-lora --limit 100
"""

import argparse
import json
import re
from pathlib import Path

from unsloth import FastModel  # must be imported before transformers/trl


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model response that may include fences or chatter."""
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    return text.strip()


def normalize(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.replace("'", '"')  # treat quote styles as equivalent
    return sql.lower()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter dir")
    ap.add_argument("--eval-file", default="data/eval.jsonl")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=None, help="Where to write predictions JSONL")
    args = ap.parse_args()

    # Loading the adapter dir directly loads base model + adapter together.
    model, tokenizer = FastModel.from_pretrained(
        args.adapter or args.model,
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastModel.for_inference(model)

    examples = []
    with open(args.eval_file) as f:
        for line in f:
            examples.append(json.loads(line))
    examples = examples[: args.limit]

    label = args.adapter or args.model
    out_path = Path(args.out or f"outputs/predictions-{Path(label).name}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    correct = 0
    with out_path.open("w") as out:
        for i, ex in enumerate(examples):
            prompt_messages = ex["messages"][:-1]
            gold = ex["messages"][-1]["content"]

            # Step 1: render the chat template to a plain string
            # (includes Gemma's <start_of_turn> markers and the <bos> token).
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # Step 2: tokenize it. Gemma 4's "tokenizer" is a multimodal
            # Processor whose first positional arg is `images`, so `text=`
            # must be passed by keyword. add_special_tokens=False because
            # the template already inserted <bos>.
            inputs = tokenizer(
                text=prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
            n_prompt_tokens = inputs["input_ids"].shape[1]
            response = tokenizer.decode(
                output_ids[0][n_prompt_tokens:], skip_special_tokens=True
            )

            pred = extract_sql(response)
            match = normalize(pred) == normalize(gold)
            correct += match

            out.write(json.dumps({
                "question": prompt_messages[-1]["content"],
                "gold": gold,
                "raw_response": response,
                "prediction": pred,
                "match": match,
            }) + "\n")

            if (i + 1) % 10 == 0:
                print(f"{i + 1}/{len(examples)}  accuracy so far: {correct / (i + 1):.1%}")

    print(f"\n=== {label} ===")
    print(f"exact match: {correct}/{len(examples)} = {correct / len(examples):.1%}")
    print(f"predictions -> {out_path}")


if __name__ == "__main__":
    main()
