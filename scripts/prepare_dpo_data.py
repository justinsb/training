"""Build DPO preference pairs by sampling the SFT model and execution-checking.

For each fresh prompt (unseen during SFT training), sample K completions at
temperature from the fine-tuned adapter, then label each one with the
execution checker (exec_score). A prompt yields a pair only when it produces
at least one verified-correct and one verified-wrong completion:

  chosen   = a completion whose results match gold on all test databases
  rejected = a completion whose results differ

Prompts where all K samples agree (all correct or all wrong) yield no pair —
preference signal lives in the inconsistency band. Prompts whose gold query
is unexecutable or non-discriminative are skipped: unreliable labels are
worse than no labels.

Output is TRL DPOTrainer conversational format:
  {"prompt": [...user messages...], "chosen": [assistant msg], "rejected": [assistant msg]}

Usage: python scripts/prepare_dpo_data.py --num-prompts 500
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from unsloth import FastModel  # must be imported before transformers/trl

import torch
from datasets import load_dataset

from eval import extract_sql
from exec_score import score_example
from prepare_data import PROMPT_TEMPLATE

# Rows 0-299 of the seed-42 shuffle are the eval split, 300-3299 the SFT
# training split (see prepare_data.py). DPO prompts start after both.
SFT_ROWS_USED = 3300


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="outputs/sql-lora")
    ap.add_argument("--num-prompts", type=int, default=500)
    ap.add_argument("--samples", type=int, default=8, help="completions per prompt")
    ap.add_argument("--temperature", type=float, default=1.0)
    # Default 1.0 = no nucleus truncation. On a sharp post-SFT distribution,
    # any top_p below the top token's probability (~0.98 here) deletes the
    # entire tail and makes sampling greedy regardless of temperature.
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data/dpo.jsonl"))
    # For all-wrong prompts, emit a pair using the dataset's gold answer as
    # chosen. Off-policy on the chosen side (the model wouldn't say it) but
    # the rejected side — where on-policyness matters most — stays honest.
    ap.add_argument("--gold-chosen", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    model, tokenizer = FastModel.from_pretrained(
        args.adapter, max_seq_length=2048, load_in_4bit=True
    )
    FastModel.for_inference(model)

    ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=args.seed)
    rows = ds.select(range(SFT_ROWS_USED, SFT_ROWS_USED + args.num_prompts))

    stats = Counter()
    success_histogram = Counter()
    distinct_histogram = Counter()  # sharpness diagnostic: unique SQL per prompt

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as out:
        for i, row in enumerate(rows):
            user_content = PROMPT_TEMPLATE.format(
                context=row["context"], question=row["question"]
            )
            prompt_text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(
                text=prompt_text, return_tensors="pt", add_special_tokens=False
            ).to(model.device)

            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.samples,
            )
            n_prompt_tokens = inputs["input_ids"].shape[1]
            candidates = [
                extract_sql(
                    tokenizer.decode(seq[n_prompt_tokens:], skip_special_tokens=True)
                )
                for seq in output_ids
            ]

            # Label every distinct candidate with the execution checker.
            verdicts = {}
            for sql in dict.fromkeys(candidates):  # dedupe, keep order
                verdicts[sql] = score_example(
                    {"question": user_content, "gold": row["answer"], "prediction": sql},
                    seed=i,
                )
            distinct_histogram[len(verdicts)] += 1

            if any(v == "gold_error" for v in verdicts.values()):
                stats["skipped_gold_error"] += 1
                continue
            if any(v == "non_discriminative" for v in verdicts.values()):
                stats["skipped_non_discriminative"] += 1
                continue

            correct = [c for c in candidates if verdicts[c] == "match"]
            wrong = [c for c in candidates if verdicts[c] == "mismatch"]
            success_histogram[len(correct)] += 1

            if not correct:
                stats["all_wrong"] += 1
                if args.gold_chosen:
                    stats["paired_gold_chosen"] += 1
                    out.write(json.dumps({
                        "prompt": [{"role": "user", "content": user_content}],
                        "chosen": [{"role": "assistant", "content": row["answer"]}],
                        "rejected": [{"role": "assistant", "content": wrong[0]}],
                    }) + "\n")
                continue
            if not wrong:
                stats["all_correct"] += 1
                continue

            stats["paired"] += 1
            out.write(json.dumps({
                "prompt": [{"role": "user", "content": user_content}],
                "chosen": [{"role": "assistant", "content": correct[0]}],
                "rejected": [{"role": "assistant", "content": wrong[0]}],
            }) + "\n")

            if (i + 1) % 25 == 0:
                print(f"{i + 1}/{len(rows)} prompts, {stats['paired']} pairs so far")

    print(f"\n=== pair yield ===")
    for key in ("paired", "paired_gold_chosen", "all_correct", "all_wrong",
                "skipped_gold_error", "skipped_non_discriminative"):
        print(f"{key}: {stats[key]}")

    print(f"\n=== success histogram (correct out of {args.samples} samples) ===")
    for k in range(args.samples + 1):
        bar = "#" * success_histogram[k]
        print(f"{k}/{args.samples}: {success_histogram[k]:4d} {bar}")

    n_prompts = sum(distinct_histogram.values())
    mean_distinct = (
        sum(k * v for k, v in distinct_histogram.items()) / n_prompts if n_prompts else 0
    )
    print(f"\n=== distinct candidates per prompt (sampling sharpness) ===")
    print(f"mean {mean_distinct:.2f} distinct out of {args.samples} samples "
          f"(1.0 = fully collapsed, {args.samples}.0 = fully diverse)")
    for k in range(1, args.samples + 1):
        bar = "#" * distinct_histogram[k]
        print(f"{k}: {distinct_histogram[k]:4d} {bar}")

    print(f"\nwrote {stats['paired']} pairs -> {args.out}")


if __name__ == "__main__":
    main()
