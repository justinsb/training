"""GRPO-train the SFT adapter with the execution checker as a live reward.

The RL loop: each step, sample a group of completions per prompt from the
*current* policy, score each with exec_score.score_example (reward 1.0 if
its results match gold on all test databases, else 0.0), and reinforce
completions that beat their group's mean. Prompts where all completions get
the same reward produce zero advantage and no gradient — watch the
reward_std / frac_reward_zero_std logs to see how much of each batch is
alive. Given the model's sharpness, expect many dead groups.

Prompts are pre-screened: gold is run against itself through the checker,
and prompts whose gold is unexecutable or non-discriminative are dropped —
a broken reward is worse than no prompt.

Smoke test:  python scripts/train_grpo.py --max-steps 3
Real run:    python scripts/train_grpo.py
"""

import argparse

from unsloth import FastModel  # must be imported before transformers/trl

from datasets import Dataset, load_dataset
from trl import GRPOConfig, GRPOTrainer

from eval import extract_sql
from exec_score import score_example
from prepare_data import PROMPT_TEMPLATE

# Rows 0-299: eval split. 300-3299: SFT training. 3300-4799: DPO mining.
GRPO_ROWS_START = 4800
DPO_ROWS_START, DPO_ROWS_END = 3300, 4800


def build_band_prompts(band_file: str, seed: int) -> Dataset:
    """Train only on the inconsistency band, harvested from a DPO pair file.

    Mined pairs exist precisely where the model sampled both right and wrong
    answers — the prompts where GRPO groups have nonzero advantage. Pairs
    whose chosen equals the dataset gold are gold-augmented (all-wrong
    prompts, dead groups) and are skipped.
    """
    import json

    ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=seed)
    rows = ds.select(range(DPO_ROWS_START, DPO_ROWS_END))
    by_content = {}
    for i, row in enumerate(rows):
        uc = PROMPT_TEMPLATE.format(context=row["context"], question=row["question"])
        by_content[uc] = (row["answer"], i)

    records, unmatched = [], 0
    seen = set()
    for line in open(band_file):
        rec = json.loads(line)
        uc = rec["prompt"][0]["content"]
        if uc in seen:
            continue
        seen.add(uc)
        if uc not in by_content:
            unmatched += 1
            continue
        gold, i = by_content[uc]
        # NOTE: pairs without a "source" field can't be told apart reliably —
        # a dialect-trained model's correct completions are usually identical
        # to gold, so chosen==gold does NOT imply gold-augmented. Keep all
        # candidates and let --screen measure band membership live.
        records.append({
            "prompt": rec["prompt"],
            "gold": gold,
            "user_content": uc,
            "db_seed": i,
            "band_source": rec.get("source", "unknown"),
        })
    # Pair files written after the "source" field was added let us drop
    # gold-augmented (all-wrong) prompts here; untagged files keep everything
    # and rely on --screen to measure band membership live.
    if any(r["band_source"] == "mined" for r in records):
        mined = [r for r in records if r["band_source"] == "mined"]
        print(f"band candidates: {len(mined)} mined kept of {len(records)} "
              f"({unmatched} unmatched)")
        return mined
    print(f"band candidates: {len(records)} untagged ({unmatched} unmatched) "
          f"— consider --screen")
    return records


def build_prompts(num_prompts: int, seed: int) -> Dataset:
    ds = load_dataset("b-mc2/sql-create-context", split="train").shuffle(seed=seed)
    rows = ds.select(range(GRPO_ROWS_START, GRPO_ROWS_START + num_prompts))

    records, dropped = [], 0
    for i, row in enumerate(rows):
        user_content = PROMPT_TEMPLATE.format(
            context=row["context"], question=row["question"]
        )
        # Screen the reward itself: gold must match gold.
        sanity = score_example(
            {"question": user_content, "gold": row["answer"], "prediction": row["answer"]},
            seed=i,
        )
        if sanity != "match":
            dropped += 1
            continue
        records.append({
            "prompt": [{"role": "user", "content": user_content}],
            "gold": row["answer"],
            "user_content": user_content,
            "db_seed": i,
        })
    print(f"prompts: {len(records)} kept, {dropped} dropped by reward screening")
    return records


def screen_prompts(model, tokenizer, records, group_size, temperature):
    """Keep only prompts where the CURRENT policy gets mixed rewards.

    Live version of the band filter (DAPO-style dynamic sampling, one round):
    sample a full group per prompt; all-correct and all-wrong prompts produce
    zero advantage in GRPO, so only mixed prompts are worth training on.
    Costs one generation pass over the candidates (~4-5 s each).
    """
    kept = []
    for idx, rec in enumerate(records):
        prompt_text = tokenizer.apply_chat_template(
            rec["prompt"], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            text=prompt_text, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=temperature,
            num_return_sequences=group_size,
        )
        n_prompt_tokens = inputs["input_ids"].shape[1]
        wins = 0
        for seq in output_ids:
            sql = extract_sql(
                tokenizer.decode(seq[n_prompt_tokens:], skip_special_tokens=True)
            )
            verdict = score_example(
                {"question": rec["user_content"], "gold": rec["gold"], "prediction": sql},
                seed=rec["db_seed"],
            )
            wins += verdict == "match"
        if 0 < wins < group_size:
            kept.append(rec)
        if (idx + 1) % 25 == 0:
            print(f"screened {idx + 1}/{len(records)}, kept {len(kept)}")
    print(f"screening: {len(kept)}/{len(records)} prompts in the current band")
    return kept


def sql_reward(prompts, completions, gold, user_content, db_seed, **kwargs):
    """1.0 if the completion's SQL produces gold's results on all test DBs."""
    rewards = []
    for completion, g, uc, s in zip(completions, gold, user_content, db_seed):
        text = completion[0]["content"] if isinstance(completion, list) else completion
        verdict = score_example(
            {"question": uc, "gold": g, "prediction": extract_sql(text)}, seed=s
        )
        rewards.append(1.0 if verdict == "match" else 0.0)
    return rewards


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="outputs/sql-lora",
                    help="Start from the SFT adapter (or outputs/sql-dpo)")
    ap.add_argument("--num-prompts", type=int, default=400)
    ap.add_argument("--band-file", default=None,
                    help="DPO pair JSONL; use its prompts as band candidates "
                         "instead of fresh rows")
    ap.add_argument("--screen", action="store_true",
                    help="Sample a group per candidate prompt with the current "
                         "policy and keep only mixed-reward (band) prompts")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--group-size", type=int, default=8,
                    help="Completions sampled per prompt (GRPO group)")
    ap.add_argument("--temperature", type=float, default=1.2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.04,
                    help="KL penalty toward the reference model")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--output-dir", default="outputs/sql-grpo")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        args.adapter,
        max_seq_length=2048,
        load_in_4bit=True,
    )

    if args.band_file:
        records = build_band_prompts(args.band_file, args.seed)
    else:
        records = build_prompts(args.num_prompts, args.seed)

    if args.screen:
        records = screen_prompts(
            model, tokenizer, records, args.group_size, args.temperature
        )

    dataset = Dataset.from_list(records)

    FastModel.for_training(model)

    # The KL reference pass (beta > 0) toggles the adapter off, which breaks
    # gradient checkpointing's forward replay (CheckpointError). With beta=0
    # there is no reference pass, so checkpointing is safe — and worth ~5-10x
    # activation memory. Modern verifiable-reward recipes (DAPO, Dr. GRPO)
    # drop the KL term anyway; our held-out eval is the drift backstop.
    use_checkpointing = args.beta == 0.0
    if use_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    else:
        model.gradient_checkpointing_disable()

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[sql_reward],
        train_dataset=dataset,
        args=GRPOConfig(
            num_generations=args.group_size,
            temperature=args.temperature,
            # Our prompts (schema + question) tokenize to ~100-200 tokens;
            # 512 is generous and saves real activation memory vs 1024.
            max_prompt_length=512,
            max_completion_length=128,
            learning_rate=args.lr,
            beta=args.beta,
            # Micro-batch of 2 completions; accumulation reassembles complete
            # groups (effective batch 16 = 2 groups of 8 — must stay divisible
            # by group size). 8 completions per backward OOMs on 24 GB with
            # checkpointing off.
            per_device_train_batch_size=4 if use_checkpointing else 2,
            gradient_accumulation_steps=4 if use_checkpointing else 8,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            gradient_checkpointing=use_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            warmup_steps=5,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            logging_steps=1,
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved GRPO adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
