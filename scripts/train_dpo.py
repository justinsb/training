"""DPO-train the SFT adapter on execution-verified preference pairs.

Continues training the LoRA adapter produced by train.py, using pairs from
prepare_dpo_data.py. The DPO loss pushes the log-prob margin between each
chosen and rejected completion apart, leashed to a reference model by beta.

Reference-model note: with a LoRA policy, TRL gets the reference for free by
disabling the adapter (ref_model=None below) — so the reference is the BASE
model, not the SFT model. This is the standard memory-saving trick; know
that it subtly changes what beta anchors to.

Watch during training (logged every step):
  rewards/margins   should grow — the pair gap is widening
  logps/chosen      often *falls* even as margins grow — the famous DPO
                    dynamic: mass leaks to sequences outside both slots

Usage: python scripts/train_dpo.py --data data/dpo.jsonl [data/more.jsonl ...]
"""

import argparse
import json

from unsloth import FastModel  # must be imported before transformers/trl

try:  # some unsloth versions need explicit DPO patching before TRL import
    from unsloth import PatchDPOTrainer
    PatchDPOTrainer()
except ImportError:
    pass

from datasets import Dataset
from trl import DPOConfig, DPOTrainer


def load_pairs(paths: list) -> Dataset:
    """Concatenate pair files, deduping by prompt (first file wins)."""
    seen = set()
    records = []
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                key = json.dumps(rec["prompt"], sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    rec.pop("source", None)  # provenance metadata, not a training column
                    records.append(rec)
    return Dataset.from_list(records)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", default="outputs/sql-lora")
    # action="extend" accepts both `--data a b c` and repeated `--data a --data b`
    ap.add_argument("--data", nargs="+", action="extend", default=None)
    ap.add_argument("--output-dir", default="outputs/sql-dpo")
    ap.add_argument("--beta", type=float, default=0.1,
                    help="KL leash: lower = more drift allowed from reference")
    ap.add_argument("--lr", type=float, default=5e-6,
                    help="DPO wants a much lower LR than SFT (5e-6 vs 2e-4)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="If > 0, stop early (overrides --epochs); for smoke tests")
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        args.adapter,  # loads base + SFT adapter; adapter stays trainable
        max_seq_length=2048,
        load_in_4bit=True,
    )
    FastModel.for_training(model)

    # Gradient checkpointing recomputes forward passes during backward, but
    # DPO's free-reference trick toggles the adapter off for the reference
    # pass — the recomputed tensors don't match and backward crashes with a
    # CheckpointError. We don't need the memory savings at this scale, so
    # turn checkpointing off entirely.
    model.gradient_checkpointing_disable()

    data_files = args.data or ["data/dpo.jsonl"]
    dataset = load_pairs(data_files)
    print(f"training on {len(dataset)} pairs from {data_files}")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # LoRA trick: reference = model with adapter disabled
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=DPOConfig(
            beta=args.beta,
            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            gradient_checkpointing=False,
            warmup_steps=2,
            optim="adamw_8bit",
            lr_scheduler_type="linear",
            logging_steps=1,  # tiny dataset: watch every step
            max_length=2048,
            max_prompt_length=1024,
            seed=42,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved DPO adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
