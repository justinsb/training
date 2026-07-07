# mightyfine — fine-tuning Gemma 4 E4B on text-to-SQL

A learning project: QLoRA fine-tune of `google/gemma-4-E4B-it` (Apache 2.0) on
[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
(~78k question+schema → SQL examples, CC-BY-4.0), using Unsloth + TRL on a
single RTX 4090.

## Workflow

```bash
# 1. Prepare data: downloads the dataset, writes chat-format JSONL
#    (3000 train / 300 held-out eval by default)
uv run scripts/prepare_data.py

# 2. Baseline: how well does the *base* model do? (~100 examples)
uv run scripts/eval.py --limit 100

# 3. Train: QLoRA, ~1 epoch. First run downloads model weights (~15 GB).
uv run scripts/train.py                  # full run
uv run scripts/train.py --max-steps 20   # smoke test

# 4. Eval the fine-tuned adapter and compare against the baseline
uv run scripts/eval.py --adapter outputs/sql-lora --limit 100
```

Predictions land in `outputs/predictions-*.jsonl` — diff the two files to see
*what* changed, not just the score.

## Knobs to experiment with (`scripts/train.py`)

| Flag | Default | Notes |
|---|---|---|
| `--lr` | 2e-4 | LoRA default; halve it if loss is spiky |
| `--epochs` | 1 | 2–3 helps until eval accuracy stops improving |
| `--rank` | 16 | LoRA capacity; try 8 or 32 |
| `--train-size` (prepare_data) | 3000 | more data ≈ better, slower |

## Scoring caveat

`eval.py` uses normalized exact match — a semantically-correct query written
differently (`SELECT count(*)` vs `SELECT COUNT(id)`) scores 0. That penalizes
the base model most (it phrases SQL its own way), which is partly the point:
the fine-tune teaches the dataset's exact style. A better metric would execute
both queries against a real schema and compare results.
