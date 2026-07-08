# mightyfine — post-training Gemma 4 E4B on text-to-SQL

A learning project covering the modern post-training arc — SFT → DPO → GRPO —
on `google/gemma-4-E4B-it` (Apache 2.0), using
[b-mc2/sql-create-context](https://huggingface.co/datasets/b-mc2/sql-create-context)
(~78k question+schema → SQL examples, CC-BY-4.0), with Unsloth + TRL on a
single RTX 4090.

## Results

Execution-verified accuracy on a held-out split (`exec_score.py`, see below):

| Model | Exec match |
|---|---|
| Base Gemma 4 E4B | 26.8% |
| + SFT (QLoRA, 3k examples, 1 epoch) | ~78% |
| + DPO (~290 execution-verified pairs) | 83.1% |
| + GRPO (execution checker as live reward) | 82.4–83.3% |

All three post-SFT methods plateau at ~83%: they each fix the *inconsistency
band* (prompts the model sometimes gets right) and none can reach the
consistent failures, which are dominated by label noise and literal formats
unknowable without seeing table contents. The ceiling is informational, not
algorithmic.

## Stage 1: SFT

```bash
uv run scripts/prepare_data.py           # dataset -> chat-format JSONL (3000 train / 300 held-out eval)
uv run scripts/eval.py --limit 100       # baseline: how good is the base model?
uv run scripts/train.py                  # QLoRA fine-tune (~15 min); --max-steps 20 to smoke test
uv run scripts/eval.py --adapter outputs/sql-lora --limit 300
```

## Scoring: string match vs execution

`eval.py` scores by normalized exact match — cheap but misleading in both
directions (penalizes equivalent SQL, forgives literal-case errors).
`exec_score.py` re-scores any predictions file offline by *executing* gold
and predicted SQL against synthetic SQLite databases (boundary values around
every numeric literal; gold-empty examples excluded as non-discriminative):

```bash
python3 scripts/exec_score.py outputs/predictions-sql-lora.jsonl
```

The same `score_example()` function is reused as the DPO pair labeler and
the GRPO reward — one checker, three jobs. Audit it before trusting it.

## Stage 2: DPO

```bash
# Mine preference pairs: sample the SFT model 8x per prompt, execution-check
# each sample, pair verified-correct vs verified-wrong. --gold-chosen also
# pairs dataset gold against the model's wrong answer on all-wrong prompts.
uv run scripts/prepare_dpo_data.py --num-prompts 1500 --temperature 1.2 --gold-chosen --out data/dpo-large.jsonl

uv run scripts/train_dpo.py --data data/dpo-large.jsonl --batch-size 1 --grad-accum 8
uv run scripts/eval.py --adapter outputs/sql-dpo --limit 300
```

What to expect: the SFT model is extremely sharp (mean ~1.3 distinct
completions per 8 samples even at temperature 1.2), so pair yield is ~9% of
prompts. In the training logs, watch `rewards/accuracies` climb from below
0.5 (within the band, the model's confident answer is usually the wrong one)
and note that gold-augmented pairs (chosen log-prob −10 to −21) mostly fail
to flip — off-policy chosen is weak signal.

## Stage 3: GRPO

```bash
# Random prompts: ~85% of groups are degenerate (all samples same reward -> zero advantage)
uv run scripts/train_grpo.py --beta 0

# Band-trained (DAPO-style): reuse DPO-mined prompts as candidates and
# re-screen against the current policy, keeping only mixed-reward prompts
uv run scripts/train_grpo.py --beta 0 --band-file data/dpo-large.jsonl --screen --epochs 3

uv run scripts/eval.py --adapter outputs/sql-grpo --limit 300
```

`--beta 0` drops the KL reference pass, which both matches modern
verifiable-reward recipes and re-enables gradient checkpointing (the
reference pass toggles the LoRA adapter off, which breaks checkpointing's
forward replay). Watch `frac_reward_zero_std` (dead-group fraction) and the
`reward` mean in the logs.

## Hard-won gotchas

- transformers v5: `apply_chat_template` returns a string by default; Gemma 4's
  "tokenizer" is a multimodal Processor (`text=` must be a keyword arg).
- Gemma 4 chat template uses `<|turn>...<turn|>` markers (not Gemma 2/3's
  `<start_of_turn>`); wrong markers in `train_on_responses_only` silently mask
  everything → `num_samples=0`.
- `top_p` below the top token's probability makes sampling greedy regardless
  of temperature. On a post-SFT model the top token is often >0.98.
- Micro-batch size is the first OOM dial: activations and logits scale with
  tokens-in-flight, not trainable parameter count.
- Always read the actual predictions before trusting a metric; our first
  string metric under-read the base model by 42 points (quote style), and our
  first execution checker over-read it (empty-result false equivalences).
