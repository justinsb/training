"""QLoRA fine-tune Gemma 4 E4B on the prepared SQL dataset with Unsloth + TRL.

Quick smoke test:   python scripts/train.py --max-steps 20
Full run (1 epoch): python scripts/train.py
"""

import argparse

from unsloth import FastModel  # must be imported before transformers/trl

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--train-file", default="data/train.jsonl")
    ap.add_argument("--output-dir", default="outputs/sql-lora")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument(
        "--max-steps", type=int, default=-1,
        help="If > 0, stop after this many steps (overrides --epochs); use for smoke tests",
    )
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(
        args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
    )

    model = FastModel.get_peft_model(
        model,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    dataset = load_dataset("json", data_files=args.train_file, split="train")

    def to_text(example: dict) -> dict:
        return {
            "text": tokenizer.apply_chat_template(
                example["messages"], tokenize=False, add_generation_prompt=False
            )
        }

    dataset = dataset.map(to_text, remove_columns=dataset.column_names)
    print("--- formatted example ---")
    print(dataset[0]["text"])
    print("-------------------------")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
            warmup_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            logging_steps=5,
            seed=42,
            output_dir=args.output_dir,
            report_to="none",
        ),
    )

    # Mask the prompt tokens so loss is computed only on the assistant's SQL,
    # not on the schema/question we feed in. These marker strings must match
    # the model's chat template exactly (Gemma 4 uses <|turn>...<turn|>;
    # Gemma 2/3 used <start_of_turn>...<end_of_turn>) — if they don't match,
    # everything gets masked and the trainer sees an empty dataset.
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    trainer.train()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"saved LoRA adapter -> {args.output_dir}")


if __name__ == "__main__":
    main()
