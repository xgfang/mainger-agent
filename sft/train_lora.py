"""
sft/train_lora.py
=================
Phase 4 of the SFT pipeline: LoRA fine-tuning of Qwen2.5-1.5B-Instruct on
verified teacher traces.

Design choices:
  - LoRA (rank 32) on attention projections only. Rank 32 is large enough
    to learn the tool-use behavior, small enough to avoid catastrophic
    forgetting of general language ability.
  - bfloat16 throughout. Matches H100/A100 native precision.
  - Loss masked to assistant turns only. We don't want the model to learn
    to predict the user's message or the (deterministic) tool results.
  - max_seq_length=4096 to accommodate full multi-turn traces with 5+
    tool calls.

Run via Slurm on GreatLakes:
  sbatch sft/train_qwen.sbatch

Or directly (e.g. in a Jupyter session with GPU allocated):
  python sft/train_lora.py \
      --base_model Qwen/Qwen2.5-1.5B-Instruct \
      --data_path  sft/data/teacher_traces_complete.jsonl \
      --output_dir sft/checkpoints/qwen-1.5b-mainger
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


# --------------------------------------------------------------------------- #
# Data loading                                                                 #
# --------------------------------------------------------------------------- #
def load_traces_as_chat_dataset(path: str) -> Dataset:
    """Read teacher_traces.jsonl and return a HF Dataset with one column
    "messages" containing the chat-format conversation per example.

    Drops error rows (those that don't have a "messages" field).
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            msgs = obj.get("messages")
            if msgs is None:
                continue
            # Filter to the message fields TRL/HuggingFace expects
            cleaned = []
            for m in msgs:
                d = {"role": m["role"]}
                if m.get("content") is not None:
                    d["content"] = m["content"]
                if m.get("tool_calls"):
                    d["tool_calls"] = m["tool_calls"]
                if m.get("tool_call_id") is not None:
                    d["tool_call_id"] = m["tool_call_id"]
                cleaned.append(d)
            rows.append({"messages": cleaned})
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--data_path", required=True,
                   help="Path to teacher_traces_complete.jsonl")
    p.add_argument("--output_dir", required=True,
                   help="Directory to save the LoRA adapter and logs")

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Optimization
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=4)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--warmup_steps", type=int, default=0,
                   help="Fixed-step warmup. If 0, --warmup_ratio is used instead.")
    p.add_argument("--warmup_ratio", type=float, default=0.1,
                   help="Fraction of total steps used for LR warmup. Ignored if --warmup_steps > 0.")
    p.add_argument("--max_seq_length", type=int, default=4096)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--seed", type=int, default=42)

    # Eval split
    p.add_argument("--eval_fraction", type=float, default=0.05)

    return p.parse_args()


def main():
    args = parse_args()

    print(f"Base model: {args.base_model}")
    print(f"Data:       {args.data_path}")
    print(f"Output:     {args.output_dir}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    print("Loading base model in bf16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False  # avoid warnings during training

    # LoRA config: target attention projections only
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Dataset
    print(f"Loading traces from {args.data_path} ...")
    full_dataset = load_traces_as_chat_dataset(args.data_path)
    print(f"  {len(full_dataset)} traces loaded")

    # Train/eval split
    if args.eval_fraction > 0 and len(full_dataset) >= 20:
        split = full_dataset.train_test_split(
            test_size=args.eval_fraction, seed=args.seed
        )
        train_ds, eval_ds = split["train"], split["test"]
        print(f"  train: {len(train_ds)}, eval: {len(eval_ds)}")
    else:
        train_ds, eval_ds = full_dataset, None
        print(f"  train: {len(train_ds)}, eval: (none)")

    # Apply chat template to produce a "text" column. TRL 0.13's SFTTrainer
    # expects pre-formatted text rather than auto-applying chat templates
    # to a "messages" column.
    print("Applying chat template ...")
    def _apply_template(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)}
    train_ds = train_ds.map(_apply_template, remove_columns=["messages"])
    if eval_ds is not None:
        eval_ds = eval_ds.map(_apply_template, remove_columns=["messages"])
    print(f"  formatted; sample length (chars): {len(train_ds[0]['text'])}")

    # SFTConfig: TRL handles chat templating from `messages` automatically
    # when the tokenizer has a chat_template (Qwen2.5 does).
    cfg = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        max_seq_length=args.max_seq_length,
        lr_scheduler_type="cosine",
        report_to="none",
        seed=args.seed,
        # Note: assistant_only_loss=True would mask loss to assistant turns
        # but requires trl >= 0.14. With trl 0.13 we accept full-sequence
        # loss; impact is small at 91 optimizer steps with LoRA r=32.
        # Memory: gradient checkpointing recomputes activations during
        # backward, ~halving activation memory at ~20% slower wallclock.
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Eval settings. The default per_device_eval_batch_size of 8 OOMs
        # at our seq length because the logits tensor (batch x seq x vocab
        # in fp32) is huge; pin to 1 and skip storing logits to fit in 44 GB.
        per_device_eval_batch_size=1,
        prediction_loss_only=True,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_total_limit=3,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    print("Starting training ...")
    trainer.train()

    print(f"Saving final adapter to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
