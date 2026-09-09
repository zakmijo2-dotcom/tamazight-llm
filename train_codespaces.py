#!/usr/bin/env python3
"""
Tamazight (Berber) LLM Training Script for GitHub Codespaces & Cloud GPUs
Supports: CUDA GPU (with bf16/fp16) or Multi-core CPU
Base Model: HuggingFaceTB/SmolLM2-135M-Instruct
"""

import os
import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tamazight LLM in Codespaces")
    parser.add_argument("--base-model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct", help="Base model")
    parser.add_argument("--data-file", type=str, default="tamazight_train.jsonl", help="Path to jsonl training data")
    parser.add_argument("--output-dir", type=str, default="tamazight-smollm-135m-final", help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-samples", type=int, default=50000, help="Max samples to train on")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repo name (e.g. username/tamazight-smollm-135m)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Tamazight LLM Training ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device == 'cuda' else os.cpu_count() + ' CPU cores'})")
    print(f"Base Model: {args.base_model}")
    print(f"Data File: {args.data_file}")

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Dataset
    print("\nLoading and formatting dataset...")
    samples = []
    with open(args.data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= args.max_samples:
                break
            d = json.loads(line)
            prompt = d.get("prompt", "")
            if prompt:
                samples.append({"text": prompt})

    dataset = Dataset.from_list(samples)
    print(f"Loaded {len(dataset):,} training samples.")

    def tokenize_func(examples):
        return tokenizer(examples["text"], max_length=128, truncation=True)

    tokenized_dataset = dataset.map(tokenize_func, batched=True, remove_columns=["text"])

    # 3. Load Model with LoRA
    print("\nLoading model into memory...")
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else (torch.float16 if device == "cuda" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training Arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        logging_steps=50,
        save_strategy="epoch",
        fp16=(device == "cuda" and dtype == torch.float16),
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    print("\nStarting Training...")
    trainer.train()

    # 5. Merge LoRA and Save
    print("\nMerging LoRA adapters into base weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved successfully to: {args.output_dir}")

    # 6. Optional: Push to Hugging Face
    if args.push_to_hub and args.hf_repo:
        print(f"\nPushing model to Hugging Face Hub: {args.hf_repo}...")
        merged_model.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        print("Model successfully published on Hugging Face!")

if __name__ == "__main__":
    main()
EOF
