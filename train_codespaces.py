#!/usr/bin/env python3
"""
Tamazight (Berber) LLM Training Pipeline
Optimized for GitHub Codespaces, Cloud GPUs, and local workstations.
Base Model: HuggingFaceTB/SmolLM2-135M-Instruct
"""

import os
import sys
import math
import json
import random
import argparse
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import LoraConfig, get_peft_model, TaskType

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tamazight Latin LLM")
    parser.add_argument("--preset", type=str, default=None, choices=["cpu-fast", "cpu-standard", "gpu-full"],
                        help="Training profile preset: 'cpu-fast' (~10 mins on CPU), 'cpu-standard' (~30 mins on CPU), 'gpu-full' (all 126k on GPU)")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="Base model ID (e.g. Qwen/Qwen2.5-0.5B-Instruct or HuggingFaceTB/SmolLM2-135M-Instruct)")
    default_data = "data/multilingual_chat.jsonl" if os.path.exists("data/multilingual_chat.jsonl") else "tamazight_train.jsonl"
    parser.add_argument("--data-file", type=str, default=default_data, help=f"Path to jsonl training corpus (default: {default_data})")
    parser.add_argument("--output-dir", type=str, default="tamazight-multilingual-llm-final", help="Final model output directory")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Optional directory to save LoRA adapter separately")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size per device")
    parser.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Peak learning rate")
    parser.add_argument("--warmup-steps", type=int, default=10, help="Warmup steps for lr schedule")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum samples to use (None for all)")
    parser.add_argument("--val-split", type=float, default=0.05, help="Validation set fraction (e.g. 0.05 for 5%%)")
    parser.add_argument("--max-length", type=int, default=None, help="Maximum sequence token length")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument("--mask-prompt", action="store_true", default=True, help="Mask user prompt tokens from loss calculation (SFT)")
    parser.add_argument("--no-mask-prompt", dest="mask_prompt", action="store_false", help="Disable prompt masking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--push-to-hub", action="store_true", help="Push final model to Hugging Face Hub")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face repository name")
    return parser.parse_args()

def prepare_balanced_samples(data_file, max_samples=None, seed=42):
    """Categorizes and balances samples to ensure high diversity."""
    print(f"Reading and balancing corpus from: {data_file}...")
    categories = {
        "dialogue": [],
        "proverbs": [],
        "wikipedia": [],
        "other": [],
        "translation": []
    }

    total_read = 0
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            total_read += 1
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                p = d.get("prompt", "")
                if not p:
                    continue
                if "Azul" in p or "amek" in p.lower():
                    categories["dialogue"].append(p)
                elif any(k in p for k in ["Asenfali", "Aserwes", "Inzi", "lemɛun"]):
                    categories["proverbs"].append(p)
                elif "wikipedia" in p.lower() or "asebter" in p.lower():
                    categories["wikipedia"].append(p)
                elif "Tasuqilt ar tmaziɣt" in p or "Tasuqilt ar tefransist" in p:
                    categories["translation"].append(p)
                else:
                    categories["other"].append(p)
            except json.JSONDecodeError:
                continue

    random.seed(seed)
    for cat in categories:
        random.shuffle(categories[cat])

    if max_samples is None or max_samples >= total_read:
        all_samples = []
        for cat in categories:
            all_samples.extend(categories[cat])
        random.shuffle(all_samples)
        return all_samples

    # Balanced allocation
    selected = []
    # Take high value categories first
    selected.extend(categories["dialogue"][:min(len(categories["dialogue"]), int(max_samples * 0.35))])
    selected.extend(categories["proverbs"][:min(len(categories["proverbs"]), int(max_samples * 0.15))])
    selected.extend(categories["wikipedia"][:min(len(categories["wikipedia"]), int(max_samples * 0.05))])
    selected.extend(categories["other"][:min(len(categories["other"]), int(max_samples * 0.20))])

    # Fill remainder with translations
    remaining = max_samples - len(selected)
    if remaining > 0:
        selected.extend(categories["translation"][:remaining])

    random.shuffle(selected)
    print(f"Curated {len(selected):,} diverse samples (Dialogues: {len(categories['dialogue'])}, Culture: {len(categories['proverbs'])}, Contextual: {len(categories['other'])}, Translations: {remaining})")
    return selected

def prepare_dataset(raw_samples, tokenizer, max_length, mask_prompt=True):
    delimiter = "<|im_start|>assistant\n"
    features = []

    for text in raw_samples:
        if mask_prompt and delimiter in text:
            parts = text.split(delimiter, 1)
            prefix = parts[0] + delimiter
            for idx in range(min(prefix_len, len(labels))):
                labels[idx] = -100

            # Ensure at least some active labels exist
            if not any(lbl != -100 for lbl in labels):
                labels = list(input_ids)
        else:
            enc = tokenizer(text, max_length=max_length, truncation=True)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            labels = list(input_ids)

        features.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        })

    return Dataset.from_list(features)

def main():
    args = parse_args()
    set_seed(args.seed)

    has_cuda = torch.cuda.is_available()
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    if has_cuda:
        device = "cuda"
        device_name = torch.cuda.get_device_name(0)
    elif has_mps:
        device = "mps"
        device_name = "Apple Silicon MPS"
    else:
        device = "cpu"
        num_cores = os.cpu_count() or 2
        torch.set_num_threads(num_cores)
        device_name = f"{num_cores} CPU cores"

    # Presets resolution
    preset = args.preset
    if preset is None:
        if device == "cpu":
            preset = "cpu-fast"
        else:
            preset = "gpu-full"

    if preset == "cpu-fast":
        epochs = args.epochs or 1
        batch_size = args.batch_size or 4
        grad_accum = args.grad_accum or 2
        max_length = args.max_length or 64
        max_samples = args.max_samples if args.max_samples is not None else 3000
        lora_modules = ["q_proj", "v_proj"]
    elif preset == "cpu-standard":
        epochs = args.epochs or 1
        batch_size = args.batch_size or 4
        grad_accum = args.grad_accum or 2
        max_length = args.max_length or 72
        max_samples = args.max_samples if args.max_samples is not None else 7500
        lora_modules = ["q_proj", "v_proj"]
    elif preset == "gpu-full":
        epochs = args.epochs or 3
        batch_size = args.batch_size or 16
        grad_accum = args.grad_accum or 2
        max_length = args.max_length or 160
        max_samples = args.max_samples  # None means all 126k
        lora_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    else:
        epochs = args.epochs or 3
        batch_size = args.batch_size or 8
        grad_accum = args.grad_accum or 2
        max_length = args.max_length or 128
        max_samples = args.max_samples
        lora_modules = ["q_proj", "v_proj"]

    print("=" * 65)
    print("  ⵣ Tamazight (Berber) LLM Optimized Training Pipeline ⵣ")
    print("=" * 65)
    print(f"Hardware:    {device} ({device_name})")
    print(f"Preset:      {preset.upper()}")
    print(f"Base Model:  {args.base_model}")
    print(f"Corpus:      {args.data_file}")
    print(f"Max Samples: {max_samples if max_samples else 'ALL (126,929)'}")
    print(f"Seq Length:  {max_length}")
    print(f"Epochs:      {epochs} | Batch Size: {batch_size} | Grad Accum: {grad_accum}")
    print(f"Effective Batch: {batch_size * grad_accum}")
    print(f"LR:          {args.lr} | Warmup Steps: {args.warmup_steps}")
    print(f"LoRA Target: {lora_modules} (r={args.lora_r}, alpha={args.lora_alpha})")
    print("=" * 65)

    # 1. Load Tokenizer
    print("\n[1/5] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Build Dataset
    print("\n[2/5] Preparing Curated Dataset...")
    samples = prepare_balanced_samples(args.data_file, max_samples=max_samples, seed=args.seed)
    full_dataset = prepare_dataset(samples, tokenizer, max_length=max_length, mask_prompt=args.mask_prompt)

    if args.val_split > 0 and len(full_dataset) > 30:
        split = full_dataset.train_test_split(test_size=args.val_split, seed=args.seed)
        train_dataset = split["train"]
        val_dataset = split["test"]
        print(f"Train samples: {len(train_dataset):,} | Validation samples: {len(val_dataset):,}")
    else:
        train_dataset = full_dataset
        val_dataset = None
        print(f"Train samples: {len(train_dataset):,} (No validation split)")

    # Compute training steps and estimated time
    effective_batch = batch_size * grad_accum
    steps_per_epoch = max(1, len(train_dataset) // effective_batch)
    total_steps = steps_per_epoch * epochs
    sec_per_step = 1.6 if device == "cpu" else 0.08
    est_total_mins = (total_steps * sec_per_step) / 60.0

    print(f"\nTraining Plan: {total_steps:,} optimization steps (~{est_total_mins:.1f} minutes estimated)")
    if device == "cpu" and total_steps > 3000:
        print("\n⚠️ WARNING: You are running a large number of steps on CPU.")
        print("  To train quickly in ~10-15 minutes, run with:")
        print("    python3 train.py --preset cpu-fast")
        print("  Or run the provided Google Colab notebook with a free GPU: Tamazight_LLM_Colab.ipynb\n")

    # 3. Load Model & Attach LoRA
    print("\n[3/5] Loading Base Model and Configuring LoRA...")
    if device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=dtype,
        low_cpu_mem_usage=True
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=lora_modules
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 4. Training Arguments & Trainer
    print("\n[4/5] Initializing Trainer...")
    os.makedirs(args.output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        logging_steps=10 if len(train_dataset) < 1000 else 25,
        eval_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="epoch",
        save_total_limit=2,
        fp16=(device == "cuda" and dtype == torch.float16),
        bf16=(device == "cuda" and dtype == torch.bfloat16),
        report_to="none",
        dataloader_num_workers=0
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        label_pad_token_id=-100,
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    print("\nStarting Model Training...")
    train_result = trainer.train()
    print("\nTraining Finished!")
    print(f"Final Train Loss: {train_result.training_loss:.4f}")

    eval_metrics = {}
    if val_dataset is not None:
        print("\nEvaluating on validation set...")
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", 0.0)
        eval_ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
        print(f"Validation Loss: {eval_loss:.4f} | Perplexity: {eval_ppl:.2f}")

    # 5. Save LoRA Adapter & Merged Model
    print("\n[5/5] Exporting Trained Model Artifacts...")
    adapter_path = args.adapter_dir or (args.output_dir + "-lora")
    print(f"Saving LoRA adapter to: {adapter_path}")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"Merging LoRA weights into standalone model weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Merged full model successfully saved to: {args.output_dir}")

    meta = {
        "base_model": args.base_model,
        "corpus_file": args.data_file,
        "preset": preset,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": args.lr,
        "lora_modules": lora_modules,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_metrics.get("eval_loss") if val_dataset is not None else None,
    }
    with open(os.path.join(args.output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("\n--- Post-Training Generation Test ---")
    merged_model.eval()
    test_prompt = "<|im_start|>user\nAzul! Amek telliḍ?<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(test_prompt, return_tensors="pt").to(merged_model.device)
    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Prompt:   Azul! Amek telliḍ?")
    print(f"Response: {gen_text.strip()}")

    if args.push_to_hub and args.hf_repo:
        print(f"\nPushing model to Hugging Face Hub: {args.hf_repo}...")
        merged_model.push_to_hub(args.hf_repo)
        tokenizer.push_to_hub(args.hf_repo)
        print("Model successfully published to Hugging Face!")

if __name__ == "__main__":
    main()
