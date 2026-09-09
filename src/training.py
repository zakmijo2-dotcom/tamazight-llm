"""
SFT Training Pipeline with Strict Assistant-Only Loss Masking.
Supports Qwen2.5-0.5B with LoRA fine-tuning, evaluation, and standalone artifact export.
"""

import os
import math
import json
import torch
from typing import Optional, Dict, Any
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    set_seed,
    PreTrainedTokenizerBase
)
from src.config import AppConfig, get_device
from src.chat_template import encode_sft_example
from src.data import load_jsonl
from src.model import load_base_model, attach_lora, merge_and_export_model

def build_hf_dataset(
    jsonl_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256,
    max_samples: Optional[int] = None
) -> Dataset:
    """
    Loads JSONL samples and applies chat-template tokenization with assistant-only loss masking.
    """
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path}")
        
    records = load_jsonl(jsonl_path)
    if max_samples and max_samples < len(records):
        records = records[:max_samples]
        
    features = []
    for r in records:
        messages = r.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant":
            continue
        try:
            feat = encode_sft_example(messages, tokenizer, max_length=max_length)
            features.append(feat)
        except Exception:
            continue
            
    if not features:
        raise ValueError(f"No valid training samples could be encoded from {jsonl_path}")
        
    return Dataset.from_list(features)

def train_model(
    config: AppConfig,
    resume_from_checkpoint: Optional[str] = None,
    max_samples: Optional[int] = None,
    epochs_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    preset: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes end-to-end SFT fine-tuning using configuration and optional runtime overrides.
    """
    set_seed(config.training.seed)
    device = get_device(config.inference.device)
    is_cpu = (device == "cpu")
    
    if preset == "cpu-fast" or (is_cpu and preset is None):
        batch_size = batch_size_override or 4
        grad_accum = 2
        epochs = epochs_override or 1
        max_length = 64
        lora_target = config.lora.cpu_target_modules
    elif preset == "cpu-standard":
        batch_size = batch_size_override or 4
        grad_accum = 4
        epochs = epochs_override or 2
        max_length = 128
        lora_target = config.lora.cpu_target_modules
    elif preset == "gpu-full":
        batch_size = batch_size_override or 16
        grad_accum = config.training.gradient_accumulation_steps
        epochs = epochs_override or config.training.epochs
        max_length = config.training.max_seq_length
        lora_target = config.lora.target_modules
    else:
        batch_size = batch_size_override or config.training.batch_size
        grad_accum = config.training.gradient_accumulation_steps
        epochs = epochs_override or config.training.epochs
        max_length = config.training.max_seq_length
        lora_target = config.lora.cpu_target_modules if is_cpu else config.lora.target_modules

    print("=" * 65)
    print("  ⵣ SFT Training Pipeline: Tamazight Multilingual LLM ⵣ")
    print("=" * 65)
    print(f"Hardware Device:  {device}")
    print(f"Base Model:       {config.model.name}")
    print(f"Training Corpus:  {config.data.train_file}")
    print(f"Validation File:  {config.data.validation_file}")
    print(f"Epochs:           {epochs} | Batch Size: {batch_size} | Grad Accum: {grad_accum}")
    print(f"Max Sequence Len: {max_length}")
    print(f"LoRA Modules:     {lora_target} (r={config.lora.r}, alpha={config.lora.alpha})")
    print(f"SFT Loss Mode:    Assistant-Only (System/User Tokens Masked to -100)")
    print("=" * 65)

    # 1. Load Base Model and Tokenizer
    model, tokenizer, selected_device, selected_dtype = load_base_model(
        config.model.name,
        device=device,
        torch_dtype=config.model.torch_dtype,
        gradient_checkpointing=config.training.gradient_checkpointing
    )
    
    # 2. Attach LoRA
    model = attach_lora(model, config.lora, is_cpu=is_cpu)
    
    # 3. Load Datasets
    print("\nTokenizing training dataset with chat template & assistant-only loss masking...")
    train_dataset = build_hf_dataset(
        config.data.train_file,
        tokenizer,
        max_length=max_length,
        max_samples=max_samples or config.data.max_samples
    )
    print(f"Loaded {len(train_dataset):,} formatted training examples.")
    
    eval_dataset = None
    if os.path.exists(config.data.validation_file):
        try:
            eval_dataset = build_hf_dataset(
                config.data.validation_file,
                tokenizer,
                max_length=max_length,
                max_samples=10 if is_cpu else None
            )
            print(f"Loaded {len(eval_dataset):,} formatted validation examples.")
        except Exception as e:
            print(f"Warning: Failed to load validation set: {e}")
            
    # 4. Trainer Setup
    os.makedirs(config.model.output_dir, exist_ok=True)
    
    training_args = TrainingArguments(
        output_dir=config.model.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=config.training.eval_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        lr_scheduler_type=config.training.lr_scheduler_type,
        logging_steps=config.training.logging_steps if len(train_dataset) > 50 else 2,
        eval_strategy=config.training.eval_strategy if eval_dataset is not None else "no",
        save_strategy=config.training.save_strategy,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end if eval_dataset is not None else False,
        metric_for_best_model=config.training.metric_for_best_model,
        fp16=(device == "cuda" and selected_dtype == torch.float16),
        bf16=(device == "cuda" and selected_dtype == torch.bfloat16),
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=(device != "cpu")
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
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("\nStarting SFT Training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    print(f"\nTraining Complete! Final Loss: {train_result.training_loss:.4f}")
    
    eval_metrics = {}
    if eval_dataset is not None:
        print("\nEvaluating model on validation split...")
        eval_metrics = trainer.evaluate()
        eval_loss = eval_metrics.get("eval_loss", float("nan"))
        ppl = math.exp(eval_loss) if eval_loss < 20 else float("inf")
        eval_metrics["perplexity"] = ppl
        print(f"Validation Loss: {eval_loss:.4f} | Perplexity: {ppl:.2f}")
        
    # 5. Export Standalone Model and Adapters
    print("\nSaving model checkpoints and standalone weights...")
    merge_and_export_model(
        model,
        tokenizer,
        output_dir=config.model.output_dir,
        adapter_dir=config.model.adapter_dir
    )
    
    # Save training metadata
    meta = {
        "base_model": config.model.name,
        "epochs": epochs,
        "batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": config.training.learning_rate,
        "lora_r": config.lora.r,
        "lora_alpha": config.lora.alpha,
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_metrics
    }
    with open(os.path.join(config.model.output_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        
    return meta
