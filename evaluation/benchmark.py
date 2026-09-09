#!/usr/bin/env python3
"""
Comprehensive Multilingual LLM Evaluation and Benchmarking Suite.
Evaluates:
  1. Perplexity (PPL) on held-out test split (data/test.jsonl)
  2. Multi-domain benchmarks (Tamazight, Arabic, French, English, Culture, Translation)
  3. Optional side-by-side comparison with Base Model
Saves reports to evaluation/results/benchmark_results.json and .csv.
"""

import os
import sys
import csv
import math
import json
import argparse
import torch
from typing import Dict, List, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config, get_device, get_torch_dtype
from src.chat_template import build_messages, format_chat_prompt
from src.data import load_jsonl

def parse_args():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Evaluate Tamazight Multilingual LLM")
    parser.add_argument("--model-path", type=str, default=None, help=f"Path to evaluation model (default: {cfg.model.output_dir} or {cfg.model.name})")
    parser.add_argument("--compare-base", action="store_true", help="Also evaluate base model and output comparison")
    parser.add_argument("--base-model", type=str, default=cfg.model.name, help="Base model ID for comparison")
    parser.add_argument("--test-file", type=str, default=cfg.data.test_file, help="Path to held-out test jsonl")
    parser.add_argument("--eval-dir", type=str, default="evaluation/datasets", help="Directory of evaluation jsonl datasets")
    parser.add_argument("--output-json", type=str, default="evaluation/results/benchmark_results.json", help="Path to save JSON report")
    parser.add_argument("--output-csv", type=str, default="evaluation/results/benchmark_summary.csv", help="Path to save CSV summary")
    parser.add_argument("--max-eval-samples", type=int, default=50, help="Max test samples for perplexity calculation")
    parser.add_argument("--device", type=str, default="auto", help="Evaluation hardware device (auto, cuda, cpu, mps)")
    return parser.parse_args()

def compute_perplexity(
    model,
    tokenizer,
    test_file: str,
    device: str,
    max_samples: int = 50
) -> Tuple[float, float]:
    """Calculates cross-entropy loss and perplexity on held-out test records."""
    if not os.path.exists(test_file):
        return float("inf"), float("nan")
        
    records = load_jsonl(test_file)
    if max_samples and max_samples < len(records):
        records = records[:max_samples]
        
    total_loss = 0.0
    total_tokens = 0
    model.eval()
    
    with torch.no_grad():
        for r in records:
            messages = r.get("messages", [])
            if not messages:
                continue
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            enc = tokenizer(text, max_length=256, truncation=True, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1]
            
            total_loss += loss * num_tokens
            total_tokens += num_tokens
            
    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return ppl, avg_loss

def evaluate_task_dataset(
    model,
    tokenizer,
    dataset_path: str,
    device: str,
    system_prompt: str
) -> Dict[str, Any]:
    """Evaluates a benchmark dataset measuring task keyword match accuracy."""
    records = load_jsonl(dataset_path)
    task_name = os.path.splitext(os.path.basename(dataset_path))[0]
    
    results = []
    correct_count = 0
    
    for r in records:
        prompt_text = r.get("prompt", "")
        keywords = r.get("keywords", [])
        ref = r.get("reference", "")
        
        messages = build_messages(prompt_text, system_prompt=system_prompt)
        full_prompt = format_chat_prompt(messages, tokenizer, add_generation_prompt=True)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )
            
        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        matched = any(kw.lower() in gen.lower() for kw in keywords) if keywords else False
        if matched:
            correct_count += 1
            
        results.append({
            "id": r.get("id"),
            "prompt": prompt_text,
            "generated": gen,
            "reference": ref,
            "matched": matched
        })
        
    accuracy = (correct_count / max(len(records), 1)) * 100
    return {
        "task": task_name,
        "total_cases": len(records),
        "passed": correct_count,
        "accuracy": accuracy,
        "details": results
    }

def evaluate_full_model(
    model_path: str,
    test_file: str,
    eval_dir: str,
    device: str,
    system_prompt: str,
    max_eval_samples: int = 50
) -> Dict[str, Any]:
    """Runs complete evaluation suite on a given model."""
    print(f"\nEvaluating Model: {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dtype = get_torch_dtype(device, "auto")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device)
    
    # 1. Perplexity
    ppl, loss = compute_perplexity(model, tokenizer, test_file, device, max_samples=max_eval_samples)
    print(f"  - Test Split Loss: {loss:.4f} | Perplexity (PPL): {ppl:.2f}")
    
    # 2. Benchmark tasks
    task_results = {}
    if os.path.exists(eval_dir):
        eval_files = sorted([f for f in os.listdir(eval_dir) if f.endswith(".jsonl")])
        for ef in eval_files:
            epath = os.path.join(eval_dir, ef)
            res = evaluate_task_dataset(model, tokenizer, epath, device, system_prompt)
            task_results[res["task"]] = res
            print(f"  - [{res['task']:<14}] Accuracy: {res['passed']}/{res['total_cases']} ({res['accuracy']:.1f}%)")
            
    # Compute overall task average
    accs = [res["accuracy"] for res in task_results.values()]
    avg_accuracy = sum(accs) / max(len(accs), 1)
    print(f"  - Overall Task Accuracy: {avg_accuracy:.1f}%\n")
    
    return {
        "model_path": model_path,
        "perplexity": ppl,
        "loss": loss,
        "average_accuracy": avg_accuracy,
        "tasks": task_results
    }

def main():
    args = parse_args()
    cfg = load_config()
    
    device = get_device(args.device)
    target_model = args.model_path or (cfg.model.output_dir if os.path.exists(os.path.join(cfg.model.output_dir, "config.json")) else cfg.model.name)
    
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    print("=" * 65)
    print("  ⵣ Multilingual Benchmark & Evaluation Suite ⵣ")
    print(f"  Target Model: {target_model}")
    print(f"  Device:       {device}")
    print("=" * 65)
    
    target_report = evaluate_full_model(
        target_model,
        test_file=args.test_file,
        eval_dir=args.eval_dir,
        device=device,
        system_prompt=cfg.system.prompt,
        max_eval_samples=args.max_eval_samples
    )
    
    base_report = None
    if args.compare_base and target_model != args.base_model:
        print("Evaluating Base Model for Comparison...")
        base_report = evaluate_full_model(
            args.base_model,
            test_file=args.test_file,
            eval_dir=args.eval_dir,
            device=device,
            system_prompt=cfg.system.prompt,
            max_eval_samples=args.max_eval_samples
        )
        
    full_output = {
        "evaluated_model": target_report,
        "base_model": base_report
    }
    
    # Save JSON report
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)
    print(f"Saved detailed JSON report to: {args.output_json}")
    
    # Save CSV summary
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "Perplexity", "Loss", "Average_Accuracy"] + list(target_report["tasks"].keys()))
        
        row_target = [
            target_report["model_path"],
            f"{target_report['perplexity']:.2f}",
            f"{target_report['loss']:.4f}",
            f"{target_report['average_accuracy']:.1f}%"
        ] + [f"{target_report['tasks'][k]['accuracy']:.1f}%" for k in target_report["tasks"]]
        writer.writerow(row_target)
        
        if base_report:
            row_base = [
                base_report["model_path"],
                f"{base_report['perplexity']:.2f}",
                f"{base_report['loss']:.4f}",
                f"{base_report['average_accuracy']:.1f}%"
            ] + [f"{base_report['tasks'].get(k, {}).get('accuracy', 0.0):.1f}%" for k in target_report["tasks"]]
            writer.writerow(row_base)
            
    print(f"Saved CSV summary table to: {args.output_csv}")

if __name__ == "__main__":
    main()
