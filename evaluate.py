#!/usr/bin/env python3
"""
Tamazight LLM Evaluation and Benchmarking Tool
Evaluates:
  1. Perplexity (PPL) on held-out Tamazight test corpus
  2. French -> Tamazight translation benchmark suite
  3. Cultural & conversational capability benchmark
"""

import os
import sys
import math
import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

TRANSLATION_BENCHMARK = [
    {"fr": "Bonjour", "expected_keywords": ["azul"]},
    {"fr": "Comment vas-tu ?", "expected_keywords": ["amek", "telliḍ"]},
    {"fr": "Merci beaucoup", "expected_keywords": ["tanemmirt"]},
    {"fr": "Au revoir", "expected_keywords": ["ar tufat", "ar timlilit", "lahna"]},
    {"fr": "Je ne sais pas", "expected_keywords": ["ur ssineɣ", "ur ẓriɣ"]},
    {"fr": "C'est bien", "expected_keywords": ["yelha", "igerrez"]},
    {"fr": "Où vas-tu ?", "expected_keywords": ["anda", "ttedduḍ", "tettedduḍ"]},
    {"fr": "Bonne nuit", "expected_keywords": ["iḍ ameggaz", "amenzu", "lahna"]},
]

MULTILINGUAL_BENCHMARK = [
    {"lang": "Tamazight", "prompt": "Azul fell-awen! Amek telliḍ?", "expected": ["azul", "labas", "telliḍ", "tanemmirt"]},
    {"lang": "Arabic", "prompt": "مرحباً! كيف حالك وماذا يمكنك أن تفعل؟", "expected": ["مرحبا", "بخير", "أستطيع", "مساعدتك", "مساعد"]},
    {"lang": "French", "prompt": "Bonjour ! Peux-tu te présenter brièvement ?", "expected": ["bonjour", "assistant", "peux", "aider", "suis"]},
    {"lang": "English", "prompt": "Hello! What can you help me with?", "expected": ["hello", "help", "assist", "can", "questions"]},
    {"lang": "Amazigh Culture", "prompt": "D acu i d Imaziɣen?", "expected": ["imaziɣen", "imezdaɣ", "tefriqt", "agafa", "zik"]},
    {"lang": "Amazigh History", "prompt": "Anwa i d Massinissa?", "expected": ["massinissa", "agellid", "numidya", "zik"]},
]
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tamazight LLM")
    default_target = "tamazight-multilingual-llm-final" if os.path.exists("tamazight-multilingual-llm-final") else "Qwen/Qwen2.5-0.5B-Instruct"
    parser.add_argument("--model-path", type=str, default=default_target, help=f"Path to model or HF repo (default: {default_target})")
    default_data = "data/multilingual_chat.jsonl" if os.path.exists("data/multilingual_chat.jsonl") else "tamazight_train.jsonl"
    parser.add_argument("--data-file", type=str, default=default_data, help=f"Path to evaluation corpus (default: {default_data})")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples for perplexity calculation")
    parser.add_argument("--output-json", type=str, default="eval_results.json", help="Path to save evaluation results")
    parser.add_argument("--device", type=str, default=None, help="Evaluation device (cuda, cpu, mps)")
    return parser.parse_args()

def calculate_perplexity(model, tokenizer, data_file, num_samples, device):
    print(f"\n--- Calculating Perplexity on {num_samples} samples ---")
    samples = []
    with open(data_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            try:
                d = json.loads(line)
                p = d.get("prompt", "")
                if p:
                    samples.append(p)
            except json.JSONDecodeError:
                continue

    if not samples:
        print("No evaluation samples found.")
        return float("inf"), 0.0

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for text in samples:
            enc = tokenizer(text, max_length=160, truncation=True, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            target_ids = input_ids.clone()
            
            outputs = model(input_ids, labels=target_ids)
            loss = outputs.loss.item()
            num_tokens = input_ids.shape[1]
            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return ppl, avg_loss

def run_translation_benchmark(model, tokenizer, device):
    print("\n--- French -> Tamazight Translation Benchmark ---")
    results = []
    matches = 0

    for item in TRANSLATION_BENCHMARK:
        fr = item["fr"]
        prompt = f'<|im_start|>user\nTasuqilt ar tmaziɣt n: "{fr}"<|im_end|>\n<|im_start|>assistant\n'
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=35,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        matched = any(kw.lower() in gen.lower() for kw in item["expected_keywords"])
        if matched:
            matches += 1

        results.append({
            "french": fr,
            "tamazight": gen,
            "expected_keywords": item["expected_keywords"],
            "matched": matched
        })
        status = "✓" if matched else "✗"
        print(f"[{status}] FR: {fr:<25} | BER: {gen}")

    accuracy = (matches / len(TRANSLATION_BENCHMARK)) * 100
    print(f"Translation Keyword Match: {matches}/{len(TRANSLATION_BENCHMARK)} ({accuracy:.1f}%)")
    return results, accuracy

def run_conversation_benchmark(model, tokenizer, device):
    print("\n--- 4-Language Conversational Benchmark (Tamazight, Arabic, French, English) ---")
    dialogues = []
    matches = 0

    for item in MULTILINGUAL_BENCHMARK:
        lang = item["lang"]
        prompt_text = item["prompt"]
        prompt = f"<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id
            )

        gen = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        matched = any(kw.lower() in gen.lower() for kw in item["expected"])
        if matched:
            matches += 1

        dialogues.append({
            "language": lang,
            "prompt": prompt_text,
            "response": gen,
            "matched": matched
        })
        status = "✓" if matched else "✗"
        print(f"\n[{status}] [{lang}]")
        print(f"Prompt:   {prompt_text}")
        print(f"Response: {gen}")

    accuracy = (matches / len(MULTILINGUAL_BENCHMARK)) * 100
    print(f"\nMultilingual Fluency Score: {matches}/{len(MULTILINGUAL_BENCHMARK)} ({accuracy:.1f}%)")
    return dialogues

def main():
    args = parse_args()
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model_target = args.model_path or "Qwen/Qwen2.5-0.5B-Instruct"
    print("=" * 60)
    print("  ⵣ Tamazight LLM Benchmark & Evaluation ⵣ")
    print(f"  Target: {model_target}")
    print(f"  Device: {device}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(model_target)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_target,
        dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)

    # 1. Perplexity
    ppl, loss = calculate_perplexity(model, tokenizer, args.data_file, args.num_samples, device)
    print(f"\nAverage Cross-Entropy Loss: {loss:.4f}")
    print(f"Model Perplexity (PPL):     {ppl:.2f}")

    # 2. Translation benchmark
    translation_results, trans_acc = run_translation_benchmark(model, tokenizer, device)

    # 3. Conversation benchmark
    conv_results = run_conversation_benchmark(model, tokenizer, device)

    # Save full report
    report = {
        "model": model_target,
        "device": device,
        "perplexity": ppl,
        "loss": loss,
        "translation_accuracy_percent": trans_acc,
        "translation_results": translation_results,
        "dialogue_results": conv_results
    }
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nFull evaluation report saved to: {args.output_json}")

if __name__ == "__main__":
    main()
