#!/usr/bin/env python3
"""
Tamazight (Berber) LLM Inference & Interactive Chat
Supports: Interactive Chat, Single Prompt Generation, Translation CLI
Base / Fine-tuned: Qwen2.5-0.5B
"""

import os
import sys
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

DEFAULT_CHECKPOINTS = ["tamazight-qwen-0.5b-final", "tamazight-multilingual-llm-final"]
DEFAULT_BASE = "Qwen/Qwen2.5-0.5B-Instruct"
FALLBACK_BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"

SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable multilingual AI assistant fluent in "
    "Tamazight (Tamaziɣt s ugemmay alatini), Arabic (العربية), French (Français), and English. "
    "Always converse and respond naturally and fluently in the language used by the user."
)
def parse_args():
    parser = argparse.ArgumentParser(description="Tamazight LLM Inference & Chat")
    parser.add_argument("--model-path", type=str, default=None, help=f"Path to local checkpoint or HF repo (default: {DEFAULT_CHECKPOINTS[0]} if exists, else {DEFAULT_BASE})")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional LoRA adapter directory")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for batch/CLI inference")
    parser.add_argument("--translate", type=str, default=None, help="French sentence to translate into Tamazight")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus sampling")
    parser.add_argument("--max-tokens", type=int, default=120, help="Max new tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty factor")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or mps")
    return parser.parse_args()

def resolve_model_path(user_path):
    if user_path:
        return user_path
    for ckpt in DEFAULT_CHECKPOINTS:
        if os.path.exists(os.path.join(ckpt, "config.json")):
            return ckpt
    return DEFAULT_BASE

def load_model_and_tokenizer(model_path, adapter_path=None, device_str=None):
    if device_str:
        device = device_str
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Loading model from: {model_path} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True
    ).to(device)

    if adapter_path and os.path.exists(adapter_path):
        from peft import PeftModel
        print(f"Applying LoRA adapter from: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path).to(device)

    model.eval()
    return model, tokenizer, device

def format_prompt(user_text, is_translation=False):
    if is_translation:
        user_message = f'Tasuqilt ar tmaziɣt n: "{user_text}"'
    else:
        user_message = user_text
    return f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

def generate_response(model, tokenizer, prompt_str, device, temperature=0.7, top_p=0.9, max_tokens=120, repetition_penalty=1.1):
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    do_sample = temperature > 0.0

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    prompt_len = inputs.input_ids.shape[1]
    new_tokens = outputs[0][prompt_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response

def interactive_chat(model, tokenizer, device, temperature=0.7, top_p=0.9, max_tokens=120, repetition_penalty=1.1):
    print("\n" + "=" * 65)
    print("  ⵣ Multilingual LLM Terminal Chat ⵣ")
    print("  Languages supported: Tamaziɣt | العربية | Français | English")
    print("  Tzemreḍ ad taruḍ s tmaziɣt, taɛrabt, tafransist neɣ teglizit.")
    print("  Type 'exit', 'quit', or 'q' to leave.")
    print("=" * 65 + "\n")
    history = ""
    while True:
        try:
            user_input = input("Kečč (You) > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTanemmirt! Ar tufat!")
            break

        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q", "ffeɣ"]:
            print("Tanemmirt! Ar tufat!")
            break
        if user_input.lower() in ["clear", "cls"]:
            history = ""
            print("[History cleared]")
            continue

        turn_prompt = format_prompt(user_input)
        prompt_with_context = history + turn_prompt

        response = generate_response(
            model, tokenizer, prompt_with_context, device,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        )

        print(f"\nQwen (Tamazight) > {response}\n")
        history += f"<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>\n"
        if len(history) > 1500:
            history = history[-1000:]

def main():
    args = parse_args()
    model_path = resolve_model_path(args.model_path)
    model, tokenizer, device = load_model_and_tokenizer(model_path, args.adapter_path, args.device)

    # 1. Translation mode
    if args.translate:
        prompt_str = format_prompt(args.translate, is_translation=True)
        response = generate_response(
            model, tokenizer, prompt_str, device,
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty
        )
        print(f"Français: {args.translate}")
        print(f"Tamaziɣt: {response}")
        return

    # 2. Single prompt mode
    if args.prompt:
        prompt_str = format_prompt(args.prompt, is_translation=False)
        response = generate_response(
            model, tokenizer, prompt_str, device,
            temperature=args.temperature, top_p=args.top_p,
            max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty
        )
        print(f"User:     {args.prompt}")
        print(f"Response: {response}")
        return

    # 3. Interactive Chat
    interactive_chat(
        model, tokenizer, device,
        temperature=args.temperature, top_p=args.top_p,
        max_tokens=args.max_tokens, repetition_penalty=args.repetition_penalty
    )

if __name__ == "__main__":
    main()
