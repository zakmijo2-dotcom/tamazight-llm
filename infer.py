#!/usr/bin/env python3
"""
Tamazight Multilingual LLM Inference CLI.
Usage:
    python infer.py --prompt "Azul fell-awen! Amek telliḍ?"
    python infer.py --translate "Bonjour mon ami"
    python infer.py  (launches interactive chat)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import load_config
from src.chat_template import build_messages
from src.inference import (
    load_inference_pipeline,
    generate_reply,
    run_interactive_chat
)

def parse_args():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Tamazight Multilingual LLM Inference")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint or HF model ID")
    parser.add_argument("--adapter-path", type=str, default=None, help="Path to optional LoRA adapter")
    parser.add_argument("--prompt", type=str, default=None, help="Single query prompt string")
    parser.add_argument("--translate", type=str, default=None, help="Sentence to translate into Tamazight")
    parser.add_argument("--temperature", type=float, default=cfg.inference.temperature, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=cfg.inference.top_p, help="Top-p nucleus sampling")
    parser.add_argument("--max-tokens", type=int, default=cfg.inference.max_new_tokens, help="Max new tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Hardware device (auto, cuda, cpu, mps)")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    
    model, tokenizer, device = load_inference_pipeline(
        model_path=args.model_path,
        adapter_path=args.adapter_path,
        device=args.device
    )
    
    # Translation helper
    if args.translate:
        prompt_text = f'Tasuqilt ar tmaziɣt n: "{args.translate}"'
        messages = build_messages(prompt_text, system_prompt=cfg.system.prompt)
        reply = generate_reply(
            model=model, tokenizer=tokenizer, messages=messages, device=device,
            temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_tokens
        )
        print(f"Source:    {args.translate}")
        print(f"Tamaziɣt:  {reply}")
        return
        
    # Single prompt
    if args.prompt:
        messages = build_messages(args.prompt, system_prompt=cfg.system.prompt)
        reply = generate_reply(
            model=model, tokenizer=tokenizer, messages=messages, device=device,
            temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_tokens
        )
        print(f"User:      {args.prompt}")
        print(f"Assistant: {reply}")
        return
        
    # Interactive chat fallback
    run_interactive_chat(
        model=model,
        tokenizer=tokenizer,
        device=device,
        system_prompt=cfg.system.prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_tokens
    )

if __name__ == "__main__":
    main()
