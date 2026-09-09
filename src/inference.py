"""
Unified Multilingual Inference Engine for Qwen2.5.
Supports interactive multi-turn terminal chat, single prompt execution, and translation.
"""

import os
import sys
import torch
from typing import Optional, List, Dict, Tuple
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from peft import PeftModel

from src.config import load_config, get_device, get_torch_dtype
from src.chat_template import build_messages, format_chat_prompt
from src.model import load_base_model

def resolve_inference_model_path(user_path: Optional[str] = None) -> str:
    """Resolves local checkpoint if available, falling back to base model."""
    if user_path and os.path.exists(user_path):
        return user_path
    cfg = load_config()
    if os.path.exists(os.path.join(cfg.model.output_dir, "config.json")):
        return cfg.model.output_dir
    return cfg.model.name

def load_inference_pipeline(
    model_path: Optional[str] = None,
    adapter_path: Optional[str] = None,
    device: str = "auto"
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, str]:
    """Loads model and tokenizer prepared for inference."""
    target_path = resolve_inference_model_path(model_path)
    print(f"Loading inference model from: {target_path}...")
    
    model, tokenizer, selected_device, _ = load_base_model(
        target_path,
        device=device,
        torch_dtype="auto"
    )
    
    if adapter_path and os.path.exists(adapter_path):
        print(f"Applying LoRA adapter: {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path).to(selected_device)
        
    model.eval()
    return model, tokenizer, selected_device

def generate_reply(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    device: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 150,
    repetition_penalty: float = 1.1
) -> str:
    """Formats messages with chat template and generates assistant response."""
    prompt_str = format_chat_prompt(messages, tokenizer, add_generation_prompt=True)
    inputs = tokenizer(prompt_str, return_tensors="pt").to(device)
    
    do_sample = temperature > 0.0
    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
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

def run_interactive_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    device: str,
    system_prompt: Optional[str] = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 150,
    repetition_penalty: float = 1.1
) -> None:
    """Runs interactive multi-turn chat session in terminal."""
    print("\n" + "=" * 65)
    print("  ⵣ Tamazight Multilingual LLM Interactive Chat ⵣ")
    print("  Languages: Tamaziɣt (Kabyle) | العربية | Français | English")
    print("  Type 'exit', 'quit', or 'q' to leave | 'clear' to reset history.")
    print("=" * 65 + "\n")
    
    history: List[Dict[str, str]] = []
    
    while True:
        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nTanemmirt! Ar tufat!")
            break
            
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit", "q", "ffeɣ"]:
            print("Tanemmirt! Ar tufat!")
            break
        if user_input.lower() in ["clear", "cls"]:
            history = []
            print("[Conversation history cleared]\n")
            continue
            
        messages = build_messages(
            user_text=user_input,
            system_prompt=system_prompt,
            history=history
        )
        
        reply = generate_reply(
            model=model,
            tokenizer=tokenizer,
            messages=messages,
            device=device,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty
        )
        
        print(f"\nAssistant > {reply}\n")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})
        
        # Keep sliding history to avoid unbounded context growth
        if len(history) > 10:
            history = history[-8:]
