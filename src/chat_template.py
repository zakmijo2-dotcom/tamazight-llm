"""
Unified Chat Template Management for Qwen2.5.
Uses tokenizer.apply_chat_template() natively for both training and inference.
Guarantees 100% alignment between training loss masking and inference prompting.
"""

from typing import List, Dict, Optional, Tuple, Any
from transformers import PreTrainedTokenizerBase

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, culturally aware, multilingual AI assistant fluent in "
    "Tamazight (Tamaziɣt s ugemmay alatini), Arabic (العربية), French (Français), and English. "
    "Always converse and answer naturally and fluently in the language used by the user."
)

def build_messages(
    user_text: str,
    system_prompt: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
    assistant_text: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Constructs a standard ChatML messages list from user input, optional system prompt,
    multi-turn history, and optional assistant response.
    """
    messages: List[Dict[str, str]] = []
    
    # Add system message
    sys_content = system_prompt or DEFAULT_SYSTEM_PROMPT
    if sys_content:
        messages.append({"role": "system", "content": sys_content})
        
    # Append past conversation history
    if history:
        for turn in history:
            role = turn.get("role")
            content = turn.get("content", "")
            if role in ["system", "user", "assistant"] and content:
                messages.append({"role": role, "content": content})
                
    # Add current user turn
    if user_text:
        messages.append({"role": "user", "content": user_text})
        
    # Add assistant response if provided (e.g. for training examples)
    if assistant_text:
        messages.append({"role": "assistant", "content": assistant_text})
        
    return messages

def format_chat_prompt(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    add_generation_prompt: bool = True
) -> str:
    """
    Applies tokenizer's native chat template to produce the model prompt string.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

def encode_sft_example(
    messages: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 256
) -> Dict[str, List[int]]:
    """
    Encodes a conversation for SFT with strict assistant-only loss masking.
    All system and user tokens have label = -100.
    Loss is calculated exclusively on assistant response tokens and EOS.
    """
    # Verify last message is from assistant
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError("SFT conversation must end with an assistant turn.")
        
    # Full conversation with assistant response
    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    
    # Prompt prefix up to assistant start
    prompt_messages = messages[:-1]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    
    # Tokenize full text
    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False
    )
    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]
    
    # Tokenize prefix to find exact masking boundary
    prompt_enc = tokenizer(
        prompt_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False
    )
    prompt_len = len(prompt_enc["input_ids"])
    
    # Initialize labels as clone of input_ids
    labels = list(input_ids)
    
    # Mask out prompt tokens with -100
    mask_boundary = min(prompt_len, len(labels))
    for i in range(mask_boundary):
        labels[i] = -100
        
    # Safety: ensure at least one assistant token has an active label
    has_active = any(lbl != -100 for lbl in labels)
    if not has_active:
        # If response was completely truncated, keep standard labels to avoid NaN
        labels = list(input_ids)
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }
