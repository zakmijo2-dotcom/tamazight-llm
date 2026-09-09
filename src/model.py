"""
Model and Tokenizer Initialization, LoRA Configuration, and Export Utilities.
Supports Qwen2.5 architecture natively with precision auto-detection.
"""

import os
import torch
from typing import Tuple, Optional, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
    PreTrainedModel
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
from src.config import LoraSettings, get_device, get_torch_dtype

def load_tokenizer(model_name_or_path: str) -> PreTrainedTokenizerBase:
    """Loads tokenizer and ensures padding tokens are properly set."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def load_base_model(
    model_name_or_path: str,
    device: str = "auto",
    torch_dtype: str = "auto",
    gradient_checkpointing: bool = False
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase, str, torch.dtype]:
    """
    Loads base causal language model and tokenizer onto target device with chosen dtype.
    """
    selected_device = get_device(device)
    selected_dtype = get_torch_dtype(selected_device, torch_dtype)
    
    tokenizer = load_tokenizer(model_name_or_path)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        dtype=selected_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(selected_device)
    
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        
    return model, tokenizer, selected_device, selected_dtype

def attach_lora(
    model: PreTrainedModel,
    lora_settings: LoraSettings,
    is_cpu: bool = False
) -> PreTrainedModel:
    """
    Attaches LoRA parameter-efficient fine-tuning adapters to target linear modules.
    On CPU, uses target_modules optimized to reduce backward-pass overhead.
    """
    target_modules = lora_settings.cpu_target_modules if is_cpu else lora_settings.target_modules
    
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_settings.r,
        lora_alpha=lora_settings.alpha,
        lora_dropout=lora_settings.dropout,
        bias=lora_settings.bias,
        target_modules=target_modules
    )
    
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model

def merge_and_export_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: str,
    adapter_dir: Optional[str] = None
) -> None:
    """
    Saves standalone merged model weights along with optional separate LoRA adapters.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save adapter if requested
    if adapter_dir:
        os.makedirs(adapter_dir, exist_ok=True)
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        print(f"LoRA adapter weights saved to: {adapter_dir}")
        
    # Merge LoRA layers into base weights
    if hasattr(model, "merge_and_unload"):
        print("Merging LoRA adapter weights into standalone model weights...")
        merged = model.merge_and_unload()
    else:
        merged = model
        
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Full standalone model successfully saved to: {output_dir}")
