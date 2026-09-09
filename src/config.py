"""
Centralized Configuration Loader for Tamazight Multilingual LLM.
Parses configs/config.yaml and provides strongly typed dataclasses with CLI override support.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml
import torch

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "config.yaml"
)

@dataclass
class ModelSettings:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    torch_dtype: str = "auto"
    output_dir: str = "tamazight-qwen-0.5b-final"
    adapter_dir: str = "tamazight-qwen-0.5b-lora"

@dataclass
class LoraSettings:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
    ])
    cpu_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj"
    ])

@dataclass
class TrainingSettings:
    epochs: int = 3
    batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_steps: int = 20
    lr_scheduler_type: str = "cosine"
    max_seq_length: int = 256
    seed: int = 42
    gradient_checkpointing: bool = False
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_strategy: str = "epoch"
    save_total_limit: int = 2
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

@dataclass
class DataSettings:
    train_file: str = "data/train.jsonl"
    validation_file: str = "data/validation.jsonl"
    test_file: str = "data/test.jsonl"
    multilingual_chat_file: str = "data/multilingual_chat.jsonl"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    val_split: float = 0.05
    test_split: float = 0.05
    max_samples: Optional[int] = None

@dataclass
class SystemSettings:
    prompt: str = (
        "You are a helpful, culturally aware, multilingual AI assistant fluent in "
        "Tamazight (Tamaziɣt s ugemmay alatini), Arabic (العربية), French (Français), and English. "
        "Always converse and answer naturally and fluently in the language used by the user."
    )
    languages: List[str] = field(default_factory=lambda: [
        "tamazight", "arabic", "french", "english"
    ])

@dataclass
class InferenceSettings:
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 150
    device: str = "auto"

@dataclass
class GgufSettings:
    outtype: str = "f16"
    quantizations: List[str] = field(default_factory=lambda: [
        "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"
    ])
    output_dir: str = "models/gguf"

@dataclass
class AppConfig:
    model: ModelSettings = field(default_factory=ModelSettings)
    lora: LoraSettings = field(default_factory=LoraSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    data: DataSettings = field(default_factory=DataSettings)
    system: SystemSettings = field(default_factory=SystemSettings)
    inference: InferenceSettings = field(default_factory=InferenceSettings)
    gguf: GgufSettings = field(default_factory=GgufSettings)

def get_device(requested: str = "auto") -> str:
    """Detects available hardware accelerator (cuda, mps, cpu)."""
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_torch_dtype(device: str, requested: str = "auto") -> torch.dtype:
    """Selects best floating point precision based on device capabilities."""
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    if requested == "float32":
        return torch.float32

    if device == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32

def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads configuration YAML file into an AppConfig instance."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    cfg = AppConfig(
        model=ModelSettings(**data.get("model", {})),
        lora=LoraSettings(**data.get("lora", {})),
        training=TrainingSettings(**data.get("training", {})),
        data=DataSettings(**data.get("data", {})),
        system=SystemSettings(**data.get("system", {})),
        inference=InferenceSettings(**data.get("inference", {})),
        gguf=GgufSettings(**data.get("gguf", {}))
    )
    return cfg
