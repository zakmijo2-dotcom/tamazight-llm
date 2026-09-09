"""
Tests for centralized configuration loading.
"""

import os
import torch
from src.config import load_config, get_device, get_torch_dtype

def test_load_default_config():
    cfg = load_config()
    assert cfg.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert "q_proj" in cfg.lora.target_modules
    assert cfg.training.epochs >= 1
    assert cfg.training.seed == 42
    assert "tamazight" in cfg.system.languages
    assert "arabic" in cfg.system.languages

def test_device_selection():
    dev = get_device("cpu")
    assert dev == "cpu"
    auto_dev = get_device("auto")
    assert auto_dev in ["cuda", "mps", "cpu"]

def test_dtype_selection():
    assert get_torch_dtype("cpu", "float32") == torch.float32
    assert get_torch_dtype("cuda", "bfloat16") == torch.bfloat16
