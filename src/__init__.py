"""
Tamazight Multilingual LLM core package.
"""

from src.config import load_config, AppConfig, get_device, get_torch_dtype

__version__ = "0.2.0"
__all__ = ["load_config", "AppConfig", "get_device", "get_torch_dtype"]
