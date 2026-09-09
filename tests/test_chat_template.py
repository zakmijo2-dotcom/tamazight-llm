"""
Tests for unified Chat Template formatting.
"""

from transformers import AutoTokenizer
from src.config import load_config
from src.chat_template import build_messages, format_chat_prompt

def test_build_messages():
    msgs = build_messages("Azul!", system_prompt="Test system.")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "Test system."
    assert msgs[1]["role"] == "user"
    assert msgs[1]["content"] == "Azul!"

def test_format_chat_prompt():
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    msgs = build_messages("Amek telliḍ?", system_prompt="Test sys")
    prompt = format_chat_prompt(msgs, tokenizer, add_generation_prompt=True)
    
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")
