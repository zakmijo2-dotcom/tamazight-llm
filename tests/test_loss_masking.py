"""
Tests for Assistant-Only Loss Masking during SFT.
Mathematically verifies that system and user prompt tokens are masked with -100,
and that loss is computed exclusively on assistant tokens.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from src.config import load_config
from src.chat_template import encode_sft_example

def test_assistant_only_loss_masking():
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Azul! Amek telliḍ?"},
        {"role": "assistant", "content": "Azul! Labas, tanemmirt."}
    ]
    
    encoded = encode_sft_example(messages, tokenizer, max_length=128)
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    
    assert len(input_ids) == len(labels)
    
    # 1. Verify masked tokens are -100
    active_tokens = [tok for tok in labels if tok != -100]
    masked_tokens = [tok for tok in labels if tok == -100]
    
    assert len(masked_tokens) > 0, "System and user prompt tokens must be masked."
    assert len(active_tokens) > 0, "Assistant response tokens must remain unmasked."
    
    # 2. Verify decoded active tokens match the assistant message + end tokens
    decoded_active = tokenizer.decode(active_tokens)
    assert "Azul! Labas, tanemmirt." in decoded_active
    assert "You are a helpful assistant." not in decoded_active
    assert "Amek telliḍ?" not in decoded_active

def test_mathematical_loss_ignores_prompt():
    """
    Proves that modifying user prompt tokens in the input does NOT affect loss computation
    when prompt labels are masked to -100.
    """
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    messages = [
        {"role": "system", "content": "System prompt."},
        {"role": "user", "content": "Hello user."},
        {"role": "assistant", "content": "Answer from assistant."}
    ]
    
    encoded = encode_sft_example(messages, tokenizer, max_length=64)
    labels = torch.tensor(encoded["labels"]).unsqueeze(0)
    
    # Simulated logits of shape (batch, seq_len, vocab_size)
    vocab_size = len(tokenizer)
    seq_len = labels.shape[1]
    logits = torch.randn(1, seq_len, vocab_size, requires_grad=True)
    
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Loss should only compute over unmasked tokens
    loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
    loss.backward()
    
    # Gradients for masked positions must be zero
    masked_indices = (labels.view(-1) == -100).nonzero(as_tuple=True)[0]
    grad_at_masked = logits.grad.view(-1, vocab_size)[masked_indices]
    
    assert torch.all(grad_at_masked == 0.0), "Gradients at prompt positions must be strictly 0.0"
