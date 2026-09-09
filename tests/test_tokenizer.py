"""
Tests for tokenizer fidelity and special character handling.
"""

from transformers import AutoTokenizer
from src.config import load_config

def test_tokenizer_roundtrip_tamazight_latin():
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    phrases = [
        "Azul fell-awen! Amek telliḍ?",
        "Tanemmirt tameqrant a gma.",
        "Imaziɣen d imezdaɣ ineṣliyen n Tefriqt n Ugafa.",
        "ɣ ḍ č ḥ ɛ ǧ ṛ ṣ ṭ ẓ Ɣ Ḍ Č Ḥ Ɛ Ǧ Ṛ Ṣ Ṭ Ẓ",
        "Sḥassfeɣ, d acu ur qebbleɣ ara da wid ur ssineɣ."
    ]
    
    for p in phrases:
        ids = tokenizer.encode(p, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        assert decoded == p, f"Decoded mismatch: '{decoded}' != '{p}'"

def test_tokenizer_multilingual():
    cfg = load_config()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    
    arabic = "مرحباً بكم في شمال إفريقيا!"
    french = "Bonjour tout le monde !"
    english = "Hello world, welcome!"
    
    for text in [arabic, french, english]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        assert len(ids) > 0
        decoded = tokenizer.decode(ids)
        assert decoded == text
