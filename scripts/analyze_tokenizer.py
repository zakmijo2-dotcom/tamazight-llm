#!/usr/bin/env python3
"""
Tokenizer Efficiency and Kabyle Linguistic Stress-Test Suite.
Calculates token compression ratios across:
  - Kabyle (Latin script with diacritics)
  - French
  - English
  - Arabic
Verifies roundtrip fidelity for special diacritics, long words, and code-switching.
"""

import os
import sys
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config

# Test parallel and equivalent sentences across the 4 languages
PARALLEL_SENTENCES = [
    {
        "kabyle": "Azul fell-awen! Amek telliḍ ass-a? Ssarameɣ teggarzed deg uxeddim-ik.",
        "french": "Bonjour à vous ! Comment allez-vous aujourd'hui ? J'espère que votre travail se passe bien.",
        "english": "Hello to you all! How are you doing today? I hope your work is going well.",
        "arabic": "مرحباً بكم جميعاً! كيف حالكم اليوم؟ أتمنى أن تكون أمور عملكم تسير على ما يرام."
    },
    {
        "kabyle": "Amezruy n Yimaziɣen di Tefriqt n Ugafa d ameqran yerna yeččur d lhiba d tlelli.",
        "french": "L'histoire des Berbères en Afrique du Nord est immense et empreinte de grandeur et de liberté.",
        "english": "The history of the Amazigh people in North Africa is vast and filled with majesty and freedom.",
        "arabic": "إن تاريخ الأمازيغ في شمال إفريقيا عريق وعظيم وحافل بالهيبة وقيم الحرية والكرامة."
    },
    {
        "kabyle": "Tamaziɣt d tutlayt tayemmat i d-yeğğan leqrun n tmusni d yidles aɣerfan.",
        "french": "Le tamazight est la langue maternelle qui a transmis des siècles de savoir et de culture populaire.",
        "english": "Tamazight is the mother tongue that has carried centuries of knowledge and folk culture.",
        "arabic": "الأمازيغية هي اللغة الأم التي حملت عبر القرون المعرفة والثقافة الشعبية الأصيلة."
    },
    {
        "kabyle": "Taddart tebna ɣef tdukli d tiwizi gar yimezdaɣ mebla ma nwala anwa yesɛan neɣ ur nesɛi.",
        "french": "Le village repose sur l'union et la solidarité (Tiwizi) entre les habitants sans distinction.",
        "english": "The village is built upon unity and mutual solidarity (Tiwizi) among all residents without distinction.",
        "arabic": "تقوم القرية على الوحدة والتضامن الجماعي (التويزة) بين جميع السكان دون تفرقة بين غني وفقير."
    }
]

# Kabyle diacritics & linguistic features
KABYLE_LINGUISTIC_TESTS = [
    ("Special Diacritics Lowercase", "ɛ ɣ ẓ ṭ ḍ č ǧ ṛ ṣ"),
    ("Special Diacritics Uppercase", "Ɛ Ɣ Ẓ Ṭ Ḍ Č Ǧ Ṛ Ṣ"),
    ("Tifinagh Alphabet", "ⴰ ⴱ ⴳ ⴷ ⴹ ⴻ ⴼ ⴽ ⵀ ⵃ ⵄ ⵅ ⵇ ⵉ ⵊ ⵍ ⵎ ⵏ ⵓ ⵔ ⵕ ⵖ ⵙ ⵚ ⵛ ⵜ ⵟ ⵡ ⵢ ⵣ ⵥ"),
    ("Long Kabyle Words", "Imeɣnasen-nneɣ Tisensegmitin Ttmeslayent-d Yettwasebded-d Tamsalt-nni"),
    ("Toponyms / Place Names", "Tizi Wezzu Vgayet Tubiret Tuvirett Larebɛa Nat Iraten At Yanni Ǧerǧer"),
    ("Code-Switching (Kabyle + FR/AR)", "Nudaɣ fell-as di l'université dɣa nnan-iyi-d d akken yella di réunion."),
    ("Grammar Particles & Affixes", "ad yaru, ur yuri ara, yura-d, yura-yas-tt, deg-s, fell-as, yid-sen")
]

def analyze_tokenizer_efficiency(tokenizer) -> Dict[str, Dict[str, float]]:
    print("=" * 70)
    print(f"  TOKENIZER EFFICIENCY ANALYSIS ({tokenizer.name_or_path})")
    print(f"  Vocabulary Size: {len(tokenizer):,} tokens")
    print("=" * 70)
    
    stats = {"kabyle": [], "french": [], "english": [], "arabic": []}
    
    for row in PARALLEL_SENTENCES:
        for lang in ["kabyle", "french", "english", "arabic"]:
            text = row[lang]
            tokens = tokenizer.encode(text, add_special_tokens=False)
            words = text.split()
            tokens_per_word = len(tokens) / max(len(words), 1)
            stats[lang].append({
                "tokens": len(tokens),
                "words": len(words),
                "chars": len(text),
                "ratio": tokens_per_word
            })
            
    summary = {}
    print(f"{'Language':<12} | {'Avg Words':<10} | {'Avg Tokens':<11} | {'Tokens/Word':<12} | {'Chars/Token':<12}")
    print("-" * 65)
    
    for lang in ["english", "french", "arabic", "kabyle"]:
        data = stats[lang]
        avg_w = sum(d["words"] for d in data) / len(data)
        avg_t = sum(d["tokens"] for d in data) / len(data)
        avg_ratio = sum(d["ratio"] for d in data) / len(data)
        avg_c_per_t = sum(d["chars"] / d["tokens"] for d in data) / len(data)
        summary[lang] = {
            "avg_words": avg_w,
            "avg_tokens": avg_t,
            "tokens_per_word": avg_ratio,
            "chars_per_token": avg_c_per_t
        }
        print(f"{lang.capitalize():<12} | {avg_w:<10.1f} | {avg_t:<11.1f} | {avg_ratio:<12.2f} | {avg_c_per_t:<12.2f}")
        
    print("=" * 65)
    return summary

def run_kabyle_linguistic_tests(tokenizer) -> bool:
    print("\n" + "=" * 70)
    print("  KABYLE PHONOLOGY & LINGUISTIC STRESS-TESTS")
    print("=" * 70)
    
    all_passed = True
    for test_name, text in KABYLE_LINGUISTIC_TESTS:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(token_ids)
        roundtrip_ok = (decoded == text)
        status = "✅ PASS" if roundtrip_ok else "❌ FAIL"
        if not roundtrip_ok:
            all_passed = False
            
        token_count = len(token_ids)
        tokens_preview = [tokenizer.decode([tid]) for tid in token_ids[:8]]
        print(f"[{status}] {test_name:<32}")
        print(f"       Input:   {text}")
        print(f"       Tokens ({token_count}): {tokens_preview}...")
        if not roundtrip_ok:
            print(f"       Decoded: {decoded}")
            
    print("-" * 70)
    if all_passed:
        print("  🎉 All Kabyle diacritics, Tifinagh, and code-switching passed with 100% fidelity!")
    else:
        print("  ⚠️ Warning: Some character roundtrips failed!")
    print("=" * 70)
    return all_passed

def main():
    cfg = load_config()
    model_name = cfg.model.name
    print(f"Loading Tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    analyze_tokenizer_efficiency(tokenizer)
    run_kabyle_linguistic_tests(tokenizer)

if __name__ == "__main__":
    main()
