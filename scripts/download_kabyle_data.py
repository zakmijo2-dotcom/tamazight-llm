#!/usr/bin/env python3
"""
Automated Kabyle/Tamazight Data Ingestion, Normalization, and Instruction Synthesis Pipeline.
Downloads and integrates all legally available Hugging Face datasets registered in configs/datasets.yaml.
"""

import os
import sys
import yaml
import json
import random
from typing import List, Dict, Any, Optional
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config
from src.data import (
    normalize_text,
    detect_script,
    detect_language,
    compute_quality_score,
    create_sample,
    deduplicate_samples,
    train_val_test_split,
    save_jsonl
)

def chatml_sample(user_text: str, asst_text: str, sys_prompt: str, source: str, lang: str = "kabyle"):
    return create_sample(
        user_text=user_text,
        assistant_text=asst_text,
        system_prompt=sys_prompt,
        source=source,
        language_override=lang
    )

def download_and_process_corpora(
    output_dir: str = "data",
    max_hca: int = 4000,
    max_ummto: int = 4000,
    max_bouira: int = 3000,
    seed: int = 42
):
    cfg = load_config()
    sys_prompt = cfg.system.prompt
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw", "huggingface"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)
    
    print("=" * 70)
    print("  ⵣ AUTOMATED KABYLE / TAMAZIGHT DATASET INGESTION & SYNTHESIS ⵣ")
    print("=" * 70)
    
    collected_samples = []
    
    # 1. Ingest Core Seed Dialogues
    from src.seed_dialogues import (
        TAMAZIGHT_CORE,
        ARABIC_CORE,
        FRENCH_CORE,
        ENGLISH_CORE,
        CROSS_LINGUAL
    )
    for u, a in TAMAZIGHT_CORE:
        s = chatml_sample(u, a, sys_prompt, "seed_kabyle", "kabyle")
        if s: collected_samples.append(s)
    for u, a in ARABIC_CORE:
        s = chatml_sample(u, a, sys_prompt, "seed_arabic", "arabic")
        if s: collected_samples.append(s)
    for u, a in FRENCH_CORE:
        s = chatml_sample(u, a, sys_prompt, "seed_french", "french")
        if s: collected_samples.append(s)
    for u, a in ENGLISH_CORE:
        s = chatml_sample(u, a, sys_prompt, "seed_english", "english")
        if s: collected_samples.append(s)
    for u, a in CROSS_LINGUAL:
        s = chatml_sample(u, a, sys_prompt, "seed_cross_lingual", "multilingual")
        if s: collected_samples.append(s)
        
    print(f"[1/7] Ingested {len(collected_samples)} curated foundational dialogues.")

    # 2. Ingest Imsidag HCA Kabyle Corpus (CC-BY-SA 4.0)
    print(f"\n[2/7] Streaming Imsidag-community/kabyle-corpus-hca (Target: {max_hca})...")
    try:
        from datasets import load_dataset
        hca_ds = load_dataset("Imsidag-community/kabyle-corpus-hca", split="train", streaming=True)
        hca_count = 0
        for row in hca_ds:
            text = normalize_text(row.get("text", ""))
            # Filter clean, substantive Kabyle sentences
            if 30 <= len(text) <= 600 and not text.startswith("http"):
                # Synthesize instruction/explanation and conversational tasks
                if hca_count % 3 == 0:
                    s = chatml_sample(
                        f"Ssummel-d aḍris-a s teqbaylit:\n\"{text}\"",
                        f"Aḍris-a yebna ɣef waya:\n{text}",
                        sys_prompt, "hca_summarization", "kabyle"
                    )
                elif hca_count % 3 == 1:
                    s = chatml_sample(
                        f"Acu i d anamek n tefyirt-agi s teqbaylit: \"{text[:120]}\"?",
                        f"Tafyirt-agi temmal-d:\n{text}",
                        sys_prompt, "hca_explanation", "kabyle"
                    )
                else:
                    s = chatml_sample(
                        f"Mmeslay-iyi-d ɣef waya s teqbaylit.",
                        f"{text}",
                        sys_prompt, "hca_reading", "kabyle"
                    )
                if s:
                    collected_samples.append(s)
                    hca_count += 1
                    if hca_count >= max_hca:
                        break
        print(f"      Ingested {hca_count:,} HCA Kabyle instruction samples.")
    except Exception as e:
        print(f"      Warning: HCA corpus ingestion failed or skipped: {e}")

    # 3. Ingest Imsidag UMMTO Kabyle Academic Corpus (CC-BY-SA 4.0)
    print(f"\n[3/7] Streaming Imsidag-community/kabyle-corpus-ummto (Target: {max_ummto})...")
    try:
        ummto_ds = load_dataset("Imsidag-community/kabyle-corpus-ummto", split="train", streaming=True)
        ummto_count = 0
        for row in ummto_ds:
            text = normalize_text(row.get("text", ""))
            if 35 <= len(text) <= 700:
                words = text.split()
                if len(words) >= 6:
                    key_word = words[0].strip(".,!?:;\"'")
                    if ummto_count % 2 == 0 and len(key_word) >= 3:
                        s = chatml_sample(
                            f"Fk-iyi-d tafyirt s wawal '{key_word}' s teqbaylit.",
                            f"Atan lemtel s wawal '{key_word}':\n{text}",
                            sys_prompt, "ummto_vocabulary", "kabyle"
                        )
                    else:
                        s = chatml_sample(
                            f"Acu i d tazwart n wawal ɣef temsalt-a s teqbaylit?",
                            f"{text}",
                            sys_prompt, "ummto_academic", "kabyle"
                        )
                    if s:
                        collected_samples.append(s)
                        ummto_count += 1
                        if ummto_count >= max_ummto:
                            break
        print(f"      Ingested {ummto_count:,} UMMTO academic Kabyle samples.")
    except Exception as e:
        print(f"      Warning: UMMTO corpus ingestion failed or skipped: {e}")

    # 4. Ingest Imsidag University of Bouira Corpus (CC-BY-SA 4.0)
    print(f"\n[4/7] Streaming Imsidag-community/kabyle-corpus-ubouira (Target: {max_bouira})...")
    try:
        bouira_ds = load_dataset("Imsidag-community/kabyle-corpus-ubouira", split="train", streaming=True)
        bouira_count = 0
        for row in bouira_ds:
            text = normalize_text(row.get("text", ""))
            if 30 <= len(text) <= 500:
                s = chatml_sample(
                    f"Aḍris s teqbaylit:",
                    f"{text}",
                    sys_prompt, "bouira_corpus", "kabyle"
                )
                if s:
                    collected_samples.append(s)
                    bouira_count += 1
                    if bouira_count >= max_bouira:
                        break
        print(f"      Ingested {bouira_count:,} Bouira Kabyle samples.")
    except Exception as e:
        print(f"      Warning: Bouira corpus ingestion failed or skipped: {e}")

    # 5. Ingest AmaWar Cultural Proverbs, Idioms & Riddles (Apache 2.0)
    print("\n[5/7] Ingesting AmaWar Cultural Wisdom & Riddles...")
    proverbs_file = os.path.join(output_dir, "raw", "proverbs.tsv")
    riddles_file = os.path.join(output_dir, "raw", "riddles.tsv")
    amawar_count = 0
    
    if os.path.exists(proverbs_file):
        with open(proverbs_file, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0] and parts[1]:
                    ber, ar_trans, ar_mean = parts[0], parts[1], parts[2]
                    # Arabic explanation
                    s1 = chatml_sample(
                        f"ما معنى هذا المثل الأمازيغي: '{ber}'؟",
                        f"معنى المثل: '{ar_trans}'. ويقصد به في الحكمة الشعبية: {ar_mean}.",
                        sys_prompt, "amawar_proverb", "arabic"
                    )
                    # Kabyle explanation
                    s2 = chatml_sample(
                        f"Acu i d anamek n yinzi-agi: '{ber}'?",
                        f"Inzi-agi yeqqar-d: '{ber}'. Anamek-is d akken: {ar_trans}.",
                        sys_prompt, "amawar_proverb", "kabyle"
                    )
                    if s1: collected_samples.append(s1); amawar_count += 1
                    if s2: collected_samples.append(s2); amawar_count += 1
                    
    if os.path.exists(riddles_file):
        with open(riddles_file, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    ber_riddle, ar_answer = parts[0], parts[1]
                    s = chatml_sample(
                        f"Fk-iyi-d timseɛreqt tamaziɣt d tririt-is.",
                        f"Timseɛreqt: {ber_riddle}\nTririt: {ar_answer}",
                        sys_prompt, "amawar_riddle", "kabyle"
                    )
                    if s: collected_samples.append(s); amawar_count += 1
    print(f"      Ingested {amawar_count:,} AmaWar cultural items.")

    # 6. Ingest TranslateWiki & Localization Strings (CC-BY 3.0 / MPL 2.0)
    print("\n[6/7] Streaming Tamazight-NLP/translatewiki...")
    try:
        tw_ds = load_dataset("Tamazight-NLP/translatewiki", split="train", streaming=True)
        tw_count = 0
        for row in tw_ds:
            src_str = normalize_text(row.get("source_string", ""))
            tgt_str = normalize_text(row.get("target_string", ""))
            if len(src_str) >= 5 and len(tgt_str) >= 3:
                s = chatml_sample(
                    f"Amek i tettaruḍ awal-agi s tmaziɣt: \"{src_str}\"?",
                    f"Tettaruḍ-t akka: \"{tgt_str}\".",
                    sys_prompt, "translatewiki", "kabyle"
                )
                if s:
                    collected_samples.append(s)
                    tw_count += 1
                    if tw_count >= 1500:
                        break
        print(f"      Ingested {tw_count:,} TranslateWiki terminology pairs.")
    except Exception as e:
        print(f"      Warning: TranslateWiki ingestion skipped: {e}")

    # 7. Ingest Encyclopedic Q&A from tamazight_train.jsonl (Kabyle Wikipedia)
    print("\n[7/7] Ingesting Kabyle Wikipedia Encyclopedic Q&A...")
    wiki_count = 0
    tamazight_train_file = "tamazight_train.jsonl"
    if os.path.exists(tamazight_train_file):
        with open(tamazight_train_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    p = d.get("prompt", "")
                    if "<|im_start|>user" in p and "<|im_start|>assistant" in p:
                        parts = p.split("<|im_start|>assistant\n")
                        user_side = parts[0].replace("<|im_start|>user\n", "").replace("<|im_end|>\n", "").strip()
                        asst_side = parts[1].replace("<|im_end|>", "").strip()
                        
                        # High quality Q&A filter
                        if user_side.startswith("Dacu i d") or user_side.startswith("Amek") or user_side.startswith("Anwa"):
                            cleaned_q = user_side.replace("_", " ")
                            s = chatml_sample(cleaned_q, asst_side, sys_prompt, "kabyle_wikipedia", "kabyle")
                            if s:
                                collected_samples.append(s)
                                wiki_count += 1
                                if wiki_count >= 3000:
                                    break
                except Exception:
                    continue
        print(f"      Ingested {wiki_count:,} Kabyle Wikipedia Q&A pairs.")

    print("\n" + "=" * 70)
    print(f"Total Raw Collected Samples: {len(collected_samples):,}")
    
    # 8. De-contaminate Against Benchmark Datasets
    eval_dir = "evaluation/datasets"
    benchmark_queries = set()
    if os.path.exists(eval_dir):
        for ef in os.listdir(eval_dir):
            if ef.endswith(".jsonl"):
                with open(os.path.join(eval_dir, ef), "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            rec = json.loads(line)
                            p = rec.get("prompt", "")
                            if p:
                                benchmark_queries.add(normalize_text(p).lower())
                                
    uncontaminated = []
    contam_count = 0
    for s in collected_samples:
        u_turn = ""
        for m in s.get("messages", []):
            if m.get("role") == "user":
                u_turn = m.get("content", "")
        if normalize_text(u_turn).lower() in benchmark_queries:
            contam_count += 1
            continue
        uncontaminated.append(s)
        
    print(f"Removed {contam_count} benchmark contamination overlaps.")
    
    # 9. Deduplication
    deduped, n_removed = deduplicate_samples(uncontaminated)
    print(f"Removed {n_removed:,} exact and normalized duplicate prompts.")
    print(f"Clean Unique High-Quality Samples: {len(deduped):,}")
    
    # 10. Save Master Processed File
    master_path = os.path.join(output_dir, "processed", "kabyle_master.jsonl")
    save_jsonl(deduped, master_path)
    print(f"Saved master clean corpus to: {master_path}")
    
    # 11. Deterministic Train/Val/Test Split
    train_set, val_set, test_set = train_val_test_split(
        deduped,
        val_ratio=0.05,
        test_ratio=0.05,
        seed=seed
    )
    
    save_jsonl(train_set, os.path.join(output_dir, "train.jsonl"))
    save_jsonl(val_set, os.path.join(output_dir, "validation.jsonl"))
    save_jsonl(test_set, os.path.join(output_dir, "test.jsonl"))
    
    print("\n" + "=" * 70)
    print("  FINAL DATASET SPLITS")
    print("=" * 70)
    print(f"  Training Split:     {len(train_set):>6,} samples (data/train.jsonl)")
    print(f"  Validation Split:   {len(val_set):>6,} samples (data/validation.jsonl)")
    print(f"  Test Split:         {len(test_set):>6,} samples (data/test.jsonl)")
    print(f"  Total Clean:        {len(deduped):>6,} samples")
    print("=" * 70)

if __name__ == "__main__":
    download_and_process_corpora()
