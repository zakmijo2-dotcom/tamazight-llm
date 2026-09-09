#!/usr/bin/env python3
"""
Master Data Preparation, Deduplication, and Splitting Pipeline.
Transforms raw corpora and TSV files into clean, metadata-enriched datasets:
  data/train.jsonl
  data/validation.jsonl
  data/test.jsonl
"""

import os
import sys
import argparse
from collections import Counter

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.data import (
    create_sample,
    deduplicate_samples,
    train_val_test_split,
    save_jsonl
)

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare, normalize, deduplicate, and split Tamazight multilingual dataset")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw TSV and JSONL data")
    parser.add_argument("--output-dir", type=str, default="data", help="Target output directory for train/val/test splits")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Validation set split ratio (default: 0.05)")
    parser.add_argument("--test-ratio", type=float, default=0.05, help="Test set split ratio (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splitting")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config()
    sys_prompt = cfg.system.prompt
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "processed"), exist_ok=True)
    
    print("=" * 65)
    print("  ⵣ Dataset Preprocessing, Normalization & Deduplication ⵣ")
    print(f"  Raw Directory:  {args.raw_dir}")
    print(f"  Output Splits:  {args.output_dir}")
    print(f"  Val / Test:     {args.val_ratio * 100:.1f}% / {args.test_ratio * 100:.1f}%")
    print("=" * 65)
    
    raw_samples = []
    
    # 1. Ingest Core Expert Multilingual Dialogues
    from src.seed_dialogues import (
        TAMAZIGHT_CORE,
        ARABIC_CORE,
        FRENCH_CORE,
        ENGLISH_CORE,
        CROSS_LINGUAL
    )
    
    for u, a in TAMAZIGHT_CORE:
        s = create_sample(u, a, sys_prompt, source="curated_expert", language_override="kabyle")
        if s: raw_samples.append(s)
        
    for u, a in ARABIC_CORE:
        s = create_sample(u, a, sys_prompt, source="curated_expert", language_override="arabic")
        if s: raw_samples.append(s)
        
    for u, a in FRENCH_CORE:
        s = create_sample(u, a, sys_prompt, source="curated_expert", language_override="french")
        if s: raw_samples.append(s)
        
    for u, a in ENGLISH_CORE:
        s = create_sample(u, a, sys_prompt, source="curated_expert", language_override="english")
        if s: raw_samples.append(s)
        
    for u, a in CROSS_LINGUAL:
        s = create_sample(u, a, sys_prompt, source="curated_cross_lingual")
        if s: raw_samples.append(s)
        
    print(f"[1/4] Ingested {len(raw_samples)} core expert dialogues.")
    
    # 2. Ingest Cultural Proverbs and Idioms from data/raw/
    proverbs_path = os.path.join(args.raw_dir, "proverbs.tsv")
    if os.path.exists(proverbs_path):
        count = 0
        with open(proverbs_path, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3 and parts[0] and parts[1]:
                    ber, ar_trans, ar_mean = parts[0], parts[1], parts[2]
                    # Arabic cultural explanation
                    s1 = create_sample(
                        f"ما معنى هذا المثل الأمازيغي: '{ber}'؟",
                        f"معنى المثل: '{ar_trans}'. ويقصد به في الحكمة الشعبية: {ar_mean}.",
                        sys_prompt, source="proverbs_tsv", language_override="arabic"
                    )
                    if s1: raw_samples.append(s1); count += 1
                    # Tamazight cultural inquiry
                    s2 = create_sample(
                        f"Acu i d anamek n yinzi-agi: '{ber}'?",
                        f"Inzi-agi yeqqar-d: '{ber}'. Anamek-is d akken: {ar_trans}.",
                        sys_prompt, source="proverbs_tsv", language_override="kabyle"
                    )
                    if s2: raw_samples.append(s2); count += 1
        print(f"[2/4] Ingested {count} cultural proverb Q&A pairs.")
        
    # 3. Ingest Encyclopedic Tamazight Q&As from tamazight_train.jsonl
    jsonl_path = "tamazight_train.jsonl"
    if os.path.exists(jsonl_path):
        wiki_count = 0
        import json
        with open(jsonl_path, "r", encoding="utf-8") as f:
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
                        
                        # Only take substantive questions, ignore bare translation echo
                        if user_side.startswith("Dacu i d") or user_side.startswith("Amek") or user_side.startswith("Anwa"):
                            cleaned_q = user_side.replace("_", " ")
                            s = create_sample(cleaned_q, asst_side, sys_prompt, source="wikipedia_kabyle", language_override="kabyle")
                            if s:
                                raw_samples.append(s)
                                wiki_count += 1
                                if wiki_count >= 3000:
                                    break
                except Exception:
                    continue
        print(f"[3/4] Ingested {wiki_count} encyclopedic Tamazight Q&A pairs.")

    print(f"\nTotal raw collected samples: {len(raw_samples):,}")
    # 4. De-contaminate against evaluation/datasets/ benchmarks
    eval_dir = "evaluation/datasets"
    benchmark_turns = set()
    if os.path.exists(eval_dir):
        from src.data import normalize_text
        for ef in os.listdir(eval_dir):
            if ef.endswith(".jsonl"):
                with open(os.path.join(eval_dir, ef), "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            d = json.loads(line)
                            p = d.get("prompt", "")
                            if p:
                                benchmark_turns.add(normalize_text(p).lower())

    clean_uncontaminated = []
    n_contam_removed = 0
    for s in raw_samples:
        u_turn = ""
        for m in s.get("messages", []):
            if m.get("role") == "user":
                u_turn = m.get("content", "")
        if normalize_text(u_turn).lower() in benchmark_turns:
            n_contam_removed += 1
            continue
        clean_uncontaminated.append(s)

    if n_contam_removed > 0:
        print(f"Removed {n_contam_removed} benchmark contamination overlaps.")

    # 5. Deduplication
    deduped_samples, n_removed = deduplicate_samples(clean_uncontaminated)
    print(f"Removed {n_removed:,} exact duplicate samples.")
    print(f"Clean unique samples: {len(deduped_samples):,}")
    # Save full processed dataset
    full_processed_path = os.path.join(args.output_dir, "processed", "full_dataset.jsonl")
    save_jsonl(deduped_samples, full_processed_path)
    print(f"Saved complete processed dataset to: {full_processed_path}")
    
    # 5. Deterministic Train/Validation/Test Split
    train_set, val_set, test_set = train_val_test_split(
        deduped_samples,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "validation.jsonl")
    test_path = os.path.join(args.output_dir, "test.jsonl")
    
    save_jsonl(train_set, train_path)
    save_jsonl(val_set, val_path)
    save_jsonl(test_set, test_path)
    
    # Statistics Breakdown
    lang_counter = Counter(s.get("language", "unknown") for s in deduped_samples)
    script_counter = Counter(s.get("script", "unknown") for s in deduped_samples)
    
    print("\n" + "=" * 65)
    print("  DATASET PIPELINE SUMMARY")
    print("=" * 65)
    print(f"  Training Split:    {len(train_set):>6,} samples ({train_path})")
    print(f"  Validation Split:  {len(val_set):>6,} samples ({val_path})")
    print(f"  Test Split:        {len(test_set):>6,} samples ({test_path})")
    print(f"  Total Clean:       {len(deduped_samples):>6,} samples")
    print("-" * 65)
    print("  Language Distribution:")
    for lang, count in lang_counter.most_common():
        print(f"    - {lang:<16}: {count:>5,} ({count * 100 / len(deduped_samples):.1f}%)")
    print("  Script Distribution:")
    for script, count in script_counter.most_common():
        print(f"    - {script:<16}: {count:>5,} ({count * 100 / len(deduped_samples):.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    main()
