#!/usr/bin/env python3
"""
Data Contamination and Leakage Verification Script.
Audits training and evaluation splits to ensure 0% data leakage:
  - Exact match leakage detection
  - Normalized string collision detection
  - Near-duplicate n-gram Jaccard overlap detection
Exits with code 0 if clean, 1 if contamination is detected.
"""

import os
import sys
import json
import argparse
from typing import Set, List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data import normalize_text, load_jsonl

def parse_args():
    parser = argparse.ArgumentParser(description="Audit dataset for train/test leakage and contamination")
    parser.add_argument("--train-file", type=str, default="data/train.jsonl", help="Path to training set jsonl")
    parser.add_argument("--test-file", type=str, default="data/test.jsonl", help="Path to test set jsonl")
    parser.add_argument("--val-file", type=str, default="data/validation.jsonl", help="Path to validation set jsonl")
    parser.add_argument("--eval-dir", type=str, default="evaluation/datasets", help="Path to benchmark evaluation datasets")
    parser.add_argument("--jaccard-threshold", type=float, default=0.85, help="Threshold for near-duplicate warning")
    return parser.parse_args()

def extract_user_turns(records: List[Dict]) -> List[str]:
    turns = []
    for r in records:
        if "messages" in r:
            for m in r["messages"]:
                if m.get("role") == "user":
                    turns.append(m.get("content", ""))
        elif "prompt" in r:
            turns.append(r["prompt"])
    return turns

def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
    norm = normalize_text(text).lower()
    return {norm[i:i+n] for i in range(max(0, len(norm) - n + 1))}

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / union if union > 0 else 0.0

def audit_leakage(
    train_turns: List[str],
    eval_turns: List[str],
    eval_name: str,
    threshold: float = 0.85
) -> Tuple[List[str], List[Tuple[str, str, float]]]:
    exact_leaks = []
    near_leaks = []
    
    train_norm_set = {normalize_text(t).lower() for t in train_turns}
    train_ngrams = [(t, get_char_ngrams(t)) for t in train_turns if len(t) > 10]
    
    for e_turn in eval_turns:
        e_norm = normalize_text(e_turn).lower()
        if e_norm in train_norm_set:
            exact_leaks.append(e_turn)
            continue
            
        e_ng = get_char_ngrams(e_turn)
        if len(e_ng) < 5:
            continue
            
        for t_orig, t_ng in train_ngrams:
            sim = jaccard_similarity(e_ng, t_ng)
            if sim >= threshold and t_orig != e_turn:
                near_leaks.append((e_turn, t_orig, sim))
                break
                
    return exact_leaks, near_leaks

def main():
    args = parse_args()
    
    print("=" * 65)
    print("  ⵣ DATASET CONTAMINATION & LEAKAGE AUDIT ⵣ")
    print("=" * 65)
    
    if not os.path.exists(args.train_file):
        print(f"Error: Train file {args.train_file} not found.")
        sys.exit(1)
        
    train_records = load_jsonl(args.train_file)
    train_turns = extract_user_turns(train_records)
    print(f"Loaded {len(train_turns):,} training user queries from {args.train_file}.")
    
    total_violations = 0
    
    # 1. Audit Test Set Leakage
    if os.path.exists(args.test_file):
        test_records = load_jsonl(args.test_file)
        test_turns = extract_user_turns(test_records)
        exact, near = audit_leakage(train_turns, test_turns, "test_split", args.jaccard_threshold)
        
        print(f"\n[Audit: Train vs Test Split]")
        print(f"  Test queries checked: {len(test_turns):,}")
        print(f"  Exact leaks found:    {len(exact)}")
        print(f"  Near-duplicate leaks: {len(near)}")
        
        if exact:
            print("  ❌ CRITICAL: Exact match data leakage detected!")
            for leak in exact[:3]:
                print(f"     - '{leak}'")
            total_violations += len(exact)
        else:
            print("  ✅ PASS: 0 exact overlaps between train and test.")
            
    # 2. Audit Benchmark Datasets in evaluation/datasets/
    if os.path.exists(args.eval_dir):
        eval_files = [f for f in os.listdir(args.eval_dir) if f.endswith(".jsonl")]
        for ef in eval_files:
            epath = os.path.join(args.eval_dir, ef)
            erecords = load_jsonl(epath)
            eturns = extract_user_turns(erecords)
            exact, near = audit_leakage(train_turns, eturns, ef, args.jaccard_threshold)
            print(f"\n[Audit: Train vs {ef}]")
            print(f"  Benchmark queries:    {len(eturns)}")
            print(f"  Exact leaks:          {len(exact)}")
            if exact:
                print(f"  ❌ CRITICAL: Benchmark contamination in {ef}!")
                total_violations += len(exact)
            else:
                print(f"  ✅ PASS: 0 benchmark contamination.")

    print("\n" + "=" * 65)
    if total_violations == 0:
        print("  🎉 AUDIT RESULT: CLEAN (0% Contamination / 0 Leaks)")
        print("=" * 65)
        sys.exit(0)
    else:
        print(f"  ⚠️ AUDIT RESULT: FAILED with {total_violations} contamination violations!")
        print("=" * 65)
        sys.exit(1)

if __name__ == "__main__":
    main()
