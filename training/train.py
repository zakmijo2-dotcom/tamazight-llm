#!/usr/bin/env python3
"""
Tamazight Multilingual LLM Training CLI Entrypoint.
Usage:
    python training/train.py --preset cpu-fast
    python training/train.py --preset gpu-full --epochs 3
"""

import os
import sys
import argparse

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.training import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Train Tamazight Multilingual LLM (Qwen2.5-0.5B)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (default: configs/config.yaml)")
    parser.add_argument("--preset", type=str, default=None, choices=["cpu-fast", "cpu-standard", "gpu-full"],
                        help="Execution preset: cpu-fast (1 epoch, light seq len), cpu-standard, gpu-full")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override per-device batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override peak learning rate")
    parser.add_argument("--max-samples", type=int, default=None, help="Override max training samples")
    parser.add_argument("--output-dir", type=str, default=None, help="Override model output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from checkpoint directory")
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    
    # Apply CLI overrides to config
    if args.output_dir:
        config.model.output_dir = args.output_dir
    if args.lr:
        config.training.learning_rate = args.lr
        
    train_model(
        config=config,
        resume_from_checkpoint=args.resume,
        max_samples=args.max_samples,
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        preset=args.preset
    )

if __name__ == "__main__":
    main()
