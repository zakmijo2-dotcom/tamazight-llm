#!/usr/bin/env python3
"""
Root entrypoint delegating to training/train.py.
"""
import os
import sys

# Ensure repository root is on Python module search path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train import main

if __name__ == "__main__":
    main()
