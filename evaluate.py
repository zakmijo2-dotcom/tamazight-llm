#!/usr/bin/env python3
"""
Root entrypoint delegating to evaluation/benchmark.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluation.benchmark import main

if __name__ == "__main__":
    main()
