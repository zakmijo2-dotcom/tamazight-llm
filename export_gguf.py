#!/usr/bin/env python3
"""
Root entrypoint delegating to scripts/export_gguf.py.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.export_gguf import main

if __name__ == "__main__":
    main()
