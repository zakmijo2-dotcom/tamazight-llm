#!/usr/bin/env python3
"""
Export trained Tamazight LLM to GGUF format for mobile (Termux) and edge deployment.
Uses llama.cpp GGUF converter.
"""

import os
import sys
import shutil
import urllib.request
import argparse
import subprocess

CONVERTER_SCRIPT = "convert_hf_to_gguf.py"
CONVERTER_URL = "https://raw.githubusercontent.com/ggerganov/llama.cpp/master/convert_hf_to_gguf.py"

def parse_args():
    parser = argparse.ArgumentParser(description="Export Tamazight LLM to GGUF format")
    default_dir = "tamazight-qwen-0.5b-final" if os.path.exists("tamazight-qwen-0.5b-final") else "tamazight-multilingual-llm-final"
    parser.add_argument("--model-dir", type=str, default=default_dir, help=f"Path to fine-tuned model directory (default: {default_dir})")
    parser.add_argument("--output-file", type=str, default=None, help="Output GGUF file path (default: tamazight-qwen-0.5b-{outtype}.gguf)")
    parser.add_argument("--outtype", type=str, default="f16", choices=["f16", "bf16", "q8_0"], help="Output weight precision type")
    return parser.parse_args()

def get_converter_path():
    bundled = os.path.join("tools", "llama.cpp", "convert_hf_to_gguf.py")
    if os.path.exists(bundled):
        return bundled
    if os.path.exists(CONVERTER_SCRIPT):
        return CONVERTER_SCRIPT
    return None

def ensure_converter():
    conv = get_converter_path()
    if conv:
        return conv
    print(f"Downloading llama.cpp converter ({CONVERTER_SCRIPT})...")
    if shutil.which("curl"):
        res = subprocess.run(["curl", "-sL", "-o", CONVERTER_SCRIPT, CONVERTER_URL])
        if res.returncode == 0 and os.path.exists(CONVERTER_SCRIPT) and os.path.getsize(CONVERTER_SCRIPT) > 1000:
            os.chmod(CONVERTER_SCRIPT, 0o755)
            print("Downloaded convert_hf_to_gguf.py successfully via curl.")
            return CONVERTER_SCRIPT
    try:
        import requests
        r = requests.get(CONVERTER_URL, timeout=30)
        with open(CONVERTER_SCRIPT, "w", encoding="utf-8") as f:
            f.write(r.text)
        os.chmod(CONVERTER_SCRIPT, 0o755)
        print("Downloaded convert_hf_to_gguf.py successfully via requests.")
        return CONVERTER_SCRIPT
    except Exception as e:
        print(f"Error downloading with requests: {e}")

    print(f"Please run manually: curl -sLO {CONVERTER_URL}")
    sys.exit(1)

def main():
    args = parse_args()
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found.")
        print("Please train the model first using: python3 train.py")
        sys.exit(1)

    out_file = args.output_file or f"tamazight-qwen-0.5b-{args.outtype}.gguf"
    converter_path = ensure_converter()

    print("=" * 60)
    print("  ⵣ Tamazight LLM GGUF Mobile / Edge Exporter ⵣ")
    print(f"  Input Model: {args.model_dir}")
    print(f"  Output GGUF: {out_file}")
    print(f"  Precision:   {args.outtype}")
    print("=" * 60)

    cmd = [
        sys.executable,
        converter_path,
        args.model_dir,
        "--outfile", out_file,
        "--outtype", args.outtype
    ]

    env = os.environ.copy()
    llama_tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "llama.cpp")
    if os.path.exists(llama_tools_dir):
        current_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{llama_tools_dir}:{current_pp}" if current_pp else llama_tools_dir

    print(f"\nRunning conversion command:\n{' '.join(cmd)}\n")
    ret = subprocess.run(cmd, env=env)
    if ret.returncode != 0:
        print("\nError during GGUF conversion!")
        sys.exit(ret.returncode)

    if os.path.exists(out_file):
        size_mb = os.path.getsize(out_file) / (1024 * 1024)
        print("\n" + "=" * 60)
        print(f"SUCCESS! GGUF model exported to: {out_file} ({size_mb:.1f} MB)")
        print("=" * 60)
        print("\n📱 To run this model on Android via Termux:")
        print("1. Install llama.cpp in Termux:")
        print("   pkg install clang cmake git")
        print("   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && cmake -B build && cmake --build build")
        print("\n2. Quantize to Q4_K_M (ultra-fast, < 350MB RAM):")
        print(f"   ./build/bin/llama-quantize {out_file} tamazight-qwen-0.5b-Q4_K_M.gguf Q4_K_M")
        print("\n3. Run inference:")
        print("   ./build/bin/llama-cli -m tamazight-qwen-0.5b-Q4_K_M.gguf \\")
        print('     -p "<|im_start|>user\\nAzul fell-awen! Amek telliḍ?<|im_end|>\\n<|im_start|>assistant\\n" \\')
        print("     -n 50 --temp 0.7")
        print("=" * 60)
    else:
        print("Error: Output GGUF file was not created.")

if __name__ == "__main__":
    main()
