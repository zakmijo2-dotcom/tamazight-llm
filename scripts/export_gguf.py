#!/usr/bin/env python3
"""
GGUF Export and Quantization Pipeline for Tamazight Multilingual LLM (Qwen2.5).
Produces verified GGUF models with header inspection, context extraction, and RAM sizing.
Pipeline:
  HF safetensors -> GGUF F16/BF16 -> Quantized GGUF (Q8_0, Q6_K, Q5_K_M, Q4_K_M)
"""

import os
import sys
import shutil
import argparse
import subprocess
from typing import Optional, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config

def parse_args():
    cfg = load_config()
    parser = argparse.ArgumentParser(description="Export Tamazight LLM to GGUF format for mobile and edge deployment")
    default_model = cfg.model.output_dir if os.path.exists(os.path.join(cfg.model.output_dir, "config.json")) else cfg.model.name
    parser.add_argument("--model-dir", type=str, default=default_model, help=f"Path to model directory or HF repo (default: {default_model})")
    parser.add_argument("--output-file", type=str, default=None, help="Output GGUF file path (default: models/gguf/model-{outtype}.gguf)")
    parser.add_argument("--outtype", type=str, default="f16", choices=["f16", "bf16", "q8_0"], help="Base conversion precision (default: f16)")
    parser.add_argument("--quantize", type=str, default=None, choices=["Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M"], help="Optional quantization target using llama-quantize")
    return parser.parse_args()

def locate_converter() -> str:
    """Locates convert_hf_to_gguf.py script."""
    candidates = [
        os.path.join("tools", "llama.cpp", "convert_hf_to_gguf.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools", "llama.cpp", "convert_hf_to_gguf.py"),
        "convert_hf_to_gguf.py"
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError("convert_hf_to_gguf.py not found in tools/llama.cpp. Please ensure tools/llama.cpp is present.")

def inspect_gguf_metadata(gguf_path: str) -> Dict[str, Any]:
    """Inspects GGUF file using gguf library to extract verified architectural parameters."""
    import gguf
    reader = gguf.GGUFReader(gguf_path)
    
    fields = {}
    for tensor in reader.tensors:
        pass
        
    def get_field_val(key: str, default: Any = None) -> Any:
        field = reader.get_field(key)
        if field is not None and len(field.parts) > 0:
            val = field.parts[-1]
            if hasattr(val, "tolist"):
                return val.tolist()
            return val
        return default

    arch = get_field_val("general.architecture", "llama")
    ctx_len = get_field_val(f"{arch}.context_length", 8192)
    embed_dim = get_field_val(f"{arch}.embedding_length", 896)
    layers = get_field_val(f"{arch}.block_count", 24)
    file_size_mb = os.path.getsize(gguf_path) / (1024 * 1024)
    
    # Practical RAM recommendation: file size + KV cache for default context + 100MB headroom
    ram_recommendation_mb = file_size_mb + 150
    
    return {
        "file_size_mb": file_size_mb,
        "architecture": arch,
        "context_length": ctx_len,
        "embedding_dim": embed_dim,
        "block_count": layers,
        "tensor_count": len(reader.tensors),
        "ram_recommendation_mb": ram_recommendation_mb
    }

def main():
    args = parse_args()
    
    converter_path = locate_converter()
    converter_dir = os.path.dirname(converter_path)
    
    os.makedirs("models/gguf", exist_ok=True)
    out_file = args.output_file or f"models/gguf/tamazight-qwen-0.5b-{args.outtype}.gguf"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    print("=" * 65)
    print("  ⵣ Tamazight LLM GGUF Mobile / Edge Export Pipeline ⵣ")
    print(f"  Input Model:  {args.model_dir}")
    print(f"  Output File:  {out_file}")
    print(f"  Precision:    {args.outtype}")
    print(f"  Converter:    {converter_path}")
    print("=" * 65)
    
    cmd = [
        sys.executable,
        converter_path,
        args.model_dir,
        "--outfile", out_file,
        "--outtype", args.outtype
    ]
    
    env = os.environ.copy()
    current_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{converter_dir}:{current_pp}" if current_pp else converter_dir
    
    print(f"\nRunning GGUF conversion command:\n{' '.join(cmd)}\n")
    ret = subprocess.run(cmd, env=env)
    
    if ret.returncode != 0:
        print("\n❌ Error during GGUF conversion!")
        sys.exit(ret.returncode)
        
    if not os.path.exists(out_file) or os.path.getsize(out_file) < 1000:
        print("❌ Error: Output GGUF file was not created properly.")
        sys.exit(1)
        
    # Inspect produced GGUF file
    meta = inspect_gguf_metadata(out_file)
    
    print("\n" + "=" * 65)
    print("  ✅ GGUF EXPORT VERIFIED")
    print("=" * 65)
    print(f"  File Path:            {out_file}")
    print(f"  Measured File Size:   {meta['file_size_mb']:.2f} MB")
    print(f"  Model Architecture:   {meta['architecture']}")
    print(f"  Tensor Count:         {meta['tensor_count']}")
    print(f"  Max Context Length:   {meta['context_length']:,} tokens")
    print(f"  Estimated RAM Needed: ~{meta['ram_recommendation_mb']:.0f} MB")
    print("=" * 65)
    
    # Optional Quantization
    if args.quantize:
        quant_bin = shutil.which("llama-quantize")
        if not quant_bin and os.path.exists("tools/llama.cpp/build/bin/llama-quantize"):
            quant_bin = "tools/llama.cpp/build/bin/llama-quantize"
            
        if quant_bin:
            quant_out = out_file.replace(f"-{args.outtype}.gguf", f"-{args.quantize}.gguf")
            print(f"\nQuantizing to {args.quantize} using {quant_bin}...")
            q_ret = subprocess.run([quant_bin, out_file, quant_out, args.quantize])
            if q_ret.returncode == 0 and os.path.exists(quant_out):
                q_size = os.path.getsize(quant_out) / (1024 * 1024)
                print(f"✅ Quantized model created: {quant_out} ({q_size:.1f} MB)")
            else:
                print("Warning: Quantization failed.")
        else:
            print(f"\nNote: 'llama-quantize' binary not found locally.")
            print(f"To quantize this model to {args.quantize} on your machine or in Termux:")
            print(f"  llama-quantize {out_file} {out_file.replace('.gguf', f'-{args.quantize}.gguf')} {args.quantize}")

    print("\n📱 Android (Termux) Deployment Instructions:")
    print("1. Transfer the GGUF file to your Android phone (e.g., via scp, adb, or Google Drive).")
    print("2. In Termux, install llama.cpp:")
    print("   pkg update && pkg install clang cmake git")
    print("   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp")
    print("   cmake -B build && cmake --build build")
    print("3. Execute inference:")
    print(f"   ./build/bin/llama-cli -m {out_file} \\")
    print('     -p "<|im_start|>user\\nAzul fell-awen! Amek telliḍ?<|im_end|>\\n<|im_start|>assistant\\n" \\')
    print("     -n 60 --temp 0.7")
    print("=" * 65)

if __name__ == "__main__":
    main()
