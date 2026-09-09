#!/usr/bin/env python3
"""
Upload trained Tamazight LLM to Hugging Face Hub
Usage:
    python3 upload_to_hf.py --repo-id "YOUR_USERNAME/tamazight-smollm-135m" --token "hf_..."
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo

MODEL_CARD_TEMPLATE = """---
language:
- ber
- kab
license: apache-2.0
tags:
- tamazight
- berber
- kabyle
- latin-script
- llm
- smollm2
- text-generation
- termux
- codespaces
datasets:
- Sifal/Kabyle-French
- Sifal/KabyleWikipedia
- Tamazight-NLP/AmaWar
pipeline_tag: text-generation
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
---

# Tamazight (Berber) SmolLM2-135M ⵣ

An efficient open-source Language Model fine-tuned on **Tamazight (Kabyle/Berber)** written in the standard **Latin alphabet (Agemmay Alatini)**.

Based on `HuggingFaceTB/SmolLM2-135M-Instruct`, optimized for high-speed inference on mobile devices (Termux/Android) and edge environments.

## Model Highlights
- **Architecture**: SmolLM2 (Llama-based autoregressive transformer)
- **Parameters**: 135 Million parameters
- **Target Language**: Tamazight / Taqbaylit (Latin script)
- **Training Data**:
  - Kabyle-French parallel translation corpus (~115,000 sentence pairs)
  - Kabyle Wikipedia articles (over 8,800 texts)
  - AmaWar cultural corpus (proverbs, expressions, riddles)
- **Deployment Targets**: GitHub Codespaces, Termux / Android via llama.cpp (GGUF), Python transformers.

## Usage with Transformers

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)

prompt = "<|im_start|>user\\nAzul! Amek telliḍ?<|im_end|>\\n<|im_start|>assistant\\n"
inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(**inputs, max_new_tokens=40, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Running on Termux (Mobile)
This model is lightweight enough to run locally inside **Termux** on Android using `llama.cpp` at ~40-60 tokens per second!
"""

def main():
    parser = argparse.ArgumentParser(description="Upload Tamazight LLM to Hugging Face")
    default_dir = "tamazight-smollm-135m-final" if os.path.exists("tamazight-smollm-135m-final") else "tamazight-smollm-135m"
    parser.add_argument("--model-dir", type=str, default=default_dir, help=f"Path to model directory (default: {default_dir})")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID (e.g. username/tamazight-smollm-135m)")
    parser.add_argument("--token", type=str, default=os.getenv("HF_TOKEN"), help="Hugging Face write token")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    args = parser.parse_args()

    if not args.token:
        print("Error: Hugging Face token is required. Pass --token 'hf_...' or set HF_TOKEN environment variable.")
        return

    print(f"Creating / verifying repo: {args.repo_id}...")
    api = HfApi(token=args.token)
    create_repo(repo_id=args.repo_id, token=args.token, private=args.private, exist_ok=True)

    # Generate Model Card
    card_path = os.path.join(args.model_dir, "README.md")
    if not os.path.exists(card_path):
        with open(card_path, "w", encoding="utf-8") as f:
            f.write(MODEL_CARD_TEMPLATE.format(repo_id=args.repo_id))

    print(f"Uploading files from {args.model_dir} to Hugging Face...")
    api.upload_folder(
        folder_path=args.model_dir,
        repo_id=args.repo_id,
        repo_type="model"
    )
    print(f"Model successfully published to: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
