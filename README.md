# Tamazight Multilingual LLM (Qwen2.5-0.5B & 1.5B) ⵣ

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/zakmijo2-dotcom/tamazight-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/zakmijo2-dotcom/tamazight-llm/actions/workflows/ci.yml)
![Language: Python](https://img.shields.io/badge/Language-Python%203.10+-blue.svg)
![Target: Android / Termux](https://img.shields.io/badge/Target-Android%20%7C%20Termux-3DDC84.svg)

A multilingual conversational language model based on **`Qwen/Qwen2.5-0.5B-Instruct`** and **`Qwen/Qwen2.5-1.5B-Instruct`**, fine-tuned for **Tamazight (Kabyle / Taqbaylit in Latin script)** while maintaining conversational competence in **Arabic**, **French**, and **English**.

Optimized for local edge inference on Android devices via **Termux** and **`llama.cpp`**.

---

## 1. What It Is

`tamazight-llm` is an open-source training, evaluation, and export pipeline for fine-tuning compact Large Language Models on the Tamazight language. The project provides:
- Clean, curated 4-language conversational datasets (Tamazight, Arabic, French, English).
- SFT (Supervised Fine-Tuning) training harness with assistant-only loss masking.
- Data contamination auditor ensuring 0% benchmark leakage.
- GGUF export and quantization tools for mobile edge execution.

---

## 2. Why It Exists

Tamazight is a low-resource language underrepresented in commercial foundation models, which often fail on basic vocabulary, confuse Amazigh dialects, or hallucinate syntax. 

This project demonstrates that fine-tuning compact (0.5B to 1.5B) parameter models using strict assistant-masked loss and high-quality, normalized corpora produces an accurate, edge-capable assistant that runs locally on commodity mobile hardware without cloud dependencies.

---

## 3. Features

- **Multilingual Support**: Trained on paired and conversational dialogues across Tamazight (Kabyle, Latin script), Modern Standard Arabic, French, and English.
- **Strict Assistant-Only Loss Masking**: User prompts and system templates are masked out (`labels = -100`) so the model only updates weights on assistant responses.
- **Contamination Guard**: Built-in `scripts/check_data_leakage.py` audit verifies 0% overlap between training samples and evaluation splits.
- **Edge Deployment (GGUF)**: Direct conversion to 4-bit quantized GGUF (`Q4_K_M`) requiring ~350 MB RAM, running CPU inference via `llama.cpp` on Android.
- **Centralized Configuration**: All hyperparameters, LoRA target modules, and dataset paths configured in `configs/config.yaml`.

---

## 4. Install and Run

### Prerequisites
- Python 3.10+
- PyTorch & Hugging Face Transformers
- `git`

### Installation

```bash
git clone https://github.com/zakmijo2-dotcom/tamazight-llm.git
cd tamazight-llm
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running Test Verification

```bash
# Verify 0% data leakage between train and benchmark splits
python scripts/check_data_leakage.py

# Run compilation check
python -m compileall src/ training/ evaluation/ scripts/ tests/

# Run unit tests
pytest -v tests/
```

---

## 5. Usage Example

### Interactive CLI Chat

```bash
python chat.py --model-path "models/tamazight-qwen-0.5b" --device cpu
```

### Mobile Execution on Android (Termux via llama.cpp)

1. Export the fine-tuned model to GGUF format:
   ```bash
   python export_gguf.py --model-dir models/tamazight-qwen-0.5b --out-dir models/gguf/
   ```
2. Quantize to 4-bit inside Termux:
   ```bash
   ./build/bin/llama-quantize models/gguf/tamazight-qwen-0.5b-f16.gguf tamazight-qwen-0.5b-Q4_K_M.gguf Q4_K_M
   ```
3. Run prompt inference:
   ```bash
   ./build/bin/llama-cli -m tamazight-qwen-0.5b-Q4_K_M.gguf \
     -p "<|im_start|>user\nAzul fell-awen! Amek telliḍ?<|im_end|>\n<|im_start|>assistant\n" \
     -n 60 --temp 0.7
   ```

---

## 6. Project Status & Limitations

- **Status**: Stable v1.0.0. Dataset pipeline, training harness, benchmark suite, and GGUF export workflows are verified and passing CI.
- **Dialect Scope**: Training data primarily covers the Kabyle (`Taqbaylit`) dialect in standard Latin orthography (`Agemmay Alatini`).
- **Hardware Profile**: The 0.5B model fits comfortably on mobile silicon with low thermal impact; complex multi-step reasoning remains bounded by model capacity.

---

## 7. Contributing

Contributions to dataset expansion, dialect evaluation benchmarks, and documentation are welcome. Please consult [CONTRIBUTING.md](CONTRIBUTING.md) and adhere to our [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## 8. License & Citation

This project is licensed under the **[Apache-2.0 License](LICENSE)**.

```bibtex
@misc{tamazight_qwen_2026,
  author = {Mijozak},
  title = {Tamazight Multilingual LLM (Qwen2.5-0.5B)},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/zakmijo2-dotcom/tamazight-llm}
}
```
