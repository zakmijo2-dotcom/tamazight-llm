# Tamazight Multilingual LLM (Qwen2.5-1.5B & 0.5B) ⵣ

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/zakmijo2-dotcom/tamazight-llm)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI](https://github.com/zakmijo2-dotcom/tamazight-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/zakmijo2-dotcom/tamazight-llm/actions/workflows/ci.yml)

A high-performance multilingual conversational language model based on **`Qwen/Qwen2.5-1.5B-Instruct`** (~1.54 Billion parameters) and **`Qwen/Qwen2.5-0.5B-Instruct`** (~490 Million parameters), optimized and fine-tuned for **Tamazight (Kabyle / Taqbaylit in Latin script)** while preserving native conversational fluency in **Arabic**, **French**, and **English**.

Designed for cloud training, local development, and mobile edge inference on Android devices via **Termux** and **`llama.cpp`**.
---

## 🌟 Key Features

1. **True Multilingual Conversational AI:**
   - **Tamazight / Taqbaylit (Latin script):** Daily dialogues, cultural heritage, proverbs, encyclopedic knowledge.
   - **Arabic (العربية الفصحى):** Native fluency in dialogue, science, culture, and history.
   - **French (Français):** Fluent conversational reasoning and cross-lingual explanation.
   - **English:** General assistant capabilities and multilingual translation support.
2. **Strict SFT Assistant-Only Loss Masking:**
   - System prompts, user queries, and template tokens are masked (`labels = -100`).
   - The model learns exclusively to generate high-quality assistant responses without repeating user queries.
3. **Robust Data Pipeline & Zero Benchmark Contamination:**
   - Automated Unicode NFKC normalization, whitespace cleanup, and prompt-level deduplication.
   - Strict 0% data leakage verified by `scripts/check_data_leakage.py` against test and benchmark splits.
4. **Mobile & Edge Optimized (GGUF):**
   - Direct export to GGUF format (`Q4_K_M` quantization ~350 MB RAM requirement).
   - Fast CPU inference on modern mobile processors via `llama.cpp`.
5. **Centralized Configuration:**
   - Single source of truth in `configs/config.yaml`. Model architectures and hyperparameters are completely configurable.

---

## 📁 Repository Structure

```text
tamazight-llm/
├── configs/
│   ├── config.yaml              # Centralized configuration (model, LoRA, training, paths)
│   └── datasets.yaml            # Official dataset registry (HCA, UMMTO, Bouira, AmaWar, etc.)
├── data/
│   ├── raw/                     # Raw original source TSVs and Hugging Face mirrors
│   ├── processed/               # Master deduplicated dataset (8,589 samples)
│   ├── train.jsonl              # Clean training split (7,731 samples)
│   ├── validation.jsonl         # Clean validation split (429 samples)
│   └── test.jsonl               # Held-out evaluation split (429 samples)
├── src/
│   ├── __init__.py
│   ├── config.py                # Typed dataclass config loader
│   ├── chat_template.py         # Qwen2.5 native ChatML template and SFT encoder
│   ├── data.py                  # Normalization, script detection, deduplication
│   ├── seed_dialogues.py        # Curated 4-language seed conversations
│   ├── model.py                 # Model loader, LoRA attachment, and artifact export
│   ├── training.py              # SFT Trainer with assistant-only loss masking
│   └── inference.py             # Multilingual inference and interactive chat engine
├── training/
│   └── train.py                 # Modular training CLI
├── evaluation/
│   ├── benchmark.py             # Evaluation suite (PPL, 4-language accuracy, translation)
│   ├── datasets/                # Standardized evaluation benchmarks (6 test sets)
│   └── results/                 # Output JSON reports and CSV summaries
├── scripts/
│   ├── download_kabyle_data.py  # Automated Hugging Face dataset download & synthesis
│   ├── prepare_data.py          # Data ingestion, deduplication, and splitting pipeline
│   ├── check_data_leakage.py    # Contamination auditor (0% leakage check)
│   ├── analyze_tokenizer.py     # Tokenizer efficiency & Kabyle phonology stress-test
│   └── export_gguf.py           # GGUF converter, quantizer, and metadata inspector
├── tests/                       # Complete pytest unit test suite (16 tests)
├── tools/llama.cpp/             # Bundled offline GGUF conversion tools
├── chat.py                      # Root interactive terminal chat launcher
├── infer.py                     # Root single-query & translation CLI
├── train.py                     # Root training launcher
├── evaluate.py                  # Root evaluation launcher
├── export_gguf.py               # Root GGUF exporter
├── upload_to_hf.py              # Safe Hugging Face Hub uploader
├── Tamazight_LLM_Colab.ipynb    # Google Colab GPU training notebook
├── requirements.txt             # Clean pip dependencies
├── MODEL_CARD.md                # Detailed model documentation and limitations
├── DATASET_CARD.md              # Detailed dataset documentation and scope
├── CONTRIBUTING.md               # Guidelines for contributors
├── CITATION.cff                 # Citation metadata
└── LICENSE                      # Apache-2.0 License
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/zakmijo2-dotcom/tamazight-llm.git
cd tamazight-llm

# Install dependencies
pip install -r requirements.txt
```

### 2. Interactive Terminal Chat (4 Languages)

Converse immediately with the model in Tamazight, Arabic, French, or English:

```bash
python chat.py
```

Or execute single queries directly:
```bash
# Tamazight query
python infer.py --prompt "Azul fell-awen! Amek telliḍ?"

# Arabic query
python infer.py --prompt "مرحباً! كيف حالك وماذا يمكنك أن تفعل؟"

# French query
python infer.py --prompt "Bonjour ! Peux-tu m'expliquer ce que tu sais faire ?"

# English query
python infer.py --prompt "Hello! Can you tell me about Amazigh heritage?"

# Translation helper
python infer.py --translate "Bonjour mon ami, comment vas-tu ?"
```

---

## 🏋️ Model Training & Fine-Tuning

Training can be conducted on local CPUs, cloud GPUs (such as Google Colab or cloud workstations), or GitHub Codespaces depending on available compute resources.

### Fast Training on CPU (Codespaces / Local Workstation)
Optimized preset with reduced sequence length and key projection LoRA layers (~10–15 minutes on a standard 2-core CPU):
```bash
python train.py --preset cpu-fast
```

### Standard Balanced Training on CPU
```bash
python train.py --preset cpu-standard --epochs 2
```

### Full GPU Training (Google Colab / Cloud Workstation)
Trains on all samples with full 7-layer LoRA projections and mixed precision:
```bash
python train.py --preset gpu-full --epochs 3 --batch-size 16 --grad-accum 2
```

*(You can also use the 1-click Jupyter notebook: `Tamazight_LLM_Colab.ipynb`).*

---

## 📊 Evaluation & Benchmarking

The evaluation framework audits both **language modeling perplexity** on held-out test splits and **multi-domain conversational accuracy** across 6 standardized test sets:

```bash
python evaluate.py
```

To run a side-by-side comparison between the fine-tuned checkpoint and the base model:
```bash
python evaluate.py --compare-base
```

Results are saved automatically to:
- `evaluation/results/benchmark_results.json`
- `evaluation/results/benchmark_summary.csv`

---

## 📱 Mobile Deployment (Android via Termux & GGUF)

The model can be converted to GGUF format for edge execution on mobile devices:

### 1. Export to GGUF
```bash
python export_gguf.py --outtype f16
```

### 2. Run on Android via Termux
1. Install `llama.cpp` inside Termux on your Android phone:
   ```bash
   pkg update && pkg install clang cmake git
   git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
   cmake -B build && cmake --build build
   ```
2. Quantize the model (e.g. to `Q4_K_M` ~350 MB RAM requirement):
   ```bash
   ./build/bin/llama-quantize models/gguf/tamazight-qwen-0.5b-f16.gguf tamazight-qwen-0.5b-Q4_K_M.gguf Q4_K_M
   ```
3. Run inference:
   ```bash
   ./build/bin/llama-cli -m tamazight-qwen-0.5b-Q4_K_M.gguf \
     -p "<|im_start|>user\nAzul fell-awen! Amek telliḍ?<|im_end|>\n<|im_start|>assistant\n" \
     -n 60 --temp 0.7
   ```

---

## 🧪 Testing & Data Verification

Run the full test suite and data contamination audit:

```bash
# Verify 0% data leakage between train, test, and benchmarks
python scripts/check_data_leakage.py

# Run unit tests (tokenizer fidelity, loss masking, config, chat template)
pytest -v tests/

# Validate Python syntax across all modules
python -m compileall src/ training/ evaluation/ scripts/ tests/
```

---

## 📤 Publishing to Hugging Face Hub

To publish your fine-tuned model and model card to the Hugging Face Hub:

```bash
export HF_TOKEN="hf_your_write_token"
python upload_to_hf.py --repo-id "YOUR_USERNAME/tamazight-qwen-0.5b"
```

---

## ⚠️ Scope & Known Limitations

- **Dialect Scope:** The Amazigh training corpus primarily reflects the **Kabyle (Taqbaylit)** dialect. Performance on other varieties (e.g., Tarifit, Tachawit, Tamasheq) may differ.
- **Orthography:** The model is fine-tuned on the standardized **Latin script (`Agemmay Alatini`)**. While Unicode normalization recognizes Tifinagh characters, output generation is optimized for Latin script.
- **Scale:** At 490 million parameters, the model is designed for mobile efficiency and lightweight deployment. Complex multi-step mathematical or reasoning tasks should be handled by larger parameter models.

---

## 📄 License & Citation

Licensed under the **[Apache-2.0 License](LICENSE)**.

```bibtex
@misc{tamazight_qwen_2026,
  author = {Mijozak},
  title = {Tamazight Multilingual LLM (Qwen2.5-0.5B)},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/zakmijo2-dotcom/tamazight-llm}
}
```
