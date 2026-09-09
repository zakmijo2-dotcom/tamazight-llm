# Model Card: Tamazight Multilingual LLM (Qwen2.5-1.5B) ⵣ

## Model Overview
- **Model Name:** Tamazight Multilingual LLM
- **Base Architecture:** `Qwen/Qwen2.5-1.5B-Instruct` (with `Qwen/Qwen2.5-0.5B-Instruct` edge profile)
- **Total Parameters:** ~1.54 Billion (1.5B class)
- **Primary Languages:** Tamazight (Kabyle / Taqbaylit in standard Latin script), Arabic (العربية), French (Français), English
- **License:** Apache 2.0
- **Fine-Tuning Method:** Supervised Fine-Tuning (SFT) with LoRA (Low-Rank Adaptation) and strict Assistant-Only loss masking
- **Deployment Targets:** Edge devices, Android (via Termux + llama.cpp), Google Colab, GitHub Codespaces, standard Linux/macOS workstations
## Intended Use
- **Conversational Assistance:** Natural dialogue in Kabyle/Tamazight, Arabic, French, and English.
- **Cultural & Linguistic Preservation:** Answering inquiries regarding Amazigh history (Numidia, Massinissa, Kahina), North African culture, proverbs, and everyday life.
- **Cross-Lingual Communication:** Assisting users across North Africa and the diaspora with multi-language Q&A.

## Out-of-Scope Use
- Automated high-stakes translation without human review.
- Dialectal generalization across non-Kabyle varieties (e.g. Tuareg, Riffian, Shilha) without domain-specific validation.
- Generating medical, legal, or financial advice.

## Training Details
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Training Strategy:** LoRA applied to linear projection layers (`q_proj`, `v_proj` on CPU; all 7 projections on GPU).
- **Loss Computation:** Strict Assistant-Only Loss Masking (`labels = -100` for all system prompts, user turns, and template delimiters).
- **Sequence Length:** Configurable (128–256 tokens).
- **Precision:** bfloat16 / float16 where supported, with float32 CPU fallback.

## Quantization & Mobile Deployment
The model converts natively to GGUF format for edge execution using `llama.cpp`:
- **F16 GGUF:** ~3.0 GB (1.5B) / ~980 MB (0.5B)
- **Q4_K_M GGUF:** ~1.1 GB (1.5B) / ~350 MB (0.5B)
- **RAM Requirements:**
  - `1.5B Q4_K_M`: Recommended for Android devices with 4GB+ RAM.
  - `0.5B Q4_K_M`: Recommended for Android devices with 2GB+ RAM.
- **Runtime Performance:** Responsive CPU generation on modern Snapdragon / Dimensity processors via `llama.cpp`.
## Limitations & Bias
- **Dialect Scope:** The Tamazight training corpus predominantly reflects the **Kabyle (Taqbaylit)** dialect written in the Latin alphabet. Performance on other varieties of Tamazight may vary.
- **Script:** Current training utilizes the standard Latin script (`Agemmay Alatini`). Tifinagh input is recognized but output generation is optimized for Latin script.
- **Parameter Scale:** At 490M parameters, the model is designed for lightweight efficiency. Complex multi-step reasoning may require larger parameter models.

## How to Cite
```bibtex
@misc{tamazight_qwen_2026,
  author = {Mijozak},
  title = {Tamazight Multilingual LLM (Qwen2.5-0.5B)},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/zakmijo2-dotcom/tamazight-llm}
}
```
