# Model Card: Tamazight Multilingual LLM (Qwen2.5-0.5B) ⵣ

## Model Overview
- **Model Name:** Tamazight Multilingual LLM
- **Base Architecture:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Total Parameters:** ~490 Million (0.5B)
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
- **Base Model:** `Qwen/Qwen2.5-0.5B-Instruct`
- **Training Strategy:** LoRA applied to linear projection layers (`q_proj`, `v_proj` on CPU; all 7 projections on GPU).
- **Loss Computation:** Strict Assistant-Only Loss Masking (`labels = -100` for all system prompts, user turns, and template delimiters).
- **Sequence Length:** Configurable (128–256 tokens).
- **Precision:** bfloat16 / float16 where supported, with float32 CPU fallback.

## Quantization & Mobile Deployment
The model converts natively to GGUF format for edge execution using `llama.cpp`:
- **F16 GGUF:** ~980 MB
- **Q4_K_M GGUF:** ~350 MB (Recommended for Android / Termux with 2GB+ RAM)
- **Runtime Performance:** Capable of responsive generation on modern mobile processors via CPU inference.

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
