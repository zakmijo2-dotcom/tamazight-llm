# Dataset Card: Tamazight Multilingual Chat Corpus ⵣ

## Dataset Summary
The **Tamazight Multilingual Chat Corpus** is an open curated conversational and instruction-tuning dataset designed to train lightweight multilingual language models capable of conversing naturally across **Tamazight (Kabyle Latin)**, **Arabic**, **French**, and **English**.

## Dataset Structure
- `data/train.jsonl`: 7,731 clean, deduplicated instruction examples.
- `data/validation.jsonl`: 429 validation examples.
- `data/test.jsonl`: 429 held-out evaluation examples.
- `data/processed/kabyle_master.jsonl`: 8,589 consolidated records.

### Sample Format
Each record includes complete ChatML messages along with language and quality metadata:
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful, culturally aware, multilingual AI assistant..."},
    {"role": "user", "content": "Azul fell-awen! Amek telliḍ?"},
    {"role": "assistant", "content": "Azul! Labas, tanemmirt tameqrant..."}
  ],
  "language": "kabyle",
  "script": "latin",
  "source": "curated_expert",
  "quality": 1.0
}
```

## Data Sources & Attribution
The corpus aggregates and cleans verified open-source initiatives registered in `configs/datasets.yaml`:
1. **HCA Kabyle Corpus (`Imsidag-community/kabyle-corpus-hca`):** Authentic Kabyle literature and cultural publications (CC-BY-SA 4.0).
2. **UMMTO University Academic Corpus (`Imsidag-community/kabyle-corpus-ummto`):** University master theses and literary texts from Tizi Ouzou (CC-BY-SA 4.0).
3. **University of Bouira Corpus (`Imsidag-community/kabyle-corpus-ubouira`):** Academic and literary Kabyle corpora (CC-BY-SA 4.0).
4. **AmaWar Cultural Corpus (`Tamazight-NLP/AmaWar`):** Traditional proverbs, idioms, and riddles with bilingual explanations (Apache 2.0).
5. **Kabyle Wikipedia (`Sifal/KabyleWikipedia`):** Encyclopedic Q&A entries covering history, geography, science, and nature.
6. **TranslateWiki & Pontoon Localizations (`Tamazight-NLP/translatewiki` & `Pontoon`):** Technical and localization terminology (CC-BY 3.0 / MPL 2.0).
7. **Expert Curated Seed Dialogues:** Handcrafted multi-turn conversational templates covering greetings, history (Numidia, Massinissa), culture (Tiwizi, Yennayer), and everyday assistance.
## Preprocessing & Curation Pipeline
1. **Normalization:** All inputs undergo Unicode **NFKC** normalization, quote standardization (`«»“”` $\rightarrow$ `"`), and whitespace collapse.
2. **Quality Filtering:** Discards samples where either turn is shorter than 3 characters, empty, or structurally corrupted.
3. **Deduplication:** Normalizes user queries and removes exact and normalized prompt collisions, retaining the highest quality turn.
4. **Leakage Elimination:** Explicitly cross-checks all evaluation benchmark questions (`evaluation/datasets/`) and removes matching training queries to guarantee **0% benchmark contamination**.

## Language & Dialect Scope
- **Tamazight Variety:** The Amazigh portion is primarily the **Kabyle (Taqbaylit)** dialect.
- **Orthography:** Written in the standardized Latin alphabet (`Agemmay Alatini`) with phonetic extensions (`č`, `ḍ`, `ɣ`, `ḥ`, `ɛ`, `ǧ`, `ṛ`, `ṣ`, `ṭ`, `ẓ`).
- **Additional Languages:** Standard Modern Arabic (العربية الفصحى), French (Français), and English.

## Licensing
The processed dataset is released under the **Apache-2.0** license, adhering to the open licenses of the upstream sources.
