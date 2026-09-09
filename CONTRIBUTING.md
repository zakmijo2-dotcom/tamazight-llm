# Contributing to Tamazight Multilingual LLM ⵣ

We welcome contributions to expand and improve Tamazight natural language processing and multilingual AI!

## How to Contribute

### 1. Adding Conversational & Cultural Data
- All new dataset entries should be added via `src/seed_dialogues.py` or through the raw ingestion pipeline in `data/raw/`.
- Ensure new Tamazight text is written in standard Latin script (`Agemmay Alatini`) with correct diacritics (`č`, `ḍ`, `ɣ`, `ḥ`, `ɛ`, `ǧ`, `ṛ`, `ṣ`, `ṭ`, `ẓ`).
- Run the preparation pipeline and verify 0% leakage:
  ```bash
  python scripts/prepare_data.py
  python scripts/check_data_leakage.py
  ```

### 2. Code Contributions
- Adhere to PEP 8 standards with clear type hints and docstrings.
- Ensure all new logic is covered by unit tests in `tests/`.
- Run pytest and byte-compilation checks before submitting a PR:
  ```bash
  python -m compileall src/ training/ evaluation/ scripts/ tests/
  pytest -v tests/
  ```

### 3. Reporting Issues
- Use GitHub Issues to report bugs, tokenization errors, or data anomalies.
- Provide minimal reproduction examples whenever possible.

## License
By contributing to this repository, you agree that your contributions will be licensed under the Apache-2.0 License.
