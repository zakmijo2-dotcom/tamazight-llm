"""
Dataset Processing, Normalization, Deduplication, and Validation Pipeline.
Supports Tamazight (Kabyle Latin & Tifinagh), Arabic, French, and English.
"""

import re
import json
import random
import hashlib
import unicodedata
from typing import List, Dict, Tuple, Optional, Any

# Common stopwords and markers for language detection
FRENCH_MARKERS = {"le", "la", "les", "du", "des", "dans", "est", "sont", "pour", "avec", "bonjour", "comment", "vous", "nous", "une", "qui"}
ENGLISH_MARKERS = {"the", "is", "are", "and", "in", "to", "what", "how", "hello", "welcome", "with", "this", "that", "from", "for"}
KABYLE_MARKERS = {"d", "n", "i", "s", "ɣer", "deg", "fell", "azul", "amek", "tanemmirt", "imaziɣen", "zik", "tameqrant", "akked", "yelha", "dacu", "anwa", "anta", "taddart", "awal", "inzi", "leḥna"}

def normalize_text(text: str) -> str:
    """
    Applies standard Unicode NFKC normalization, strips trailing whitespace,
    and collapses multiple whitespace sequences.
    """
    if not text:
        return ""
    # Unicode NFKC normalization
    norm = unicodedata.normalize("NFKC", text)
    # Normalize varied quotes to standard quotes
    norm = re.sub(r'[\u201C\u201D\u00AB\u00BB]', '"', norm)
    norm = re.sub(r'[\u2018\u2019]', "'", norm)
    # Collapse multiple spaces and tabs (preserving single newlines)
    lines = [re.sub(r'[ \t]+', ' ', line).strip() for line in norm.splitlines()]
    # Remove excessive blank lines
    cleaned_lines = []
    prev_empty = False
    for line in lines:
        if not line:
            if not prev_empty:
                cleaned_lines.append("")
            prev_empty = True
        else:
            cleaned_lines.append(line)
            prev_empty = False
    return "\n".join(cleaned_lines).strip()

def detect_script(text: str) -> str:
    """
    Detects dominant script: tifinagh, arabic, latin, or mixed.
    """
    tifinagh_count = sum(1 for c in text if 0x2D30 <= ord(c) <= 0x2D7F)
    arabic_count = sum(1 for c in text if (0x0600 <= ord(c) <= 0x06FF or 0x0750 <= ord(c) <= 0x077F or 0x08A0 <= ord(c) <= 0x08FF))
    latin_count = sum(1 for c in text if c.isascii() and c.isalpha())
    
    total_letters = tifinagh_count + arabic_count + latin_count
    if total_letters == 0:
        return "latin"
        
    if tifinagh_count / total_letters > 0.4:
        return "tifinagh"
    if arabic_count / total_letters > 0.4:
        return "arabic"
    if latin_count / total_letters > 0.6:
        return "latin"
    return "mixed"

def detect_language(text: str, default: str = "kabyle") -> str:
    """
    Classifies text into kabyle, arabic, french, english, or multilingual.
    """
    script = detect_script(text)
    if script == "arabic":
        return "arabic"
    if script == "tifinagh":
        return "tamazight"
        
    words = set(re.findall(r'\b[a-zA-ZɣḍčḥɛǧṛṣṭẓƔḌČḤƐǦṚṢṬẒ\-]+\b', text.lower()))
    if not words:
        return default
        
    kabyle_score = len(words.intersection(KABYLE_MARKERS))
    french_score = len(words.intersection(FRENCH_MARKERS))
    english_score = len(words.intersection(ENGLISH_MARKERS))
    
    scores = [
        (kabyle_score, "kabyle"),
        (french_score, "french"),
        (english_score, "english")
    ]
    scores.sort(key=lambda x: x[0], reverse=True)
    
    if scores[0][0] > 0:
        return scores[0][1]
    return default

def compute_quality_score(user_text: str, assistant_text: str) -> float:
    """
    Computes a heuristic quality score between 0.0 and 1.0 based on:
    - Minimum length
    - Non-repetitiveness
    - Punctuation and formatting
    """
    u_len = len(user_text.strip())
    a_len = len(assistant_text.strip())
    
    # Reject empty or very short turns
    if u_len < 3 or a_len < 3:
        return 0.0
        
    score = 0.8  # baseline for valid non-empty pair
    
    # Bonus for substantive, well-formed answers
    if 20 <= a_len <= 1500:
        score += 0.1
    if any(p in assistant_text for p in [".", "!", "?", ":"]):
        score += 0.05
    if len(set(assistant_text.split())) / max(len(assistant_text.split()), 1) > 0.5:
        score += 0.05  # good vocabulary variety (not a repetition loop)
        
    return min(score, 1.0)

def create_sample(
    user_text: str,
    assistant_text: str,
    system_prompt: str,
    source: str = "curated",
    language_override: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Validates, normalizes, and packages a conversation with structured metadata.
    Returns None if sample is invalid or corrupted.
    """
    norm_u = normalize_text(user_text)
    norm_a = normalize_text(assistant_text)
    
    quality = compute_quality_score(norm_u, norm_a)
    if quality <= 0.0:
        return None
        
    lang = language_override or detect_language(norm_u)
    script = detect_script(norm_u + " " + norm_a)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": norm_u},
        {"role": "assistant", "content": norm_a}
    ]
    
    return {
        "messages": messages,
        "language": lang,
        "script": script,
        "source": source,
        "quality": round(quality, 2)
    }

def deduplicate_samples(samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """
    Deduplicates samples using normalized user queries to guarantee that
    no identical user prompt exists twice in the dataset.
    If a prompt is duplicated, keeps the sample with the highest quality score.
    Returns (deduplicated_samples, number_of_duplicates_removed).
    """
    best_by_prompt: Dict[str, Dict[str, Any]] = {}
    total_seen = 0
    
    for s in samples:
        messages = s.get("messages", [])
        u_content = ""
        for m in messages:
            if m.get("role") == "user":
                u_content = m.get("content", "").strip()
                
        if not u_content:
            continue
            
        total_seen += 1
        prompt_key = normalize_text(u_content).lower()
        
        if prompt_key not in best_by_prompt:
            best_by_prompt[prompt_key] = s
        else:
            # Keep the one with higher quality score or longer response
            prev_qual = best_by_prompt[prompt_key].get("quality", 0.0)
            curr_qual = s.get("quality", 0.0)
            if curr_qual > prev_qual:
                best_by_prompt[prompt_key] = s
                
    deduped = list(best_by_prompt.values())
    removed = total_seen - len(deduped)
    return deduped, removed

def train_val_test_split(
    samples: List[Dict[str, Any]],
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits samples deterministically into train, validation, and test sets.
    """
    random.seed(seed)
    shuffled = list(samples)
    random.shuffle(shuffled)
    
    n_total = len(shuffled)
    n_val = max(1, int(n_total * val_ratio))
    n_test = max(1, int(n_total * test_ratio))
    
    test_set = shuffled[:n_test]
    val_set = shuffled[n_test:n_test + n_val]
    train_set = shuffled[n_test + n_val:]
    
    return train_set, val_set, test_set

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Loads records from a JSONL file."""
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def save_jsonl(records: List[Dict[str, Any]], file_path: str) -> None:
    """Saves records to a JSONL file."""
    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
