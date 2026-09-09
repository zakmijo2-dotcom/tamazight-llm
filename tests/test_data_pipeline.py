"""
Tests for Data Processing, Normalization, Script Detection, and Deduplication.
"""

from src.data import (
    normalize_text,
    detect_script,
    detect_language,
    compute_quality_score,
    create_sample,
    deduplicate_samples,
    train_val_test_split
)

def test_normalize_text():
    raw = "  Azul   fell-awen  !  \n\n\n  Amek   telliḍ ?  "
    norm = normalize_text(raw)
    assert norm == "Azul fell-awen !\n\nAmek telliḍ ?"

def test_detect_script():
    assert detect_script("Azul fell-awen!") == "latin"
    assert detect_script("مرحباً بكم") == "arabic"
    assert detect_script("ⴰⵣⵓⵍ ⴼⵍⵍ-ⴰⵡⵏ") == "tifinagh"

def test_detect_language():
    assert detect_language("Azul fell-awen! Amek telliḍ?") == "kabyle"
    assert detect_language("مرحباً كيف حالك؟") == "arabic"
    assert detect_language("Bonjour comment allez-vous ?") == "french"
    assert detect_language("Hello what is the capital of Canada?") == "english"

def test_compute_quality_score():
    assert compute_quality_score("", "") == 0.0
    assert compute_quality_score("a", "b") == 0.0
    assert compute_quality_score("Azul!", "Azul! Labas, tanemmirt.") >= 0.8

def test_deduplicate_samples():
    s1 = create_sample("Azul!", "Labas 1.", "sys", source="src1")
    s2 = create_sample("azul!", "Labas 2 longer and better.", "sys", source="src2")
    s3 = create_sample("Amek telliḍ?", "Labas.", "sys", source="src3")
    
    samples = [s1, s2, s3]
    deduped, removed = deduplicate_samples(samples)
    assert len(deduped) == 2
    assert removed == 1

def test_train_val_test_split():
    samples = [create_sample(f"User {i}", f"Assistant {i}", "sys") for i in range(100)]
    train, val, test = train_val_test_split(samples, val_ratio=0.1, test_ratio=0.1, seed=42)
    assert len(test) == 10
    assert len(val) == 10
    assert len(train) == 80
    assert set(id(s) for s in train).isdisjoint(set(id(s) for s in test))
