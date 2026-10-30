"""Regression tests for the lightweight Arabic analyzer."""

import pytest

from app.utils.arabic_morphology import LightArabicAnalyzer


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("أ إ آ", "ا ا ا"),
        ("ة ى ؤ ئ", "ه ي و ي"),
        ("سَـلَامٌ", "سلام"),
        ("  أ\tإ\nآ  ", "ا ا ا"),
        ("أ؟?", "ا؟?"),
    ],
)
def test_normalize_restores_arabic_and_preserves_punctuation(text, expected):
    assert LightArabicAnalyzer().normalize(text) == expected


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("الكتاب", "كتاب"),
        ("والكلمات", "كلم"),
        ("بالمدارس", "مدارس"),
        ("مكتوب", "مكتوب"),
        ("مدرسة", "مدرس"),
        ("معلمون", "معلم"),
        ("كتابات", "كتاب"),
        ("البيات", "بيات"),
        ("العب", "العب"),
        ("والالكتاب", "الكتاب"),
        ("كتاباتها", "كتابات"),
    ],
)
def test_extract_roots_returns_light_stem_keys(word, expected):
    assert LightArabicAnalyzer().extract_roots(word) == {expected}


def test_extract_roots_ignores_words_shorter_than_four_after_normalizing():
    assert LightArabicAnalyzer().extract_roots("كِتْبٌ قلم") == set()
