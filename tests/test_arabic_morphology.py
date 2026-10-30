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
