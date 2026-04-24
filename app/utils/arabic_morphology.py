"""
Arabic morphological analysis utilities.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Set

logger = logging.getLogger(__name__)


class ArabicMorphologicalAnalyzer:
    """
    Analyzer for Arabic text morphology using CAMEL Tools.

    CAMEL Tools is imported lazily so the rest of the project can run even when
    the optional dependency is unavailable.
    """

    def __init__(self, db_path: str = "arablmsr"):
        try:
            from camel_tools.morphology.analyzer import Analyzer
            from camel_tools.morphology.database import MorphologyDB
        except ImportError as exc:
            raise RuntimeError(
                "camel-tools is required for ArabicMorphologicalAnalyzer. "
                "Install it with `pip install camel-tools`."
            ) from exc

        try:
            db = MorphologyDB(db_path)
            self.analyzer = Analyzer(db)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize morphological analyzer: {str(exc)}"
            ) from exc

    def analyze(self, word: str) -> List[Dict[str, str]]:
        if not word or not isinstance(word, str):
            return []

        try:
            analyses = self.analyzer.analyze(word)
            results = []
            for analysis in analyses:
                if isinstance(analysis, dict):
                    results.append(
                        {
                            "lemma": analysis.get("lemma"),
                            "root": analysis.get("root"),
                            "pos": analysis.get("pos"),
                            "pattern": analysis.get("pattern"),
                        }
                    )
                else:
                    results.append(
                        {
                            "lemma": getattr(analysis, "lemma", None),
                            "root": getattr(analysis, "root", None),
                            "pos": getattr(analysis, "pos", None),
                            "pattern": getattr(analysis, "pattern", None),
                        }
                    )
            return results
        except Exception:
            return []

    def get_roots(self, words: List[str]) -> Set[str]:
        roots: Set[str] = set()
        for word in words:
            for analysis in self.analyze(word):
                root = analysis.get("root")
                if root:
                    roots.add(root)
        return roots

    def get_lemmas(self, words: List[str]) -> Set[str]:
        lemmas: Set[str] = set()
        for word in words:
            for analysis in self.analyze(word):
                lemma = analysis.get("lemma")
                if lemma:
                    lemmas.add(lemma)
        return lemmas

    def get_pos_tags(self, words: List[str]) -> List[str]:
        pos_tags: List[str] = []
        for word in words:
            analyses = self.analyze(word)
            pos_tags.append(analyses[0].get("pos", "unknown") if analyses else "unknown")
        return pos_tags

    def extract_roots_from_text(self, text: str) -> Set[str]:
        if not text:
            return set()
        return self.get_roots(text.split())

    def lemmatize(self, word: str) -> str:
        analyses = self.analyze(word)
        if analyses:
            return analyses[0].get("lemma") or word
        return word


class LightArabicAnalyzer:
    """Lightweight Arabic analyzer without external NLP dependencies."""

    def __init__(self):
        self.arabic_pattern = re.compile(r"[\u0600-\u06FF]+")

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        text = text.replace("?", "?").replace("?", "?").replace("?", "?")
        text = text.replace("?", "?").replace("?", "?").replace("?", "?")
        text = text.replace("?", "?").replace("?", "")
        text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def extract_roots(self, text: str) -> Set[str]:
        if not text:
            return set()

        text = self.normalize(text)
        roots: Set[str] = set()

        for word in text.split():
            word = word.strip()
            if len(word) < 3:
                continue
            if word.startswith("?") and len(word) == 5:
                roots.add(word[1:4])
            elif word.startswith("?") and len(word) == 5:
                roots.add(word[1:4])
            elif len(word) == 3:
                roots.add(word)
            elif len(word) == 4:
                if word[1] == word[2]:
                    roots.add(word[0] + word[1] + word[3])
                else:
                    roots.add(word[:3])
            elif len(word) >= 6:
                roots.add(word[:3])

        return roots

    def get_word_overlap(self, text1: str, text2: str) -> float:
        words1 = set(self.normalize(text1).split())
        words2 = set(self.normalize(text2).split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)

    def get_bm25_score(
        self,
        query: str,
        text: str,
        avg_doc_len: float = 100.0,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        query_terms = self.normalize(query).split()
        text_terms = self.normalize(text).split()

        if not query_terms or not text_terms:
            return 0.0

        doc_len = len(text_terms)
        frequencies: Dict[str, int] = {}
        for term in text_terms:
            frequencies[term] = frequencies.get(term, 0) + 1

        score = 0.0
        for term in query_terms:
            if term in frequencies:
                tf = frequencies[term]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
                score += numerator / denominator

        return score

    def get_exact_match_bonus(self, query: str, text: str) -> float:
        query_terms = self.normalize(query).split()
        text_norm = self.normalize(text)

        if not query_terms:
            return 0.0

        matches = sum(1 for term in query_terms if term in text_norm)
        return matches / len(query_terms)

    def get_ngram_overlap(self, text1: str, text2: str, n: int = 2) -> float:
        words1 = self.normalize(text1).split()
        words2 = self.normalize(text2).split()

        if len(words1) < n or len(words2) < n:
            return 0.0

        ngrams1 = {tuple(words1[i : i + n]) for i in range(len(words1) - n + 1)}
        ngrams2 = {tuple(words2[i : i + n]) for i in range(len(words2) - n + 1)}

        if not ngrams1 and not ngrams2:
            return 0.0

        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        return len(intersection) / len(union) if union else 0.0

    def get_root_overlap(self, text1: str, text2: str) -> float:
        roots1 = self.extract_roots(text1)
        roots2 = self.extract_roots(text2)

        if not roots1 or not roots2:
            return 0.0

        intersection = roots1 & roots2
        union = roots1 | roots2
        return len(intersection) / len(union)

    def get_comprehensive_score(self, query: str, text: str) -> Dict[str, float]:
        return {
            "word_overlap": self.get_word_overlap(query, text),
            "root_overlap": self.get_root_overlap(query, text),
            "bm25": self.get_bm25_score(query, text),
            "exact_match": self.get_exact_match_bonus(query, text),
            "ngram_2": self.get_ngram_overlap(query, text, 2),
            "ngram_3": self.get_ngram_overlap(query, text, 3),
        }
