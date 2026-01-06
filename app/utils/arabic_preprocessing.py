"""
Arabic text preprocessing utilities.

This module provides functions for normalizing and preprocessing Arabic text,
including letter normalization, diacritic removal, and whitespace handling.
"""

import re


def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text by standardizing letters and removing diacritics.
    
    This function performs the following operations:
    1. Normalizes Arabic letter variants to their base forms:
       - أ, إ, آ → ا (alif variants)
       - ى → ي (alif maksura to ya)
       - ؤ → و (hamza on waw to waw)
       - ئ → ي (hamza on ya to ya)
       - ة → ه (ta marbuta to ha)
    2. Removes all Arabic diacritics (tashkeel):
       - Fatha (َ), Damma (ُ), Kasra (ِ)
       - Shadda (ّ), Sukoon (ْ)
       - Tanween variants (ً, ٌ, ٍ)
       - Other diacritical marks
    3. Removes elongation marks (ـ)
    4. Normalizes whitespace (multiple spaces to single space)
    5. Strips leading and trailing whitespace
    
    Args:
        text: Input Arabic text string to normalize.
        
    Returns:
        Normalized Arabic text string with standardized letters,
        no diacritics, and normalized whitespace.
        
    Example:
        >>> normalize_arabic("السَّلامُ عَلَيْكُمْ")
        'السلام عليكم'
        >>> normalize_arabic("أهلاً وسهلاً")
        'اهلا وسهلا'
    """
    if not text:
        return ""
    
    # Normalize Arabic letter variants
    # أ, إ, آ → ا
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
    # ى → ي
    text = text.replace('ى', 'ي')
    # ؤ → و
    text = text.replace('ؤ', 'و')
    # ئ → ي
    text = text.replace('ئ', 'ي')
    # ة → ه
    text = text.replace('ة', 'ه')
    
    # Remove Arabic diacritics (tashkeel)
    # Unicode ranges for Arabic diacritics
    diacritics_pattern = re.compile(
        r'[\u064B-\u065F\u0670\u0640]'  # Includes fatha, damma, kasra, shadda, sukoon, tanween, etc.
    )
    text = diacritics_pattern.sub('', text)
    
    # Remove elongation marks (tatweel)
    text = text.replace('ـ', '')
    
    # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading and trailing whitespace
    text = text.strip()
    
    return text


def preprocess_text(text: str) -> str:
    """
    Preprocess Arabic text for RAG pipeline.
    
    This function applies normalization and can be extended with additional
    preprocessing steps such as:
    - Tokenization
    - Stop word removal
    - Stemming/Lemmatization
    - Special character handling
    
    Args:
        text: Input Arabic text string to preprocess.
        
    Returns:
        Preprocessed Arabic text string ready for embedding or retrieval.
        
    Example:
        >>> preprocess_text("السَّلامُ عَلَيْكُمْ")
        'السلام عليكم'
    """
    if not text:
        return ""
    
    # Apply Arabic normalization
    text = normalize_arabic(text)
    
    # Additional preprocessing steps can be added here
    # e.g., stop word removal, stemming, etc.
    
    return text

