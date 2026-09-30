"""
Unit tests for TextPreprocessor.
"""

import pytest
from modules.preprocessing import TextPreprocessor

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_unicode_normalization(preprocessor):
    text = "Hello\u00A0World" # non-breaking space
    normalized = preprocessor.normalize_unicode(text)
    assert "Hello" in normalized and "World" in normalized

def test_metadata_extraction(preprocessor):
    text = "Check out https://example.com and follow @user on #AI"
    meta = preprocessor.extract_metadata(text)
    assert "https://example.com" in meta['urls']
    assert "@user" in meta['mentions']
    assert "#AI" in meta['hashtags']

def test_model_input_preserves_emojis(preprocessor):
    text = "  I LOVE this product!!! 😍❤️   "
    prepared = preprocessor.prepare_model_input(text)
    assert "😍" in prepared
    assert "❤️" in prepared
    assert "LOVE" in prepared
    assert prepared == "I LOVE this product!!! 😍❤️"

def test_text_mining_stream(preprocessor):
    text = "I love amazing products and great services!"
    res = preprocessor.process_for_text_mining(text)
    assert "love" in res['clean_tokens']
    assert "amazing" in res['clean_tokens']
    assert "product" in res['lemmas'] or "products" in res['clean_tokens']
    assert "and" in res['stop_words_found']
