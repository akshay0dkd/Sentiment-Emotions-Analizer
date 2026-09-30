"""
Unit tests for FeatureExtractor.
"""

import pytest
from modules.feature_extraction import FeatureExtractor
from modules.preprocessing import TextPreprocessor

@pytest.fixture
def extractor():
    return FeatureExtractor()

@pytest.fixture
def preprocessor():
    return TextPreprocessor()

def test_feature_counts(extractor, preprocessor):
    text = "WOW! This is amazing, but is it true???"
    preproc = preprocessor.process_for_text_mining(text)
    features = extractor.extract_features(text, preproc, emoji_count=0)
    
    stats = features['statistics']
    surf = features['surface_counts']
    lex = features['lexical_cues']
    
    assert stats['word_count'] > 0
    assert surf['exclamation_count'] == 1
    assert surf['question_count'] == 3
    assert surf['uppercase_words'] >= 1 # WOW
    assert "amazing" in lex['positive_cues']
    assert "but" in lex['contrastive_cues']
    assert lex['has_contrastive'] is True

def test_empty_features(extractor):
    empty_res = extractor.extract_features("", {}, emoji_count=0)
    assert empty_res['statistics']['word_count'] == 0
    assert empty_res['statistics']['character_count'] == 0
