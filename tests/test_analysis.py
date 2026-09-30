"""
Unit tests for TF-IDF and Model Inference interfaces.
"""

import pytest
from modules.tfidf_analyzer import TfidfAnalyzer
from modules.sentiment_analyzer import SentimentAnalyzer
from modules.emotion_analyzer import EmotionAnalyzer

def test_tfidf_extraction():
    tfidf = TfidfAnalyzer()
    terms = tfidf.extract_top_terms("love product customer service fantastic", top_k=3)
    assert len(terms) > 0
    assert any("love" in t['term'] or "product" in t['term'] for t in terms)

def test_sentiment_analyzer_mock_pipeline():
    # Mock pipeline function returning standard Hugging Face pipeline structure
    def mock_pipe(text):
        return [[
            {'label': 'positive', 'score': 0.88},
            {'label': 'neutral', 'score': 0.08},
            {'label': 'negative', 'score': 0.04}
        ]]
        
    analyzer = SentimentAnalyzer(pipeline_fn=lambda: mock_pipe)
    result = analyzer.analyze("I love this!", feature_data={})
    assert result['dominant_sentiment'] == 'Positive'
    assert result['confidence'] == 88.0
    assert result['probabilities']['Positive'] == 0.88

def test_emotion_analyzer_mock_pipeline():
    def mock_pipe(text):
        return [[
            {'label': 'joy', 'score': 0.82},
            {'label': 'surprise', 'score': 0.10},
            {'label': 'neutral', 'score': 0.08}
        ]]
        
    analyzer = EmotionAnalyzer(pipeline_fn=lambda: mock_pipe)
    result = analyzer.analyze("So thrilled today!")
    assert result['dominant_emotion'] == 'Joy'
    assert result['confidence'] == 82.0
