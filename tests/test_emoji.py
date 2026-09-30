"""
Unit tests for EmojiAnalyzer.
"""

import pytest
from modules.emoji_analyzer import EmojiAnalyzer

@pytest.fixture
def emoji_analyzer():
    return EmojiAnalyzer()

def test_emoji_detection(emoji_analyzer):
    text = "I love this product 😍❤️🔥"
    res = emoji_analyzer.analyze_emojis(text)
    
    assert res['total_emoji_count'] == 3
    assert res['unique_emoji_count'] == 3
    assert "😍" in res['emojis_detected']
    assert "❤️" in res['emojis_detected']
    assert "🔥" in res['emojis_detected']
    assert res['emoji_sentiment_summary'] == "Positive"

def test_no_emojis(emoji_analyzer):
    text = "Plain text without any emojis at all."
    res = emoji_analyzer.analyze_emojis(text)
    assert res['total_emoji_count'] == 0
    assert len(res['emojis_detected']) == 0
