"""
Emoji Analysis Module.
Detects emojis, computes frequencies, extracts Unicode descriptions,
and provides auxiliary sentiment/emotion interpretations.
"""

from typing import Dict, List, Any
import emoji
from collections import Counter

# Sentiment & Emotion polarity mapping for common emojis
EMOJI_SEMANTIC_MAP = {
    # Joy / Love / Positive
    '😍': {'sentiment': 'Positive', 'emotion': 'Love / Admiration', 'category': 'Affection'},
    '🥰': {'sentiment': 'Positive', 'emotion': 'Love / Warmth', 'category': 'Affection'},
    '❤️': {'sentiment': 'Positive', 'emotion': 'Love / Passion', 'category': 'Affection'},
    '💖': {'sentiment': 'Positive', 'emotion': 'Love / Excitement', 'category': 'Affection'},
    '💕': {'sentiment': 'Positive', 'emotion': 'Affection', 'category': 'Affection'},
    '😊': {'sentiment': 'Positive', 'emotion': 'Joy / Contentment', 'category': 'Happiness'},
    '😄': {'sentiment': 'Positive', 'emotion': 'Joy / Happiness', 'category': 'Happiness'},
    '😀': {'sentiment': 'Positive', 'emotion': 'Joy', 'category': 'Happiness'},
    '😁': {'sentiment': 'Positive', 'emotion': 'Cheerfulness', 'category': 'Happiness'},
    '😂': {'sentiment': 'Positive', 'emotion': 'Humor / Amusement', 'category': 'Laughter'},
    '🤣': {'sentiment': 'Positive', 'emotion': 'Intense Amusement', 'category': 'Laughter'},
    '🔥': {'sentiment': 'Positive', 'emotion': 'Excitement / Hot / Trendy', 'category': 'Enthusiasm'},
    '🎉': {'sentiment': 'Positive', 'emotion': 'Celebration / Joy', 'category': 'Celebration'},
    '✨': {'sentiment': 'Positive', 'emotion': 'Magic / Delight', 'category': 'Delight'},
    '👏': {'sentiment': 'Positive', 'emotion': 'Appreciation / Praise', 'category': 'Approval'},
    '👍': {'sentiment': 'Positive', 'emotion': 'Approval / Agreement', 'category': 'Approval'},
    '💯': {'sentiment': 'Positive', 'emotion': 'Perfection / Agreement', 'category': 'Emphasis'},
    '🙌': {'sentiment': 'Positive', 'emotion': 'Gratitude / Celebration', 'category': 'Celebration'},

    # Negative / Anger / Sadness / Disappointment
    '😡': {'sentiment': 'Negative', 'emotion': 'Anger / Rage', 'category': 'Anger'},
    '😠': {'sentiment': 'Negative', 'emotion': 'Anger / Irritation', 'category': 'Anger'},
    '🤬': {'sentiment': 'Negative', 'emotion': 'Intense Rage / Profanity', 'category': 'Anger'},
    '😢': {'sentiment': 'Negative', 'emotion': 'Sadness / Grief', 'category': 'Sadness'},
    '😭': {'sentiment': 'Negative', 'emotion': 'Intense Sadness / Overwhelmed', 'category': 'Sadness'},
    '😞': {'sentiment': 'Negative', 'emotion': 'Disappointment / Regret', 'category': 'Sadness'},
    '😔': {'sentiment': 'Negative', 'emotion': 'Pensive Sadness', 'category': 'Sadness'},
    '💔': {'sentiment': 'Negative', 'emotion': 'Heartbreak / Grief', 'category': 'Sadness'},
    '🤮': {'sentiment': 'Negative', 'emotion': 'Disgust / Revulsion', 'category': 'Disgust'},
    '🤢': {'sentiment': 'Negative', 'emotion': 'Disgust / Sickness', 'category': 'Disgust'},
    '👎': {'sentiment': 'Negative', 'emotion': 'Disapproval / Dislike', 'category': 'Disapproval'},
    '😒': {'sentiment': 'Negative', 'emotion': 'Annoyance / Discontent', 'category': 'Discontent'},
    '😩': {'sentiment': 'Negative', 'emotion': 'Weariness / Frustration', 'category': 'Frustration'},
    '😤': {'sentiment': 'Negative', 'emotion': 'Frustration / Huffing', 'category': 'Frustration'},

    # Fear / Surprise / Irony / Ambiguous
    '😱': {'sentiment': 'Negative', 'emotion': 'Fear / Shock', 'category': 'Fear'},
    '😨': {'sentiment': 'Negative', 'emotion': 'Fear / Anxiety', 'category': 'Fear'},
    '😰': {'sentiment': 'Negative', 'emotion': 'Anxiety / Stress', 'category': 'Fear'},
    '😲': {'sentiment': 'Neutral', 'emotion': 'Surprise / Astonishment', 'category': 'Surprise'},
    '😮': {'sentiment': 'Neutral', 'emotion': 'Surprise / Wonder', 'category': 'Surprise'},
    '🤯': {'sentiment': 'Neutral', 'emotion': 'Mind Blown / Shock', 'category': 'Surprise'},
    '🙃': {'sentiment': 'Neutral', 'emotion': 'Irony / Sarcasm / Silly', 'category': 'Irony'},
    '😐': {'sentiment': 'Neutral', 'emotion': 'Neutral / Indifference', 'category': 'Neutral'},
    '🤔': {'sentiment': 'Neutral', 'emotion': 'Thoughtful / Puzzled', 'category': 'Thought'}
}


class EmojiAnalyzer:
    """
    Extracts and semantically characterizes emojis in text.
    """

    def analyze_emojis(self, text: str) -> Dict[str, Any]:
        """
        Analyzes all emojis present in the text string.
        """
        if not text:
            return self._empty_result()

        # Extract emojis using the emoji library
        emoji_list = emoji.emoji_list(text)
        detected_chars = [item['emoji'] for item in emoji_list]
        total_count = len(detected_chars)
        
        if total_count == 0:
            return self._empty_result()

        # Frequency counter
        freq_counter = Counter(detected_chars)
        
        unique_emojis_data = []
        sentiment_tallies = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        
        for char, count in freq_counter.most_common():
            # Get official demographic / descriptive name
            raw_name = emoji.demojize(char).replace(':', '').replace('_', ' ').title()
            
            # Semantic interpretation
            semantic_info = EMOJI_SEMANTIC_MAP.get(char, {
                'sentiment': 'Neutral',
                'emotion': 'General Emotion',
                'category': 'Miscellaneous'
            })
            
            s_pol = semantic_info['sentiment']
            sentiment_tallies[s_pol] = sentiment_tallies.get(s_pol, 0) + count
            
            unique_emojis_data.append({
                'emoji': char,
                'count': count,
                'description': raw_name,
                'sentiment': s_pol,
                'emotion': semantic_info['emotion'],
                'category': semantic_info['category']
            })

        # Predominant emoji polarity
        predominant_sentiment = max(sentiment_tallies, key=sentiment_tallies.get)
        if sum(sentiment_tallies.values()) == 0 or (sentiment_tallies['Positive'] == sentiment_tallies['Negative'] and sentiment_tallies['Positive'] > 0):
            predominant_sentiment = 'Mixed / Balanced'

        return {
            'total_emoji_count': total_count,
            'unique_emoji_count': len(freq_counter),
            'emojis_detected': detected_chars,
            'emoji_breakdown': unique_emojis_data,
            'emoji_sentiment_summary': predominant_sentiment,
            'emoji_string': " ".join(detected_chars)
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            'total_emoji_count': 0,
            'unique_emoji_count': 0,
            'emojis_detected': [],
            'emoji_breakdown': [],
            'emoji_sentiment_summary': 'No Emojis Detected',
            'emoji_string': ''
        }
