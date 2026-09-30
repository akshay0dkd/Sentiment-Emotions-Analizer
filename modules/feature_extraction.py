"""
Feature Extraction Module.
Computes surface text statistics, linguistic counts, punctuation frequencies,
and lexical polarity markers.
"""

import re
from typing import Dict, Any, List
import string

# Curated lexicons for auxiliary text mining indicators
POSITIVE_LEXICON = {
    'good', 'great', 'excellent', 'amazing', 'love', 'loved', 'loving', 'wonderful',
    'fantastic', 'awesome', 'best', 'super', 'happy', 'pleased', 'delighted',
    'perfect', 'brilliant', 'outstanding', 'flawless', 'exceptional', 'positive',
    'beautiful', 'recommend', 'favorite', 'enjoyed', 'satisfying', 'top-notch'
}

NEGATIVE_LEXICON = {
    'bad', 'terrible', 'horrible', 'awful', 'worst', 'poor', 'hate', 'hated',
    'disappointed', 'disappointing', 'useless', 'broken', 'slow', 'delay', 'delayed',
    'annoying', 'angry', 'sucks', 'fail', 'failed', 'failure', 'waste', 'pathetic',
    'unacceptable', 'unpleasant', 'rude', 'frustrated', 'frustrating', 'problem'
}

CONTRASTIVE_CONJUNCTIONS = {
    'but', 'however', 'although', 'though', 'yet', 'despite', 'nevertheless', 'nonetheless'
}

INTENSIFIERS = {
    'very', 'extremely', 'absolutely', 'completely', 'totally', 'really', 'so',
    'highly', 'deeply', 'incredibly', 'exceptionally', 'utterly'
}


class FeatureExtractor:
    """
    Extracts comprehensive Text Mining surface, statistical, and lexical features.
    """

    def __init__(self):
        self.punctuation_set = set(string.punctuation)

    def extract_features(self, text: str, preprocessed_data: Dict[str, Any], emoji_count: int = 0) -> Dict[str, Any]:
        """
        Extracts comprehensive features from raw text and preprocessed tokens.
        """
        if not text:
            return self._empty_features()

        char_count = len(text)
        char_no_spaces = len(text.replace(" ", ""))
        words = text.split()
        word_count = len(words)
        
        # Sentence splitting using regex for robustness
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        sentence_count = max(1, len(sentences)) if text.strip() else 0

        # Word lengths
        word_lengths = [len(w) for w in words]
        avg_word_length = round(sum(word_lengths) / word_count, 2) if word_count > 0 else 0.0

        # Uppercase / Capitalization patterns (stripping surrounding punctuation like WOW!)
        clean_words = [w.strip(string.punctuation) for w in words]
        uppercase_words = [cw for cw in clean_words if cw.isupper() and len(cw) > 1 and cw.isalpha()]
        uppercase_count = len(uppercase_words)
        uppercase_ratio = round((uppercase_count / word_count) * 100, 1) if word_count > 0 else 0.0

        # Punctuation counts
        punct_count = sum(1 for c in text if c in self.punctuation_set)
        exclamation_count = text.count('!')
        question_count = text.count('?')
        ellipsis_count = len(re.findall(r'\.{2,}', text))

        # Metadata counts from preprocessor
        metadata = preprocessed_data.get('metadata', {})
        url_count = len(metadata.get('urls', []))
        mention_count = len(metadata.get('mentions', []))
        hashtag_count = len(metadata.get('hashtags', []))

        # Lexical feature matches (using clean tokens)
        clean_tokens = preprocessed_data.get('clean_tokens', [])
        clean_tokens_set = set(clean_tokens)
        
        positive_cues = list(clean_tokens_set.intersection(POSITIVE_LEXICON))
        negative_cues = list(clean_tokens_set.intersection(NEGATIVE_LEXICON))
        contrastive_cues = list(clean_tokens_set.intersection(CONTRASTIVE_CONJUNCTIONS))
        intensifier_cues = list(clean_tokens_set.intersection(INTENSIFIERS))

        return {
            'statistics': {
                'character_count': char_count,
                'character_no_spaces': char_no_spaces,
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_word_length': avg_word_length,
                'emoji_count': emoji_count,
            },
            'surface_counts': {
                'uppercase_words': uppercase_count,
                'uppercase_ratio_pct': uppercase_ratio,
                'punctuation_count': punct_count,
                'exclamation_count': exclamation_count,
                'question_count': question_count,
                'ellipsis_count': ellipsis_count,
                'url_count': url_count,
                'mention_count': mention_count,
                'hashtag_count': hashtag_count,
            },
            'lexical_cues': {
                'positive_cues': positive_cues,
                'negative_cues': negative_cues,
                'contrastive_cues': contrastive_cues,
                'intensifier_cues': intensifier_cues,
                'has_contrastive': len(contrastive_cues) > 0,
                'is_potentially_mixed': len(positive_cues) > 0 and len(negative_cues) > 0
            }
        }

    def _empty_features(self) -> Dict[str, Any]:
        return {
            'statistics': {
                'character_count': 0,
                'character_no_spaces': 0,
                'word_count': 0,
                'sentence_count': 0,
                'avg_word_length': 0.0,
                'emoji_count': 0,
            },
            'surface_counts': {
                'uppercase_words': 0,
                'uppercase_ratio_pct': 0.0,
                'punctuation_count': 0,
                'exclamation_count': 0,
                'question_count': 0,
                'ellipsis_count': 0,
                'url_count': 0,
                'mention_count': 0,
                'hashtag_count': 0,
            },
            'lexical_cues': {
                'positive_cues': [],
                'negative_cues': [],
                'contrastive_cues': [],
                'intensifier_cues': [],
                'has_contrastive': False,
                'is_potentially_mixed': False
            }
        }
