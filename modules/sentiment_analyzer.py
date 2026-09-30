"""
Sentiment Analyzer Module.
Executes Transformer-based sentiment classification, computes class probabilities,
confidence intervals, and analyzes mixed/contrastive sentiment signals.
"""

from typing import Dict, Any, List
from models.model_loader import ModelRegistry, MODEL_METADATA


class SentimentAnalyzer:
    """
    Performs Transformer-based Sentiment Analysis using CardiffNLP Twitter-RoBERTa.
    """

    def __init__(self, pipeline_fn=None):
        self.pipeline_fn = pipeline_fn or ModelRegistry.load_sentiment_pipeline
        self.metadata = MODEL_METADATA["sentiment"]

    def analyze(self, text: str, feature_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Runs neural sentiment classification on text.
        """
        if not text or not text.strip():
            return self._empty_response()

        pipe = self.pipeline_fn()
        # Truncate to 500 characters or reasonable token length for safety
        model_output = pipe(text[:1000])

        # model_output with top_k=None returns list of dicts: [{'label': 'positive', 'score': 0.85}, ...]
        if isinstance(model_output, list) and len(model_output) > 0 and isinstance(model_output[0], list):
            scores_list = model_output[0]
        elif isinstance(model_output, list):
            scores_list = model_output
        else:
            scores_list = [model_output]

        # Standardize labels to Title Case
        probabilities = {}
        for item in scores_list:
            raw_label = str(item['label']).lower()
            score = float(item['score'])
            
            # Map labels
            if 'pos' in raw_label:
                probabilities['Positive'] = score
            elif 'neg' in raw_label:
                probabilities['Negative'] = score
            elif 'neu' in raw_label:
                probabilities['Neutral'] = score
            else:
                probabilities[raw_label.capitalize()] = score

        # Ensure all 3 canonical classes exist
        for cat in ['Positive', 'Neutral', 'Negative']:
            if cat not in probabilities:
                probabilities[cat] = 0.0

        # Sort and find dominant label
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        dominant_label, dominant_score = sorted_probs[0]

        # Mixed sentiment detection logic
        mixed_analysis = self._detect_mixed_sentiment(probabilities, feature_data)

        # Color and Emoji icon mappings for UI
        color_map = {
            'Positive': '#28a745', # Emerald Green
            'Negative': '#dc3545', # Coral Red
            'Neutral': '#6c757d',  # Slate Grey
            'Mixed': '#ffc107'     # Amber Yellow
        }
        icon_map = {
            'Positive': '🟢 Positive',
            'Negative': '🔴 Negative',
            'Neutral': '⚪ Neutral',
            'Mixed': '🟡 Mixed / Contrastive'
        }

        return {
            'dominant_sentiment': dominant_label,
            'confidence': round(dominant_score * 100, 1),
            'confidence_ratio': round(dominant_score, 4),
            'display_label': icon_map.get(dominant_label, dominant_label),
            'color': color_map.get(dominant_label, '#4f46e5'),
            'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
            'probabilities_pct': {k: round(v * 100, 1) for k, v in probabilities.items()},
            'mixed_analysis': mixed_analysis,
            'model_info': {
                'name': self.metadata['name'],
                'id': self.metadata['huggingface_id']
            }
        }

    def _detect_mixed_sentiment(self, probabilities: Dict[str, float], feature_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Derives mixed sentiment indicators using contrastive cues and model score balance.
        """
        pos_score = probabilities.get('Positive', 0.0)
        neg_score = probabilities.get('Negative', 0.0)
        
        has_contrastive = False
        pos_cues = []
        neg_cues = []
        
        if feature_data and 'lexical_cues' in feature_data:
            lex = feature_data['lexical_cues']
            has_contrastive = lex.get('has_contrastive', False)
            pos_cues = lex.get('positive_cues', [])
            neg_cues = lex.get('negative_cues', [])

        is_balanced = (pos_score > 0.25 and neg_score > 0.25)
        is_mixed = (has_contrastive and len(pos_cues) > 0 and len(neg_cues) > 0) or is_balanced

        explanation = "Single dominant sentiment signal detected."
        if is_mixed:
            explanation = "Contrastive signals detected (e.g., both positive and negative lexical markers or competing model probabilities)."

        return {
            'is_mixed': is_mixed,
            'has_contrastive_conjunctions': has_contrastive,
            'positive_indicators': pos_cues,
            'negative_indicators': neg_cues,
            'explanation': explanation
        }

    def _empty_response(self) -> Dict[str, Any]:
        return {
            'dominant_sentiment': 'Neutral',
            'confidence': 0.0,
            'confidence_ratio': 0.0,
            'display_label': '⚪ Neutral',
            'color': '#6c757d',
            'probabilities': {'Positive': 0.0, 'Neutral': 1.0, 'Negative': 0.0},
            'probabilities_pct': {'Positive': 0.0, 'Neutral': 100.0, 'Negative': 0.0},
            'mixed_analysis': {
                'is_mixed': False,
                'has_contrastive_conjunctions': False,
                'positive_indicators': [],
                'negative_indicators': [],
                'explanation': 'Empty input.'
            },
            'model_info': {
                'name': self.metadata['name'],
                'id': self.metadata['huggingface_id']
            }
        }
