"""
Emotion Analyzer Module.
Executes Transformer-based multi-class emotion classification across Ekman's emotion taxonomy.
"""

from typing import Dict, Any, List
from models.model_loader import ModelRegistry, MODEL_METADATA

EMOTION_ICONS = {
    'Joy': '😊 Joy',
    'Sadness': '😢 Sadness',
    'Anger': '😡 Anger',
    'Fear': '😨 Fear',
    'Surprise': '😲 Surprise',
    'Disgust': '🤢 Disgust',
    'Neutral': '😐 Neutral'
}

EMOTION_COLORS = {
    'Joy': '#10B981',       # Emerald green
    'Sadness': '#3B82F6',   # Blue
    'Anger': '#EF4444',     # Red
    'Fear': '#8B5CF6',      # Purple
    'Surprise': '#F59E0B',  # Amber
    'Disgust': '#84CC16',   # Lime
    'Neutral': '#9CA3AF'    # Gray
}


class EmotionAnalyzer:
    """
    Performs Transformer-based Emotion Classification using j-hartmann/emotion-english-distilroberta-base.
    """

    def __init__(self, pipeline_fn=None):
        self.pipeline_fn = pipeline_fn or ModelRegistry.load_emotion_pipeline
        self.metadata = MODEL_METADATA["emotion"]

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Runs neural emotion classification on input text.
        """
        if not text or not text.strip():
            return self._empty_response()

        pipe = self.pipeline_fn()
        model_output = pipe(text[:1000])

        if isinstance(model_output, list) and len(model_output) > 0 and isinstance(model_output[0], list):
            scores_list = model_output[0]
        elif isinstance(model_output, list):
            scores_list = model_output
        else:
            scores_list = [model_output]

        # Extract probabilities for standard Ekman labels
        probabilities = {}
        for item in scores_list:
            raw_label = str(item['label']).lower()
            score = float(item['score'])
            norm_label = self.metadata['display_labels'].get(raw_label, raw_label.capitalize())
            probabilities[norm_label] = score

        # Ensure all registered labels are present
        for exp_label in self.metadata['display_labels'].values():
            if exp_label not in probabilities:
                probabilities[exp_label] = 0.0

        # Sort descending
        ranked_emotions = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        dominant_label, dominant_score = ranked_emotions[0]

        return {
            'dominant_emotion': dominant_label,
            'confidence': round(dominant_score * 100, 1),
            'confidence_ratio': round(dominant_score, 4),
            'display_label': EMOTION_ICONS.get(dominant_label, dominant_label),
            'color': EMOTION_COLORS.get(dominant_label, '#6366F1'),
            'probabilities': {k: round(v, 4) for k, v in probabilities.items()},
            'probabilities_pct': {k: round(v * 100, 1) for k, v in probabilities.items()},
            'ranked_distribution': [
                {
                    'emotion': k,
                    'display': EMOTION_ICONS.get(k, k),
                    'score': round(v, 4),
                    'percentage': round(v * 100, 1),
                    'color': EMOTION_COLORS.get(k, '#6366F1')
                }
                for k, v in ranked_emotions
            ],
            'model_info': {
                'name': self.metadata['name'],
                'id': self.metadata['huggingface_id']
            }
        }

    def _empty_response(self) -> Dict[str, Any]:
        return {
            'dominant_emotion': 'Neutral',
            'confidence': 0.0,
            'confidence_ratio': 0.0,
            'display_label': '😐 Neutral',
            'color': '#9CA3AF',
            'probabilities': {k: 0.0 for k in self.metadata['display_labels'].values()},
            'probabilities_pct': {k: 0.0 for k in self.metadata['display_labels'].values()},
            'ranked_distribution': [],
            'model_info': {
                'name': self.metadata['name'],
                'id': self.metadata['huggingface_id']
            }
        }
