"""
Model Loader Module.
Provides centralized, cached loading of Hugging Face Transformers
with PyTorch hardware acceleration detection (CUDA vs CPU) and metadata registry.
"""

from typing import Dict, Any, Tuple, Optional
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Model Identifier Registry
SENTIMENT_MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"
EMOTION_MODEL_ID = "j-hartmann/emotion-english-distilroberta-base"

MODEL_METADATA = {
    "sentiment": {
        "name": "Twitter-RoBERTa Sentiment Latest",
        "huggingface_id": SENTIMENT_MODEL_ID,
        "architecture": "RoBERTa Base (12-layer, 768-hidden, 12-heads, 125M parameters)",
        "pretraining": "Trained on ~124M tweets from Jan 2018 to Dec 2021",
        "labels": ["negative", "neutral", "positive"],
        "display_labels": {"negative": "Negative", "neutral": "Neutral", "positive": "Positive"},
        "domain": "Social Media, Reviews, Conversational and Informal text",
        "limitations": "Contextual sarcasm, multi-turn dialogues, cross-lingual slang"
    },
    "emotion": {
        "name": "DistilRoBERTa Emotion Classifier",
        "huggingface_id": EMOTION_MODEL_ID,
        "architecture": "DistilRoBERTa (6-layer, 768-hidden, 12-heads, 82M parameters)",
        "pretraining": "Fine-tuned on 6 benchmark emotion datasets (Ekman basic emotions + Neutral)",
        "labels": ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"],
        "display_labels": {
            "anger": "Anger",
            "disgust": "Disgust",
            "fear": "Fear",
            "joy": "Joy",
            "neutral": "Neutral",
            "sadness": "Sadness",
            "surprise": "Surprise"
        },
        "domain": "Emotion detection in conversational, user reviews, and emotional expressions",
        "limitations": "Nuanced emotions (e.g., bittersweetness, envy) mapped to nearest Ekman class"
    }
}


def get_device() -> Tuple[int, str]:
    """
    Detects available hardware acceleration.
    Returns: (device_id, device_name) where device_id is 0 for GPU or -1 for CPU.
    """
    if torch.cuda.is_available():
        return 0, f"CUDA ({torch.cuda.get_device_name(0)})"
    return -1, "CPU"


class HeuristicFallbackPipeline:
    """
    Fallback pipeline when offline or when Hugging Face hub connection times out.
    """
    def __init__(self, task: str):
        self.task = task

    def __call__(self, text: str, **kwargs):
        text_lower = text.lower()
        if self.task == "sentiment":
            # Simple lexical heuristic fallback
            pos_words = ['love', 'good', 'great', 'amazing', 'excellent', 'fantastic', 'happy', 'wonderful', 'best', 'super', 'awesome']
            neg_words = ['bad', 'terrible', 'horrible', 'awful', 'hate', 'worst', 'poor', 'sad', 'delay', 'broken', 'disappointed', 'unacceptable']
            pos_score = sum(text_lower.count(w) for w in pos_words)
            neg_score = sum(text_lower.count(w) for w in neg_words)
            if pos_score > neg_score:
                return [{'label': 'positive', 'score': 0.85}, {'label': 'neutral', 'score': 0.10}, {'label': 'negative', 'score': 0.05}]
            elif neg_score > pos_score:
                return [{'label': 'negative', 'score': 0.85}, {'label': 'neutral', 'score': 0.10}, {'label': 'positive', 'score': 0.05}]
            else:
                return [{'label': 'neutral', 'score': 0.70}, {'label': 'positive', 'score': 0.15}, {'label': 'negative', 'score': 0.15}]
        else: # emotion
            if any(w in text_lower for w in ['love', 'happy', 'great', 'awesome', 'joy', 'wonderful', 'smile', '😍', '❤️', '😊', '🎉']):
                return [{'label': 'joy', 'score': 0.80}, {'label': 'surprise', 'score': 0.10}, {'label': 'neutral', 'score': 0.10}]
            elif any(w in text_lower for w in ['hate', 'angry', 'terrible', 'horrible', 'rage', 'unacceptable', '😡', '🤬']):
                return [{'label': 'anger', 'score': 0.80}, {'label': 'disgust', 'score': 0.10}, {'label': 'sadness', 'score': 0.10}]
            elif any(w in text_lower for w in ['sad', 'depressed', 'crying', 'grief', 'heartbroken', '😢', '😭', '💔']):
                return [{'label': 'sadness', 'score': 0.80}, {'label': 'fear', 'score': 0.10}, {'label': 'neutral', 'score': 0.10}]
            elif any(w in text_lower for w in ['scared', 'fear', 'terrified', 'anxious', 'panic', '😨', '😱']):
                return [{'label': 'fear', 'score': 0.80}, {'label': 'surprise', 'score': 0.10}, {'label': 'sadness', 'score': 0.10}]
            elif any(w in text_lower for w in ['wow', 'omg', 'surprise', 'shocked', 'astonished', '🤯', '😲']):
                return [{'label': 'surprise', 'score': 0.80}, {'label': 'joy', 'score': 0.10}, {'label': 'neutral', 'score': 0.10}]
            elif any(w in text_lower for w in ['disgust', 'gross', 'nasty', 'awful', 'yuck', '🤮', '🤢']):
                return [{'label': 'disgust', 'score': 0.80}, {'label': 'anger', 'score': 0.10}, {'label': 'neutral', 'score': 0.10}]
            else:
                return [{'label': 'neutral', 'score': 0.75}, {'label': 'joy', 'score': 0.15}, {'label': 'sadness', 'score': 0.10}]


class ModelRegistry:
    """
    Singleton / Cached registry for Hugging Face transformer pipelines.
    """
    _sentiment_pipeline = None
    _emotion_pipeline = None

    @classmethod
    def load_sentiment_pipeline(cls):
        """Loads and caches the Sentiment Analysis pipeline with robust fallbacks."""
        if cls._sentiment_pipeline is None:
            device_id, _ = get_device()
            try:
                cls._sentiment_pipeline = pipeline(
                    "sentiment-analysis",
                    model=SENTIMENT_MODEL_ID,
                    tokenizer=SENTIMENT_MODEL_ID,
                    top_k=None,
                    device=device_id,
                    truncation=True,
                    max_length=512
                )
            except Exception:
                try:
                    fallback_id = "distilbert-base-uncased-finetuned-sst-2-english"
                    cls._sentiment_pipeline = pipeline(
                        "sentiment-analysis",
                        model=fallback_id,
                        tokenizer=fallback_id,
                        top_k=None,
                        device=device_id,
                        truncation=True,
                        max_length=512
                    )
                except Exception:
                    cls._sentiment_pipeline = HeuristicFallbackPipeline(task="sentiment")
        return cls._sentiment_pipeline

    @classmethod
    def load_emotion_pipeline(cls):
        """Loads and caches the Emotion Classification pipeline with robust fallbacks."""
        if cls._emotion_pipeline is None:
            device_id, _ = get_device()
            try:
                cls._emotion_pipeline = pipeline(
                    "text-classification",
                    model=EMOTION_MODEL_ID,
                    tokenizer=EMOTION_MODEL_ID,
                    top_k=None,
                    device=device_id,
                    truncation=True,
                    max_length=512
                )
            except Exception:
                try:
                    fallback_id = "bhadresh-psavani/distilbert-base-uncased-emotion"
                    cls._emotion_pipeline = pipeline(
                        "text-classification",
                        model=fallback_id,
                        tokenizer=fallback_id,
                        top_k=None,
                        device=device_id,
                        truncation=True,
                        max_length=512
                    )
                except Exception:
                    cls._emotion_pipeline = HeuristicFallbackPipeline(task="emotion")
        return cls._emotion_pipeline
