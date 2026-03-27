"""
model.py — Sentiment model wrapper for both transformer-based and lexicon-based inference.
"""

from __future__ import annotations
from typing import Dict, Any


class SentimentModel:
    """
    Unified interface for sentiment analysis models.

    Supports:
      - HuggingFace transformer pipelines (DistilBERT, RoBERTa, etc.)
      - VADER (rule-based, no GPU needed)
    """

    LABEL_MAP = {
        # HuggingFace SST-2 labels
        "LABEL_0": "NEGATIVE",
        "LABEL_1": "POSITIVE",
        # Cardiff NLP RoBERTa labels
        "negative": "NEGATIVE",
        "neutral": "NEUTRAL",
        "positive": "POSITIVE",
    }

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._pipeline = None
        self._vader = None
        self._load()

    def _load(self) -> None:
        if self.model_name == "vader":
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self._vader = SentimentIntensityAnalyzer()
        else:
            from transformers import pipeline
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                truncation=True,
                max_length=512,
            )

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Run inference on a single text string.

        Returns:
            dict with keys:
              - label (str): "POSITIVE", "NEGATIVE", or "NEUTRAL"
              - score (float): confidence in [0, 1]
              - model (str): model identifier used
        """
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.5, "model": self.model_name}

        if self._vader:
            return self._predict_vader(text)
        return self._predict_transformer(text)

    def _predict_transformer(self, text: str) -> Dict[str, Any]:
        raw = self._pipeline(text)[0]
        label = self.LABEL_MAP.get(raw["label"], raw["label"].upper())
        return {"label": label, "score": float(raw["score"]), "model": self.model_name}

    def _predict_vader(self, text: str) -> Dict[str, Any]:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            label, score = "POSITIVE", (compound + 1) / 2
        elif compound <= -0.05:
            label, score = "NEGATIVE", (1 - compound) / 2
        else:
            label, score = "NEUTRAL", 0.5 + abs(compound)
        return {"label": label, "score": min(score, 1.0), "model": "vader"}

    def predict_batch(self, texts: list[str]) -> list[Dict[str, Any]]:
        """Run inference on a list of texts."""
        return [self.predict(t) for t in texts]
