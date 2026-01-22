"""
Sentiment analysis feature engineering for ML models.
Uses both lexicon-based and transformer-based approaches.

Provides sentiment features that can be used alongside technical indicators
to improve prediction accuracy.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Try to import transformers for FinBERT
_transformers_available = False
_finbert_model = None
_finbert_tokenizer = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _transformers_available = True
except ImportError:
    logger.info("Transformers not available, using lexicon-based sentiment only")


class FinancialSentimentAnalyzer:
    """
    Financial sentiment analyzer using FinBERT model.

    FinBERT is a BERT model fine-tuned on financial text,
    providing more accurate sentiment for financial news.
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(self, use_transformers: bool = True):
        """
        Initialize sentiment analyzer.

        Args:
            use_transformers: Whether to use FinBERT (requires transformers library)
        """
        self.use_transformers = use_transformers and _transformers_available
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load_model(self):
        """Lazy load the FinBERT model."""
        global _finbert_model, _finbert_tokenizer

        if not self.use_transformers:
            return

        if _finbert_model is not None:
            self._model = _finbert_model
            self._tokenizer = _finbert_tokenizer
            return

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            logger.info("Loading FinBERT model...")

            self._tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_NAME)

            # Use GPU if available
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model.to(self._device)
            self._model.eval()

            # Cache globally
            _finbert_model = self._model
            _finbert_tokenizer = self._tokenizer

            logger.info(f"FinBERT model loaded on {self._device}")

        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}")
            self.use_transformers = False

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            Dict with sentiment scores:
            - sentiment: -1 (negative) to +1 (positive)
            - positive: probability of positive
            - negative: probability of negative
            - neutral: probability of neutral
        """
        if self.use_transformers:
            return self._analyze_with_finbert(text)
        else:
            return self._analyze_with_lexicon(text)

    def _analyze_with_finbert(self, text: str) -> Dict[str, float]:
        """Analyze using FinBERT model."""
        self._load_model()

        if self._model is None:
            return self._analyze_with_lexicon(text)

        try:
            import torch

            # Truncate text to max length
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            # FinBERT outputs: [positive, negative, neutral]
            probs = probs.cpu().numpy()[0]

            positive = float(probs[0])
            negative = float(probs[1])
            neutral = float(probs[2])

            # Convert to single sentiment score
            sentiment = positive - negative

            return {
                "sentiment": sentiment,
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
                "method": "finbert",
            }

        except Exception as e:
            logger.error(f"FinBERT analysis failed: {e}")
            return self._analyze_with_lexicon(text)

    def _analyze_with_lexicon(self, text: str) -> Dict[str, float]:
        """Analyze using lexicon-based approach (fallback)."""
        import re

        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        # Financial sentiment lexicons
        bullish_words = {
            "surge", "rally", "soar", "jump", "gain", "rise", "climb",
            "bullish", "optimistic", "demand", "buying", "support",
            "breakout", "momentum", "upside", "strength", "positive",
            "record", "high", "beat", "exceed", "growth", "expansion",
            "recovery", "boost", "advance", "upgrade", "outperform",
            "profit", "earnings", "dividend", "opportunity"
        }

        bearish_words = {
            "drop", "fall", "plunge", "decline", "crash", "bearish",
            "pessimistic", "selling", "pressure", "breakdown", "weak",
            "downside", "negative", "low", "miss", "below", "loss",
            "recession", "contraction", "downgrade", "underperform",
            "concern", "fear", "risk", "uncertainty", "volatile",
            "warning", "caution", "crisis", "default"
        }

        bullish_count = sum(1 for w in words if w in bullish_words)
        bearish_count = sum(1 for w in words if w in bearish_words)

        total = bullish_count + bearish_count
        if total == 0:
            sentiment = 0.0
            positive = 0.33
            negative = 0.33
            neutral = 0.34
        else:
            sentiment = (bullish_count - bearish_count) / total
            positive = bullish_count / (total + 1)
            negative = bearish_count / (total + 1)
            neutral = 1 - positive - negative

        return {
            "sentiment": max(-1, min(1, sentiment)),
            "positive": positive,
            "negative": negative,
            "neutral": max(0, neutral),
            "method": "lexicon",
        }

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """
        Analyze sentiment of multiple texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment dicts
        """
        if not self.use_transformers or len(texts) <= 5:
            return [self.analyze_text(t) for t in texts]

        # Batch processing for FinBERT
        self._load_model()

        if self._model is None:
            return [self._analyze_with_lexicon(t) for t in texts]

        try:
            import torch

            results = []
            batch_size = 8

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self._tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

                probs = probs.cpu().numpy()

                for p in probs:
                    results.append({
                        "sentiment": float(p[0] - p[1]),
                        "positive": float(p[0]),
                        "negative": float(p[1]),
                        "neutral": float(p[2]),
                        "method": "finbert",
                    })

            return results

        except Exception as e:
            logger.error(f"Batch FinBERT analysis failed: {e}")
            return [self._analyze_with_lexicon(t) for t in texts]


class SentimentFeatureEngine:
    """
    Generate sentiment features for ML models.

    Combines news sentiment with price data to create
    features that can improve prediction accuracy.
    """

    def __init__(self):
        self.analyzer = FinancialSentimentAnalyzer()

    def create_sentiment_features(
        self,
        news_articles: List[Dict[str, Any]],
        lookback_hours: int = 24,
    ) -> Dict[str, float]:
        """
        Create sentiment features from news articles.

        Args:
            news_articles: List of article dicts with 'title', 'description', 'published_at'
            lookback_hours: Hours of news to consider

        Returns:
            Dict of sentiment features
        """
        if not news_articles:
            return self._empty_features()

        # Filter by time
        cutoff = datetime.now() - timedelta(hours=lookback_hours)
        recent = [
            a for a in news_articles
            if a.get("published_at", datetime.now()) >= cutoff
        ]

        if not recent:
            return self._empty_features()

        # Analyze each article
        texts = [
            f"{a.get('title', '')} {a.get('description', '')}"
            for a in recent
        ]
        sentiments = self.analyzer.analyze_batch(texts)

        # Aggregate features
        sentiment_scores = [s["sentiment"] for s in sentiments]
        positive_scores = [s["positive"] for s in sentiments]
        negative_scores = [s["negative"] for s in sentiments]

        features = {
            # Basic sentiment
            "sentiment_mean": float(np.mean(sentiment_scores)),
            "sentiment_std": float(np.std(sentiment_scores)) if len(sentiment_scores) > 1 else 0,
            "sentiment_min": float(np.min(sentiment_scores)),
            "sentiment_max": float(np.max(sentiment_scores)),

            # Positive/Negative ratios
            "positive_ratio": float(np.mean(positive_scores)),
            "negative_ratio": float(np.mean(negative_scores)),

            # Counts
            "news_count": len(recent),
            "bullish_news_count": sum(1 for s in sentiment_scores if s > 0.2),
            "bearish_news_count": sum(1 for s in sentiment_scores if s < -0.2),

            # Momentum (recent vs older)
            "sentiment_momentum": self._calculate_momentum(recent, sentiments),

            # Consensus strength
            "sentiment_consensus": 1 - float(np.std(sentiment_scores)) if len(sentiment_scores) > 1 else 0.5,
        }

        return features

    def _calculate_momentum(
        self,
        articles: List[Dict],
        sentiments: List[Dict],
    ) -> float:
        """Calculate sentiment momentum (recent vs older)."""
        if len(articles) < 4:
            return 0.0

        mid = len(articles) // 2
        recent_sentiment = np.mean([s["sentiment"] for s in sentiments[:mid]])
        older_sentiment = np.mean([s["sentiment"] for s in sentiments[mid:]])

        return float(recent_sentiment - older_sentiment)

    def _empty_features(self) -> Dict[str, float]:
        """Return empty features when no data available."""
        return {
            "sentiment_mean": 0.0,
            "sentiment_std": 0.0,
            "sentiment_min": 0.0,
            "sentiment_max": 0.0,
            "positive_ratio": 0.33,
            "negative_ratio": 0.33,
            "news_count": 0,
            "bullish_news_count": 0,
            "bearish_news_count": 0,
            "sentiment_momentum": 0.0,
            "sentiment_consensus": 0.5,
        }

    def add_sentiment_to_df(
        self,
        df: pd.DataFrame,
        sentiment_features: Dict[str, float],
    ) -> pd.DataFrame:
        """
        Add sentiment features to price dataframe.

        Since news doesn't change as frequently as price,
        sentiment features are constant for recent rows.

        Args:
            df: DataFrame with price data
            sentiment_features: Dict of sentiment features

        Returns:
            DataFrame with sentiment columns added
        """
        df = df.copy()

        for feature_name, value in sentiment_features.items():
            df[f"news_{feature_name}"] = value

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Get list of sentiment feature column names."""
        return [
            "news_sentiment_mean",
            "news_sentiment_std",
            "news_sentiment_min",
            "news_sentiment_max",
            "news_positive_ratio",
            "news_negative_ratio",
            "news_count",
            "news_bullish_news_count",
            "news_bearish_news_count",
            "news_sentiment_momentum",
            "news_sentiment_consensus",
        ]


# Singleton instances
sentiment_analyzer = FinancialSentimentAnalyzer()
sentiment_feature_engine = SentimentFeatureEngine()
