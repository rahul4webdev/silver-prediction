# Features module
from app.ml.features.technical import TechnicalIndicators, add_technical_features

# Optional sentiment imports
try:
    from app.ml.features.sentiment import (
        FinancialSentimentAnalyzer,
        SentimentFeatureEngine,
        sentiment_analyzer,
        sentiment_feature_engine,
    )
except ImportError:
    FinancialSentimentAnalyzer = None
    SentimentFeatureEngine = None
    sentiment_analyzer = None
    sentiment_feature_engine = None

__all__ = [
    "TechnicalIndicators",
    "add_technical_features",
    "FinancialSentimentAnalyzer",
    "SentimentFeatureEngine",
    "sentiment_analyzer",
    "sentiment_feature_engine",
]
