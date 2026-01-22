"""
News sentiment API endpoints.
Provides access to news sentiment analysis for silver and gold.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query

from app.services.news_sentiment import news_sentiment_service, SentimentResult, NewsArticle
from app.ml.features.sentiment import sentiment_feature_engine

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sentiment")


@router.get("/current", response_model=Dict[str, Any])
async def get_current_sentiment(
    asset: str = Query(default="silver", description="Asset to analyze (silver, gold)"),
    lookback_days: int = Query(default=3, ge=1, le=7, description="Days of news to analyze"),
):
    """
    Get current news sentiment for an asset.

    Returns aggregated sentiment analysis from recent news articles.
    """
    try:
        sentiment = await news_sentiment_service.get_sentiment(
            asset=asset,
            lookback_days=lookback_days,
        )

        return {
            "status": "success",
            "asset": asset,
            "lookback_days": lookback_days,
            "sentiment": {
                "overall": sentiment.overall_sentiment,
                "label": sentiment.sentiment_label,
                "confidence": sentiment.confidence,
            },
            "stats": {
                "article_count": sentiment.article_count,
                "bullish_count": sentiment.bullish_count,
                "bearish_count": sentiment.bearish_count,
                "neutral_count": sentiment.neutral_count,
                "average_relevance": sentiment.average_relevance,
            },
            "timestamp": sentiment.timestamp.isoformat(),
            "articles": [
                {
                    "title": article.title,
                    "source": article.source,
                    "published_at": article.published_at.isoformat(),
                    "url": article.url,
                    "sentiment_score": article.sentiment_score,
                    "relevance_score": article.relevance_score,
                }
                for article in sentiment.articles[:10]  # Top 10 articles
            ],
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment: {e}")
        return {
            "status": "error",
            "message": str(e),
            "asset": asset,
        }


@router.get("/features", response_model=Dict[str, Any])
async def get_sentiment_features(
    asset: str = Query(default="silver", description="Asset to analyze"),
):
    """
    Get sentiment features formatted for ML models.

    Returns features that can be used in prediction models.
    """
    try:
        sentiment = await news_sentiment_service.get_sentiment(asset=asset)

        # Convert to ML features
        features = news_sentiment_service.sentiment_to_features(sentiment)

        return {
            "status": "success",
            "asset": asset,
            "features": features,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment features: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/history", response_model=Dict[str, Any])
async def get_sentiment_history(
    asset: str = Query(default="silver", description="Asset to analyze"),
    days: int = Query(default=7, ge=1, le=30, description="Days of history"),
):
    """
    Get sentiment history over time.

    Note: This requires sentiment data to be stored in the database.
    Currently returns recent sentiment snapshots.
    """
    try:
        # Get sentiment for different time windows
        history = []

        for lookback in [1, 2, 3, 5, 7]:
            if lookback <= days:
                sentiment = await news_sentiment_service.get_sentiment(
                    asset=asset,
                    lookback_days=lookback,
                )
                history.append({
                    "lookback_days": lookback,
                    "sentiment": sentiment.overall_sentiment,
                    "label": sentiment.sentiment_label,
                    "confidence": sentiment.confidence,
                    "article_count": sentiment.article_count,
                })

        return {
            "status": "success",
            "asset": asset,
            "history": history,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment history: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/analyze", response_model=Dict[str, Any])
async def analyze_text(
    text: str = Query(..., description="Text to analyze for sentiment"),
):
    """
    Analyze sentiment of custom text.

    Useful for testing the sentiment analyzer with specific headlines.
    """
    try:
        from app.ml.features.sentiment import sentiment_analyzer

        result = sentiment_analyzer.analyze_text(text)

        return {
            "status": "success",
            "text": text,
            "sentiment": result["sentiment"],
            "positive": result["positive"],
            "negative": result["negative"],
            "neutral": result["neutral"],
            "method": result["method"],
        }

    except Exception as e:
        logger.error(f"Error analyzing text: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
