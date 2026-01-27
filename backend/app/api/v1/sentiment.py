"""
News sentiment API endpoints.
Provides access to news sentiment analysis for silver and gold.

Now includes endpoints for:
- Current sentiment (live fetch)
- Stored sentiment history (from database)
- Stored articles (from database)
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.services.news_sentiment import news_sentiment_service, SentimentResult, NewsArticle
from app.ml.features.sentiment import sentiment_feature_engine
from app.models.database import get_db
from app.models.sentiment import SentimentSnapshot, NewsArticle as NewsArticleModel

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


@router.get("/snapshots", response_model=Dict[str, Any])
async def get_sentiment_snapshots(
    asset: str = Query(default="silver", description="Asset to get snapshots for"),
    days: int = Query(default=7, ge=1, le=30, description="Days of history"),
    limit: int = Query(default=100, le=500, description="Maximum snapshots to return"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get stored sentiment snapshots from the database.

    Returns time-series sentiment data for charting and analysis.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = (
            select(SentimentSnapshot)
            .where(
                SentimentSnapshot.asset == asset,
                SentimentSnapshot.timestamp >= cutoff,
            )
            .order_by(desc(SentimentSnapshot.timestamp))
            .limit(limit)
        )

        result = await db.execute(query)
        snapshots = result.scalars().all()

        return {
            "status": "success",
            "asset": asset,
            "days": days,
            "count": len(snapshots),
            "snapshots": [s.to_dict() for s in snapshots],
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment snapshots: {e}")
        return {
            "status": "error",
            "message": str(e),
            "snapshots": [],
        }


@router.get("/articles", response_model=Dict[str, Any])
async def get_stored_articles(
    asset: str = Query(default="silver", description="Asset to get articles for"),
    days: int = Query(default=3, ge=1, le=14, description="Days of history"),
    limit: int = Query(default=50, le=200, description="Maximum articles to return"),
    min_relevance: float = Query(default=0.3, ge=0, le=1, description="Minimum relevance score"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get stored news articles from the database.

    Returns articles that have been analyzed and stored.
    """
    try:
        cutoff = datetime.utcnow() - timedelta(days=days)

        query = (
            select(NewsArticleModel)
            .where(
                NewsArticleModel.asset == asset,
                NewsArticleModel.published_at >= cutoff,
                NewsArticleModel.relevance_score >= min_relevance,
            )
            .order_by(desc(NewsArticleModel.published_at))
            .limit(limit)
        )

        result = await db.execute(query)
        articles = result.scalars().all()

        return {
            "status": "success",
            "asset": asset,
            "days": days,
            "count": len(articles),
            "articles": [a.to_dict() for a in articles],
        }

    except Exception as e:
        logger.error(f"Error fetching stored articles: {e}")
        return {
            "status": "error",
            "message": str(e),
            "articles": [],
        }


@router.post("/sync", response_model=Dict[str, Any])
async def trigger_sentiment_sync(
    asset: str = Query(default="silver", description="Asset to sync"),
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger sentiment sync.

    Fetches latest news, analyzes sentiment, and stores in database.
    """
    try:
        result = await news_sentiment_service.fetch_and_save_sentiment(
            db, asset=asset, lookback_days=3
        )
        return result

    except Exception as e:
        logger.error(f"Error syncing sentiment: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/stats", response_model=Dict[str, Any])
async def get_sentiment_stats(
    asset: str = Query(default="silver", description="Asset to get stats for"),
    db: AsyncSession = Depends(get_db),
):
    """
    Get sentiment storage statistics.

    Shows how much sentiment data has been collected.
    """
    try:
        # Count snapshots
        snapshot_count = await db.execute(
            select(func.count(SentimentSnapshot.id))
            .where(SentimentSnapshot.asset == asset)
        )
        total_snapshots = snapshot_count.scalar()

        # Get date range for snapshots
        snapshot_range = await db.execute(
            select(
                func.min(SentimentSnapshot.timestamp),
                func.max(SentimentSnapshot.timestamp),
            ).where(SentimentSnapshot.asset == asset)
        )
        min_ts, max_ts = snapshot_range.first()

        # Count articles
        article_count = await db.execute(
            select(func.count(NewsArticleModel.id))
            .where(NewsArticleModel.asset == asset)
        )
        total_articles = article_count.scalar()

        # Get latest snapshot
        latest = await db.execute(
            select(SentimentSnapshot)
            .where(SentimentSnapshot.asset == asset)
            .order_by(desc(SentimentSnapshot.timestamp))
            .limit(1)
        )
        latest_snapshot = latest.scalar_one_or_none()

        return {
            "status": "success",
            "asset": asset,
            "snapshots": {
                "total": total_snapshots,
                "oldest": min_ts.isoformat() if min_ts else None,
                "newest": max_ts.isoformat() if max_ts else None,
            },
            "articles": {
                "total": total_articles,
            },
            "latest_sentiment": latest_snapshot.to_dict() if latest_snapshot else None,
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment stats: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
