"""
News sentiment data models for storing and analyzing sentiment over time.

Tables:
1. news_articles: Individual articles fetched from news sources
2. sentiment_snapshots: Aggregated sentiment at regular intervals
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class NewsArticle(Base):
    """
    Stores individual news articles fetched from various sources.

    Used for:
    - Historical analysis of news impact on prices
    - Training sentiment models
    - Backtesting sentiment-based predictions
    """

    __tablename__ = "news_articles"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Article identification
    url: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)

    # Timestamps
    published_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    # Asset relevance
    asset: Mapped[str] = mapped_column(
        String(20), nullable=False, index=True,
        comment="Asset this article relates to (silver, gold)"
    )
    relevance_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="0-1 score indicating how relevant this article is to the asset"
    )

    # Sentiment analysis
    sentiment_score: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="-1 (bearish) to +1 (bullish)"
    )
    sentiment_label: Mapped[str] = mapped_column(
        String(20), nullable=False, default="neutral",
        comment="bearish, neutral, or bullish"
    )

    # Keyword matches for analysis
    keyword_matches: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True,
        comment="JSON list of matched keywords"
    )

    # Processing status
    is_processed: Mapped[bool] = mapped_column(Boolean, default=True)

    __table_args__ = (
        Index("idx_news_articles_asset_published", "asset", "published_at"),
        Index("idx_news_articles_sentiment", "asset", "sentiment_score"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "description": self.description,
            "source": self.source,
            "published_at": self.published_at.isoformat(),
            "fetched_at": self.fetched_at.isoformat(),
            "asset": self.asset,
            "relevance_score": self.relevance_score,
            "sentiment_score": self.sentiment_score,
            "sentiment_label": self.sentiment_label,
        }


class SentimentSnapshot(Base):
    """
    Stores aggregated sentiment at regular intervals.

    Snapshots are taken every 30 minutes and include:
    - Overall sentiment score
    - Article counts and breakdown
    - Rolling averages for trend analysis

    Used for:
    - Adding sentiment features to ML training
    - Historical sentiment trend analysis
    - Correlation studies with price movements
    """

    __tablename__ = "sentiment_snapshots"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )

    # Aggregated sentiment
    overall_sentiment: Mapped[float] = mapped_column(
        Float, nullable=False,
        comment="-1 (bearish) to +1 (bullish)"
    )
    sentiment_label: Mapped[str] = mapped_column(
        String(20), nullable=False,
        comment="bearish, neutral, or bullish"
    )
    confidence: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="0-1 confidence in the sentiment assessment"
    )

    # Article breakdown
    article_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bullish_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bearish_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    neutral_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Rolling metrics (for trend analysis)
    avg_relevance: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0,
        comment="Average relevance score of articles"
    )
    sentiment_7d_avg: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="7-day rolling average sentiment"
    )
    sentiment_momentum: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True,
        comment="Current sentiment minus 7d average (momentum)"
    )

    # Source breakdown
    google_news_count: Mapped[int] = mapped_column(Integer, default=0)
    newsapi_count: Mapped[int] = mapped_column(Integer, default=0)

    # Metadata
    lookback_days: Mapped[int] = mapped_column(
        Integer, nullable=False, default=3,
        comment="Number of days of news analyzed"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint("asset", "timestamp", name="uq_sentiment_snapshot_asset_time"),
        Index("idx_sentiment_snapshot_lookup", "asset", "timestamp"),
    )

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "asset": self.asset,
            "timestamp": self.timestamp.isoformat(),
            "overall_sentiment": self.overall_sentiment,
            "sentiment_label": self.sentiment_label,
            "confidence": self.confidence,
            "article_count": self.article_count,
            "bullish_count": self.bullish_count,
            "bearish_count": self.bearish_count,
            "neutral_count": self.neutral_count,
            "avg_relevance": self.avg_relevance,
            "sentiment_7d_avg": self.sentiment_7d_avg,
            "sentiment_momentum": self.sentiment_momentum,
        }

    def to_features(self) -> dict:
        """
        Convert to feature dictionary for ML training.

        Returns dict with normalized features ready for model input.
        """
        return {
            "news_sentiment": self.overall_sentiment,
            "news_confidence": self.confidence,
            "news_bullish_ratio": self.bullish_count / max(self.article_count, 1),
            "news_bearish_ratio": self.bearish_count / max(self.article_count, 1),
            "news_article_count": min(self.article_count / 20, 1.0),  # Normalized
            "news_avg_relevance": self.avg_relevance,
            "news_sentiment_7d_avg": self.sentiment_7d_avg or 0.0,
            "news_sentiment_momentum": self.sentiment_momentum or 0.0,
        }
