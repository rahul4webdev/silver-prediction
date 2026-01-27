"""
News sentiment analysis service for silver/gold price prediction.
Fetches financial news and analyzes sentiment to provide features for ML models.

Uses multiple news sources:
1. Google News RSS - Free, no API key needed
2. NewsAPI.org - General financial news (optional API key)
3. Finnhub - Commodities news (free tier API key)
4. Reddit - Social sentiment from r/silverbugs, r/wallstreetsilver
5. Kitco - Precious metals focused news (RSS feeds)
6. Alpha Vantage - Market news with built-in sentiment (free tier API key)
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

import aiohttp
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import new news sources
try:
    from app.services.news_sources.finnhub import FinnhubNewsFetcher
    from app.services.news_sources.reddit import RedditNewsFetcher
    from app.services.news_sources.kitco import KitcoNewsFetcher
    from app.services.news_sources.alpha_vantage import AlphaVantageNewsFetcher
    NEWS_SOURCES_AVAILABLE = True
except ImportError:
    NEWS_SOURCES_AVAILABLE = False
    logger.warning("Additional news sources not available")


class NewsArticle(BaseModel):
    """Represents a news article."""
    title: str
    description: Optional[str] = None
    source: str
    published_at: datetime
    url: str
    sentiment_score: Optional[float] = None  # -1 (bearish) to +1 (bullish)
    relevance_score: Optional[float] = None  # 0 to 1


class SentimentResult(BaseModel):
    """Aggregated sentiment result."""
    overall_sentiment: float  # -1 to +1
    sentiment_label: str  # bearish, neutral, bullish
    confidence: float  # 0 to 1
    article_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    average_relevance: float
    timestamp: datetime
    articles: List[NewsArticle]


class NewsSentimentService:
    """
    Service for fetching and analyzing news sentiment for commodities.

    Features:
    - Multi-source news aggregation
    - Keyword-based relevance filtering
    - Simple lexicon-based sentiment analysis
    - Caching to avoid API rate limits
    """

    # Keywords for filtering relevant articles
    SILVER_KEYWORDS = [
        "silver", "slv", "precious metal", "comex silver", "mcx silver",
        "silver price", "silver futures", "white metal", "industrial metal",
        "silver demand", "silver supply", "silver mining", "silver etf"
    ]

    GOLD_KEYWORDS = [
        "gold", "gld", "precious metal", "comex gold", "mcx gold",
        "gold price", "gold futures", "yellow metal", "gold demand",
        "gold supply", "gold mining", "gold etf", "bullion"
    ]

    # General commodity keywords that affect silver/gold
    MACRO_KEYWORDS = [
        "inflation", "fed", "federal reserve", "interest rate", "dollar",
        "treasury", "bond yield", "risk", "hedge", "safe haven",
        "central bank", "monetary policy", "recession", "economic",
        "geopolitical", "china", "india", "jewelry demand"
    ]

    # Sentiment lexicons for financial context
    BULLISH_WORDS = {
        "surge", "rally", "soar", "jump", "gain", "rise", "climb",
        "bullish", "optimistic", "demand", "buying", "support",
        "breakout", "momentum", "upside", "strength", "positive",
        "record", "high", "beat", "exceed", "growth", "expansion",
        "recovery", "boost", "advance", "upgrade", "outperform"
    }

    BEARISH_WORDS = {
        "drop", "fall", "plunge", "decline", "crash", "bearish",
        "pessimistic", "selling", "pressure", "breakdown", "weak",
        "downside", "negative", "low", "miss", "below", "loss",
        "recession", "contraction", "downgrade", "underperform",
        "concern", "fear", "risk", "uncertainty", "volatile"
    }

    INTENSITY_MODIFIERS = {
        "very": 1.5, "extremely": 2.0, "slightly": 0.5, "somewhat": 0.7,
        "highly": 1.5, "significantly": 1.5, "sharply": 1.8, "strongly": 1.5
    }

    def __init__(
        self,
        newsapi_key: Optional[str] = None,
        finnhub_api_key: Optional[str] = None,
        alpha_vantage_api_key: Optional[str] = None,
        reddit_client_id: Optional[str] = None,
        reddit_client_secret: Optional[str] = None,
        cache_ttl_minutes: int = 15,
    ):
        """
        Initialize the news sentiment service.

        Args:
            newsapi_key: Optional API key for NewsAPI.org
            finnhub_api_key: Optional API key for Finnhub
            alpha_vantage_api_key: Optional API key for Alpha Vantage
            reddit_client_id: Optional Reddit app client ID
            reddit_client_secret: Optional Reddit app client secret
            cache_ttl_minutes: Cache time-to-live in minutes
        """
        self.newsapi_key = newsapi_key
        self.finnhub_api_key = finnhub_api_key
        self.alpha_vantage_api_key = alpha_vantage_api_key
        self.reddit_client_id = reddit_client_id
        self.reddit_client_secret = reddit_client_secret
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, tuple] = {}  # {key: (timestamp, data)}

        # Initialize additional news fetchers
        self._finnhub = None
        self._reddit = None
        self._kitco = None
        self._alpha_vantage = None

        if NEWS_SOURCES_AVAILABLE:
            if finnhub_api_key:
                self._finnhub = FinnhubNewsFetcher(api_key=finnhub_api_key)
                logger.info("Finnhub news source enabled")

            # Reddit works without API key (limited) or with credentials (better)
            self._reddit = RedditNewsFetcher(
                client_id=reddit_client_id,
                client_secret=reddit_client_secret,
            )
            logger.info(f"Reddit news source enabled (OAuth: {bool(reddit_client_id)})")

            # Kitco doesn't need API key
            self._kitco = KitcoNewsFetcher()
            logger.info("Kitco news source enabled")

            if alpha_vantage_api_key:
                self._alpha_vantage = AlphaVantageNewsFetcher(api_key=alpha_vantage_api_key)
                logger.info("Alpha Vantage news source enabled")

    def _get_cache(self, key: str) -> Optional[Any]:
        """Get cached data if not expired."""
        if key in self._cache:
            timestamp, data = self._cache[key]
            if datetime.now() - timestamp < self.cache_ttl:
                return data
            del self._cache[key]
        return None

    def _set_cache(self, key: str, data: Any) -> None:
        """Set cache with current timestamp."""
        self._cache[key] = (datetime.now(), data)

    async def fetch_google_news_rss(
        self,
        query: str,
        language: str = "en",
    ) -> List[NewsArticle]:
        """
        Fetch news from Google News RSS feed (free, no API key needed).

        Args:
            query: Search query
            language: Language code

        Returns:
            List of news articles
        """
        articles = []

        try:
            encoded_query = quote_plus(query)
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl={language}&gl=US&ceid=US:en"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"Google News RSS returned {response.status}")
                        return articles

                    content = await response.text()

                    # Parse XML (simple regex parsing to avoid lxml dependency)
                    items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)

                    for item in items[:20]:  # Limit to 20 articles
                        title_match = re.search(r'<title>(.*?)</title>', item)
                        link_match = re.search(r'<link>(.*?)</link>', item)
                        pub_date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)
                        source_match = re.search(r'<source[^>]*>(.*?)</source>', item)

                        if title_match and link_match:
                            # Parse publication date
                            pub_date = datetime.now()
                            if pub_date_match:
                                try:
                                    # Format: "Tue, 21 Jan 2025 10:30:00 GMT"
                                    date_str = pub_date_match.group(1)
                                    pub_date = datetime.strptime(
                                        date_str, "%a, %d %b %Y %H:%M:%S %Z"
                                    )
                                except ValueError:
                                    pass

                            # Clean title (remove CDATA and HTML entities)
                            title = title_match.group(1)
                            title = re.sub(r'<!\[CDATA\[(.*?)\]\]>', r'\1', title)
                            title = title.replace('&amp;', '&').replace('&quot;', '"')

                            articles.append(NewsArticle(
                                title=title,
                                description=None,
                                source=source_match.group(1) if source_match else "Google News",
                                published_at=pub_date,
                                url=link_match.group(1),
                            ))

        except asyncio.TimeoutError:
            logger.warning("Google News RSS request timed out")
        except Exception as e:
            logger.error(f"Error fetching Google News: {e}")

        return articles

    async def fetch_newsapi(
        self,
        query: str,
        from_date: Optional[datetime] = None,
    ) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI.org (requires API key).

        Args:
            query: Search query
            from_date: Start date for articles

        Returns:
            List of news articles
        """
        if not self.newsapi_key:
            return []

        articles = []

        try:
            if from_date is None:
                from_date = datetime.now() - timedelta(days=7)

            url = "https://newsapi.org/v2/everything"
            params = {
                "q": query,
                "from": from_date.strftime("%Y-%m-%d"),
                "sortBy": "publishedAt",
                "language": "en",
                "apiKey": self.newsapi_key,
                "pageSize": 50,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status != 200:
                        logger.warning(f"NewsAPI returned {response.status}")
                        return articles

                    data = await response.json()

                    for article in data.get("articles", []):
                        pub_date = datetime.now()
                        if article.get("publishedAt"):
                            try:
                                pub_date = datetime.fromisoformat(
                                    article["publishedAt"].replace("Z", "+00:00")
                                )
                            except ValueError:
                                pass

                        articles.append(NewsArticle(
                            title=article.get("title", ""),
                            description=article.get("description"),
                            source=article.get("source", {}).get("name", "Unknown"),
                            published_at=pub_date,
                            url=article.get("url", ""),
                        ))

        except asyncio.TimeoutError:
            logger.warning("NewsAPI request timed out")
        except Exception as e:
            logger.error(f"Error fetching NewsAPI: {e}")

        return articles

    def calculate_relevance(
        self,
        article: NewsArticle,
        asset: str = "silver",
    ) -> float:
        """
        Calculate relevance score for an article.

        Args:
            article: News article
            asset: Asset to check relevance for

        Returns:
            Relevance score 0-1
        """
        text = f"{article.title} {article.description or ''}".lower()

        # Get relevant keywords based on asset
        if asset.lower() == "silver":
            primary_keywords = self.SILVER_KEYWORDS
        elif asset.lower() == "gold":
            primary_keywords = self.GOLD_KEYWORDS
        else:
            primary_keywords = self.SILVER_KEYWORDS + self.GOLD_KEYWORDS

        # Check primary keywords
        primary_matches = sum(1 for kw in primary_keywords if kw in text)

        # Check macro keywords
        macro_matches = sum(1 for kw in self.MACRO_KEYWORDS if kw in text)

        # Calculate relevance score
        # Primary keywords are more important
        primary_score = min(primary_matches / 2, 1.0) * 0.7
        macro_score = min(macro_matches / 3, 1.0) * 0.3

        return primary_score + macro_score

    def analyze_sentiment(self, text: str) -> float:
        """
        Analyze sentiment of text using lexicon-based approach.

        Args:
            text: Text to analyze

        Returns:
            Sentiment score -1 (bearish) to +1 (bullish)
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)

        bullish_score = 0.0
        bearish_score = 0.0

        for i, word in enumerate(words):
            # Check for intensity modifiers
            modifier = 1.0
            if i > 0:
                prev_word = words[i - 1]
                modifier = self.INTENSITY_MODIFIERS.get(prev_word, 1.0)

            if word in self.BULLISH_WORDS:
                bullish_score += modifier
            elif word in self.BEARISH_WORDS:
                bearish_score += modifier

        total = bullish_score + bearish_score
        if total == 0:
            return 0.0

        # Normalize to -1 to +1 range
        sentiment = (bullish_score - bearish_score) / total
        return max(-1.0, min(1.0, sentiment))

    async def get_sentiment(
        self,
        asset: str = "silver",
        lookback_days: int = 3,
    ) -> SentimentResult:
        """
        Get aggregated sentiment for an asset.

        Fetches from multiple sources:
        1. Google News RSS (always available)
        2. NewsAPI (if API key configured)
        3. Finnhub (if API key configured)
        4. Reddit (always available, better with credentials)
        5. Kitco (always available, RSS feeds)
        6. Alpha Vantage (if API key configured)

        Args:
            asset: Asset to analyze (silver, gold)
            lookback_days: Days of news to analyze

        Returns:
            Aggregated sentiment result
        """
        cache_key = f"sentiment_{asset}_{lookback_days}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        # Build search query
        if asset.lower() == "silver":
            queries = ["silver price", "silver futures", "precious metals silver"]
        elif asset.lower() == "gold":
            queries = ["gold price", "gold futures", "precious metals gold"]
        else:
            queries = [f"{asset} price", f"{asset} commodity"]

        # Fetch articles from multiple sources concurrently
        all_articles = []
        source_stats = {"google": 0, "newsapi": 0, "finnhub": 0, "reddit": 0, "kitco": 0, "alphavantage": 0}

        # Create tasks for all sources
        tasks = []

        # Google News RSS (always)
        for query in queries:
            tasks.append(("google", self.fetch_google_news_rss(query)))

        # NewsAPI if key available
        if self.newsapi_key:
            tasks.append(("newsapi", self.fetch_newsapi(
                queries[0],
                from_date=datetime.now() - timedelta(days=lookback_days),
            )))

        # Finnhub if available
        if self._finnhub:
            tasks.append(("finnhub", self._finnhub.fetch_news()))

        # Reddit (works without API key)
        if self._reddit:
            if asset.lower() == "silver":
                tasks.append(("reddit", self._reddit.fetch_silver_posts(limit_per_sub=15)))
            else:
                tasks.append(("reddit", self._reddit.fetch_gold_posts(limit_per_sub=15)))

        # Kitco (no API key needed)
        if self._kitco:
            if asset.lower() == "silver":
                tasks.append(("kitco", self._kitco.fetch_silver_news()))
            else:
                tasks.append(("kitco", self._kitco.fetch_gold_news()))

        # Alpha Vantage if available
        if self._alpha_vantage:
            if asset.lower() == "silver":
                tasks.append(("alphavantage", self._alpha_vantage.fetch_silver_news()))
            else:
                tasks.append(("alphavantage", self._alpha_vantage.fetch_gold_news()))

        # Execute all tasks concurrently
        for source, task in tasks:
            try:
                articles = await task
                if articles:
                    # Convert to NewsArticle format
                    for article_data in articles:
                        if isinstance(article_data, dict):
                            article = NewsArticle(
                                title=article_data.get("title", ""),
                                description=article_data.get("description"),
                                source=article_data.get("source", source),
                                published_at=article_data.get("published_at", datetime.now()),
                                url=article_data.get("url", ""),
                                # Use pre-computed sentiment from Alpha Vantage if available
                                sentiment_score=article_data.get("av_sentiment_score"),
                            )
                            all_articles.append(article)
                            source_stats[source] += 1
                        elif isinstance(article_data, NewsArticle):
                            all_articles.append(article_data)
                            source_stats[source] += 1
            except Exception as e:
                logger.warning(f"Failed to fetch from {source}: {e}")

        # Filter by date
        cutoff = datetime.now() - timedelta(days=lookback_days)
        recent_articles = [a for a in all_articles if a.published_at >= cutoff]

        # Remove duplicates by title similarity
        unique_articles = []
        seen_titles = set()
        for article in recent_articles:
            # Normalize title for comparison
            normalized = re.sub(r'[^\w\s]', '', article.title.lower())[:50]
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_articles.append(article)

        # Calculate relevance and sentiment for each article
        analyzed_articles = []
        for article in unique_articles:
            relevance = self.calculate_relevance(article, asset)

            # Only include relevant articles
            if relevance >= 0.3:
                text = f"{article.title} {article.description or ''}"
                sentiment = self.analyze_sentiment(text)

                article.relevance_score = relevance
                article.sentiment_score = sentiment
                analyzed_articles.append(article)

        # Sort by relevance
        analyzed_articles.sort(key=lambda x: x.relevance_score or 0, reverse=True)

        # Calculate aggregate sentiment
        if not analyzed_articles:
            result = SentimentResult(
                overall_sentiment=0.0,
                sentiment_label="neutral",
                confidence=0.0,
                article_count=0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                average_relevance=0.0,
                timestamp=datetime.now(),
                articles=[],
            )
        else:
            # Weight sentiment by relevance
            total_weight = sum(a.relevance_score or 1 for a in analyzed_articles)
            weighted_sentiment = sum(
                (a.sentiment_score or 0) * (a.relevance_score or 1)
                for a in analyzed_articles
            ) / total_weight

            # Count sentiment categories
            bullish = sum(1 for a in analyzed_articles if (a.sentiment_score or 0) > 0.1)
            bearish = sum(1 for a in analyzed_articles if (a.sentiment_score or 0) < -0.1)
            neutral = len(analyzed_articles) - bullish - bearish

            # Determine label
            if weighted_sentiment > 0.15:
                label = "bullish"
            elif weighted_sentiment < -0.15:
                label = "bearish"
            else:
                label = "neutral"

            # Confidence based on article count and sentiment consistency
            consistency = abs(bullish - bearish) / max(len(analyzed_articles), 1)
            article_factor = min(len(analyzed_articles) / 10, 1.0)
            confidence = consistency * 0.7 + article_factor * 0.3

            avg_relevance = sum(a.relevance_score or 0 for a in analyzed_articles) / len(analyzed_articles)

            result = SentimentResult(
                overall_sentiment=weighted_sentiment,
                sentiment_label=label,
                confidence=confidence,
                article_count=len(analyzed_articles),
                bullish_count=bullish,
                bearish_count=bearish,
                neutral_count=neutral,
                average_relevance=avg_relevance,
                timestamp=datetime.now(),
                articles=analyzed_articles[:10],  # Return top 10 articles
            )

        # Log source statistics
        active_sources = {k: v for k, v in source_stats.items() if v > 0}
        logger.info(
            f"Sentiment for {asset}: {result.sentiment_label} "
            f"({result.overall_sentiment:.3f}), "
            f"{result.article_count} articles from {len(active_sources)} sources: {active_sources}"
        )

        self._set_cache(cache_key, result)
        return result

    def sentiment_to_features(self, sentiment: SentimentResult) -> Dict[str, float]:
        """
        Convert sentiment result to features for ML models.

        Args:
            sentiment: Sentiment result

        Returns:
            Dict of feature values
        """
        return {
            "news_sentiment": sentiment.overall_sentiment,
            "news_confidence": sentiment.confidence,
            "news_bullish_ratio": sentiment.bullish_count / max(sentiment.article_count, 1),
            "news_bearish_ratio": sentiment.bearish_count / max(sentiment.article_count, 1),
            "news_article_count": min(sentiment.article_count / 20, 1.0),  # Normalized
            "news_avg_relevance": sentiment.average_relevance,
        }

    async def save_articles_to_db(
        self,
        db,  # AsyncSession
        articles: List[NewsArticle],
        asset: str = "silver",
    ) -> int:
        """
        Save news articles to the database.

        Args:
            db: Database session
            articles: List of NewsArticle objects
            asset: Asset the articles relate to

        Returns:
            Number of articles saved (new ones only)
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from app.models.sentiment import NewsArticle as NewsArticleModel

        if not articles:
            return 0

        saved_count = 0

        for article in articles:
            try:
                # Determine sentiment label
                if article.sentiment_score and article.sentiment_score > 0.15:
                    label = "bullish"
                elif article.sentiment_score and article.sentiment_score < -0.15:
                    label = "bearish"
                else:
                    label = "neutral"

                record = {
                    "url": article.url[:500],  # Truncate if needed
                    "title": article.title[:500],
                    "description": article.description[:2000] if article.description else None,
                    "source": article.source[:100],
                    "published_at": article.published_at,
                    "fetched_at": datetime.now(),
                    "asset": asset,
                    "relevance_score": article.relevance_score or 0.0,
                    "sentiment_score": article.sentiment_score or 0.0,
                    "sentiment_label": label,
                    "is_processed": True,
                }

                stmt = pg_insert(NewsArticleModel).values(record)
                stmt = stmt.on_conflict_do_nothing(index_elements=["url"])

                result = await db.execute(stmt)
                if result.rowcount > 0:
                    saved_count += 1

            except Exception as e:
                logger.warning(f"Failed to save article {article.url}: {e}")

        await db.commit()
        return saved_count

    async def save_sentiment_snapshot(
        self,
        db,  # AsyncSession
        sentiment: SentimentResult,
        asset: str = "silver",
    ) -> bool:
        """
        Save a sentiment snapshot to the database.

        Args:
            db: Database session
            sentiment: SentimentResult to save
            asset: Asset the sentiment relates to

        Returns:
            True if saved successfully
        """
        from sqlalchemy import select, func
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        from app.models.sentiment import SentimentSnapshot

        try:
            # Calculate 7-day rolling average from previous snapshots
            seven_days_ago = datetime.now() - timedelta(days=7)
            avg_query = await db.execute(
                select(func.avg(SentimentSnapshot.overall_sentiment))
                .where(
                    SentimentSnapshot.asset == asset,
                    SentimentSnapshot.timestamp >= seven_days_ago,
                )
            )
            sentiment_7d_avg = avg_query.scalar()

            # Calculate momentum (current - 7d average)
            sentiment_momentum = None
            if sentiment_7d_avg is not None:
                sentiment_momentum = sentiment.overall_sentiment - sentiment_7d_avg

            # Count sources - track all sources now
            google_count = sum(
                1 for a in sentiment.articles if "Google" in a.source
            )
            # Count other sources (Finnhub, Reddit, Kitco, Alpha Vantage counted in newsapi_count for now)
            other_count = sum(
                1 for a in sentiment.articles
                if any(src in a.source for src in ["Finnhub", "Reddit", "Kitco", "AlphaVantage"])
            )
            newsapi_count = sentiment.article_count - google_count - other_count

            record = {
                "asset": asset,
                "timestamp": sentiment.timestamp,
                "overall_sentiment": sentiment.overall_sentiment,
                "sentiment_label": sentiment.sentiment_label,
                "confidence": sentiment.confidence,
                "article_count": sentiment.article_count,
                "bullish_count": sentiment.bullish_count,
                "bearish_count": sentiment.bearish_count,
                "neutral_count": sentiment.neutral_count,
                "avg_relevance": sentiment.average_relevance,
                "sentiment_7d_avg": sentiment_7d_avg,
                "sentiment_momentum": sentiment_momentum,
                "google_news_count": google_count,
                "newsapi_count": newsapi_count,
                "lookback_days": 3,
            }

            stmt = pg_insert(SentimentSnapshot).values(record)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_sentiment_snapshot_asset_time",
                set_={
                    "overall_sentiment": stmt.excluded.overall_sentiment,
                    "sentiment_label": stmt.excluded.sentiment_label,
                    "confidence": stmt.excluded.confidence,
                    "article_count": stmt.excluded.article_count,
                    "bullish_count": stmt.excluded.bullish_count,
                    "bearish_count": stmt.excluded.bearish_count,
                    "neutral_count": stmt.excluded.neutral_count,
                    "avg_relevance": stmt.excluded.avg_relevance,
                    "sentiment_7d_avg": stmt.excluded.sentiment_7d_avg,
                    "sentiment_momentum": stmt.excluded.sentiment_momentum,
                },
            )

            await db.execute(stmt)
            await db.commit()

            logger.info(
                f"Saved sentiment snapshot for {asset}: "
                f"{sentiment.sentiment_label} ({sentiment.overall_sentiment:.3f}), "
                f"{sentiment.article_count} articles"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to save sentiment snapshot: {e}")
            await db.rollback()
            return False

    async def fetch_and_save_sentiment(
        self,
        db,  # AsyncSession
        asset: str = "silver",
        lookback_days: int = 3,
    ) -> Dict[str, Any]:
        """
        Fetch current sentiment and save to database.

        This is the main method called by the scheduler.

        Args:
            db: Database session
            asset: Asset to analyze
            lookback_days: Days of news to analyze

        Returns:
            Dict with results
        """
        try:
            # Fetch sentiment (uses cache if available)
            sentiment = await self.get_sentiment(asset=asset, lookback_days=lookback_days)

            # Save individual articles
            articles_saved = await self.save_articles_to_db(db, sentiment.articles, asset)

            # Save aggregated snapshot
            snapshot_saved = await self.save_sentiment_snapshot(db, sentiment, asset)

            return {
                "status": "success",
                "asset": asset,
                "sentiment_label": sentiment.sentiment_label,
                "overall_sentiment": sentiment.overall_sentiment,
                "article_count": sentiment.article_count,
                "articles_saved": articles_saved,
                "snapshot_saved": snapshot_saved,
            }

        except Exception as e:
            logger.error(f"Failed to fetch and save sentiment: {e}")
            return {"status": "error", "error": str(e)}

    async def get_historical_sentiment_features(
        self,
        db,  # AsyncSession
        asset: str,
        timestamp: datetime,
        lookback_hours: int = 24,
    ) -> Optional[Dict[str, float]]:
        """
        Get sentiment features from historical snapshots for a given timestamp.

        Used to add sentiment features to historical training data.

        Args:
            db: Database session
            asset: Asset to get sentiment for
            timestamp: Point in time to get sentiment for
            lookback_hours: How many hours back to look for a snapshot

        Returns:
            Dict of sentiment features or None if not available
        """
        from sqlalchemy import select
        from app.models.sentiment import SentimentSnapshot

        try:
            cutoff = timestamp - timedelta(hours=lookback_hours)

            result = await db.execute(
                select(SentimentSnapshot)
                .where(
                    SentimentSnapshot.asset == asset,
                    SentimentSnapshot.timestamp <= timestamp,
                    SentimentSnapshot.timestamp >= cutoff,
                )
                .order_by(SentimentSnapshot.timestamp.desc())
                .limit(1)
            )

            snapshot = result.scalar_one_or_none()

            if snapshot:
                return snapshot.to_features()

            return None

        except Exception as e:
            logger.warning(f"Failed to get historical sentiment: {e}")
            return None


# Create singleton with config
def _create_service() -> NewsSentimentService:
    """Create service with API keys from config."""
    try:
        from app.core.config import settings

        # Log which sources are available
        sources = []
        if settings.newsapi_key:
            sources.append("NewsAPI")
        if getattr(settings, "finnhub_api_key", None):
            sources.append("Finnhub")
        if getattr(settings, "reddit_client_id", None):
            sources.append("Reddit(OAuth)")
        else:
            sources.append("Reddit(public)")
        sources.append("Kitco")  # Always available
        sources.append("Google News")  # Always available
        if settings.alpha_vantage_api_key:
            sources.append("Alpha Vantage")

        logger.info(f"News sentiment sources: {', '.join(sources)}")

        return NewsSentimentService(
            newsapi_key=settings.newsapi_key,
            finnhub_api_key=getattr(settings, "finnhub_api_key", None),
            alpha_vantage_api_key=settings.alpha_vantage_api_key,
            reddit_client_id=getattr(settings, "reddit_client_id", None),
            reddit_client_secret=getattr(settings, "reddit_client_secret", None),
        )
    except Exception as e:
        logger.warning(f"Failed to load settings for news service: {e}")
        return NewsSentimentService()


# Singleton instance
news_sentiment_service = _create_service()
