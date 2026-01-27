"""
Alpha Vantage news sentiment fetcher.

Alpha Vantage provides:
- News sentiment API with built-in sentiment scores
- Free tier: 25 requests/day
- Topics include: commodity_markets, financial_markets, economy_fiscal, etc.

API Key: Free at https://www.alphavantage.co/support/#api-key
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class AlphaVantageNewsFetcher:
    """
    Fetches news with sentiment from Alpha Vantage.

    Unique feature: Provides sentiment scores directly from API.
    Free tier: 25 requests/day.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Topics relevant to precious metals
    COMMODITY_TOPICS = [
        "commodity_markets",
        "financial_markets",
        "economy_monetary",
        "economy_fiscal",
    ]

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Alpha Vantage fetcher.

        Args:
            api_key: Alpha Vantage API key (free at alphavantage.co)
        """
        self.api_key = api_key

    async def fetch_news_sentiment(
        self,
        tickers: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        time_from: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[dict]:
        """
        Fetch news with sentiment analysis.

        Args:
            tickers: List of tickers (e.g., ["SLV", "GLD", "SI=F"])
            topics: List of topics (e.g., ["commodity_markets"])
            time_from: Start time for news
            limit: Max articles to return

        Returns:
            List of article dicts with sentiment scores
        """
        if not self.api_key:
            logger.debug("Alpha Vantage API key not configured, skipping")
            return []

        articles = []

        try:
            params = {
                "function": "NEWS_SENTIMENT",
                "apikey": self.api_key,
                "limit": min(limit, 200),  # API max is 200
            }

            # Add tickers if specified
            if tickers:
                params["tickers"] = ",".join(tickers)

            # Add topics if specified
            if topics:
                params["topics"] = ",".join(topics)
            elif not tickers:
                # Default to commodity topics
                params["topics"] = ",".join(self.COMMODITY_TOPICS)

            # Add time filter
            if time_from:
                params["time_from"] = time_from.strftime("%Y%m%dT%H%M")

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=20
                ) as response:
                    if response.status == 429:
                        logger.warning("Alpha Vantage rate limit reached")
                        return articles

                    if response.status != 200:
                        logger.warning(f"Alpha Vantage returned {response.status}")
                        return articles

                    data = await response.json()

                    # Check for API limit message
                    if "Note" in data or "Information" in data:
                        logger.warning(f"Alpha Vantage limit: {data.get('Note') or data.get('Information')}")
                        return articles

                    feed = data.get("feed", [])

                    for item in feed:
                        # Parse timestamp
                        time_str = item.get("time_published", "")
                        pub_date = datetime.now()
                        if time_str:
                            try:
                                # Format: "20250127T103000"
                                pub_date = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
                            except ValueError:
                                pass

                        # Get overall sentiment
                        overall_sentiment = item.get("overall_sentiment_score", 0)
                        sentiment_label = item.get("overall_sentiment_label", "Neutral")

                        # Get ticker-specific sentiment if available
                        ticker_sentiment = None
                        for ts in item.get("ticker_sentiment", []):
                            ticker = ts.get("ticker", "")
                            if ticker in ["SLV", "GLD", "SI=F", "GC=F"]:
                                ticker_sentiment = {
                                    "ticker": ticker,
                                    "score": float(ts.get("ticker_sentiment_score", 0)),
                                    "label": ts.get("ticker_sentiment_label", "Neutral"),
                                    "relevance": float(ts.get("relevance_score", 0)),
                                }
                                break

                        # Filter for precious metals relevance
                        title = item.get("title", "").lower()
                        summary = item.get("summary", "").lower()
                        text = f"{title} {summary}"

                        keywords = [
                            "silver", "gold", "precious metal", "commodity",
                            "bullion", "metal", "slv", "gld", "mining",
                            "inflation", "fed", "dollar"
                        ]

                        is_relevant = (
                            ticker_sentiment is not None or
                            any(kw in text for kw in keywords)
                        )

                        if is_relevant:
                            articles.append({
                                "title": item.get("title", ""),
                                "description": item.get("summary"),
                                "url": item.get("url", ""),
                                "published_at": pub_date,
                                "source": f"AlphaVantage/{item.get('source', 'Unknown')}",
                                # Alpha Vantage specific sentiment data
                                "av_sentiment_score": overall_sentiment,
                                "av_sentiment_label": sentiment_label,
                                "ticker_sentiment": ticker_sentiment,
                            })

        except asyncio.TimeoutError:
            logger.warning("Alpha Vantage request timed out")
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage news: {e}")

        logger.info(f"Alpha Vantage: fetched {len(articles)} relevant articles")
        return articles

    async def fetch_silver_news(self) -> List[dict]:
        """
        Fetch silver-specific news with sentiment.

        Returns:
            List of silver news articles with sentiment
        """
        # SLV = iShares Silver Trust ETF
        # SI=F = Silver futures (COMEX)
        return await self.fetch_news_sentiment(
            tickers=["SLV", "SI=F"],
            topics=["commodity_markets"],
            time_from=datetime.now() - timedelta(days=7),
        )

    async def fetch_gold_news(self) -> List[dict]:
        """
        Fetch gold-specific news with sentiment.

        Returns:
            List of gold news articles with sentiment
        """
        # GLD = SPDR Gold Trust ETF
        # GC=F = Gold futures (COMEX)
        return await self.fetch_news_sentiment(
            tickers=["GLD", "GC=F"],
            topics=["commodity_markets"],
            time_from=datetime.now() - timedelta(days=7),
        )

    async def fetch_commodity_news(self) -> List[dict]:
        """
        Fetch general commodity market news.

        Returns:
            List of commodity news articles
        """
        return await self.fetch_news_sentiment(
            topics=self.COMMODITY_TOPICS,
            time_from=datetime.now() - timedelta(days=3),
        )

    def convert_av_sentiment(self, av_score: float) -> float:
        """
        Convert Alpha Vantage sentiment score to our -1 to +1 scale.

        Alpha Vantage uses:
        - Bearish: x <= -0.35
        - Somewhat-Bearish: -0.35 < x <= -0.15
        - Neutral: -0.15 < x < 0.15
        - Somewhat-Bullish: 0.15 <= x < 0.35
        - Bullish: x >= 0.35

        Args:
            av_score: Alpha Vantage sentiment score (-1 to 1)

        Returns:
            Normalized sentiment score (-1 to 1)
        """
        # Alpha Vantage scores are already in -1 to 1 range
        # but tend to be conservative, so we can amplify slightly
        return max(-1.0, min(1.0, av_score * 1.2))
