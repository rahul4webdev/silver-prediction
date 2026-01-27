"""
Finnhub news fetcher for commodities news.

Finnhub provides free tier with:
- 60 API calls/minute
- General news and company news
- Market news by category

API Key: Free at https://finnhub.io/register
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class FinnhubNewsFetcher:
    """
    Fetches news from Finnhub API.

    Free tier: 60 calls/minute, general market news.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Finnhub fetcher.

        Args:
            api_key: Finnhub API key (free at finnhub.io)
        """
        self.api_key = api_key

    async def fetch_news(
        self,
        category: str = "general",
        min_id: int = 0,
    ) -> List[dict]:
        """
        Fetch general market news.

        Args:
            category: News category (general, forex, crypto, merger)
            min_id: Minimum news ID for pagination

        Returns:
            List of article dicts with title, description, url, published_at, source
        """
        if not self.api_key:
            logger.debug("Finnhub API key not configured, skipping")
            return []

        articles = []

        try:
            url = f"{self.BASE_URL}/news"
            params = {
                "category": category,
                "token": self.api_key,
            }
            if min_id > 0:
                params["minId"] = min_id

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 429:
                        logger.warning("Finnhub rate limit reached")
                        return articles

                    if response.status != 200:
                        logger.warning(f"Finnhub returned {response.status}")
                        return articles

                    data = await response.json()

                    for item in data:
                        # Filter for commodities/precious metals news
                        headline = item.get("headline", "").lower()
                        summary = item.get("summary", "").lower()
                        text = f"{headline} {summary}"

                        # Check if relevant to precious metals
                        keywords = [
                            "silver", "gold", "precious metal", "commodity",
                            "inflation", "fed", "interest rate", "dollar",
                            "comex", "bullion", "metals"
                        ]

                        if any(kw in text for kw in keywords):
                            pub_date = datetime.fromtimestamp(item.get("datetime", 0))

                            articles.append({
                                "title": item.get("headline", ""),
                                "description": item.get("summary"),
                                "url": item.get("url", ""),
                                "published_at": pub_date,
                                "source": f"Finnhub/{item.get('source', 'Unknown')}",
                            })

        except asyncio.TimeoutError:
            logger.warning("Finnhub request timed out")
        except Exception as e:
            logger.error(f"Error fetching Finnhub news: {e}")

        logger.info(f"Finnhub: fetched {len(articles)} relevant articles")
        return articles

    async def fetch_company_news(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None,
    ) -> List[dict]:
        """
        Fetch news for a specific company/symbol.

        Args:
            symbol: Stock symbol (e.g., SLV, GLD for ETFs)
            from_date: Start date
            to_date: End date

        Returns:
            List of article dicts
        """
        if not self.api_key:
            return []

        articles = []

        try:
            if from_date is None:
                from_date = datetime.now() - timedelta(days=7)
            if to_date is None:
                to_date = datetime.now()

            url = f"{self.BASE_URL}/company-news"
            params = {
                "symbol": symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": self.api_key,
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status != 200:
                        return articles

                    data = await response.json()

                    for item in data:
                        pub_date = datetime.fromtimestamp(item.get("datetime", 0))

                        articles.append({
                            "title": item.get("headline", ""),
                            "description": item.get("summary"),
                            "url": item.get("url", ""),
                            "published_at": pub_date,
                            "source": f"Finnhub/{item.get('source', 'Unknown')}",
                        })

        except Exception as e:
            logger.error(f"Error fetching Finnhub company news: {e}")

        return articles
