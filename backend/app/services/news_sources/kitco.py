"""
Kitco News scraper for precious metals news.

Kitco is a leading precious metals information source with:
- Real-time gold/silver prices
- Expert analysis and commentary
- Mining news
- Market updates

No API key required - uses RSS feeds and web scraping.
"""

import asyncio
import logging
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import urljoin

import aiohttp

logger = logging.getLogger(__name__)


class KitcoNewsFetcher:
    """
    Fetches news from Kitco via RSS feeds and web scraping.

    No API key required - Kitco provides public RSS feeds.
    """

    # Kitco news page URLs (RSS feeds no longer available)
    NEWS_PAGES = {
        "silver": "https://www.kitco.com/news/category/commodities/silver",
        "gold": "https://www.kitco.com/news/category/commodities/gold",
        "commodities": "https://www.kitco.com/news/category/commodities",
        "news": "https://www.kitco.com/news/",
        "opinion": "https://www.kitco.com/opinion",
    }

    def __init__(self):
        """Initialize Kitco news fetcher."""
        pass

    async def fetch_news_page(self, page_url: str) -> List[dict]:
        """
        Fetch and parse a Kitco news page.

        Args:
            page_url: URL of the news page

        Returns:
            List of article dicts
        """
        articles = []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(page_url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        logger.warning(f"Kitco page returned {response.status}")
                        return articles

                    html = await response.text()

                    # Extract article URLs from the page
                    # Pattern: /news/article/YYYY-MM-DD/title-slug or /opinion/YYYY-MM-DD/title-slug
                    article_patterns = [
                        r'href="(/news/article/(\d{4}-\d{2}-\d{2})/([^"]+))"',
                        r'href="(/opinion/(\d{4}-\d{2}-\d{2})/([^"]+))"',
                        r'href="(/news/off-the-wire/(\d{4}-\d{2}-\d{2})/([^"]+))"',
                    ]

                    seen_urls = set()

                    for pattern in article_patterns:
                        matches = re.findall(pattern, html)

                        for match in matches[:20]:  # Limit per pattern
                            url_path, date_str, slug = match
                            full_url = f"https://www.kitco.com{url_path}"

                            if full_url in seen_urls:
                                continue
                            seen_urls.add(full_url)

                            # Parse date from URL
                            try:
                                pub_date = datetime.strptime(date_str, "%Y-%m-%d")
                            except ValueError:
                                pub_date = datetime.now()

                            # Convert slug to readable title
                            title = slug.replace("-", " ").title()
                            title = title[:100]  # Limit title length

                            articles.append({
                                "title": title,
                                "description": None,
                                "url": full_url,
                                "published_at": pub_date,
                                "source": "Kitco",
                            })

        except asyncio.TimeoutError:
            logger.warning("Kitco page request timed out")
        except Exception as e:
            logger.error(f"Error fetching Kitco page: {e}")

        return articles

    async def fetch_silver_news(self) -> List[dict]:
        """
        Fetch silver-specific news from Kitco.

        Returns:
            List of silver news articles
        """
        articles = []
        seen_urls = set()

        # Fetch from silver category page
        silver_articles = await self.fetch_news_page(self.NEWS_PAGES["silver"])
        for article in silver_articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                articles.append(article)

        # Also get from main news page and filter for silver
        news_articles = await self.fetch_news_page(self.NEWS_PAGES["news"])
        for article in news_articles:
            if article["url"] in seen_urls:
                continue
            text = article["title"].lower()
            if any(kw in text for kw in ["silver", "slv", "precious metal", "white metal"]):
                seen_urls.add(article["url"])
                articles.append(article)

        logger.info(f"Kitco: fetched {len(articles)} silver articles")
        return articles

    async def fetch_gold_news(self) -> List[dict]:
        """
        Fetch gold-specific news from Kitco.

        Returns:
            List of gold news articles
        """
        articles = []
        seen_urls = set()

        # Fetch from gold category page
        gold_articles = await self.fetch_news_page(self.NEWS_PAGES["gold"])
        for article in gold_articles:
            if article["url"] not in seen_urls:
                seen_urls.add(article["url"])
                articles.append(article)

        # Also get from main news page and filter for gold
        news_articles = await self.fetch_news_page(self.NEWS_PAGES["news"])
        for article in news_articles:
            if article["url"] in seen_urls:
                continue
            text = article["title"].lower()
            if any(kw in text for kw in ["gold", "gld", "precious metal", "yellow metal", "bullion"]):
                seen_urls.add(article["url"])
                articles.append(article)

        logger.info(f"Kitco: fetched {len(articles)} gold articles")
        return articles

    async def fetch_all_precious_metals_news(self) -> List[dict]:
        """
        Fetch all precious metals news from Kitco.

        Returns:
            Combined list of articles
        """
        all_articles = []
        seen_urls = set()

        # Fetch from all pages
        for page_name, page_url in self.NEWS_PAGES.items():
            try:
                articles = await self.fetch_news_page(page_url)

                for article in articles:
                    if article["url"] not in seen_urls:
                        seen_urls.add(article["url"])
                        article["source"] = f"Kitco/{page_name.capitalize()}"
                        all_articles.append(article)

                # Small delay between requests
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.warning(f"Failed to fetch Kitco {page_name}: {e}")

        logger.info(f"Kitco: fetched {len(all_articles)} total articles")
        return all_articles
