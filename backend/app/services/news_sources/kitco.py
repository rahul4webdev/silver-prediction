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

    # Kitco RSS feeds
    RSS_FEEDS = {
        "latest": "https://www.kitco.com/rss/news.xml",
        "gold": "https://www.kitco.com/rss/gold.xml",
        "silver": "https://www.kitco.com/rss/silver.xml",
        "mining": "https://www.kitco.com/rss/mining.xml",
        "commentary": "https://www.kitco.com/rss/commentary.xml",
    }

    # Alternative news page URLs if RSS fails
    NEWS_PAGES = {
        "silver": "https://www.kitco.com/news/silver/",
        "gold": "https://www.kitco.com/news/gold/",
        "markets": "https://www.kitco.com/news/markets/",
    }

    def __init__(self):
        """Initialize Kitco news fetcher."""
        pass

    async def fetch_rss_feed(self, feed_url: str) -> List[dict]:
        """
        Fetch and parse a Kitco RSS feed.

        Args:
            feed_url: URL of the RSS feed

        Returns:
            List of article dicts
        """
        articles = []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (compatible; SilverPrediction/1.0)",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(feed_url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        logger.warning(f"Kitco RSS returned {response.status}")
                        return articles

                    content = await response.text()

                    # Parse RSS XML (simple regex parsing)
                    items = re.findall(r'<item>(.*?)</item>', content, re.DOTALL)

                    for item in items[:30]:  # Limit to 30 articles
                        title_match = re.search(r'<title>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</title>', item)
                        link_match = re.search(r'<link>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</link>', item)
                        desc_match = re.search(r'<description>(?:<!\[CDATA\[)?(.*?)(?:\]\]>)?</description>', item, re.DOTALL)
                        pub_date_match = re.search(r'<pubDate>(.*?)</pubDate>', item)

                        if title_match and link_match:
                            # Parse publication date
                            pub_date = datetime.now()
                            if pub_date_match:
                                try:
                                    # Format: "Mon, 27 Jan 2025 10:30:00 GMT"
                                    date_str = pub_date_match.group(1).strip()
                                    pub_date = datetime.strptime(
                                        date_str, "%a, %d %b %Y %H:%M:%S %Z"
                                    )
                                except ValueError:
                                    try:
                                        # Try alternative format
                                        pub_date = datetime.strptime(
                                            date_str, "%a, %d %b %Y %H:%M:%S %z"
                                        )
                                    except ValueError:
                                        pass

                            # Clean title and description
                            title = title_match.group(1).strip()
                            title = re.sub(r'<[^>]+>', '', title)  # Remove HTML tags
                            title = title.replace('&amp;', '&').replace('&quot;', '"')
                            title = title.replace('&#39;', "'").replace('&lt;', '<')
                            title = title.replace('&gt;', '>')

                            description = None
                            if desc_match:
                                description = desc_match.group(1).strip()
                                description = re.sub(r'<[^>]+>', '', description)
                                description = description.replace('&amp;', '&')
                                description = description[:500] if len(description) > 500 else description

                            articles.append({
                                "title": title,
                                "description": description,
                                "url": link_match.group(1).strip(),
                                "published_at": pub_date,
                                "source": "Kitco",
                            })

        except asyncio.TimeoutError:
            logger.warning("Kitco RSS request timed out")
        except Exception as e:
            logger.error(f"Error fetching Kitco RSS: {e}")

        return articles

    async def fetch_silver_news(self) -> List[dict]:
        """
        Fetch silver-specific news from Kitco.

        Returns:
            List of silver news articles
        """
        articles = []

        # Try silver RSS feed first
        silver_articles = await self.fetch_rss_feed(self.RSS_FEEDS["silver"])
        articles.extend(silver_articles)

        # Also get from latest feed (may have silver content)
        latest_articles = await self.fetch_rss_feed(self.RSS_FEEDS["latest"])

        # Filter latest for silver-related
        for article in latest_articles:
            text = f"{article['title']} {article.get('description', '')}".lower()
            if any(kw in text for kw in ["silver", "slv", "precious metal", "white metal"]):
                if article not in articles:
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

        # Try gold RSS feed
        gold_articles = await self.fetch_rss_feed(self.RSS_FEEDS["gold"])
        articles.extend(gold_articles)

        # Also get from latest feed
        latest_articles = await self.fetch_rss_feed(self.RSS_FEEDS["latest"])

        # Filter latest for gold-related
        for article in latest_articles:
            text = f"{article['title']} {article.get('description', '')}".lower()
            if any(kw in text for kw in ["gold", "gld", "precious metal", "yellow metal", "bullion"]):
                if article not in articles:
                    articles.append(article)

        logger.info(f"Kitco: fetched {len(articles)} gold articles")
        return articles

    async def fetch_all_precious_metals_news(self) -> List[dict]:
        """
        Fetch all precious metals news from multiple Kitco feeds.

        Returns:
            Combined list of articles from all feeds
        """
        all_articles = []
        seen_urls = set()

        # Fetch from all relevant feeds
        for feed_name in ["silver", "gold", "latest", "commentary"]:
            try:
                articles = await self.fetch_rss_feed(self.RSS_FEEDS[feed_name])

                for article in articles:
                    if article["url"] not in seen_urls:
                        seen_urls.add(article["url"])
                        article["source"] = f"Kitco/{feed_name.capitalize()}"
                        all_articles.append(article)

                # Small delay between requests
                await asyncio.sleep(0.3)

            except Exception as e:
                logger.warning(f"Failed to fetch Kitco {feed_name}: {e}")

        logger.info(f"Kitco: fetched {len(all_articles)} total articles")
        return all_articles

    async def scrape_news_page(self, page_url: str) -> List[dict]:
        """
        Scrape news from a Kitco news page (fallback if RSS fails).

        Args:
            page_url: URL of the news page

        Returns:
            List of article dicts
        """
        articles = []

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(page_url, headers=headers, timeout=15) as response:
                    if response.status != 200:
                        return articles

                    html = await response.text()

                    # Extract article links and titles (basic pattern)
                    # Pattern for Kitco article links
                    pattern = r'<a[^>]+href="(/news/article/[^"]+)"[^>]*>([^<]+)</a>'
                    matches = re.findall(pattern, html, re.IGNORECASE)

                    for link, title in matches[:20]:
                        full_url = urljoin(page_url, link)
                        title = title.strip()

                        if title and len(title) > 10:  # Skip very short titles
                            articles.append({
                                "title": title,
                                "description": None,
                                "url": full_url,
                                "published_at": datetime.now(),
                                "source": "Kitco",
                            })

        except Exception as e:
            logger.error(f"Error scraping Kitco page: {e}")

        return articles
