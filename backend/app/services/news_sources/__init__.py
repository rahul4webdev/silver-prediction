"""
News sources for sentiment analysis.

This module provides multiple news fetchers:
1. Finnhub - Free commodities news API
2. Reddit - Social sentiment from precious metals communities
3. Kitco - Precious metals focused news
4. Alpha Vantage - Market news with sentiment
"""

from app.services.news_sources.finnhub import FinnhubNewsFetcher
from app.services.news_sources.reddit import RedditNewsFetcher
from app.services.news_sources.kitco import KitcoNewsFetcher
from app.services.news_sources.alpha_vantage import AlphaVantageNewsFetcher

__all__ = [
    "FinnhubNewsFetcher",
    "RedditNewsFetcher",
    "KitcoNewsFetcher",
    "AlphaVantageNewsFetcher",
]
