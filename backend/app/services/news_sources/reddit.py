"""
Reddit news/sentiment fetcher for precious metals communities.

Subreddits monitored:
- r/silverbugs - Silver stacking community
- r/wallstreetsilver - Silver investing/speculation
- r/gold - Gold discussion
- r/commodities - General commodities

Reddit API:
- With OAuth: 60 requests/minute
- Without OAuth (public JSON): Limited but works for basic fetching
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class RedditNewsFetcher:
    """
    Fetches posts and sentiment from Reddit precious metals communities.

    Can work without API key using public JSON endpoints (rate limited).
    With OAuth credentials, gets higher limits.
    """

    # Subreddits focused on precious metals
    SILVER_SUBREDDITS = ["silverbugs", "wallstreetsilver", "silver"]
    GOLD_SUBREDDITS = ["gold", "goldbugs"]
    GENERAL_SUBREDDITS = ["commodities", "investing", "stocks"]

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        """
        Initialize Reddit fetcher.

        Args:
            client_id: Reddit app client ID (optional)
            client_secret: Reddit app client secret (optional)
        """
        self.client_id = client_id
        self.client_secret = client_secret
        self._access_token = None
        self._token_expires = None

    async def _get_access_token(self) -> Optional[str]:
        """Get OAuth access token if credentials are configured."""
        if not self.client_id or not self.client_secret:
            return None

        # Check if token is still valid
        if self._access_token and self._token_expires:
            if datetime.now() < self._token_expires:
                return self._access_token

        try:
            auth = aiohttp.BasicAuth(self.client_id, self.client_secret)
            data = {
                "grant_type": "client_credentials",
            }
            headers = {"User-Agent": "SilverPrediction/1.0"}

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://www.reddit.com/api/v1/access_token",
                    auth=auth,
                    data=data,
                    headers=headers,
                    timeout=10,
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self._access_token = result.get("access_token")
                        expires_in = result.get("expires_in", 3600)
                        self._token_expires = datetime.now() + timedelta(seconds=expires_in - 60)
                        return self._access_token

        except Exception as e:
            logger.warning(f"Failed to get Reddit OAuth token: {e}")

        return None

    async def fetch_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "hot",
        limit: int = 25,
        time_filter: str = "week",
    ) -> List[dict]:
        """
        Fetch posts from a subreddit.

        Args:
            subreddit: Subreddit name (without r/)
            sort: Sort method (hot, new, top, rising)
            limit: Number of posts to fetch
            time_filter: Time filter for top sort (hour, day, week, month, year, all)

        Returns:
            List of post dicts with title, text, url, published_at, source, score
        """
        posts = []

        try:
            # Try OAuth first, fall back to public JSON
            token = await self._get_access_token()

            if token:
                # OAuth endpoint
                base_url = "https://oauth.reddit.com"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "SilverPrediction/1.0 (by /u/silver_prediction_bot)",
                }
                url = f"{base_url}/r/{subreddit}/{sort}.json"
            else:
                # Use old.reddit.com which is more permissive for public JSON access
                base_url = "https://old.reddit.com"
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible)",
                    "Accept": "application/json",
                }
                url = f"{base_url}/r/{subreddit}/{sort}.json"
            params = {"limit": limit, "t": time_filter}

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=15
                ) as response:
                    if response.status == 429:
                        logger.warning(f"Reddit rate limit for r/{subreddit}")
                        return posts

                    if response.status != 200:
                        logger.warning(f"Reddit r/{subreddit} returned {response.status}")
                        return posts

                    data = await response.json()

                    for child in data.get("data", {}).get("children", []):
                        post = child.get("data", {})

                        # Skip pinned/stickied posts
                        if post.get("stickied"):
                            continue

                        # Get post text (selftext for text posts)
                        text = post.get("selftext", "")
                        if len(text) > 500:
                            text = text[:500] + "..."

                        # Parse timestamp
                        created_utc = post.get("created_utc", 0)
                        pub_date = datetime.fromtimestamp(created_utc)

                        posts.append({
                            "title": post.get("title", ""),
                            "description": text if text else None,
                            "url": f"https://reddit.com{post.get('permalink', '')}",
                            "published_at": pub_date,
                            "source": f"Reddit/r/{subreddit}",
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "upvote_ratio": post.get("upvote_ratio", 0.5),
                        })

        except asyncio.TimeoutError:
            logger.warning(f"Reddit r/{subreddit} request timed out")
        except Exception as e:
            logger.error(f"Error fetching Reddit r/{subreddit}: {e}")

        return posts

    async def fetch_silver_posts(self, limit_per_sub: int = 20) -> List[dict]:
        """
        Fetch posts from silver-focused subreddits.

        Args:
            limit_per_sub: Posts per subreddit

        Returns:
            Combined list of posts from all silver subreddits
        """
        all_posts = []

        for subreddit in self.SILVER_SUBREDDITS:
            posts = await self.fetch_subreddit_posts(
                subreddit, sort="hot", limit=limit_per_sub
            )
            all_posts.extend(posts)

            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)

        logger.info(f"Reddit: fetched {len(all_posts)} silver posts")
        return all_posts

    async def fetch_gold_posts(self, limit_per_sub: int = 20) -> List[dict]:
        """
        Fetch posts from gold-focused subreddits.

        Args:
            limit_per_sub: Posts per subreddit

        Returns:
            Combined list of posts from all gold subreddits
        """
        all_posts = []

        for subreddit in self.GOLD_SUBREDDITS:
            posts = await self.fetch_subreddit_posts(
                subreddit, sort="hot", limit=limit_per_sub
            )
            all_posts.extend(posts)

            await asyncio.sleep(0.5)

        logger.info(f"Reddit: fetched {len(all_posts)} gold posts")
        return all_posts

    async def search_reddit(
        self,
        query: str,
        subreddit: Optional[str] = None,
        sort: str = "relevance",
        time_filter: str = "week",
        limit: int = 25,
    ) -> List[dict]:
        """
        Search Reddit for specific terms.

        Args:
            query: Search query
            subreddit: Limit to specific subreddit (optional)
            sort: Sort by (relevance, hot, new, top, comments)
            time_filter: Time period (hour, day, week, month, year, all)
            limit: Max results

        Returns:
            List of matching posts
        """
        posts = []

        try:
            token = await self._get_access_token()

            if token:
                base_url = "https://oauth.reddit.com"
                headers = {
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "SilverPrediction/1.0 (by /u/silver_prediction_bot)",
                }
            else:
                # Use old.reddit.com for better public access
                base_url = "https://old.reddit.com"
                headers = {
                    "User-Agent": "Mozilla/5.0 (compatible)",
                    "Accept": "application/json",
                }

            if subreddit:
                url = f"{base_url}/r/{subreddit}/search.json"
                params = {
                    "q": query,
                    "restrict_sr": "on",
                    "sort": sort,
                    "t": time_filter,
                    "limit": limit,
                }
            else:
                url = f"{base_url}/search.json"
                params = {
                    "q": query,
                    "sort": sort,
                    "t": time_filter,
                    "limit": limit,
                }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, params=params, headers=headers, timeout=15
                ) as response:
                    if response.status != 200:
                        return posts

                    data = await response.json()

                    for child in data.get("data", {}).get("children", []):
                        post = child.get("data", {})

                        text = post.get("selftext", "")
                        if len(text) > 500:
                            text = text[:500] + "..."

                        created_utc = post.get("created_utc", 0)
                        pub_date = datetime.fromtimestamp(created_utc)

                        posts.append({
                            "title": post.get("title", ""),
                            "description": text if text else None,
                            "url": f"https://reddit.com{post.get('permalink', '')}",
                            "published_at": pub_date,
                            "source": f"Reddit/r/{post.get('subreddit', 'unknown')}",
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                        })

        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")

        return posts
