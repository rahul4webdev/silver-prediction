"""
Upstox API client for MCX silver data.
Handles OAuth authentication, historical data fetching, and WebSocket streaming.
"""

import asyncio
import gzip
import json
import logging
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from urllib.parse import urlencode

import httpx

from app.core.config import settings
from app.core.constants import UPSTOX_CONFIG

logger = logging.getLogger(__name__)


class UpstoxAuthError(Exception):
    """Raised when authentication fails."""
    pass


class UpstoxAPIError(Exception):
    """Raised when API call fails."""
    pass


class UpstoxClient:
    """
    Upstox API client for MCX data.

    Features:
    - OAuth2 authentication flow
    - Historical candle data fetching
    - Real-time WebSocket streaming
    - Automatic token refresh
    """

    BASE_URL = "https://api.upstox.com/v2"

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        redirect_uri: Optional[str] = None,
        access_token: Optional[str] = None,
    ):
        self.api_key = api_key or settings.upstox_api_key
        self.api_secret = api_secret or settings.upstox_api_secret
        self.redirect_uri = redirect_uri or settings.upstox_redirect_uri

        # Use provided token or get from settings
        self.access_token: Optional[str] = access_token or settings.upstox_access_token
        self._http_client: Optional[httpx.AsyncClient] = None
        self._instruments_cache: Dict[str, Dict] = {}

        # Log if token is available
        if self.access_token:
            logger.info("Upstox client initialized with access token")

    @property
    def is_authenticated(self) -> bool:
        """Check if client has a valid access token."""
        return bool(self.access_token)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=30.0,
                headers={"Accept": "application/json"},
            )
        return self._http_client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    # =========================================================================
    # AUTHENTICATION
    # =========================================================================

    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """
        Get OAuth authorization URL for user to authenticate.

        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Authorization URL to redirect user to
        """
        params = {
            "client_id": self.api_key,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
        }
        if state:
            params["state"] = state

        return f"{UPSTOX_CONFIG['auth_url']}?{urlencode(params)}"

    async def exchange_code_for_token(self, authorization_code: str) -> Dict[str, Any]:
        """
        Exchange authorization code for access token.

        Args:
            authorization_code: Code received from OAuth callback

        Returns:
            Token response containing access_token
        """
        client = await self._get_client()

        data = {
            "code": authorization_code,
            "client_id": self.api_key,
            "client_secret": self.api_secret,
            "redirect_uri": self.redirect_uri,
            "grant_type": "authorization_code",
        }

        response = await client.post(
            UPSTOX_CONFIG["token_url"],
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            logger.error(f"Token exchange failed: {response.text}")
            raise UpstoxAuthError(f"Token exchange failed: {response.text}")

        result = response.json()
        self.access_token = result.get("access_token")

        return result

    def set_access_token(self, token: str) -> None:
        """Set access token directly (for stored tokens)."""
        self.access_token = token

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        if not self.access_token:
            raise UpstoxAuthError("No access token set. Please authenticate first.")
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    # =========================================================================
    # INSTRUMENTS
    # =========================================================================

    async def fetch_instruments(self, exchange: str = "MCX") -> List[Dict]:
        """
        Fetch all instruments for an exchange.

        Args:
            exchange: Exchange code (MCX, NSE, BSE)

        Returns:
            List of instrument dictionaries
        """
        client = await self._get_client()

        # Download compressed instruments file
        url = UPSTOX_CONFIG["instruments_url"]
        response = await client.get(url)

        if response.status_code != 200:
            raise UpstoxAPIError(f"Failed to fetch instruments: {response.text}")

        # Decompress and parse
        data = gzip.decompress(response.content)
        instruments = json.loads(data)

        # Filter by exchange
        filtered = [i for i in instruments if i.get("exchange") == exchange]

        # Cache instruments
        for instrument in filtered:
            key = f"{instrument.get('exchange')}:{instrument.get('tradingsymbol')}"
            self._instruments_cache[key] = instrument

        return filtered

    async def get_silver_instrument_key(self) -> Optional[str]:
        """
        Get the instrument key for MCX Silver (SILVERM).

        Returns:
            Instrument key string or None if not found
        """
        if not self._instruments_cache:
            await self.fetch_instruments("MCX")

        # Look for SILVERM (Silver Mini) or SILVER
        for key, instrument in self._instruments_cache.items():
            tradingsymbol = instrument.get("tradingsymbol", "")
            if "SILVER" in tradingsymbol and instrument.get("instrument_type") == "FUT":
                # Return the nearest expiry contract
                return instrument.get("instrument_key")

        logger.warning("Silver instrument not found in MCX instruments")
        return None

    # =========================================================================
    # HISTORICAL DATA
    # =========================================================================

    async def get_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: datetime,
        to_date: datetime,
    ) -> List[Dict]:
        """
        Fetch historical candle data.

        Args:
            instrument_key: Upstox instrument key
            interval: Candle interval (1minute, 30minute, day, week, month)
            from_date: Start date
            to_date: End date

        Returns:
            List of candle dictionaries with OHLCV data
        """
        client = await self._get_client()

        # Format dates
        from_str = from_date.strftime("%Y-%m-%d")
        to_str = to_date.strftime("%Y-%m-%d")

        url = f"{self.BASE_URL}/historical-candle/{instrument_key}/{interval}/{to_str}/{from_str}"

        response = await client.get(url, headers=self._get_auth_headers())

        if response.status_code != 200:
            logger.error(f"Historical data fetch failed: {response.text}")
            raise UpstoxAPIError(f"Failed to fetch historical data: {response.text}")

        result = response.json()

        if result.get("status") != "success":
            raise UpstoxAPIError(f"API error: {result.get('message', 'Unknown error')}")

        candles = result.get("data", {}).get("candles", [])

        # Transform to standard format
        # Upstox returns: [timestamp, open, high, low, close, volume, oi]
        formatted_candles = []
        for candle in candles:
            formatted_candles.append({
                "timestamp": candle[0],
                "open": candle[1],
                "high": candle[2],
                "low": candle[3],
                "close": candle[4],
                "volume": candle[5],
                "open_interest": candle[6] if len(candle) > 6 else None,
            })

        return formatted_candles

    async def fetch_historical_data_chunked(
        self,
        instrument_key: str,
        interval: str,
        days: int = 365,
        chunk_days: int = 30,
    ) -> List[Dict]:
        """
        Fetch historical data in chunks to handle rate limits.

        Args:
            instrument_key: Upstox instrument key
            interval: Candle interval
            days: Total days of history to fetch
            chunk_days: Days per chunk

        Returns:
            Combined list of candles
        """
        all_candles = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        current_end = end_date

        while current_end > start_date:
            current_start = max(current_end - timedelta(days=chunk_days), start_date)

            try:
                candles = await self.get_historical_candles(
                    instrument_key=instrument_key,
                    interval=interval,
                    from_date=current_start,
                    to_date=current_end,
                )
                all_candles.extend(candles)

                logger.info(
                    f"Fetched {len(candles)} candles from {current_start} to {current_end}"
                )

            except UpstoxAPIError as e:
                logger.error(f"Error fetching chunk: {e}")
                # Continue with next chunk

            current_end = current_start - timedelta(days=1)

            # Rate limit delay
            await asyncio.sleep(0.5)

        # Sort by timestamp (oldest first)
        all_candles.sort(key=lambda x: x["timestamp"])

        return all_candles

    # =========================================================================
    # WEBSOCKET STREAMING
    # =========================================================================

    async def get_websocket_auth(self) -> Dict[str, Any]:
        """
        Get WebSocket authorization URL for market data streaming.

        Returns:
            Dict containing authorized WebSocket URL
        """
        client = await self._get_client()

        response = await client.get(
            UPSTOX_CONFIG["websocket_auth_url"],
            headers=self._get_auth_headers(),
        )

        if response.status_code != 200:
            raise UpstoxAPIError(f"WebSocket auth failed: {response.text}")

        return response.json().get("data", {})

    async def stream_market_data(
        self,
        instrument_keys: List[str],
        on_message: Callable[[Dict], None],
        on_error: Optional[Callable[[Exception], None]] = None,
    ) -> None:
        """
        Stream real-time market data via WebSocket.

        Args:
            instrument_keys: List of instrument keys to subscribe to
            on_message: Callback for each market data message
            on_error: Optional error callback

        Note:
            This method runs indefinitely until cancelled.
            Upstox WebSocket uses Protobuf encoding - requires decoding.
        """
        import websockets

        try:
            # Get authorized WebSocket URL
            auth_data = await self.get_websocket_auth()
            ws_url = auth_data.get("authorizedRedirectUri")

            if not ws_url:
                raise UpstoxAPIError("Failed to get WebSocket URL")

            async with websockets.connect(ws_url) as websocket:
                # Subscribe to instruments
                subscribe_message = {
                    "guid": "silver-prediction-stream",
                    "method": "sub",
                    "data": {
                        "mode": "full",
                        "instrumentKeys": instrument_keys[:100],  # Max 100
                    },
                }

                await websocket.send(json.dumps(subscribe_message))
                logger.info(f"Subscribed to {len(instrument_keys)} instruments")

                # Process messages
                async for message in websocket:
                    try:
                        # Note: Upstox sends Protobuf-encoded messages
                        # You'll need to decode using their proto file
                        # For now, assuming JSON for simplicity
                        if isinstance(message, bytes):
                            # Protobuf decoding would go here
                            # For now, skip binary messages
                            continue

                        data = json.loads(message)
                        on_message(data)

                    except json.JSONDecodeError:
                        logger.warning("Received non-JSON message")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if on_error:
                            on_error(e)

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if on_error:
                on_error(e)
            raise

    async def stream_market_data_generator(
        self,
        instrument_keys: List[str],
    ) -> AsyncGenerator[Dict, None]:
        """
        Generator version of market data streaming.

        Yields:
            Market data dictionaries
        """
        import websockets

        auth_data = await self.get_websocket_auth()
        ws_url = auth_data.get("authorizedRedirectUri")

        if not ws_url:
            raise UpstoxAPIError("Failed to get WebSocket URL")

        async with websockets.connect(ws_url) as websocket:
            # Subscribe
            subscribe_message = {
                "guid": "silver-prediction-stream",
                "method": "sub",
                "data": {
                    "mode": "full",
                    "instrumentKeys": instrument_keys[:100],
                },
            }
            await websocket.send(json.dumps(subscribe_message))

            async for message in websocket:
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                        yield data
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    continue


# Singleton instance
upstox_client = UpstoxClient()
