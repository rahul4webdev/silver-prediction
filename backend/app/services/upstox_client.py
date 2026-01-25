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
import pandas as pd

from app.core.config import settings
from app.core.constants import UPSTOX_CONFIG

logger = logging.getLogger(__name__)


# Interval mapping: our format -> Upstox format
UPSTOX_INTERVAL_MAP = {
    "1m": "1minute",
    "5m": "5minute",
    "15m": "15minute",
    "30m": "30minute",
    "1h": "60minute",
    "4h": "day",  # Upstox doesn't have 4h, we'll aggregate from daily
    "1d": "day",
    "daily": "day",
    "1wk": "week",
    "1mo": "month",
}


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

        # Cache instruments - use trading_symbol (Upstox's field name)
        for instrument in filtered:
            trading_symbol = instrument.get("trading_symbol") or instrument.get("tradingsymbol", "")
            key = f"{instrument.get('exchange')}:{trading_symbol}"
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

        # Look for active SILVERM futures contract
        silver_futures = []
        now = datetime.now()

        for key, instrument in self._instruments_cache.items():
            # Upstox uses 'trading_symbol' (with underscore) not 'tradingsymbol'
            trading_symbol = instrument.get("trading_symbol", "") or instrument.get("tradingsymbol", "")
            instrument_type = instrument.get("instrument_type", "")
            name = instrument.get("name", "")
            asset_symbol = instrument.get("asset_symbol", "")

            # Look for SILVERM (Silver Mini) futures - check name, asset_symbol, or trading_symbol
            # Must match SILVERM but NOT SILVERMIC (Silver Micro)
            trading_symbol_upper = trading_symbol.upper()
            is_silver_mini = (
                # Match SILVERM but exclude SILVERMIC
                (trading_symbol_upper.startswith("SILVERM") and "SILVERMIC" not in trading_symbol_upper) or
                (asset_symbol == "SILVERM" and "SILVERMIC" not in trading_symbol_upper) or
                (name == "SILVER" and asset_symbol == "SILVERM")
            )

            if is_silver_mini and instrument_type == "FUT":
                # Check expiry - can be milliseconds timestamp or ISO string
                expiry_val = instrument.get("expiry")
                expiry = None

                if expiry_val:
                    try:
                        if isinstance(expiry_val, (int, float)):
                            # Milliseconds timestamp
                            expiry = datetime.fromtimestamp(expiry_val / 1000)
                        elif isinstance(expiry_val, str):
                            expiry = datetime.fromisoformat(expiry_val.replace("Z", "+00:00"))
                    except (ValueError, OSError):
                        pass

                if expiry is None or expiry > now:
                    silver_futures.append({
                        "instrument_key": instrument.get("instrument_key"),
                        "trading_symbol": trading_symbol,
                        "expiry": expiry,
                        "name": name,
                    })

        # Sort by expiry and return nearest
        if silver_futures:
            silver_futures.sort(key=lambda x: x["expiry"] or datetime.max)
            selected = silver_futures[0]
            logger.info(f"Selected MCX Silver: {selected['trading_symbol']} (key: {selected['instrument_key']})")
            return selected["instrument_key"]

        # Fallback: Look for any SILVER futures (not just SILVERM) but exclude SILVERMIC
        for key, instrument in self._instruments_cache.items():
            trading_symbol = instrument.get("trading_symbol", "") or instrument.get("tradingsymbol", "")
            trading_symbol_upper = trading_symbol.upper()
            name = instrument.get("name", "")
            # Exclude SILVERMIC from fallback as well
            if (name == "SILVER" or "SILVER" in trading_symbol_upper) and \
               "SILVERMIC" not in trading_symbol_upper and \
               instrument.get("instrument_type") == "FUT":
                logger.info(f"Fallback MCX Silver: {trading_symbol}")
                return instrument.get("instrument_key")

        logger.warning("Silver instrument not found in MCX instruments")
        return None

    async def get_all_silver_instrument_keys(self) -> List[Dict[str, Any]]:
        """
        Get instrument keys for all MCX Silver contracts (SILVER, SILVERM, SILVERMIC).

        Returns:
            List of dicts with instrument_key, trading_symbol, contract_type, expiry, lot_size
        """
        if not self._instruments_cache:
            await self.fetch_instruments("MCX")

        now = datetime.now()
        silver_contracts = []

        for key, instrument in self._instruments_cache.items():
            trading_symbol = instrument.get("trading_symbol", "") or instrument.get("tradingsymbol", "")
            trading_symbol_upper = trading_symbol.upper()
            instrument_type = instrument.get("instrument_type", "")
            name = instrument.get("name", "")
            asset_symbol = instrument.get("asset_symbol", "")

            # Only process futures contracts
            if instrument_type != "FUT":
                continue

            # Determine contract type
            contract_type = None
            if trading_symbol_upper.startswith("SILVERMIC") or "SILVERMIC" in trading_symbol_upper:
                contract_type = "SILVERMIC"
            elif trading_symbol_upper.startswith("SILVERM") and "SILVERMIC" not in trading_symbol_upper:
                contract_type = "SILVERM"
            elif trading_symbol_upper.startswith("SILVER") and "SILVERM" not in trading_symbol_upper and "SILVERMIC" not in trading_symbol_upper:
                contract_type = "SILVER"
            elif name == "SILVER":
                # Fallback: check asset_symbol
                if asset_symbol == "SILVERMIC":
                    contract_type = "SILVERMIC"
                elif asset_symbol == "SILVERM":
                    contract_type = "SILVERM"
                elif asset_symbol == "SILVER":
                    contract_type = "SILVER"

            if not contract_type:
                continue

            # Parse expiry date
            expiry_val = instrument.get("expiry")
            expiry = None

            if expiry_val:
                try:
                    if isinstance(expiry_val, (int, float)):
                        # Milliseconds timestamp
                        expiry = datetime.fromtimestamp(expiry_val / 1000)
                    elif isinstance(expiry_val, str):
                        expiry = datetime.fromisoformat(expiry_val.replace("Z", "+00:00"))
                except (ValueError, OSError):
                    pass

            # Skip expired contracts
            if expiry and expiry < now:
                continue

            silver_contracts.append({
                "instrument_key": instrument.get("instrument_key"),
                "trading_symbol": trading_symbol,
                "contract_type": contract_type,
                "expiry": expiry,
                "lot_size": instrument.get("lot_size"),
            })

        # Sort by expiry date (nearest first) - ascending order
        # This ensures contracts expiring soonest are selected first
        silver_contracts.sort(
            key=lambda x: x["expiry"] or datetime.max
        )

        logger.info(f"Found {len(silver_contracts)} active silver contracts")
        return silver_contracts

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
    # HIGH-LEVEL DATA FETCHING (For Data Sync Service)
    # =========================================================================

    async def get_mcx_silver_data(
        self,
        interval: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch MCX Silver historical data.

        This is the main method to use for syncing MCX data.
        It handles instrument key resolution and data fetching.
        For intervals not natively supported by Upstox (1h, 4h), it fetches
        30m data and aggregates it.

        Args:
            interval: Candle interval (30m, 1h, 4h, 1d, etc.)
            start_date: Start date
            end_date: End date (defaults to now)

        Returns:
            DataFrame with OHLCV data
        """
        if not self.is_authenticated:
            raise UpstoxAuthError("Not authenticated. Please set access token first.")

        # Get silver instrument key
        instrument_key = await self.get_silver_instrument_key()
        if not instrument_key:
            raise UpstoxAPIError("Could not find MCX Silver instrument")

        end_date = end_date or datetime.now()

        # Handle intervals that need aggregation (1h, 4h)
        # Upstox only supports: 1minute, 30minute, day, week, month
        needs_aggregation = interval in ["1h", "4h"]

        if needs_aggregation:
            # Fetch 30m data and aggregate
            upstox_interval = "30minute"
            logger.info(f"Interval {interval} not natively supported by Upstox, fetching 30m data to aggregate")
        else:
            # Map to Upstox interval
            upstox_interval = UPSTOX_INTERVAL_MAP.get(interval, interval)
            if upstox_interval not in ["1minute", "30minute", "day", "week", "month"]:
                logger.warning(f"Unknown interval {interval}, defaulting to 30minute")
                upstox_interval = "30minute"

        # Calculate days to fetch
        days = (end_date - start_date).days + 1

        # Check Upstox retention limits
        retention_days = UPSTOX_CONFIG["retention"].get(upstox_interval.replace("minute", "minute"), 365)
        if days > retention_days:
            logger.warning(f"Requested {days} days but Upstox only retains {retention_days} for {upstox_interval}")
            start_date = end_date - timedelta(days=retention_days - 1)

        # Fetch data in chunks
        candles = await self.fetch_historical_data_chunked(
            instrument_key=instrument_key,
            interval=upstox_interval,
            days=days,
            chunk_days=30,
        )

        if not candles:
            logger.warning("No MCX silver data returned from Upstox")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(candles)

        # Ensure proper timestamp format
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Aggregate if needed
        if needs_aggregation and not df.empty:
            df = self._aggregate_candles(df, interval)
            logger.info(f"Aggregated to {len(df)} {interval} candles")
        else:
            logger.info(f"Fetched {len(df)} MCX silver candles from Upstox")

        return df

    def _aggregate_candles(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Aggregate candles to a larger interval.

        Args:
            df: DataFrame with OHLCV data (expects 30m candles)
            target_interval: Target interval (1h, 4h)

        Returns:
            Aggregated DataFrame
        """
        if df.empty:
            return df

        # Ensure timestamp is datetime and set as index
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")

        # Determine aggregation rule
        if target_interval == "1h":
            rule = "1h"  # pandas rule for 1 hour
        elif target_interval == "4h":
            rule = "4h"  # pandas rule for 4 hours
        else:
            return df.reset_index()

        # Aggregate OHLCV data
        agg_df = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        # Reset index and rename
        agg_df = agg_df.reset_index()

        # Keep open_interest if present
        if "open_interest" in df.columns:
            oi_df = df["open_interest"].resample(rule).last().reset_index()
            agg_df["open_interest"] = oi_df["open_interest"]

        return agg_df

    async def get_live_quote(self, instrument_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Get live quote for MCX Silver.

        Args:
            instrument_key: Optional instrument key (auto-resolves if not provided)

        Returns:
            Dict with current price data
        """
        if not self.is_authenticated:
            raise UpstoxAuthError("Not authenticated")

        if not instrument_key:
            instrument_key = await self.get_silver_instrument_key()
            if not instrument_key:
                raise UpstoxAPIError("Could not find MCX Silver instrument")

        client = await self._get_client()

        url = f"{self.BASE_URL}/market-quote/quotes"
        params = {"instrument_key": instrument_key}

        response = await client.get(
            url,
            params=params,
            headers=self._get_auth_headers(),
        )

        if response.status_code != 200:
            raise UpstoxAPIError(f"Quote fetch failed: {response.text}")

        result = response.json()
        if result.get("status") != "success":
            raise UpstoxAPIError(f"API error: {result.get('message')}")

        data = result.get("data", {})

        # Upstox returns data with a dynamic key like "MCX_FO:SILVERM26APRFUT"
        # instead of the instrument_key we passed, so get the first (and only) item
        quote = {}
        symbol_key = instrument_key
        if data:
            symbol_key = list(data.keys())[0]
            quote = data[symbol_key]

        # Get price from different possible fields
        price = quote.get("last_price")
        if price is None:
            # Try to use close from ohlc if last_price not available
            ohlc = quote.get("ohlc", {})
            price = ohlc.get("close") or ohlc.get("open")

        return {
            "symbol": symbol_key,
            "price": price,
            "open": quote.get("ohlc", {}).get("open"),
            "high": quote.get("ohlc", {}).get("high"),
            "low": quote.get("ohlc", {}).get("low"),
            "close": quote.get("ohlc", {}).get("close"),
            "change": quote.get("net_change"),
            "change_percent": quote.get("percentage_change"),
            "volume": quote.get("volume"),
            "timestamp": datetime.now(),
            "market": "mcx",
            "currency": "INR",
        }

    async def verify_authentication(self) -> Dict[str, Any]:
        """
        Verify that the access token is valid.

        Returns:
            Dict with authentication status
        """
        if not self.access_token:
            return {
                "authenticated": False,
                "reason": "no_token",
                "message": "No access token set",
            }

        try:
            client = await self._get_client()

            # Try to get profile to verify token
            response = await client.get(
                f"{self.BASE_URL}/user/profile",
                headers=self._get_auth_headers(),
            )

            if response.status_code == 200:
                data = response.json()
                return {
                    "authenticated": True,
                    "user": data.get("data", {}),
                    "message": "Token is valid",
                }
            elif response.status_code == 401:
                return {
                    "authenticated": False,
                    "reason": "token_expired",
                    "message": "Access token has expired. Need to re-authenticate.",
                }
            else:
                return {
                    "authenticated": False,
                    "reason": "unknown",
                    "message": f"API returned status {response.status_code}",
                }

        except Exception as e:
            return {
                "authenticated": False,
                "reason": "error",
                "message": str(e),
            }

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
