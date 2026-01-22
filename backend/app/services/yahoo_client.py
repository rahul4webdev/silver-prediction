"""
Yahoo Finance client for COMEX silver data and correlated factors.
Provides historical data and near-real-time quotes.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class YahooFinanceError(Exception):
    """Raised when Yahoo Finance API call fails."""
    pass


class YahooFinanceClient:
    """
    Yahoo Finance client for COMEX and market factor data.

    Features:
    - Historical OHLCV data
    - Multiple timeframe support
    - Correlated factors fetching
    - Caching for rate limit management
    """

    # Symbol mappings
    SYMBOLS = {
        # Precious metals - International
        "silver": "SI=F",       # Silver Futures (COMEX)
        "gold": "GC=F",         # Gold Futures (COMEX)

        # Indian Silver/Gold ETFs (MCX proxy)
        "silver_mcx": "SILVERBEES.NS",  # Nippon India Silver ETF
        "silver_etf": "SILVERBEES.NS",  # Alias
        "gold_mcx": "GOLDBEES.NS",      # Nippon India Gold ETF
        "gold_etf": "GOLDBEES.NS",      # Alias

        # Correlated factors
        "dxy": "DX-Y.NYB",      # US Dollar Index
        "us10y": "^TNX",        # 10-Year Treasury Yield
        "vix": "^VIX",          # Volatility Index
        "copper": "HG=F",       # Copper Futures
        "spx": "^GSPC",         # S&P 500
        "crude": "CL=F",        # Crude Oil

        # Currency
        "usdinr": "USDINR=X",   # USD/INR Exchange Rate
    }

    # Interval mappings (yfinance format)
    INTERVALS = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "30m": "30m",
        "1h": "60m",
        "4h": "1h",  # Will aggregate 4x 1h candles
        "1d": "1d",
        "daily": "1d",
        "1wk": "1wk",
        "1mo": "1mo",
    }

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._cache_expiry: Dict[str, datetime] = {}

    def _get_symbol(self, asset: str) -> str:
        """Get Yahoo Finance symbol for asset."""
        return self.SYMBOLS.get(asset.lower(), asset)

    def _get_interval(self, interval: str) -> str:
        """Convert interval to yfinance format."""
        return self.INTERVALS.get(interval.lower(), interval)

    async def get_historical_data(
        self,
        symbol: str,
        interval: str = "30m",
        period: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.

        Args:
            symbol: Asset symbol or code (e.g., "silver", "SI=F")
            interval: Candle interval (30m, 1h, 1d, etc.)
            period: Period string (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max)
            start: Start datetime (alternative to period)
            end: End datetime

        Returns:
            DataFrame with OHLCV data
        """
        yf_symbol = self._get_symbol(symbol)
        yf_interval = self._get_interval(interval)

        try:
            ticker = yf.Ticker(yf_symbol)

            if period:
                df = ticker.history(period=period, interval=yf_interval)
            elif start and end:
                df = ticker.history(start=start, end=end, interval=yf_interval)
            elif start:
                df = ticker.history(start=start, interval=yf_interval)
            else:
                # Default to 1 year for daily, 60 days for intraday
                if yf_interval in ["1d", "1wk", "1mo"]:
                    df = ticker.history(period="1y", interval=yf_interval)
                else:
                    df = ticker.history(period="60d", interval=yf_interval)

            if df.empty:
                logger.warning(f"No data returned for {yf_symbol}")
                return pd.DataFrame()

            # Standardize column names
            df = df.reset_index()
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]

            # Rename datetime column
            if "date" in df.columns:
                df = df.rename(columns={"date": "timestamp"})
            if "datetime" in df.columns:
                df = df.rename(columns={"datetime": "timestamp"})

            # Ensure timezone-aware timestamps
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {yf_symbol}: {e}")
            raise YahooFinanceError(f"Failed to fetch data: {e}")

    async def get_silver_comex(
        self,
        interval: str = "30m",
        days: int = 365,
    ) -> pd.DataFrame:
        """
        Fetch COMEX silver historical data.

        Args:
            interval: Candle interval
            days: Number of days of history

        Returns:
            DataFrame with silver OHLCV data
        """
        start = datetime.now() - timedelta(days=days)
        return await self.get_historical_data(
            symbol="silver",
            interval=interval,
            start=start,
        )

    async def get_current_price(self, symbol: str) -> Dict[str, Any]:
        """
        Get current price and quote data.

        Args:
            symbol: Asset symbol

        Returns:
            Dict with current price data
        """
        yf_symbol = self._get_symbol(symbol)

        try:
            ticker = yf.Ticker(yf_symbol)
            info = ticker.info

            return {
                "symbol": yf_symbol,
                "price": info.get("regularMarketPrice") or info.get("previousClose"),
                "change": info.get("regularMarketChange"),
                "change_percent": info.get("regularMarketChangePercent"),
                "high": info.get("dayHigh"),
                "low": info.get("dayLow"),
                "open": info.get("open"),
                "previous_close": info.get("previousClose"),
                "volume": info.get("volume"),
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error fetching price for {yf_symbol}: {e}")
            raise YahooFinanceError(f"Failed to fetch price: {e}")

    async def get_usdinr_rate(self) -> float:
        """
        Get current USD/INR exchange rate.

        Returns:
            Exchange rate as float
        """
        try:
            ticker = yf.Ticker("USDINR=X")
            info = ticker.info
            rate = info.get("regularMarketPrice") or info.get("previousClose") or 83.5
            return float(rate)
        except Exception as e:
            logger.warning(f"Failed to fetch USD/INR rate: {e}, using default 83.5")
            return 83.5

    async def get_silver_mcx(
        self,
        interval: str = "30m",
        days: int = 60,
    ) -> pd.DataFrame:
        """
        Fetch MCX Silver data using Silver Bees ETF as proxy.

        This uses SILVERBEES.NS (Nippon India Silver ETF) from NSE
        which closely tracks MCX silver prices.

        Args:
            interval: Candle interval
            days: Number of days of history

        Returns:
            DataFrame with silver OHLCV data in INR
        """
        start = datetime.now() - timedelta(days=days)
        return await self.get_historical_data(
            symbol="silver_mcx",
            interval=interval,
            start=start,
        )

    async def get_silver_price_inr(self) -> Dict[str, Any]:
        """
        Get current silver price in INR.

        First tries Silver Bees ETF, then falls back to
        COMEX silver with USD/INR conversion.

        Returns:
            Dict with silver price data in INR
        """
        # Try Silver Bees ETF first (actual INR price)
        try:
            etf_price = await self.get_current_price("silver_mcx")
            if etf_price and etf_price.get("price"):
                # Silver Bees is per gram, convert to per kg for MCX equivalent
                # Note: Silver Bees tracks 1 gram of silver
                price_per_gram = etf_price["price"]
                price_per_kg = price_per_gram * 1000

                return {
                    "symbol": "SILVERBEES.NS",
                    "price": price_per_kg,
                    "price_per_gram": price_per_gram,
                    "currency": "INR",
                    "source": "silver_bees_etf",
                    "change": etf_price.get("change"),
                    "change_percent": etf_price.get("change_percent"),
                    "timestamp": datetime.now(),
                }
        except Exception as e:
            logger.warning(f"Silver Bees fetch failed: {e}")

        # Fallback to COMEX + USD/INR conversion
        try:
            comex_price = await self.get_current_price("silver")
            usd_inr = await self.get_usdinr_rate()

            if comex_price and comex_price.get("price"):
                # COMEX silver is per troy ounce, convert to per kg
                # 1 troy ounce = 31.1035 grams
                # 1 kg = 1000 grams = 32.1507 troy ounces
                price_per_oz_usd = comex_price["price"]
                price_per_kg_inr = price_per_oz_usd * 32.1507 * usd_inr

                return {
                    "symbol": "SI=F",
                    "price": round(price_per_kg_inr, 2),
                    "price_per_oz_usd": price_per_oz_usd,
                    "usd_inr_rate": usd_inr,
                    "currency": "INR",
                    "source": "comex_converted",
                    "change": comex_price.get("change"),
                    "change_percent": comex_price.get("change_percent"),
                    "timestamp": datetime.now(),
                }
        except Exception as e:
            logger.error(f"COMEX fallback failed: {e}")
            raise YahooFinanceError(f"Failed to fetch silver price: {e}")

    async def get_multiple_symbols(
        self,
        symbols: List[str],
        interval: str = "1d",
        period: str = "1mo",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols efficiently.

        Args:
            symbols: List of symbols
            interval: Candle interval
            period: Time period

        Returns:
            Dict mapping symbol to DataFrame
        """
        results = {}
        yf_interval = self._get_interval(interval)

        # yfinance supports batch downloads
        yf_symbols = [self._get_symbol(s) for s in symbols]
        symbol_str = " ".join(yf_symbols)

        try:
            data = yf.download(
                symbol_str,
                period=period,
                interval=yf_interval,
                group_by="ticker",
                threads=True,
            )

            for i, symbol in enumerate(symbols):
                yf_symbol = yf_symbols[i]
                if len(symbols) == 1:
                    df = data.copy()
                else:
                    df = data[yf_symbol].copy() if yf_symbol in data.columns.get_level_values(0) else pd.DataFrame()

                if not df.empty:
                    df = df.reset_index()
                    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                    if "date" in df.columns:
                        df = df.rename(columns={"date": "timestamp"})
                    results[symbol] = df

        except Exception as e:
            logger.error(f"Error in batch download: {e}")
            # Fallback to individual downloads
            for symbol in symbols:
                try:
                    df = await self.get_historical_data(symbol, interval, period)
                    results[symbol] = df
                except Exception:
                    results[symbol] = pd.DataFrame()

        return results

    async def get_correlated_factors(
        self,
        interval: str = "1d",
        period: str = "3mo",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch all correlated factors for silver analysis.

        Returns:
            Dict mapping factor name to DataFrame
        """
        factors = ["dxy", "gold", "us10y", "vix", "copper", "spx", "crude"]
        return await self.get_multiple_symbols(factors, interval, period)

    async def calculate_correlations(
        self,
        silver_df: pd.DataFrame,
        factors_data: Dict[str, pd.DataFrame],
        window: int = 30,
    ) -> Dict[str, float]:
        """
        Calculate rolling correlations between silver and factors.

        Args:
            silver_df: Silver price DataFrame
            factors_data: Dict of factor DataFrames
            window: Rolling window size

        Returns:
            Dict of factor correlations
        """
        correlations = {}

        silver_returns = silver_df["close"].pct_change().dropna()

        for factor_name, factor_df in factors_data.items():
            if factor_df.empty or "close" not in factor_df.columns:
                continue

            factor_returns = factor_df["close"].pct_change().dropna()

            # Align dates
            aligned = pd.concat([silver_returns, factor_returns], axis=1, join="inner")
            aligned.columns = ["silver", factor_name]

            if len(aligned) > window:
                # Calculate rolling correlation
                rolling_corr = aligned["silver"].rolling(window).corr(aligned[factor_name])
                correlations[factor_name] = rolling_corr.iloc[-1] if not pd.isna(rolling_corr.iloc[-1]) else 0.0
            else:
                # Simple correlation for short periods
                correlations[factor_name] = aligned["silver"].corr(aligned[factor_name])

        return correlations

    def convert_to_candles(self, df: pd.DataFrame) -> List[Dict]:
        """
        Convert DataFrame to list of candle dictionaries.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            List of candle dicts
        """
        candles = []

        for _, row in df.iterrows():
            candle = {
                "timestamp": row["timestamp"].isoformat() if hasattr(row["timestamp"], "isoformat") else str(row["timestamp"]),
                "open": float(row["open"]) if pd.notna(row["open"]) else None,
                "high": float(row["high"]) if pd.notna(row["high"]) else None,
                "low": float(row["low"]) if pd.notna(row["low"]) else None,
                "close": float(row["close"]) if pd.notna(row["close"]) else None,
                "volume": int(row["volume"]) if "volume" in row and pd.notna(row["volume"]) else 0,
            }
            candles.append(candle)

        return candles


# Singleton instance
yahoo_client = YahooFinanceClient()
