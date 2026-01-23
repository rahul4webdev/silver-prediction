"""
Macro economic data service for silver/gold price prediction.
Fetches DXY (Dollar Index), Fear & Greed Index, and COT data.

These macro factors have strong correlations with precious metal prices:
- DXY: Strong inverse correlation with silver/gold
- Fear & Greed: Market sentiment indicator
- COT: Institutional positioning data
"""

import asyncio
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)


class MacroDataService:
    """
    Service for fetching macro economic data that affects silver/gold prices.

    Data sources:
    - DXY: Yahoo Finance (DX-Y.NYB)
    - Fear & Greed Index: CNN API
    - COT: CFTC weekly reports
    """

    def __init__(self, cache_ttl_minutes: int = 30):
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache: Dict[str, tuple] = {}

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

    async def get_dxy(self) -> Dict[str, Any]:
        """
        Get US Dollar Index (DXY) data from Yahoo Finance.

        Returns:
            Dict with current DXY value, change, and historical data
        """
        cache_key = "dxy"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf

            # Fetch DXY data
            ticker = yf.Ticker("DX-Y.NYB")

            # Get current price
            info = ticker.info
            current_price = info.get("regularMarketPrice") or info.get("previousClose", 0)
            prev_close = info.get("previousClose", current_price)

            # Get historical data for trend
            hist = ticker.history(period="1mo", interval="1d")

            if hist.empty:
                # Fallback: try different symbol
                ticker = yf.Ticker("DX=F")
                info = ticker.info
                current_price = info.get("regularMarketPrice") or info.get("previousClose", 0)
                prev_close = info.get("previousClose", current_price)
                hist = ticker.history(period="1mo", interval="1d")

            change = current_price - prev_close
            change_percent = (change / prev_close * 100) if prev_close else 0

            # Calculate trend indicators
            if not hist.empty:
                sma_5 = hist["Close"].tail(5).mean()
                sma_20 = hist["Close"].tail(20).mean()
                trend = "strengthening" if sma_5 > sma_20 else "weakening"

                # Recent history for charting
                recent_history = [
                    {"date": idx.strftime("%Y-%m-%d"), "close": float(row["Close"])}
                    for idx, row in hist.tail(30).iterrows()
                ]
            else:
                trend = "unknown"
                recent_history = []

            result = {
                "symbol": "DXY",
                "name": "US Dollar Index",
                "current": float(current_price),
                "previous_close": float(prev_close),
                "change": float(change),
                "change_percent": float(change_percent),
                "trend": trend,
                "timestamp": datetime.now().isoformat(),
                "history": recent_history,
                "correlation_note": "Inverse correlation with silver (-0.7 to -0.8)",
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error fetching DXY: {e}")
            return {
                "symbol": "DXY",
                "error": str(e),
                "current": 0,
                "timestamp": datetime.now().isoformat(),
            }

    async def get_fear_greed_index(self) -> Dict[str, Any]:
        """
        Get CNN Fear & Greed Index.

        The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed).
        During extreme fear, precious metals often rally (safe haven).

        Returns:
            Dict with current index value and classification
        """
        cache_key = "fear_greed"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            # CNN Fear & Greed API endpoint
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        raise Exception(f"CNN API returned {response.status}")

                    data = await response.json()

            # Extract current value
            fear_greed_data = data.get("fear_and_greed", {})
            current_value = fear_greed_data.get("score", 50)
            previous_value = fear_greed_data.get("previous_close", current_value)

            # Classify the value
            if current_value <= 25:
                classification = "Extreme Fear"
                signal = "bullish"  # Good for precious metals
            elif current_value <= 40:
                classification = "Fear"
                signal = "slightly_bullish"
            elif current_value <= 60:
                classification = "Neutral"
                signal = "neutral"
            elif current_value <= 75:
                classification = "Greed"
                signal = "slightly_bearish"
            else:
                classification = "Extreme Greed"
                signal = "bearish"  # Risk-on, bad for precious metals

            # Get historical data if available
            history = []
            if "fear_and_greed_historical" in data:
                hist_data = data["fear_and_greed_historical"].get("data", [])
                history = [
                    {"date": item.get("x"), "value": item.get("y")}
                    for item in hist_data[-30:]  # Last 30 days
                ]

            result = {
                "name": "Fear & Greed Index",
                "current": float(current_value),
                "previous": float(previous_value),
                "change": float(current_value - previous_value),
                "classification": classification,
                "signal_for_silver": signal,
                "timestamp": datetime.now().isoformat(),
                "history": history,
                "interpretation": f"{classification} - {'Bullish' if signal in ['bullish', 'slightly_bullish'] else 'Bearish' if signal in ['bearish', 'slightly_bearish'] else 'Neutral'} for precious metals",
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error fetching Fear & Greed Index: {e}")
            # Return neutral default
            return {
                "name": "Fear & Greed Index",
                "current": 50,
                "classification": "Neutral",
                "signal_for_silver": "neutral",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            }

    async def get_cot_data(self, asset: str = "silver") -> Dict[str, Any]:
        """
        Get CFTC Commitment of Traders (COT) data.

        COT shows positioning of commercial hedgers, large speculators,
        and small speculators. Extreme positioning often signals reversals.

        Args:
            asset: 'silver' or 'gold'

        Returns:
            Dict with COT positioning data
        """
        cache_key = f"cot_{asset}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            # CFTC commodity codes
            commodity_codes = {
                "silver": "084691",  # Silver COMEX
                "gold": "088691",    # Gold COMEX
            }

            code = commodity_codes.get(asset.lower(), "084691")

            # Quandl/CFTC data URL (free, no API key needed for basic access)
            # Using the disaggregated futures-only report
            current_year = datetime.now().year
            url = f"https://www.cftc.gov/dea/newcot/{current_year}/futures/financial_lf.txt"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status != 200:
                        raise Exception(f"CFTC returned {response.status}")

                    content = await response.text()

            # Parse the data (this is a complex text format)
            # For now, return estimated data structure
            # In production, you'd parse the actual CFTC file

            # Alternative: Use Quandl's parsed COT data
            quandl_url = f"https://data.nasdaq.com/api/v3/datasets/CFTC/{code}_F_L_ALL.json?rows=10"

            async with aiohttp.ClientSession() as session:
                async with session.get(quandl_url, timeout=10) as response:
                    if response.status == 200:
                        quandl_data = await response.json()
                        dataset = quandl_data.get("dataset", {})
                        data = dataset.get("data", [])
                        columns = dataset.get("column_names", [])

                        if data and columns:
                            latest = dict(zip(columns, data[0]))

                            # Extract key positions
                            commercial_long = latest.get("Commercial Positions-Long", 0)
                            commercial_short = latest.get("Commercial Positions-Short", 0)
                            noncommercial_long = latest.get("Noncommercial Positions-Long", 0)
                            noncommercial_short = latest.get("Noncommercial Positions-Short", 0)

                            net_commercial = commercial_long - commercial_short
                            net_speculative = noncommercial_long - noncommercial_short

                            # Interpret positioning
                            if net_speculative > 50000:
                                speculator_signal = "extremely_long"
                                interpretation = "Speculators extremely long - potential reversal"
                            elif net_speculative > 20000:
                                speculator_signal = "long"
                                interpretation = "Speculators net long - bullish momentum"
                            elif net_speculative < -20000:
                                speculator_signal = "short"
                                interpretation = "Speculators net short - bearish momentum"
                            elif net_speculative < -50000:
                                speculator_signal = "extremely_short"
                                interpretation = "Speculators extremely short - potential reversal"
                            else:
                                speculator_signal = "neutral"
                                interpretation = "Balanced positioning"

                            result = {
                                "asset": asset,
                                "report_date": latest.get("Date", ""),
                                "commercial": {
                                    "long": commercial_long,
                                    "short": commercial_short,
                                    "net": net_commercial,
                                },
                                "speculative": {
                                    "long": noncommercial_long,
                                    "short": noncommercial_short,
                                    "net": net_speculative,
                                },
                                "signal": speculator_signal,
                                "interpretation": interpretation,
                                "timestamp": datetime.now().isoformat(),
                            }

                            self._set_cache(cache_key, result)
                            return result

            # Fallback: return structure with no data
            return {
                "asset": asset,
                "error": "COT data not available",
                "signal": "neutral",
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error fetching COT data: {e}")
            return {
                "asset": asset,
                "error": str(e),
                "signal": "neutral",
                "timestamp": datetime.now().isoformat(),
            }

    async def get_treasury_yields(self) -> Dict[str, Any]:
        """
        Get US Treasury yields (10Y, 2Y).

        Higher real yields are typically bearish for gold/silver.
        Inverted yield curve signals recession risk (bullish for metals).

        Returns:
            Dict with treasury yield data
        """
        cache_key = "treasury"
        cached = self._get_cache(cache_key)
        if cached:
            return cached

        try:
            import yfinance as yf

            # Fetch 10Y and 2Y yields
            tnx = yf.Ticker("^TNX")  # 10-Year
            twy = yf.Ticker("^IRX")  # 13-Week T-Bill (proxy for short-term)

            tnx_info = tnx.info
            twy_info = twy.info

            yield_10y = tnx_info.get("regularMarketPrice", 0) or tnx_info.get("previousClose", 0)
            yield_3m = twy_info.get("regularMarketPrice", 0) or twy_info.get("previousClose", 0)

            # Yield curve (10Y - 3M)
            yield_curve = yield_10y - yield_3m

            if yield_curve < 0:
                curve_status = "inverted"
                signal = "bullish"  # Recession risk, good for metals
            elif yield_curve < 0.5:
                curve_status = "flat"
                signal = "neutral"
            else:
                curve_status = "normal"
                signal = "neutral"

            result = {
                "10y_yield": float(yield_10y),
                "3m_yield": float(yield_3m),
                "yield_curve": float(yield_curve),
                "curve_status": curve_status,
                "signal_for_silver": signal,
                "timestamp": datetime.now().isoformat(),
                "interpretation": f"Yield curve {curve_status} ({yield_curve:.2f}%)",
            }

            self._set_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error fetching treasury yields: {e}")
            return {
                "error": str(e),
                "signal_for_silver": "neutral",
                "timestamp": datetime.now().isoformat(),
            }

    async def get_all_macro_data(self, asset: str = "silver") -> Dict[str, Any]:
        """
        Get all macro data in one call.

        Returns:
            Dict with all macro indicators
        """
        # Fetch all data concurrently
        dxy, fear_greed, cot, treasury = await asyncio.gather(
            self.get_dxy(),
            self.get_fear_greed_index(),
            self.get_cot_data(asset),
            self.get_treasury_yields(),
            return_exceptions=True,
        )

        # Handle exceptions
        if isinstance(dxy, Exception):
            dxy = {"error": str(dxy)}
        if isinstance(fear_greed, Exception):
            fear_greed = {"error": str(fear_greed)}
        if isinstance(cot, Exception):
            cot = {"error": str(cot)}
        if isinstance(treasury, Exception):
            treasury = {"error": str(treasury)}

        # Calculate composite signal
        signals = []

        # DXY signal (inverse correlation)
        if dxy.get("trend") == "weakening":
            signals.append(1)  # Bullish for silver
        elif dxy.get("trend") == "strengthening":
            signals.append(-1)  # Bearish for silver
        else:
            signals.append(0)

        # Fear & Greed signal
        fg_signal = fear_greed.get("signal_for_silver", "neutral")
        if fg_signal == "bullish":
            signals.append(1)
        elif fg_signal == "slightly_bullish":
            signals.append(0.5)
        elif fg_signal == "bearish":
            signals.append(-1)
        elif fg_signal == "slightly_bearish":
            signals.append(-0.5)
        else:
            signals.append(0)

        # COT signal
        cot_signal = cot.get("signal", "neutral")
        if cot_signal in ["extremely_short"]:
            signals.append(1)  # Contrarian bullish
        elif cot_signal in ["short"]:
            signals.append(0.5)
        elif cot_signal in ["extremely_long"]:
            signals.append(-1)  # Contrarian bearish
        elif cot_signal in ["long"]:
            signals.append(-0.5)
        else:
            signals.append(0)

        # Treasury signal
        treasury_signal = treasury.get("signal_for_silver", "neutral")
        if treasury_signal == "bullish":
            signals.append(1)
        elif treasury_signal == "bearish":
            signals.append(-1)
        else:
            signals.append(0)

        # Average signal
        avg_signal = sum(signals) / len(signals) if signals else 0

        if avg_signal > 0.3:
            composite = "bullish"
        elif avg_signal < -0.3:
            composite = "bearish"
        else:
            composite = "neutral"

        return {
            "asset": asset,
            "timestamp": datetime.now().isoformat(),
            "dxy": dxy,
            "fear_greed": fear_greed,
            "cot": cot,
            "treasury": treasury,
            "composite_signal": {
                "direction": composite,
                "score": float(avg_signal),
                "interpretation": f"Macro factors are {composite} for {asset}",
            },
        }

    def macro_to_features(self, macro_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Convert macro data to ML features.

        Returns:
            Dict of feature values for ML models
        """
        features = {}

        # DXY features
        dxy = macro_data.get("dxy", {})
        features["dxy_current"] = dxy.get("current", 100) / 100  # Normalize around 1
        features["dxy_change_pct"] = dxy.get("change_percent", 0) / 100
        features["dxy_trend"] = 1 if dxy.get("trend") == "weakening" else -1 if dxy.get("trend") == "strengthening" else 0

        # Fear & Greed features
        fg = macro_data.get("fear_greed", {})
        features["fear_greed"] = fg.get("current", 50) / 100  # Normalize to 0-1
        features["fear_greed_change"] = fg.get("change", 0) / 100

        # COT features
        cot = macro_data.get("cot", {})
        spec = cot.get("speculative", {})
        features["cot_spec_net"] = spec.get("net", 0) / 100000  # Normalize
        cot_signal = cot.get("signal", "neutral")
        features["cot_signal"] = (
            1 if cot_signal in ["extremely_short", "short"]
            else -1 if cot_signal in ["extremely_long", "long"]
            else 0
        )

        # Treasury features
        treasury = macro_data.get("treasury", {})
        features["yield_10y"] = treasury.get("10y_yield", 4) / 10  # Normalize
        features["yield_curve"] = treasury.get("yield_curve", 0) / 5

        # Composite
        composite = macro_data.get("composite_signal", {})
        features["macro_composite"] = composite.get("score", 0)

        return features


# Singleton instance
macro_data_service = MacroDataService()
