"""
Multi-timeframe confluence detection service.
Identifies when multiple timeframes align for stronger signals.

Confluence occurs when predictions across different intervals
(30m, 1h, 4h, daily) agree on direction with high confidence.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.predictions import Prediction

logger = logging.getLogger(__name__)


class ConfluenceDetector:
    """
    Detect multi-timeframe confluence in predictions.

    Features:
    - Check alignment across 30m, 1h, 4h, daily
    - Calculate confluence strength
    - Generate confluence signals
    - Track confluence accuracy over time
    """

    INTERVALS = ["30m", "1h", "4h", "1d"]
    MIN_CONFIDENCE_THRESHOLD = 0.55  # Minimum confidence to include

    def __init__(self):
        self.confluence_history: List[Dict] = []

    async def get_latest_predictions(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
    ) -> Dict[str, Optional[Dict]]:
        """
        Get latest prediction for each interval.
        """
        predictions = {}

        for interval in self.INTERVALS:
            # Map interval names
            interval_db = "daily" if interval == "1d" else interval

            query = (
                select(Prediction)
                .where(
                    and_(
                        Prediction.asset == asset,
                        Prediction.market == market,
                        Prediction.interval == interval_db,
                        Prediction.target_time > datetime.utcnow(),  # Still valid
                    )
                )
                .order_by(Prediction.created_at.desc())
                .limit(1)
            )

            result = await db.execute(query)
            pred = result.scalar_one_or_none()

            if pred:
                predictions[interval] = {
                    "id": str(pred.id),
                    "direction": pred.predicted_direction,
                    "confidence": float(pred.direction_confidence),
                    "predicted_price": float(pred.predicted_price),
                    "current_price": float(pred.current_price),
                    "target_time": pred.target_time.isoformat(),
                    "interval": interval,
                }
            else:
                predictions[interval] = None

        return predictions

    async def detect_confluence(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
    ) -> Dict[str, Any]:
        """
        Detect if multiple timeframes are aligned.

        Returns:
            Dict with confluence analysis
        """
        predictions = await self.get_latest_predictions(db, asset, market)

        # Filter valid predictions with sufficient confidence
        valid_preds = {
            k: v for k, v in predictions.items()
            if v and v["confidence"] >= self.MIN_CONFIDENCE_THRESHOLD
        }

        if len(valid_preds) < 2:
            return {
                "status": "insufficient_data",
                "message": f"Only {len(valid_preds)} valid predictions available",
                "confluence_detected": False,
                "predictions": predictions,
            }

        # Check direction alignment
        directions = [p["direction"] for p in valid_preds.values()]
        bullish_count = sum(1 for d in directions if d == "bullish")
        bearish_count = sum(1 for d in directions if d == "bearish")

        total = len(directions)
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total

        # Determine confluence
        if bullish_ratio >= 0.75:
            confluence_direction = "bullish"
            alignment_ratio = bullish_ratio
            aligned_intervals = [k for k, v in valid_preds.items() if v["direction"] == "bullish"]
        elif bearish_ratio >= 0.75:
            confluence_direction = "bearish"
            alignment_ratio = bearish_ratio
            aligned_intervals = [k for k, v in valid_preds.items() if v["direction"] == "bearish"]
        else:
            confluence_direction = "mixed"
            alignment_ratio = max(bullish_ratio, bearish_ratio)
            aligned_intervals = []

        confluence_detected = confluence_direction in ["bullish", "bearish"]

        # Calculate average confidence of aligned predictions
        if aligned_intervals:
            avg_confidence = sum(
                valid_preds[i]["confidence"] for i in aligned_intervals
            ) / len(aligned_intervals)
        else:
            avg_confidence = 0

        # Calculate confluence strength (0-1)
        # Based on: alignment ratio, number of intervals, average confidence
        if confluence_detected:
            strength = (
                alignment_ratio * 0.4 +
                (len(aligned_intervals) / len(self.INTERVALS)) * 0.3 +
                avg_confidence * 0.3
            )
        else:
            strength = 0

        # Generate signal quality
        if strength >= 0.8:
            signal_quality = "strong"
        elif strength >= 0.6:
            signal_quality = "moderate"
        elif strength >= 0.4:
            signal_quality = "weak"
        else:
            signal_quality = "none"

        result = {
            "status": "success",
            "asset": asset,
            "market": market,
            "timestamp": datetime.now().isoformat(),
            "confluence_detected": confluence_detected,
            "direction": confluence_direction,
            "alignment_ratio": alignment_ratio,
            "aligned_intervals": aligned_intervals,
            "total_intervals": len(valid_preds),
            "avg_confidence": avg_confidence,
            "strength": strength,
            "signal_quality": signal_quality,
            "predictions": predictions,
            "recommendation": self._generate_recommendation(
                confluence_detected, confluence_direction, strength, aligned_intervals
            ),
        }

        # Store in history
        self.confluence_history.append({
            "timestamp": datetime.now(),
            "result": result,
        })

        # Keep only last 100 entries
        if len(self.confluence_history) > 100:
            self.confluence_history = self.confluence_history[-100:]

        return result

    def _generate_recommendation(
        self,
        detected: bool,
        direction: str,
        strength: float,
        intervals: List[str],
    ) -> str:
        """Generate trading recommendation based on confluence."""
        if not detected:
            return "No confluence - wait for clearer signals or use single-timeframe analysis"

        if strength >= 0.8:
            return (
                f"STRONG {direction.upper()} confluence across {', '.join(intervals)}. "
                f"Consider entering with confidence, use appropriate position sizing."
            )
        elif strength >= 0.6:
            return (
                f"MODERATE {direction.upper()} confluence on {', '.join(intervals)}. "
                f"Proceed with caution, consider smaller position size."
            )
        else:
            return (
                f"WEAK {direction.upper()} confluence. "
                f"Low conviction signal, wait for stronger alignment."
            )

    async def get_confluence_history(
        self,
        db: AsyncSession,
        asset: str = "silver",
        market: str = "mcx",
        hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """
        Get confluence detection history.
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        return [
            entry["result"]
            for entry in self.confluence_history
            if entry["timestamp"] >= cutoff
            and entry["result"].get("asset") == asset
            and entry["result"].get("market") == market
        ]


class CorrelationAnalyzer:
    """
    Analyze correlations between silver and related assets.

    Tracks real-time correlations with:
    - Gold (positive correlation)
    - DXY (negative correlation)
    - S&P 500 (risk appetite indicator)
    """

    CORRELATION_ASSETS = {
        "gold": {"symbol": "GC=F", "expected": "positive", "typical": 0.85},
        "dxy": {"symbol": "DX-Y.NYB", "expected": "negative", "typical": -0.75},
        "sp500": {"symbol": "^GSPC", "expected": "variable", "typical": 0.3},
        "copper": {"symbol": "HG=F", "expected": "positive", "typical": 0.6},
        "oil": {"symbol": "CL=F", "expected": "variable", "typical": 0.4},
    }

    async def calculate_correlations(
        self,
        lookback_days: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate correlations between silver and related assets.
        """
        try:
            import yfinance as yf
            import numpy as np

            # Fetch silver data
            silver = yf.Ticker("SI=F")
            silver_hist = silver.history(period=f"{lookback_days}d")["Close"]

            if silver_hist.empty:
                return {"error": "Could not fetch silver data"}

            correlations = {}

            for asset_name, config in self.CORRELATION_ASSETS.items():
                try:
                    ticker = yf.Ticker(config["symbol"])
                    hist = ticker.history(period=f"{lookback_days}d")["Close"]

                    if not hist.empty and len(hist) > 10:
                        # Align data
                        common_dates = silver_hist.index.intersection(hist.index)
                        if len(common_dates) > 10:
                            silver_aligned = silver_hist.loc[common_dates]
                            asset_aligned = hist.loc[common_dates]

                            # Calculate correlation
                            corr = np.corrcoef(
                                silver_aligned.pct_change().dropna(),
                                asset_aligned.pct_change().dropna()
                            )[0, 1]

                            # Determine if correlation is as expected
                            expected = config["expected"]
                            typical = config["typical"]

                            if expected == "positive":
                                status = "normal" if corr > 0.5 else "diverging"
                            elif expected == "negative":
                                status = "normal" if corr < -0.3 else "diverging"
                            else:
                                status = "normal"

                            correlations[asset_name] = {
                                "correlation": float(corr),
                                "expected": expected,
                                "typical": typical,
                                "status": status,
                                "data_points": len(common_dates),
                            }

                except Exception as e:
                    logger.warning(f"Could not calculate {asset_name} correlation: {e}")
                    correlations[asset_name] = {"error": str(e)}

            # Calculate overall market regime
            gold_corr = correlations.get("gold", {}).get("correlation", 0.85)
            dxy_corr = correlations.get("dxy", {}).get("correlation", -0.75)

            if gold_corr > 0.9 and dxy_corr < -0.7:
                regime = "classic_safe_haven"
                description = "Silver moving with gold, inverse to dollar - classic safe haven behavior"
            elif gold_corr > 0.7 and dxy_corr > 0:
                regime = "risk_on"
                description = "Unusual positive correlation with dollar - industrial demand driven"
            elif gold_corr < 0.5:
                regime = "divergent"
                description = "Silver diverging from gold - watch for industrial factors"
            else:
                regime = "normal"
                description = "Correlations within typical ranges"

            return {
                "status": "success",
                "lookback_days": lookback_days,
                "timestamp": datetime.now().isoformat(),
                "correlations": correlations,
                "market_regime": {
                    "type": regime,
                    "description": description,
                },
            }

        except Exception as e:
            logger.error(f"Error calculating correlations: {e}")
            return {"error": str(e)}


# Singleton instances
confluence_detector = ConfluenceDetector()
correlation_analyzer = CorrelationAnalyzer()
