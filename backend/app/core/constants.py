"""
Constants and asset configurations for the prediction system.
Defines supported assets, markets, prediction intervals, and correlated factors.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Market(str, Enum):
    """Supported markets."""
    MCX = "mcx"
    COMEX = "comex"


class Asset(str, Enum):
    """Supported assets."""
    SILVER = "silver"
    GOLD = "gold"  # Future support


class PredictionInterval(str, Enum):
    """Supported prediction intervals."""
    THIRTY_MIN = "30m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    DAILY = "daily"


class PredictionDirection(str, Enum):
    """Prediction direction outcomes."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


@dataclass
class MarketConfig:
    """Configuration for a market."""
    exchange: str
    symbol: str
    trading_hours_start: str  # Local time
    trading_hours_end: str
    timezone: str
    currency: str


@dataclass
class AssetConfig:
    """Configuration for an asset across markets."""
    name: str
    markets: Dict[Market, MarketConfig]
    correlated_factors: List[str]
    volatility_multiplier: float


# =============================================================================
# MARKET CONFIGURATIONS
# =============================================================================

MCX_SILVER_CONFIG = MarketConfig(
    exchange="MCX",
    symbol="SILVERM",  # Will be resolved to actual instrument key
    trading_hours_start="09:00",
    trading_hours_end="23:30",
    timezone="Asia/Kolkata",
    currency="INR",
)

COMEX_SILVER_CONFIG = MarketConfig(
    exchange="COMEX",
    symbol="SI=F",  # Yahoo Finance symbol
    trading_hours_start="18:00",  # Previous day
    trading_hours_end="17:00",  # Next day (nearly 24h)
    timezone="America/New_York",
    currency="USD",
)

MCX_GOLD_CONFIG = MarketConfig(
    exchange="MCX",
    symbol="GOLDM",
    trading_hours_start="09:00",
    trading_hours_end="23:30",
    timezone="Asia/Kolkata",
    currency="INR",
)

COMEX_GOLD_CONFIG = MarketConfig(
    exchange="COMEX",
    symbol="GC=F",
    trading_hours_start="18:00",
    trading_hours_end="17:00",
    timezone="America/New_York",
    currency="USD",
)


# =============================================================================
# ASSET CONFIGURATIONS
# =============================================================================

ASSET_CONFIGS: Dict[Asset, AssetConfig] = {
    Asset.SILVER: AssetConfig(
        name="Silver",
        markets={
            Market.MCX: MCX_SILVER_CONFIG,
            Market.COMEX: COMEX_SILVER_CONFIG,
        },
        correlated_factors=[
            "DXY",      # US Dollar Index (inverse)
            "GC=F",     # Gold (positive)
            "HG=F",     # Copper (industrial proxy)
            "^TNX",     # 10-Year Treasury Yield (inverse)
            "^VIX",     # Volatility Index
        ],
        volatility_multiplier=2.0,  # Silver is ~2x more volatile than gold
    ),
    Asset.GOLD: AssetConfig(
        name="Gold",
        markets={
            Market.MCX: MCX_GOLD_CONFIG,
            Market.COMEX: COMEX_GOLD_CONFIG,
        },
        correlated_factors=[
            "DXY",
            "SI=F",     # Silver
            "^TNX",
            "^VIX",
        ],
        volatility_multiplier=1.0,
    ),
}


# =============================================================================
# PREDICTION INTERVAL CONFIGURATIONS
# =============================================================================

@dataclass
class IntervalConfig:
    """Configuration for a prediction interval."""
    minutes: int
    model_weights: Dict[str, float]
    features_type: str
    confidence_factor: float
    use_case: str


INTERVAL_CONFIGS: Dict[PredictionInterval, IntervalConfig] = {
    PredictionInterval.THIRTY_MIN: IntervalConfig(
        minutes=30,
        model_weights={"prophet": 0.2, "lstm": 0.5, "xgboost": 0.3},
        features_type="technical_only",
        confidence_factor=0.85,
        use_case="Scalping, quick entries/exits",
    ),
    PredictionInterval.ONE_HOUR: IntervalConfig(
        minutes=60,
        model_weights={"prophet": 0.3, "lstm": 0.4, "xgboost": 0.3},
        features_type="technical_volume",
        confidence_factor=0.80,
        use_case="Intraday trading",
    ),
    PredictionInterval.FOUR_HOUR: IntervalConfig(
        minutes=240,
        model_weights={"prophet": 0.4, "lstm": 0.3, "xgboost": 0.3},
        features_type="technical_fundamental",
        confidence_factor=0.70,
        use_case="Swing trading",
    ),
    PredictionInterval.DAILY: IntervalConfig(
        minutes=1440,
        model_weights={"prophet": 0.5, "lstm": 0.25, "xgboost": 0.25},
        features_type="all_features",
        confidence_factor=0.60,
        use_case="Position trading",
    ),
}


# =============================================================================
# CORRELATED FACTORS (What moves silver)
# =============================================================================

SILVER_FACTORS = {
    # Inverse correlation - Strong
    "DXY": {
        "name": "US Dollar Index",
        "symbol": "DX-Y.NYB",
        "correlation": "inverse",
        "strength": "strong",
        "description": "Strong inverse correlation - weaker dollar = higher silver",
    },
    "US10Y": {
        "name": "10-Year Treasury Yield",
        "symbol": "^TNX",
        "correlation": "inverse",
        "strength": "moderate",
        "description": "Higher real yields = lower silver",
    },
    # Positive correlation - Strong
    "GOLD": {
        "name": "Gold Spot Price",
        "symbol": "GC=F",
        "correlation": "positive",
        "strength": "strong",
        "description": "Strong positive - precious metals move together",
    },
    # Industrial demand proxies
    "COPPER": {
        "name": "Copper Futures",
        "symbol": "HG=F",
        "correlation": "positive",
        "strength": "moderate",
        "description": "Industrial demand indicator",
    },
    # Volatility
    "VIX": {
        "name": "CBOE Volatility Index",
        "symbol": "^VIX",
        "correlation": "positive",
        "strength": "moderate",
        "description": "Fear gauge - uncertainty can drive silver higher",
    },
    # Risk appetite
    "SPX": {
        "name": "S&P 500",
        "symbol": "^GSPC",
        "correlation": "mixed",
        "strength": "weak",
        "description": "Risk-on/risk-off dynamics",
    },
}


# =============================================================================
# UPSTOX API CONFIGURATION
# =============================================================================

UPSTOX_CONFIG = {
    "auth_url": "https://api.upstox.com/v2/login/authorization/dialog",
    "token_url": "https://api.upstox.com/v2/login/authorization/token",
    "websocket_auth_url": "https://api.upstox.com/v3/feed/market-data-feed/authorize",
    "historical_candle_url": "https://api.upstox.com/v2/historical-candle",
    "instruments_url": "https://assets.upstox.com/market-quote/instruments/exchange/complete.json.gz",

    # Rate limits
    "requests_per_second": 50,
    "requests_per_minute": 500,
    "websocket_max_instruments": 100,

    # Data retention (Upstox limits)
    "retention": {
        "1minute": 30,      # days
        "30minute": 365,    # days
        "day": 365,         # days
        "week": 3650,       # days (10 years)
        "month": 3650,      # days (10 years)
    },
}


# =============================================================================
# TECHNICAL INDICATORS CONFIGURATION
# =============================================================================

TECHNICAL_INDICATORS = {
    "trend": [
        ("SMA", 20),
        ("SMA", 50),
        ("SMA", 200),
        ("EMA", 12),
        ("EMA", 26),
    ],
    "momentum": [
        ("RSI", 14),
        ("MACD", (12, 26, 9)),
        ("STOCH", (14, 3, 3)),
    ],
    "volatility": [
        ("ATR", 14),
        ("BBANDS", (20, 2)),
    ],
    "volume": [
        ("OBV", None),
        ("VOLUME_SMA", 20),
    ],
}


# =============================================================================
# CONFIDENCE INTERVAL PERCENTILES
# =============================================================================

CONFIDENCE_INTERVALS = {
    "ci_50": 0.50,
    "ci_80": 0.80,
    "ci_95": 0.95,
}


# =============================================================================
# MODEL PERFORMANCE THRESHOLDS
# =============================================================================

PERFORMANCE_THRESHOLDS = {
    "min_direction_accuracy": 0.52,  # Must beat random
    "target_direction_accuracy": 0.55,
    "min_ci_80_coverage": 0.75,  # 80% CI should contain at least 75%
    "max_mape": 5.0,  # Maximum acceptable MAPE %
    "retrain_trigger_accuracy_drop": 0.05,  # Retrain if accuracy drops 5%
}
