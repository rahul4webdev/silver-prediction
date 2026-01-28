"""
Technical indicators for feature engineering.
Calculates RSI, MACD, Bollinger Bands, ATR, and other indicators.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators for price data.

    All methods return the input DataFrame with additional indicator columns.
    """

    @staticmethod
    def sma(df: pd.DataFrame, column: str = "close", periods: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """
        Calculate Simple Moving Averages.

        Args:
            df: DataFrame with price data
            column: Column to calculate SMA for
            periods: List of periods

        Returns:
            DataFrame with SMA columns added
        """
        for period in periods:
            df[f"sma_{period}"] = df[column].rolling(window=period).mean()
        return df

    @staticmethod
    def ema(df: pd.DataFrame, column: str = "close", periods: List[int] = [12, 26]) -> pd.DataFrame:
        """
        Calculate Exponential Moving Averages.
        """
        for period in periods:
            df[f"ema_{period}"] = df[column].ewm(span=period, adjust=False).mean()
        return df

    @staticmethod
    def rsi(df: pd.DataFrame, column: str = "close", period: int = 14) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = df[column].diff()

        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        return df

    @staticmethod
    def macd(
        df: pd.DataFrame,
        column: str = "close",
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal)
        Histogram = MACD Line - Signal Line
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()

        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        return df

    @staticmethod
    def bollinger_bands(
        df: pd.DataFrame,
        column: str = "close",
        period: int = 20,
        std_dev: float = 2.0,
    ) -> pd.DataFrame:
        """
        Calculate Bollinger Bands.

        Middle Band = SMA(period)
        Upper Band = Middle Band + (std_dev * StdDev)
        Lower Band = Middle Band - (std_dev * StdDev)
        """
        sma = df[column].rolling(window=period).mean()
        std = df[column].rolling(window=period).std()

        df["bb_middle"] = sma
        df["bb_upper"] = sma + (std_dev * std)
        df["bb_lower"] = sma - (std_dev * std)
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
        df["bb_percent"] = (df[column] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

        return df

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average True Range (volatility indicator).

        True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
        ATR = SMA(True Range, period)
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f"atr_{period}"] = true_range.rolling(window=period).mean()

        return df

    @staticmethod
    def stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3,
        smooth_k: int = 3,
    ) -> pd.DataFrame:
        """
        Calculate Stochastic Oscillator.

        %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        %D = SMA(%K, d_period)
        """
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()

        stoch_k = ((df["close"] - low_min) / (high_max - low_min)) * 100
        df["stoch_k"] = stoch_k.rolling(window=smooth_k).mean()
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()

        return df

    @staticmethod
    def obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate On-Balance Volume.

        OBV increases when close > previous close
        OBV decreases when close < previous close
        """
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        df["obv"] = obv
        return df

    @staticmethod
    def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Volume Simple Moving Average."""
        df[f"volume_sma_{period}"] = df["volume"].rolling(window=period).mean()
        df["volume_ratio"] = df["volume"] / df[f"volume_sma_{period}"]
        return df

    @staticmethod
    def price_momentum(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate price momentum (rate of change).

        Momentum = (Price / Price_n_periods_ago - 1) * 100
        """
        for period in periods:
            df[f"momentum_{period}"] = (df["close"] / df["close"].shift(period) - 1) * 100
        return df

    @staticmethod
    def support_resistance(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Identify support and resistance levels using rolling high/low.
        """
        df["resistance"] = df["high"].rolling(window=window).max()
        df["support"] = df["low"].rolling(window=window).min()
        df["resistance_distance"] = (df["resistance"] - df["close"]) / df["close"] * 100
        df["support_distance"] = (df["close"] - df["support"]) / df["close"] * 100

        return df

    @staticmethod
    def higher_highs_lower_lows(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        Detect higher highs and lower lows patterns.
        """
        # Higher high: current high > previous lookback highs
        df["higher_high"] = df["high"] > df["high"].rolling(window=lookback).max().shift(1)

        # Lower low: current low < previous lookback lows
        df["lower_low"] = df["low"] < df["low"].rolling(window=lookback).min().shift(1)

        return df

    @staticmethod
    def returns(df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """
        Calculate percentage returns over various periods.
        Returns are a key feature for predicting future price movements.
        """
        for period in periods:
            df[f"return_{period}"] = df["close"].pct_change(period) * 100
        return df

    @staticmethod
    def volatility(df: pd.DataFrame, periods: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Calculate historical volatility (rolling standard deviation of returns).
        """
        returns = df["close"].pct_change()
        for period in periods:
            df[f"volatility_{period}"] = returns.rolling(window=period).std() * 100
        return df

    @staticmethod
    def price_position(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Calculate where price is relative to recent range (0-1).
        Useful for mean reversion and breakout detection.
        """
        high_max = df["high"].rolling(window=period).max()
        low_min = df["low"].rolling(window=period).min()
        df["price_position"] = (df["close"] - low_min) / (high_max - low_min)
        df["price_from_high"] = (high_max - df["close"]) / df["close"] * 100
        df["price_from_low"] = (df["close"] - low_min) / df["close"] * 100
        return df

    @staticmethod
    def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate Average Directional Index (ADX) - trend strength indicator.
        ADX > 25 indicates strong trend, < 20 indicates weak/no trend.
        """
        high = df["high"]
        low = df["low"]
        close = df["close"]

        # +DM and -DM
        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed averages
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di

        return df

    @staticmethod
    def gap_detection(df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect price gaps between sessions (overnight gaps).
        Gaps often indicate strong momentum.
        """
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1) * 100
        df["gap_up"] = (df["gap"] > 0.1).astype(int)
        df["gap_down"] = (df["gap"] < -0.1).astype(int)
        return df

    @staticmethod
    def range_features(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate range-based features (ATR ratio, range expansion).
        """
        df["range"] = df["high"] - df["low"]
        df["range_pct"] = df["range"] / df["close"] * 100
        df["avg_range"] = df["range"].rolling(window=period).mean()
        df["range_expansion"] = df["range"] / df["avg_range"]
        return df

    @staticmethod
    def candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candlestick pattern features.
        """
        body = df["close"] - df["open"]
        full_range = df["high"] - df["low"]

        # Body size relative to range
        df["body_size"] = body.abs() / full_range.replace(0, np.nan)

        # Upper and lower wicks
        df["upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / full_range.replace(0, np.nan)
        df["lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / full_range.replace(0, np.nan)

        # Bullish/bearish candle
        df["bullish_candle"] = (body > 0).astype(int)

        return df

    @staticmethod
    def vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Volume Weighted Average Price.

        VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        """
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return df

    @classmethod
    def calculate_all(
        cls,
        df: pd.DataFrame,
        include_volume: bool = True,
        include_momentum: bool = True,
        max_lookback: int = None,
    ) -> pd.DataFrame:
        """
        Calculate all technical indicators.

        Args:
            df: DataFrame with OHLCV data
            include_volume: Include volume-based indicators
            include_momentum: Include momentum indicators
            max_lookback: Maximum lookback period to use (auto-detected if None)

        Returns:
            DataFrame with all indicator columns
        """
        df = df.copy()
        data_len = len(df)

        # Ensure required columns exist
        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Auto-detect max lookback based on data length
        # We need to keep at least 50% of data after dropna
        if max_lookback is None:
            max_lookback = min(200, data_len // 2)

        # Adapt SMA periods based on data length
        if max_lookback >= 200:
            sma_periods = [20, 50, 200]
        elif max_lookback >= 50:
            sma_periods = [10, 20, 50]
        else:
            sma_periods = [5, 10, 20]

        # Trend indicators with adaptive periods
        df = cls.sma(df, periods=sma_periods)
        df = cls.ema(df, periods=[12, 26])

        # Momentum indicators
        df = cls.rsi(df)
        df = cls.macd(df)
        df = cls.stochastic(df)

        if include_momentum:
            df = cls.price_momentum(df)

        # Volatility indicators
        df = cls.bollinger_bands(df)
        df = cls.atr(df)
        df = cls.volatility(df)

        # Volume indicators
        if include_volume and "volume" in df.columns:
            df = cls.obv(df)
            df = cls.volume_sma(df)
            df = cls.vwap(df)

        # Price action
        df = cls.support_resistance(df)
        df = cls.higher_highs_lower_lows(df)

        # Returns and price position
        df = cls.returns(df)
        df = cls.price_position(df)

        # Trend strength
        df = cls.adx(df)

        # Range and gap features
        df = cls.range_features(df)
        df = cls.gap_detection(df)

        # Candlestick patterns
        df = cls.candle_patterns(df)

        return df

    @staticmethod
    def get_feature_columns() -> List[str]:
        """Get list of all feature column names."""
        return [
            # Trend
            "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26",

            # Momentum
            "rsi_14",
            "macd", "macd_signal", "macd_hist",
            "stoch_k", "stoch_d",
            "momentum_5", "momentum_10", "momentum_20",

            # Volatility
            "bb_middle", "bb_upper", "bb_lower", "bb_width", "bb_percent",
            "atr_14",
            "volatility_5", "volatility_10", "volatility_20",

            # Volume
            "obv", "volume_sma_20", "volume_ratio", "vwap",

            # Price action
            "resistance", "support",
            "resistance_distance", "support_distance",
            "higher_high", "lower_low",

            # Returns
            "return_1", "return_5", "return_10", "return_20",

            # Price position
            "price_position", "price_from_high", "price_from_low",

            # Trend strength (ADX)
            "adx", "plus_di", "minus_di",

            # Range features
            "range", "range_pct", "avg_range", "range_expansion",

            # Gap detection
            "gap", "gap_up", "gap_down",

            # Candlestick patterns
            "body_size", "upper_wick", "lower_wick", "bullish_candle",
        ]


def add_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to add all technical features to a DataFrame.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with technical indicator columns
    """
    return TechnicalIndicators.calculate_all(df)
