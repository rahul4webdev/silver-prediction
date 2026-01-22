"""
Random Forest model for price prediction.
Robust ensemble method that handles non-linear relationships well.

NOTE: This model predicts percentage returns, not absolute prices.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from app.ml.models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest model for price prediction.

    Strengths:
    - Robust to overfitting
    - Handles non-linear relationships
    - Provides feature importance
    - No need for extensive hyperparameter tuning
    - Works well with limited data

    Limitations:
    - May not extrapolate well beyond training data
    - Can be slower for very large datasets
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        lookback_periods: int = 15,
    ):
        super().__init__("random_forest")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.lookback_periods = lookback_periods

        self.model: Optional[RandomForestRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}
        self._last_price: Optional[float] = None

    def _create_return_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ) -> Tuple[pd.DataFrame, np.ndarray, float]:
        """Create return-based features for prediction."""
        result = df.copy()
        prices = result[target_col].values
        last_price = float(prices[-1])

        # Calculate returns
        returns = np.zeros(len(prices))
        returns[1:] = np.diff(prices) / prices[:-1] * 100

        # Target returns (next period)
        target_returns = np.zeros(len(prices))
        target_returns[:-1] = returns[1:]

        # Return lags
        for i in range(1, self.lookback_periods + 1):
            result[f"return_lag_{i}"] = pd.Series(returns).shift(i)

        # Rolling statistics on returns
        returns_series = pd.Series(returns)
        for window in [5, 10, 20]:
            result[f"return_mean_{window}"] = returns_series.rolling(window).mean()
            result[f"return_std_{window}"] = returns_series.rolling(window).std()
            result[f"return_min_{window}"] = returns_series.rolling(window).min()
            result[f"return_max_{window}"] = returns_series.rolling(window).max()
            result[f"return_skew_{window}"] = returns_series.rolling(window).skew()

        # Price momentum
        for window in [5, 10, 20]:
            result[f"momentum_{window}"] = result[target_col].pct_change(window) * 100

        # Price position relative to range
        for window in [10, 20]:
            rolling_min = result[target_col].rolling(window).min()
            rolling_max = result[target_col].rolling(window).max()
            rolling_range = rolling_max - rolling_min
            result[f"price_position_{window}"] = (result[target_col] - rolling_min) / (rolling_range + 1e-8)

        # Volatility features
        result["daily_range_pct"] = (result["high"] - result["low"]) / result[target_col] * 100
        result["body_size_pct"] = abs(result["close"] - result["open"]) / result[target_col] * 100
        result["upper_shadow_pct"] = (result["high"] - result[["open", "close"]].max(axis=1)) / result[target_col] * 100
        result["lower_shadow_pct"] = (result[["open", "close"]].min(axis=1) - result["low"]) / result[target_col] * 100

        # Volume features (if available)
        if "volume" in result.columns:
            result["volume_change_pct"] = result["volume"].pct_change() * 100
            result["volume_change_pct"] = result["volume_change_pct"].clip(-100, 100)
            result["volume_ma_ratio"] = result["volume"] / result["volume"].rolling(10).mean()

        # Technical indicators (if available)
        if "rsi_14" in result.columns:
            result["rsi_centered"] = result["rsi_14"] - 50

        if "macd" in result.columns:
            result["macd_pct"] = result["macd"] / result[target_col] * 100

        # Time features
        if "timestamp" in result.columns:
            ts = pd.to_datetime(result["timestamp"])
            result["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
            result["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
            result["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
            result["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        return result, target_returns, last_price

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        is_training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Prepare features for Random Forest."""
        # Create features
        df_features, target_returns, last_price = self._create_return_features(df, target_col)

        # Drop rows with NaN
        valid_mask = ~df_features.isna().any(axis=1)

        # Select feature columns
        exclude_cols = [
            "timestamp", "date", "id", target_col,
            "open", "high", "low", "close", "volume",
            "hour", "day_of_week", "day_of_month", "month",
        ]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        # Filter to numeric columns
        feature_cols = [c for c in feature_cols if df_features[c].dtype in ["float64", "int64", "float32", "int32"]]

        if is_training:
            self.feature_columns = feature_cols

        # Apply valid mask
        df_features = df_features[valid_mask]
        target_returns = target_returns[valid_mask.values]

        X = df_features[self.feature_columns].values
        y = target_returns

        return X, y, last_price

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """Train Random Forest model on returns."""
        logger.info(f"Training Random Forest model on {len(df)} samples")

        # Prepare features
        X, y, self._last_price = self._prepare_features(df, target_col, is_training=True)

        # Remove last row (target is unknown)
        X = X[:-1]
        y = y[:-1]

        # Handle any remaining NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create and train model
        self.model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(X_train, y_train)

        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            col: float(imp) for col, imp in zip(self.feature_columns, importance)
        }

        # Calculate validation metrics
        y_pred = self.model.predict(X_val)

        mae = np.mean(np.abs(y_pred - y_val))
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))

        # Direction accuracy
        direction_actual = np.sign(y_val)
        direction_pred = np.sign(y_pred)
        direction_accuracy = np.mean(direction_actual == direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "direction_accuracy": float(direction_accuracy),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": len(self.feature_columns),
            "prediction_type": "returns",
        }

        self.is_trained = True
        self.last_trained = datetime.now()

        logger.info(f"Random Forest training complete. Direction accuracy: {direction_accuracy:.2%}, MAE: {mae:.4f}%")
        logger.info(f"Top 5 features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
    ) -> PredictionResult:
        """Make prediction with Random Forest."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare features
        X, _, last_data_price = self._prepare_features(df, is_training=False)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X)

        # Get last sample
        X_last = X_scaled[-1:]

        # Use provided current_price or fall back
        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        # Predict return
        predicted_return = float(self.model.predict(X_last)[0])

        # Estimate uncertainty using individual tree predictions
        tree_predictions = np.array([tree.predict(X_last)[0] for tree in self.model.estimators_])
        return_std = float(np.std(tree_predictions))

        # Convert to price
        predicted_price = current_price * (1 + predicted_return / 100)
        std_dev = current_price * return_std / 100

        # Confidence intervals
        ci_50, ci_80, ci_95 = self.calculate_confidence_intervals(predicted_price, std_dev)

        # Direction
        direction, direction_prob = self.determine_direction(current_price, predicted_price)

        # Enhance direction probability based on tree agreement
        bullish_trees = np.sum(tree_predictions > 0)
        bearish_trees = np.sum(tree_predictions < 0)
        tree_agreement = max(bullish_trees, bearish_trees) / len(tree_predictions)
        direction_prob = (direction_prob + tree_agreement) / 2

        # Target time
        if "timestamp" in df.columns:
            last_time = pd.to_datetime(df["timestamp"].iloc[-1])
        else:
            last_time = datetime.now()

        if len(df) > 1 and "timestamp" in df.columns:
            time_diff = pd.to_datetime(df["timestamp"]).diff().median()
            if pd.notna(time_diff):
                target_time = last_time + time_diff * horizon
            else:
                target_time = last_time + timedelta(hours=horizon)
        else:
            target_time = last_time + timedelta(hours=horizon)

        return PredictionResult(
            predicted_price=predicted_price,
            std_dev=std_dev,
            ci_50=ci_50,
            ci_80=ci_80,
            ci_95=ci_95,
            direction=direction,
            direction_probability=direction_prob,
            model_name=self.model_name,
            prediction_time=datetime.now(),
            target_time=target_time,
            features_used=list(self.feature_importance.keys())[:10],
        )

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N feature importances."""
        sorted_importance = sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return dict(sorted_importance[:top_n])

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "feature_importance": self.feature_importance,
                "is_trained": self.is_trained,
                "last_trained": self.last_trained,
                "training_metrics": self.training_metrics,
                "last_price": self._last_price,
                "config": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "lookback_periods": self.lookback_periods,
                },
            }, f)

        logger.info(f"Random Forest model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.feature_importance = data["feature_importance"]
        self.is_trained = data["is_trained"]
        self.last_trained = data["last_trained"]
        self.training_metrics = data["training_metrics"]
        self._last_price = data.get("last_price")

        config = data["config"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]
        self.lookback_periods = config["lookback_periods"]

        logger.info(f"Random Forest model loaded from {path}")
