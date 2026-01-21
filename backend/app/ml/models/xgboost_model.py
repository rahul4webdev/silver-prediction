"""
XGBoost model for feature-based prediction.
Provides interpretable feature importance and handles non-linear relationships.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from app.ml.models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost model for price prediction.

    Strengths:
    - Handles non-linear relationships
    - Provides feature importance
    - Fast training and inference
    - Robust to overfitting with proper tuning

    Limitations:
    - Doesn't capture sequential patterns directly
    - Requires feature engineering
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 1,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        lookback_periods: int = 10,
    ):
        super().__init__("xgboost")

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.lookback_periods = lookback_periods

        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_columns: List[str] = []
        self.feature_importance: Dict[str, float] = {}

    def _create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ) -> pd.DataFrame:
        """
        Create lag features for prediction.
        """
        result = df.copy()

        # Price lags
        for i in range(1, self.lookback_periods + 1):
            result[f"close_lag_{i}"] = result[target_col].shift(i)
            result[f"return_lag_{i}"] = result[target_col].pct_change(i)

        # Rolling statistics
        for window in [5, 10, 20]:
            result[f"rolling_mean_{window}"] = result[target_col].rolling(window).mean()
            result[f"rolling_std_{window}"] = result[target_col].rolling(window).std()
            result[f"rolling_min_{window}"] = result[target_col].rolling(window).min()
            result[f"rolling_max_{window}"] = result[target_col].rolling(window).max()

        # Price relative to moving averages
        if "sma_20" in df.columns:
            result["price_to_sma_20"] = result[target_col] / result["sma_20"]
        if "sma_50" in df.columns:
            result["price_to_sma_50"] = result[target_col] / result["sma_50"]

        # Volatility features
        result["daily_range"] = (result["high"] - result["low"]) / result[target_col]
        result["daily_return"] = result[target_col].pct_change()

        # Time features (if timestamp available)
        if "timestamp" in result.columns:
            ts = pd.to_datetime(result["timestamp"])
            result["hour"] = ts.dt.hour
            result["day_of_week"] = ts.dt.dayofweek
            result["day_of_month"] = ts.dt.day
            result["month"] = ts.dt.month

        return result

    def _prepare_features(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        is_training: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features for XGBoost.
        """
        # Create lag features
        df_features = self._create_lag_features(df, target_col)

        # Drop rows with NaN (from lag creation)
        df_features = df_features.dropna()

        # Select feature columns (exclude target and non-features)
        exclude_cols = ["timestamp", "date", "id", target_col]
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        # Filter to numeric columns only
        feature_cols = [c for c in feature_cols if df_features[c].dtype in ["float64", "int64", "float32", "int32"]]

        if is_training:
            self.feature_columns = feature_cols

        # Extract features and target
        X = df_features[self.feature_columns].values
        y = df_features[target_col].values

        return X, y

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """
        Train XGBoost model.

        Args:
            df: DataFrame with features
            target_col: Column to predict
            validation_split: Fraction for validation

        Returns:
            Training metrics
        """
        logger.info(f"Training XGBoost model on {len(df)} samples")

        # Prepare features
        X, y = self._prepare_features(df, target_col, is_training=True)

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        split_idx = int(len(X_scaled) * (1 - validation_split))
        X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Create and train model
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=42,
            n_jobs=-1,
        )

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Get feature importance
        importance = self.model.feature_importances_
        self.feature_importance = {
            col: float(imp) for col, imp in zip(self.feature_columns, importance)
        }

        # Calculate validation metrics
        y_pred = self.model.predict(X_val)

        mae = np.mean(np.abs(y_pred - y_val))
        rmse = np.sqrt(np.mean((y_pred - y_val) ** 2))
        mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100

        # Direction accuracy
        direction_actual = np.sign(np.diff(y_val))
        direction_pred = np.sign(np.diff(y_pred))
        direction_accuracy = np.mean(direction_actual == direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "n_features": len(self.feature_columns),
        }

        self.is_trained = True
        self.last_trained = datetime.now()

        logger.info(f"XGBoost training complete. MAPE: {mape:.2f}%, Direction: {direction_accuracy:.2%}")
        logger.info(f"Top 5 features: {sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]}")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
        n_estimators_for_std: int = 50,
    ) -> PredictionResult:
        """
        Make prediction with XGBoost.

        Uses early stopping iterations to estimate uncertainty.

        Args:
            df: Recent data with features
            horizon: Periods ahead (note: XGBoost predicts next period)
            current_price: Current price for direction
            n_estimators_for_std: Number of estimators for std estimation

        Returns:
            PredictionResult
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare features
        X, _ = self._prepare_features(df, is_training=False)
        X_scaled = self.scaler.transform(X)

        # Get last sample for prediction
        X_last = X_scaled[-1:, :]

        # Main prediction
        predicted_price = float(self.model.predict(X_last)[0])

        # Estimate uncertainty using different numbers of trees
        predictions = []
        for n_trees in range(max(10, self.n_estimators - n_estimators_for_std), self.n_estimators + 1, 5):
            pred = float(self.model.predict(X_last, iteration_range=(0, n_trees))[0])
            predictions.append(pred)

        std_dev = float(np.std(predictions)) if len(predictions) > 1 else abs(predicted_price * 0.01)

        # Calculate confidence intervals
        ci_50, ci_80, ci_95 = self.calculate_confidence_intervals(predicted_price, std_dev)

        # Determine direction
        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        direction, direction_prob = self.determine_direction(current_price, predicted_price)

        # Calculate target time
        if "timestamp" in df.columns:
            last_time = pd.to_datetime(df["timestamp"].iloc[-1])
        else:
            last_time = datetime.now()

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
            features_used=self.feature_columns[:10],  # Top 10 features
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
                "config": {
                    "n_estimators": self.n_estimators,
                    "max_depth": self.max_depth,
                    "learning_rate": self.learning_rate,
                    "lookback_periods": self.lookback_periods,
                },
            }, f)

        logger.info(f"XGBoost model saved to {path}")

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

        config = data["config"]
        self.n_estimators = config["n_estimators"]
        self.max_depth = config["max_depth"]
        self.learning_rate = config["learning_rate"]
        self.lookback_periods = config["lookback_periods"]

        logger.info(f"XGBoost model loaded from {path}")
