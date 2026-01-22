"""
LSTM (Long Short-Term Memory) model for sequence-based prediction.
Captures complex temporal patterns in price data.

NOTE: This model predicts percentage returns, not absolute prices.
This works better for commodities like silver where the absolute
price level changes over time.
"""

import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.ml.models.base import BaseModel, PredictionResult

logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """
    LSTM Neural Network for time series prediction.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM output: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the last time step output
        last_output = lstm_out[:, -1, :]

        # Pass through fully connected layers
        output = self.fc(last_output)

        return output


class LSTMModel(BaseModel):
    """
    LSTM model for price prediction.

    Predicts percentage returns instead of absolute prices for better
    performance on commodities.

    Strengths:
    - Captures complex sequential patterns
    - Good for short-term predictions
    - Can learn from multiple features
    - Returns-based prediction normalizes across different price levels

    Limitations:
    - Requires more data to train effectively
    - Can overfit on small datasets
    - Slower to train than traditional models
    """

    def __init__(
        self,
        sequence_length: int = 60,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        device: Optional[str] = None,
    ):
        super().__init__("lstm")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.model: Optional[LSTMNetwork] = None
        self.scaler_X: Optional[Any] = None
        self.feature_columns: List[str] = []
        self._last_price: Optional[float] = None

    def _prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Prepare sequences for LSTM training using returns.

        Returns:
            X: (num_samples, sequence_length, num_features) - feature sequences
            y: (num_samples,) - target returns (percentage)
            last_price: float - last price in the data
        """
        if feature_cols is None:
            # Use OHLCV by default
            feature_cols = ["open", "high", "low", "close", "volume"]
            feature_cols = [c for c in feature_cols if c in df.columns]

        self.feature_columns = feature_cols

        # Get price data
        prices = df[target_col].values
        last_price = float(prices[-1])

        # Calculate percentage returns for target
        # Returns[i] = (Price[i] - Price[i-1]) / Price[i-1] * 100
        target_returns = np.diff(prices) / prices[:-1] * 100

        # Calculate percentage returns for OHLC features
        # This normalizes the features across different price levels
        features_df = df[feature_cols].copy()

        # For price columns, convert to returns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in features_df.columns:
                col_values = features_df[col].values
                col_returns = np.zeros_like(col_values)
                col_returns[1:] = np.diff(col_values) / col_values[:-1] * 100
                features_df[col] = col_returns

        # For volume, convert to percentage change
        if "volume" in features_df.columns:
            vol = features_df["volume"].values
            vol_pct = np.zeros_like(vol, dtype=float)
            vol_pct[1:] = np.diff(vol) / (vol[:-1] + 1e-8) * 100  # Avoid div by zero
            vol_pct = np.clip(vol_pct, -100, 100)  # Clip extreme values
            features_df["volume"] = vol_pct

        features = features_df.values

        # Remove first row (no return for first data point)
        features = features[1:]

        # Normalize features (returns are already on similar scale, but normalization helps)
        from sklearn.preprocessing import StandardScaler

        self.scaler_X = StandardScaler()
        features_scaled = self.scaler_X.fit_transform(features)

        # Create sequences
        # X[i] = features[i-seq_len:i], y[i] = return[i]
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i - self.sequence_length:i])
            y.append(target_returns[i - 1])  # -1 because returns array is one shorter

        return np.array(X), np.array(y), last_price

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train LSTM model on percentage returns.

        Args:
            df: DataFrame with features
            target_col: Column to predict
            validation_split: Fraction for validation
            feature_cols: Feature columns to use

        Returns:
            Training metrics
        """
        logger.info(f"Training LSTM model on {len(df)} samples (using returns)")

        # Prepare data
        X, y, self._last_price = self._prepare_sequences(df, target_col, feature_cols)

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_size = X.shape[2]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Loss and optimizer - use MSE for returns
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t).squeeze()
                val_loss = criterion(val_outputs, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{self.epochs} - "
                    f"Train Loss: {np.mean(train_losses):.6f}, Val Loss: {val_loss:.6f}"
                )

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Calculate final metrics on returns
        self.model.eval()
        with torch.no_grad():
            val_pred_returns = self.model(X_val_t).squeeze().cpu().numpy()
        val_actual_returns = y_val

        # MAE on returns (percentage points)
        mae = np.mean(np.abs(val_pred_returns - val_actual_returns))

        # RMSE on returns
        rmse = np.sqrt(np.mean((val_pred_returns - val_actual_returns) ** 2))

        # MAPE relative to mean return magnitude (returns can be 0, so standard MAPE doesn't work)
        mean_return_mag = np.mean(np.abs(val_actual_returns))
        mape = (mae / mean_return_mag * 100) if mean_return_mag > 0 else 100

        # Direction accuracy - this is what matters most
        direction_actual = np.sign(val_actual_returns)
        direction_pred = np.sign(val_pred_returns)
        direction_accuracy = np.mean(direction_actual == direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),  # Relative to mean return magnitude
            "direction_accuracy": float(direction_accuracy),
            "best_val_loss": float(best_val_loss),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "prediction_type": "returns",
        }

        self.is_trained = True
        self.last_trained = datetime.now()

        logger.info(f"LSTM training complete. Direction accuracy: {direction_accuracy:.2%}, MAE: {mae:.4f}%")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
        n_samples: int = 100,
    ) -> PredictionResult:
        """
        Make prediction with LSTM using Monte Carlo dropout for uncertainty.

        Predicts return and converts back to price using current_price.

        Args:
            df: Recent data with features
            horizon: Periods ahead
            current_price: Current price for conversion and direction
            n_samples: Number of MC samples for uncertainty

        Returns:
            PredictionResult
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare input sequence - convert to returns
        features_df = df[self.feature_columns].copy()

        # Convert price columns to returns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            if col in features_df.columns:
                col_values = features_df[col].values
                col_returns = np.zeros_like(col_values, dtype=float)
                col_returns[1:] = np.diff(col_values) / col_values[:-1] * 100
                features_df[col] = col_returns

        # Convert volume to percentage change
        if "volume" in features_df.columns:
            vol = features_df["volume"].values
            vol_pct = np.zeros_like(vol, dtype=float)
            vol_pct[1:] = np.diff(vol) / (vol[:-1] + 1e-8) * 100
            vol_pct = np.clip(vol_pct, -100, 100)
            features_df["volume"] = vol_pct

        features = features_df.values[1:]  # Skip first row (no return)
        features_scaled = self.scaler_X.transform(features)

        # Take last sequence_length rows
        if len(features_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length + 1} rows for prediction")

        sequence = features_scaled[-self.sequence_length:]
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Use provided current_price or fall back to last price in data
        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        # Monte Carlo dropout for uncertainty estimation
        return_predictions = []
        self.model.train()  # Enable dropout

        with torch.no_grad():
            for _ in range(n_samples):
                pred_return = self.model(X).squeeze().cpu().numpy()
                return_predictions.append(float(pred_return))

        self.model.eval()

        # Calculate statistics on returns
        return_predictions = np.array(return_predictions)
        predicted_return = float(np.mean(return_predictions))
        return_std = float(np.std(return_predictions))

        # Convert return prediction to price
        # If predicted return is +1%, price goes up 1%
        predicted_price = current_price * (1 + predicted_return / 100)

        # Convert return std to price std
        std_dev = current_price * return_std / 100

        # Calculate confidence intervals
        ci_50, ci_80, ci_95 = self.calculate_confidence_intervals(predicted_price, std_dev)

        # Determine direction
        direction, direction_prob = self.determine_direction(current_price, predicted_price)

        # Calculate target time based on interval
        if "timestamp" in df.columns:
            last_time = pd.to_datetime(df["timestamp"].iloc[-1])
        else:
            last_time = datetime.now()

        # Determine time delta based on data frequency
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
            features_used=self.feature_columns + ["returns"],
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "scaler_X": self.scaler_X,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained,
            "training_metrics": self.training_metrics,
            "last_price": self._last_price,
            "config": {
                "sequence_length": self.sequence_length,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }, path)

        logger.info(f"LSTM model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        # Note: weights_only=False is required because we save sklearn scalers
        # This is safe as we only load our own saved models
        data = torch.load(path, map_location=self.device, weights_only=False)

        self.scaler_X = data["scaler_X"]
        self.feature_columns = data["feature_columns"]
        self.is_trained = data["is_trained"]
        self.last_trained = data["last_trained"]
        self.training_metrics = data["training_metrics"]
        self._last_price = data.get("last_price")

        config = data["config"]
        self.sequence_length = config["sequence_length"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]

        if data["model_state"]:
            input_size = len(self.feature_columns)
            self.model = LSTMNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
            self.model.load_state_dict(data["model_state"])

        logger.info(f"LSTM model loaded from {path}")
