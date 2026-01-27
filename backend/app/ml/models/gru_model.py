"""
GRU (Gated Recurrent Unit) model for time series prediction.
A simpler alternative to LSTM that often performs comparably with fewer parameters.

NOTE: This model predicts percentage returns, not absolute prices.
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


class GRUNetwork(nn.Module):
    """GRU neural network for sequence prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(
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
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRU output
        gru_out, _ = self.gru(x)

        # Use last time step
        last_output = gru_out[:, -1, :]

        # Fully connected layers
        output = self.fc(last_output)

        return output


class GRUModel(BaseModel):
    """
    GRU model for price prediction using returns.

    Strengths:
    - Simpler than LSTM with comparable performance
    - Faster training and inference
    - Fewer parameters to tune
    - Good for shorter sequences

    Limitations:
    - May not capture very long-term dependencies as well as LSTM
    """

    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 20,  # Reduced from 50 to speed up training
    ):
        super().__init__("gru")

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

        self.model: Optional[GRUNetwork] = None
        self.scaler_X = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._last_price: Optional[float] = None

    def _prepare_return_data(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Prepare return-based features and targets."""
        prices = df[target_col].values
        last_price = float(prices[-1])

        # Calculate returns
        returns = np.zeros(len(prices))
        returns[1:] = np.diff(prices) / prices[:-1] * 100

        # Create features from returns
        features = []
        for i in range(len(returns)):
            row_features = []

            # Return lags
            for lag in range(1, min(11, i + 1)):
                row_features.append(returns[i - lag] if i >= lag else 0)

            # Pad if needed
            while len(row_features) < 10:
                row_features.append(0)

            # Rolling statistics
            if i >= 5:
                row_features.append(np.mean(returns[i-4:i+1]))
                row_features.append(np.std(returns[i-4:i+1]) if np.std(returns[i-4:i+1]) > 0 else 0.01)
            else:
                row_features.extend([0, 0.01])

            if i >= 10:
                row_features.append(np.mean(returns[i-9:i+1]))
                row_features.append(np.std(returns[i-9:i+1]) if np.std(returns[i-9:i+1]) > 0 else 0.01)
            else:
                row_features.extend([0, 0.01])

            # Price position (normalized)
            if i >= 10:
                min_p = np.min(prices[i-9:i+1])
                max_p = np.max(prices[i-9:i+1])
                pos = (prices[i] - min_p) / (max_p - min_p + 1e-8)
                row_features.append(pos)
            else:
                row_features.append(0.5)

            features.append(row_features)

        features = np.array(features)

        # Target: next period return
        target_returns = np.zeros(len(returns))
        target_returns[:-1] = returns[1:]

        return features, target_returns, last_price

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU."""
        X, y = [], []
        for i in range(self.sequence_length, len(features) - 1):
            X.append(features[i - self.sequence_length:i])
            y.append(targets[i])
        return np.array(X), np.array(y)

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
    ) -> Dict[str, float]:
        """Train GRU model on returns."""
        logger.info(f"Training GRU model on {len(df)} samples")

        # Prepare data
        features, targets, self._last_price = self._prepare_return_data(df, target_col)

        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        features_scaled = self.scaler_X.fit_transform(features)

        # Create sequences
        X, y = self._create_sequences(features_scaled, targets)

        if len(X) < 50:
            raise ValueError(f"Insufficient data: {len(X)} sequences, need at least 50")

        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(-1).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(-1).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Initialize model
        input_size = X.shape[2]
        self.model = GRUNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

        # Training setup
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t).item()

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= 10:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # Load best model
        if best_model_state:
            self.model.load_state_dict(best_model_state)

        # Calculate metrics
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_val_t).cpu().numpy().flatten()
            y_actual = y_val

        mae = np.mean(np.abs(y_pred - y_actual))
        rmse = np.sqrt(np.mean((y_pred - y_actual) ** 2))

        direction_pred = np.sign(y_pred)
        direction_actual = np.sign(y_actual)
        direction_accuracy = np.mean(direction_pred == direction_actual)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "direction_accuracy": float(direction_accuracy),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "prediction_type": "returns",
        }

        self.is_trained = True
        self.last_trained = datetime.now()

        logger.info(f"GRU training complete. Direction accuracy: {direction_accuracy:.2%}, MAE: {mae:.4f}%")

        return self.training_metrics

    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        current_price: Optional[float] = None,
    ) -> PredictionResult:
        """Make prediction with GRU."""
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare features
        features, _, last_data_price = self._prepare_return_data(df)
        features_scaled = self.scaler_X.transform(features)

        # Get sequence for prediction
        X = features_scaled[-self.sequence_length:].reshape(1, self.sequence_length, -1)
        X_t = torch.FloatTensor(X).to(self.device)

        # Use provided current_price or fall back
        if current_price is None:
            current_price = float(df["close"].iloc[-1])

        # Predict
        self.model.eval()
        with torch.no_grad():
            predicted_return = float(self.model(X_t).cpu().numpy()[0, 0])

        # Estimate uncertainty via dropout (MC Dropout)
        self.model.train()  # Enable dropout
        return_predictions = []
        for _ in range(20):
            with torch.no_grad():
                pred = float(self.model(X_t).cpu().numpy()[0, 0])
                return_predictions.append(pred)
        self.model.eval()

        return_std = float(np.std(return_predictions)) if len(return_predictions) > 1 else abs(predicted_return * 0.5)

        # Convert to price
        predicted_price = current_price * (1 + predicted_return / 100)
        std_dev = current_price * return_std / 100

        # Confidence intervals
        ci_50, ci_80, ci_95 = self.calculate_confidence_intervals(predicted_price, std_dev)

        # Direction
        direction, direction_prob = self.determine_direction(current_price, predicted_price)

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
            features_used=["returns", "rolling_stats"],
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "model_state": self.model.state_dict() if self.model else None,
                "scaler_X": self.scaler_X,
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
            }, f)

        logger.info(f"GRU model saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        config = data["config"]
        self.sequence_length = config["sequence_length"]
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]

        self.scaler_X = data["scaler_X"]
        self.is_trained = data["is_trained"]
        self.last_trained = data["last_trained"]
        self.training_metrics = data["training_metrics"]
        self._last_price = data.get("last_price")

        if data["model_state"]:
            # Recreate model architecture
            input_size = 15  # Based on feature count
            self.model = GRUNetwork(
                input_size=input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)
            self.model.load_state_dict(data["model_state"])
            self.model.eval()

        logger.info(f"GRU model loaded from {path}")
