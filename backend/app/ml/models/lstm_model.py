"""
LSTM (Long Short-Term Memory) model for sequence-based prediction.
Captures complex temporal patterns in price data.
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

    Strengths:
    - Captures complex sequential patterns
    - Good for short-term predictions
    - Can learn from multiple features

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
        self.scaler_y: Optional[Any] = None
        self.feature_columns: List[str] = []

    def _prepare_sequences(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.

        Returns:
            X: (num_samples, sequence_length, num_features)
            y: (num_samples,)
        """
        if feature_cols is None:
            # Use OHLCV by default
            feature_cols = ["open", "high", "low", "close", "volume"]
            feature_cols = [c for c in feature_cols if c in df.columns]

        self.feature_columns = feature_cols

        # Extract features
        features = df[feature_cols].values
        target = df[target_col].values

        # Normalize features
        from sklearn.preprocessing import MinMaxScaler

        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

        features_scaled = self.scaler_X.fit_transform(features)
        target_scaled = self.scaler_y.fit_transform(target.reshape(-1, 1)).flatten()

        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i - self.sequence_length:i])
            y.append(target_scaled[i])

        return np.array(X), np.array(y)

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "close",
        validation_split: float = 0.2,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Train LSTM model.

        Args:
            df: DataFrame with features
            target_col: Column to predict
            validation_split: Fraction for validation
            feature_cols: Feature columns to use

        Returns:
            Training metrics
        """
        logger.info(f"Training LSTM model on {len(df)} samples")

        # Prepare data
        X, y = self._prepare_sequences(df, target_col, feature_cols)

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

        # Loss and optimizer
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

        # Calculate final metrics
        self.model.eval()
        with torch.no_grad():
            val_pred = self.model(X_val_t).squeeze().cpu().numpy()

        # Inverse transform predictions
        val_pred_inv = self.scaler_y.inverse_transform(val_pred.reshape(-1, 1)).flatten()
        y_val_inv = self.scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

        mae = np.mean(np.abs(val_pred_inv - y_val_inv))
        rmse = np.sqrt(np.mean((val_pred_inv - y_val_inv) ** 2))
        mape = np.mean(np.abs((y_val_inv - val_pred_inv) / y_val_inv)) * 100

        # Direction accuracy
        direction_actual = np.sign(np.diff(y_val_inv))
        direction_pred = np.sign(np.diff(val_pred_inv))
        direction_accuracy = np.mean(direction_actual == direction_pred)

        self.training_metrics = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
            "best_val_loss": float(best_val_loss),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
        }

        self.is_trained = True
        self.last_trained = datetime.now()

        logger.info(f"LSTM training complete. MAPE: {mape:.2f}%, Direction: {direction_accuracy:.2%}")

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

        Args:
            df: Recent data with features
            horizon: Periods ahead
            current_price: Current price for direction
            n_samples: Number of MC samples for uncertainty

        Returns:
            PredictionResult
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model must be trained before prediction")

        # Prepare input sequence
        features = df[self.feature_columns].values
        features_scaled = self.scaler_X.transform(features)

        # Take last sequence_length rows
        if len(features_scaled) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows for prediction")

        sequence = features_scaled[-self.sequence_length:]
        X = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # Monte Carlo dropout for uncertainty estimation
        predictions = []
        self.model.train()  # Enable dropout

        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.model(X).squeeze().cpu().numpy()
                pred_inv = self.scaler_y.inverse_transform([[pred]])[0, 0]
                predictions.append(pred_inv)

        self.model.eval()

        # Calculate statistics
        predictions = np.array(predictions)
        predicted_price = float(np.mean(predictions))
        std_dev = float(np.std(predictions))

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
            features_used=self.feature_columns,
        )

    def save(self, path: str) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_state": self.model.state_dict() if self.model else None,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "feature_columns": self.feature_columns,
            "is_trained": self.is_trained,
            "last_trained": self.last_trained,
            "training_metrics": self.training_metrics,
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
        data = torch.load(path, map_location=self.device)

        self.scaler_X = data["scaler_X"]
        self.scaler_y = data["scaler_y"]
        self.feature_columns = data["feature_columns"]
        self.is_trained = data["is_trained"]
        self.last_trained = data["last_trained"]
        self.training_metrics = data["training_metrics"]

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
