"""
Prediction model for storing and tracking all predictions.
Includes prediction details, confidence intervals, and verification results.
"""

from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional
from uuid import uuid4

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class Prediction(Base):
    """
    Stores every prediction made by the system.
    Includes prediction details, confidence intervals, and verification results.

    This is the CRITICAL table for tracking prediction accuracy and building trust.
    """

    __tablename__ = "predictions"

    # Primary key
    id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        primary_key=True,
        default=lambda: str(uuid4()),
    )

    # Creation timestamp (primary time column for TimescaleDB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        index=True,
    )

    # ==========================================================================
    # WHAT WAS PREDICTED
    # ==========================================================================
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Contract info (MCX silver has multiple contracts: SILVER, SILVERM, SILVERMIC)
    instrument_key: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        index=True,
        comment="Upstox instrument key (e.g., MCX_FO|451669)",
    )
    contract_type: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        index=True,
        comment="Contract type (SILVER, SILVERM, SILVERMIC)",
    )
    trading_symbol: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Human-readable trading symbol (e.g., SILVERM FUT 27 FEB 26)",
    )

    # ==========================================================================
    # PREDICTION DETAILS
    # ==========================================================================
    prediction_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        comment="When prediction was made",
    )
    target_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
        comment="When prediction is for (prediction_time + interval)",
    )
    current_price: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="Price at prediction time",
    )
    predicted_price: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="Predicted price at target time",
    )
    predicted_direction: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="bullish, bearish, or neutral",
    )
    direction_confidence: Mapped[Decimal] = mapped_column(
        Numeric(5, 4),
        nullable=False,
        comment="Confidence in direction (0.0 to 1.0)",
    )

    # ==========================================================================
    # CONFIDENCE INTERVALS
    # ==========================================================================
    ci_50_lower: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="50% confidence interval lower bound",
    )
    ci_50_upper: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="50% confidence interval upper bound",
    )
    ci_80_lower: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="80% confidence interval lower bound",
    )
    ci_80_upper: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="80% confidence interval upper bound",
    )
    ci_95_lower: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="95% confidence interval lower bound",
    )
    ci_95_upper: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="95% confidence interval upper bound",
    )

    # ==========================================================================
    # MODEL INFORMATION
    # ==========================================================================
    model_weights: Mapped[Dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        comment="Weights used for each model in ensemble",
    )
    features_used: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB,
        nullable=True,
        comment="List of features used for prediction",
    )
    model_version: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Version of the model used",
    )

    # ==========================================================================
    # RESULT TRACKING (Filled after target_time passes)
    # ==========================================================================
    actual_price: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 4),
        nullable=True,
        comment="Actual price at target_time",
    )
    is_direction_correct: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="Did we predict direction correctly?",
    )
    price_error: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 4),
        nullable=True,
        comment="predicted_price - actual_price",
    )
    price_error_percent: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Error as percentage of current_price",
    )
    within_ci_50: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="Was actual price within 50% CI?",
    )
    within_ci_80: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="Was actual price within 80% CI?",
    )
    within_ci_95: Mapped[Optional[bool]] = mapped_column(
        Boolean,
        nullable=True,
        comment="Was actual price within 95% CI?",
    )
    verified_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When result was recorded",
    )
    verification_notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Any notes about verification (e.g., data issues)",
    )

    # ==========================================================================
    # CONSTRAINTS AND INDEXES
    # ==========================================================================
    __table_args__ = (
        UniqueConstraint(
            "asset", "market", "interval", "prediction_time", "target_time",
            name="uq_prediction_asset_market_interval_times"
        ),
        Index(
            "idx_predictions_verification",
            "asset", "market", "interval", "verified_at",
        ),
        Index(
            "idx_predictions_pending",
            "target_time",
            postgresql_where="verified_at IS NULL",
        ),
        Index(
            "idx_predictions_accuracy",
            "asset", "market", "interval", "is_direction_correct",
            postgresql_where="verified_at IS NOT NULL",
        ),
    )

    def __repr__(self) -> str:
        status = "verified" if self.verified_at else "pending"
        result = ""
        if self.is_direction_correct is not None:
            result = " correct" if self.is_direction_correct else " wrong"
        contract = f", contract={self.contract_type}" if self.contract_type else ""
        return (
            f"<Prediction("
            f"id={self.id[:8]}..., "
            f"asset={self.asset}, "
            f"market={self.market}, "
            f"interval={self.interval}{contract}, "
            f"direction={self.predicted_direction}, "
            f"status={status}{result}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "asset": self.asset,
            "market": self.market,
            "interval": self.interval,
            # Contract info
            "instrument_key": self.instrument_key,
            "contract_type": self.contract_type,
            "trading_symbol": self.trading_symbol,
            # Prediction details
            "prediction_time": self.prediction_time.isoformat(),
            "target_time": self.target_time.isoformat(),
            "current_price": float(self.current_price),
            "predicted_price": float(self.predicted_price),
            "predicted_direction": self.predicted_direction,
            "direction_confidence": float(self.direction_confidence),
            "confidence_intervals": {
                "ci_50": {
                    "lower": float(self.ci_50_lower),
                    "upper": float(self.ci_50_upper),
                },
                "ci_80": {
                    "lower": float(self.ci_80_lower),
                    "upper": float(self.ci_80_upper),
                },
                "ci_95": {
                    "lower": float(self.ci_95_lower),
                    "upper": float(self.ci_95_upper),
                },
            },
            "model_weights": self.model_weights,
            "model_version": self.model_version,
            # Verification results (if available)
            "verification": {
                "actual_price": float(self.actual_price) if self.actual_price else None,
                "is_direction_correct": self.is_direction_correct,
                "price_error": float(self.price_error) if self.price_error else None,
                "price_error_percent": float(self.price_error_percent) if self.price_error_percent else None,
                "within_ci_50": self.within_ci_50,
                "within_ci_80": self.within_ci_80,
                "within_ci_95": self.within_ci_95,
                "verified_at": self.verified_at.isoformat() if self.verified_at else None,
            } if self.verified_at else None,
        }

    def verify(
        self,
        actual_price: float,
    ) -> None:
        """
        Verify this prediction against the actual price.
        Call this when target_time has passed.
        """
        self.actual_price = Decimal(str(actual_price))
        self.verified_at = datetime.utcnow()

        # Calculate error
        self.price_error = self.predicted_price - self.actual_price
        self.price_error_percent = (
            (self.price_error / self.current_price) * 100
        )

        # Check direction
        actual_direction = (
            "bullish" if actual_price > float(self.current_price)
            else "bearish" if actual_price < float(self.current_price)
            else "neutral"
        )
        self.is_direction_correct = (
            self.predicted_direction == actual_direction
            or (self.predicted_direction == "neutral" and abs(self.price_error_percent) < Decimal("0.1"))
        )

        # Check confidence intervals
        self.within_ci_50 = self.ci_50_lower <= self.actual_price <= self.ci_50_upper
        self.within_ci_80 = self.ci_80_lower <= self.actual_price <= self.ci_80_upper
        self.within_ci_95 = self.ci_95_lower <= self.actual_price <= self.ci_95_upper

    @property
    def is_verified(self) -> bool:
        """Check if prediction has been verified."""
        return self.verified_at is not None

    @property
    def is_pending(self) -> bool:
        """Check if prediction is pending verification."""
        return self.verified_at is None and self.target_time < datetime.utcnow()
