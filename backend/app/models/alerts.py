"""
Database models for price alerts and trade journal.
"""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, Optional

from sqlalchemy import Column, String, DateTime, Numeric, Boolean, Text, JSON, Enum as SQLEnum
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.models.database import Base


class AlertStatus(enum.Enum):
    """Alert status enum."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    CANCELLED = "cancelled"
    EXPIRED = "expired"


class AlertType(enum.Enum):
    """Alert type enum."""
    PRICE_ABOVE = "price_above"
    PRICE_BELOW = "price_below"
    PERCENT_CHANGE = "percent_change"
    CONFLUENCE = "confluence"


class PriceAlert(Base):
    """
    Price alert model for user-defined price targets.
    """
    __tablename__ = "price_alerts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Alert configuration
    asset = Column(String(20), nullable=False, default="silver")
    market = Column(String(10), nullable=False)  # mcx, comex
    alert_type = Column(SQLEnum(AlertType), nullable=False)
    target_price = Column(Numeric(12, 4), nullable=False)
    current_price_at_creation = Column(Numeric(12, 4), nullable=False)

    # Status tracking
    status = Column(SQLEnum(AlertStatus), default=AlertStatus.ACTIVE, nullable=False)
    triggered_at = Column(DateTime, nullable=True)
    triggered_price = Column(Numeric(12, 4), nullable=True)

    # Optional settings
    note = Column(Text, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    notify_telegram = Column(Boolean, default=True)
    notify_email = Column(Boolean, default=False)

    # User identification (optional, for multi-user support)
    user_id = Column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<PriceAlert {self.id[:8]}: {self.asset}/{self.market} {self.alert_type.value} {self.target_price}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "asset": self.asset,
            "market": self.market,
            "alert_type": self.alert_type.value,
            "target_price": float(self.target_price),
            "current_price_at_creation": float(self.current_price_at_creation),
            "status": self.status.value,
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "triggered_price": float(self.triggered_price) if self.triggered_price else None,
            "note": self.note,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    def check_trigger(self, current_price: float) -> bool:
        """
        Check if alert should be triggered based on current price.
        """
        if self.status != AlertStatus.ACTIVE:
            return False

        target = float(self.target_price)

        if self.alert_type == AlertType.PRICE_ABOVE:
            return current_price >= target
        elif self.alert_type == AlertType.PRICE_BELOW:
            return current_price <= target
        elif self.alert_type == AlertType.PERCENT_CHANGE:
            creation_price = float(self.current_price_at_creation)
            change_pct = abs(current_price - creation_price) / creation_price * 100
            return change_pct >= target

        return False

    def trigger(self, triggered_price: float) -> None:
        """Mark alert as triggered."""
        self.status = AlertStatus.TRIGGERED
        self.triggered_at = datetime.utcnow()
        self.triggered_price = Decimal(str(triggered_price))


class TradeJournal(Base):
    """
    Trade journal model for tracking user trades.
    """
    __tablename__ = "trade_journal"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Trade details
    asset = Column(String(20), nullable=False, default="silver")
    market = Column(String(10), nullable=False)
    trade_type = Column(String(10), nullable=False)  # buy, sell
    quantity = Column(Numeric(12, 4), nullable=True)

    # Prices
    entry_price = Column(Numeric(12, 4), nullable=False)
    exit_price = Column(Numeric(12, 4), nullable=True)
    stop_loss = Column(Numeric(12, 4), nullable=True)
    take_profit = Column(Numeric(12, 4), nullable=True)

    # Timing
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)

    # P&L
    pnl_amount = Column(Numeric(12, 4), nullable=True)
    pnl_percent = Column(Numeric(8, 4), nullable=True)

    # System prediction at entry
    prediction_id = Column(UUID(as_uuid=True), nullable=True)
    prediction_direction = Column(String(10), nullable=True)
    prediction_confidence = Column(Numeric(5, 4), nullable=True)
    followed_prediction = Column(Boolean, nullable=True)  # Did user follow system?

    # Notes and analysis
    notes = Column(Text, nullable=True)
    tags = Column(JSON, nullable=True)  # ["swing", "breakout", etc.]
    emotions = Column(String(50), nullable=True)  # "confident", "fearful", etc.
    lessons_learned = Column(Text, nullable=True)

    # User identification
    user_id = Column(String(50), nullable=True)

    def __repr__(self) -> str:
        return f"<Trade {self.id[:8]}: {self.trade_type} {self.asset} @ {self.entry_price}>"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self.id),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "asset": self.asset,
            "market": self.market,
            "trade_type": self.trade_type,
            "quantity": float(self.quantity) if self.quantity else None,
            "entry_price": float(self.entry_price),
            "exit_price": float(self.exit_price) if self.exit_price else None,
            "stop_loss": float(self.stop_loss) if self.stop_loss else None,
            "take_profit": float(self.take_profit) if self.take_profit else None,
            "entry_time": self.entry_time.isoformat() if self.entry_time else None,
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "pnl_amount": float(self.pnl_amount) if self.pnl_amount else None,
            "pnl_percent": float(self.pnl_percent) if self.pnl_percent else None,
            "prediction_direction": self.prediction_direction,
            "prediction_confidence": float(self.prediction_confidence) if self.prediction_confidence else None,
            "followed_prediction": self.followed_prediction,
            "notes": self.notes,
            "tags": self.tags,
        }

    def calculate_pnl(self) -> None:
        """Calculate P&L when trade is closed."""
        if self.exit_price and self.entry_price:
            entry = float(self.entry_price)
            exit_p = float(self.exit_price)

            if self.trade_type == "buy":
                self.pnl_amount = Decimal(str(exit_p - entry))
                self.pnl_percent = Decimal(str((exit_p - entry) / entry * 100))
            else:  # sell/short
                self.pnl_amount = Decimal(str(entry - exit_p))
                self.pnl_percent = Decimal(str((entry - exit_p) / entry * 100))
