"""
Market factors model for storing correlated market data.
Used for fundamental analysis and multi-factor predictions.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import (
    BigInteger,
    DateTime,
    Index,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class MarketFactor(Base):
    """
    Stores correlated market factor data (DXY, Gold, VIX, etc.).
    Used for fundamental analysis in predictions.
    """

    __tablename__ = "market_factors"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Factor identification
    factor_code: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        index=True,
        comment="Factor code (e.g., DXY, GOLD, VIX)",
    )
    symbol: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        comment="Yahoo Finance symbol",
    )

    # Timestamp (primary time column for TimescaleDB)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # Price/Value data
    value: Mapped[Decimal] = mapped_column(
        Numeric(14, 4),
        nullable=False,
        comment="Factor value at timestamp",
    )
    change: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(10, 4),
        nullable=True,
        comment="Change from previous period",
    )
    change_percent: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(8, 4),
        nullable=True,
        comment="Percentage change from previous period",
    )

    # Additional data for indices
    open: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    high: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    low: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Metadata
    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="yahoo",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )

    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint(
            "factor_code", "timestamp",
            name="uq_market_factor_code_timestamp"
        ),
        Index(
            "idx_market_factors_lookup",
            "factor_code", "timestamp",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<MarketFactor("
            f"factor_code={self.factor_code}, "
            f"timestamp={self.timestamp}, "
            f"value={self.value}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "factor_code": self.factor_code,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "value": float(self.value),
            "change": float(self.change) if self.change else None,
            "change_percent": float(self.change_percent) if self.change_percent else None,
            "source": self.source,
        }


class UpstoxToken(Base):
    """
    Stores Upstox OAuth tokens for API access.
    Tokens are valid for 1 year but need to be refreshed.
    """

    __tablename__ = "upstox_tokens"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Token data
    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    token_type: Mapped[str] = mapped_column(String(20), nullable=False, default="Bearer")

    # Token metadata
    expires_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Status
    is_active: Mapped[bool] = mapped_column(
        default=True,
        comment="Whether this token is currently active",
    )

    def __repr__(self) -> str:
        return f"<UpstoxToken(id={self.id}, active={self.is_active})>"

    @property
    def is_expired(self) -> bool:
        """Check if token is expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
