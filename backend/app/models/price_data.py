"""
Price data model for storing OHLCV data from MCX and COMEX.
Optimized for time-series queries using TimescaleDB.
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
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.models.database import Base


class PriceData(Base):
    """
    Stores OHLCV (Open, High, Low, Close, Volume) price data.
    This is the primary table for historical and real-time price data.
    """

    __tablename__ = "price_data"

    # Primary key - composite of asset, market, interval, timestamp
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Asset identification
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Timestamp (primary time column for TimescaleDB)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    # OHLCV Data
    open: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    high: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    low: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    close: Mapped[Decimal] = mapped_column(Numeric(14, 4), nullable=False)
    volume: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)

    # Additional fields
    open_interest: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    vwap: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
    )
    source: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="upstox",
    )

    # Unique constraint to prevent duplicates
    __table_args__ = (
        UniqueConstraint(
            "asset", "market", "interval", "timestamp",
            name="uq_price_data_asset_market_interval_timestamp"
        ),
        Index(
            "idx_price_data_lookup",
            "asset", "market", "interval", "timestamp",
        ),
        Index(
            "idx_price_data_timestamp_desc",
            timestamp.desc(),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<PriceData("
            f"asset={self.asset}, "
            f"market={self.market}, "
            f"interval={self.interval}, "
            f"timestamp={self.timestamp}, "
            f"close={self.close}"
            f")>"
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "asset": self.asset,
            "market": self.market,
            "interval": self.interval,
            "timestamp": self.timestamp.isoformat(),
            "open": float(self.open),
            "high": float(self.high),
            "low": float(self.low),
            "close": float(self.close),
            "volume": self.volume,
            "open_interest": self.open_interest,
            "vwap": float(self.vwap) if self.vwap else None,
            "source": self.source,
        }

    @classmethod
    def from_candle(
        cls,
        asset: str,
        market: str,
        interval: str,
        timestamp: datetime,
        open_price: float,
        high_price: float,
        low_price: float,
        close_price: float,
        volume: int,
        source: str = "upstox",
        open_interest: Optional[int] = None,
        vwap: Optional[float] = None,
    ) -> "PriceData":
        """Create PriceData from candle data."""
        return cls(
            asset=asset,
            market=market,
            interval=interval,
            timestamp=timestamp,
            open=Decimal(str(open_price)),
            high=Decimal(str(high_price)),
            low=Decimal(str(low_price)),
            close=Decimal(str(close_price)),
            volume=volume,
            source=source,
            open_interest=open_interest,
            vwap=Decimal(str(vwap)) if vwap else None,
        )
