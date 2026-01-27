"""
Price data model for storing OHLCV data from MCX and COMEX.
Optimized for time-series queries using TimescaleDB.

Updated to support per-contract data for MCX (SILVER, SILVERM, SILVERMIC).
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

    For MCX market, data is stored per-contract with instrument_key differentiation.
    This allows separate training data for SILVER (30kg), SILVERM (5kg), SILVERMIC (1kg).
    """

    __tablename__ = "price_data"

    # Primary key - composite of asset, market, interval, timestamp
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Asset identification
    asset: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    market: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    interval: Mapped[str] = mapped_column(String(10), nullable=False, index=True)

    # Contract identification (for MCX - multiple contracts per asset)
    # For COMEX, these remain NULL as there's only one contract
    instrument_key: Mapped[Optional[str]] = mapped_column(
        String(50), nullable=True, index=True,
        comment="Upstox instrument key (e.g., MCX_FO|451666)"
    )
    contract_type: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True,
        comment="Contract type: SILVER, SILVERM, SILVERMIC"
    )
    trading_symbol: Mapped[Optional[str]] = mapped_column(
        String(100), nullable=True,
        comment="Human-readable symbol (e.g., SILVERM FUT 27 FEB 26)"
    )
    expiry: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="Contract expiry date"
    )

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

    # Unique constraints and indexes
    # NOTE: We keep the old constraint for backward compatibility during transition.
    # The new constraint includes instrument_key for per-contract uniqueness.
    # After migration, the old constraint can be dropped.
    __table_args__ = (
        # OLD constraint: kept for backward compatibility with existing code
        # This allows COMEX data (NULL instrument_key) to continue working
        UniqueConstraint(
            "asset", "market", "interval", "timestamp",
            name="uq_price_data_asset_market_interval_timestamp"
        ),
        # Index for general queries (COMEX or when contract doesn't matter)
        Index(
            "idx_price_data_lookup",
            "asset", "market", "interval", "timestamp",
        ),
        Index(
            "idx_price_data_timestamp_desc",
            timestamp.desc(),
        ),
        # Index for contract type queries
        Index(
            "idx_price_data_contract_type",
            "asset", "market", "contract_type",
        ),
        # Index for instrument_key lookups
        Index(
            "idx_price_data_instrument_key",
            "instrument_key",
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
            "instrument_key": self.instrument_key,
            "contract_type": self.contract_type,
            "trading_symbol": self.trading_symbol,
            "expiry": self.expiry.isoformat() if self.expiry else None,
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
        instrument_key: Optional[str] = None,
        contract_type: Optional[str] = None,
        trading_symbol: Optional[str] = None,
        expiry: Optional[datetime] = None,
    ) -> "PriceData":
        """Create PriceData from candle data with optional contract info."""
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
            instrument_key=instrument_key,
            contract_type=contract_type,
            trading_symbol=trading_symbol,
            expiry=expiry,
        )
