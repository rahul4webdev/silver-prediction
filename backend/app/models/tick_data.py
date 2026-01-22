"""
Tick data model for storing real-time price updates (every second).
Used for high-frequency model training and analysis.
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


class TickData(Base):
    """
    Stores real-time tick data from WebSocket streams.

    This captures every price update for high-frequency analysis.
    Data is collected via Upstox WebSocket for MCX and polling for COMEX.
    """

    __tablename__ = "tick_data"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    asset: Mapped[str] = mapped_column(String(20), nullable=False)  # silver, gold
    market: Mapped[str] = mapped_column(String(10), nullable=False)  # mcx, comex
    symbol: Mapped[str] = mapped_column(String(50), nullable=False)  # Full symbol

    # Timestamp (millisecond precision for ticks)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # Price data
    ltp: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 4), nullable=True, comment="Last Traded Price"
    )
    ltq: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="Last Traded Quantity"
    )

    # OHLC (intraday)
    open: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    high: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    low: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    close: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)

    # Volume
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    oi: Mapped[Optional[int]] = mapped_column(
        BigInteger, nullable=True, comment="Open Interest"
    )

    # Bid/Ask spread
    bid_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    bid_qty: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    ask_price: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    ask_qty: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Change
    change: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    change_percent: Mapped[Optional[Decimal]] = mapped_column(Numeric(8, 4), nullable=True)

    # Metadata
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )
    source: Mapped[str] = mapped_column(String(20), default="upstox")

    __table_args__ = (
        # Index for fast time-series queries
        Index("ix_tick_data_asset_market_timestamp", "asset", "market", "timestamp"),
        # Index for recent data queries
        Index("ix_tick_data_timestamp", "timestamp"),
        # Composite index for symbol lookups
        Index("ix_tick_data_symbol_timestamp", "symbol", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<TickData {self.symbol} @ {self.timestamp}: {self.ltp}>"


class TickDataAggregated(Base):
    """
    Aggregated tick data at various intervals (1s, 5s, 10s, 1m).
    Pre-computed for faster model training queries.
    """

    __tablename__ = "tick_data_aggregated"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Identification
    asset: Mapped[str] = mapped_column(String(20), nullable=False)
    market: Mapped[str] = mapped_column(String(10), nullable=False)
    interval: Mapped[str] = mapped_column(String(10), nullable=False)  # 1s, 5s, 10s, 1m

    # Time bucket
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # OHLCV
    open: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    high: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    low: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    close: Mapped[Optional[Decimal]] = mapped_column(Numeric(14, 4), nullable=True)
    volume: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)

    # Statistics
    tick_count: Mapped[int] = mapped_column(BigInteger, default=0)
    vwap: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 4), nullable=True, comment="Volume Weighted Avg Price"
    )
    avg_spread: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(14, 4), nullable=True, comment="Average bid-ask spread"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow
    )

    __table_args__ = (
        UniqueConstraint(
            "asset", "market", "interval", "timestamp",
            name="uq_tick_aggregated_asset_market_interval_timestamp"
        ),
        Index("ix_tick_aggregated_timestamp", "timestamp"),
    )

    def __repr__(self) -> str:
        return f"<TickDataAggregated {self.asset}/{self.market} {self.interval} @ {self.timestamp}>"
