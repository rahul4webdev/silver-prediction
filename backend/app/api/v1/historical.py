"""
Historical data API endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.price_data import PriceData

router = APIRouter(prefix="/historical")


@router.get("/{asset}/{market}")
async def get_historical_data(
    asset: str,
    market: str,
    interval: str = Query("30m", description="Candle interval"),
    limit: int = Query(100, le=1000, description="Number of candles"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get historical OHLCV data for an asset/market.
    """
    conditions = [
        PriceData.asset == asset,
        PriceData.market == market,
        PriceData.interval == interval,
    ]

    if start_date:
        conditions.append(PriceData.timestamp >= start_date)
    if end_date:
        conditions.append(PriceData.timestamp <= end_date)

    query = (
        select(PriceData)
        .where(and_(*conditions))
        .order_by(desc(PriceData.timestamp))
        .limit(limit)
    )

    result = await db.execute(query)
    rows = result.scalars().all()

    # Sort by timestamp (oldest first) for charting
    candles = sorted([r.to_dict() for r in rows], key=lambda x: x["timestamp"])

    return {
        "asset": asset,
        "market": market,
        "interval": interval,
        "count": len(candles),
        "candles": candles,
    }


@router.get("/{asset}/{market}/latest")
async def get_latest_price(
    asset: str,
    market: str,
    interval: str = Query("30m"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get the latest price data.
    """
    query = (
        select(PriceData)
        .where(
            and_(
                PriceData.asset == asset,
                PriceData.market == market,
                PriceData.interval == interval,
            )
        )
        .order_by(desc(PriceData.timestamp))
        .limit(1)
    )

    result = await db.execute(query)
    price_data = result.scalar_one_or_none()

    if not price_data:
        raise HTTPException(status_code=404, detail="No price data found")

    return price_data.to_dict()


@router.get("/{asset}/{market}/stats")
async def get_price_stats(
    asset: str,
    market: str,
    interval: str = Query("30m"),
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get price statistics for the specified period.
    """
    since = datetime.utcnow() - timedelta(days=period_days)

    query = (
        select(
            func.min(PriceData.low).label("low"),
            func.max(PriceData.high).label("high"),
            func.avg(PriceData.close).label("avg_close"),
            func.count(PriceData.id).label("count"),
            func.sum(PriceData.volume).label("total_volume"),
        )
        .where(
            and_(
                PriceData.asset == asset,
                PriceData.market == market,
                PriceData.interval == interval,
                PriceData.timestamp >= since,
            )
        )
    )

    result = await db.execute(query)
    row = result.one()

    # Get first and last prices for change calculation
    first_query = (
        select(PriceData.close)
        .where(
            and_(
                PriceData.asset == asset,
                PriceData.market == market,
                PriceData.interval == interval,
                PriceData.timestamp >= since,
            )
        )
        .order_by(PriceData.timestamp)
        .limit(1)
    )

    last_query = (
        select(PriceData.close)
        .where(
            and_(
                PriceData.asset == asset,
                PriceData.market == market,
                PriceData.interval == interval,
                PriceData.timestamp >= since,
            )
        )
        .order_by(desc(PriceData.timestamp))
        .limit(1)
    )

    first_result = await db.execute(first_query)
    first_price = first_result.scalar_one_or_none()

    last_result = await db.execute(last_query)
    last_price = last_result.scalar_one_or_none()

    change = None
    change_percent = None
    if first_price and last_price:
        change = float(last_price) - float(first_price)
        change_percent = (change / float(first_price)) * 100

    return {
        "asset": asset,
        "market": market,
        "interval": interval,
        "period_days": period_days,
        "stats": {
            "low": float(row.low) if row.low else None,
            "high": float(row.high) if row.high else None,
            "avg_close": float(row.avg_close) if row.avg_close else None,
            "candle_count": row.count,
            "total_volume": row.total_volume,
            "price_change": change,
            "change_percent": change_percent,
        },
    }


@router.get("/factors")
async def get_market_factors(
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get correlated market factors data.
    """
    from app.models.market_factors import MarketFactor

    since = datetime.utcnow() - timedelta(days=period_days)

    query = (
        select(MarketFactor)
        .where(MarketFactor.timestamp >= since)
        .order_by(desc(MarketFactor.timestamp))
        .limit(100)
    )

    result = await db.execute(query)
    factors = result.scalars().all()

    # Group by factor code
    grouped = {}
    for f in factors:
        if f.factor_code not in grouped:
            grouped[f.factor_code] = []
        grouped[f.factor_code].append(f.to_dict())

    return {
        "period_days": period_days,
        "factors": grouped,
    }
