"""
API endpoints for tick data management and statistics.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db
from app.models.tick_data import TickData, TickDataAggregated
from app.services.tick_collector import tick_collector

router = APIRouter(prefix="/ticks")


@router.get("/stats")
async def get_tick_stats(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get tick collection statistics.

    Returns:
        - Collector runtime stats (ticks received, stored, errors)
        - Database stats (total ticks, today's ticks, latest tick)
    """
    # Get collector stats
    collector_stats = tick_collector.stats

    try:
        # Count total ticks
        total_result = await db.execute(
            select(func.count(TickData.id))
        )
        total_ticks = total_result.scalar() or 0

        # Get latest tick
        latest_result = await db.execute(
            select(TickData)
            .order_by(TickData.timestamp.desc())
            .limit(1)
        )
        latest_tick = latest_result.scalar_one_or_none()

        # Get oldest tick
        oldest_result = await db.execute(
            select(TickData)
            .order_by(TickData.timestamp.asc())
            .limit(1)
        )
        oldest_tick = oldest_result.scalar_one_or_none()

        # Count today's ticks
        today_start = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        today_result = await db.execute(
            select(func.count(TickData.id))
            .where(TickData.timestamp >= today_start)
        )
        today_ticks = today_result.scalar() or 0

        # Count aggregated candles
        agg_result = await db.execute(
            select(
                TickDataAggregated.interval,
                func.count(TickDataAggregated.id)
            ).group_by(TickDataAggregated.interval)
        )
        aggregated_counts = {row[0]: row[1] for row in agg_result.fetchall()}

        return {
            "collector": {
                "is_running": tick_collector.is_running,
                "ticks_received": collector_stats.get("ticks_received", 0),
                "ticks_stored": collector_stats.get("ticks_stored", 0),
                "errors": collector_stats.get("errors", 0),
                "last_tick_time": collector_stats.get("last_tick_time").isoformat()
                    if collector_stats.get("last_tick_time") else None,
                "connected_since": collector_stats.get("connected_since").isoformat()
                    if collector_stats.get("connected_since") else None,
            },
            "database": {
                "total_ticks": total_ticks,
                "today_ticks": today_ticks,
                "oldest_tick": oldest_tick.timestamp.isoformat() if oldest_tick else None,
                "latest_tick": latest_tick.timestamp.isoformat() if latest_tick else None,
                "latest_ltp": float(latest_tick.ltp) if latest_tick and latest_tick.ltp else None,
                "latest_symbol": latest_tick.symbol if latest_tick else None,
            },
            "aggregated_candles": aggregated_counts,
        }

    except Exception as e:
        return {
            "collector": collector_stats,
            "database": {"error": str(e)},
            "aggregated_candles": {},
        }


@router.get("/recent")
async def get_recent_ticks(
    db: AsyncSession = Depends(get_db),
    limit: int = Query(100, le=1000),
    asset: str = Query("silver"),
    market: str = Query("mcx"),
) -> Dict[str, Any]:
    """
    Get recent tick data.

    Args:
        limit: Number of ticks to return (max 1000)
        asset: Asset to filter by (default: silver)
        market: Market to filter by (default: mcx)

    Returns:
        List of recent ticks
    """
    result = await db.execute(
        select(TickData)
        .where(TickData.asset == asset, TickData.market == market)
        .order_by(TickData.timestamp.desc())
        .limit(limit)
    )
    ticks = result.scalars().all()

    return {
        "count": len(ticks),
        "asset": asset,
        "market": market,
        "ticks": [
            {
                "timestamp": tick.timestamp.isoformat(),
                "ltp": float(tick.ltp) if tick.ltp else None,
                "ltq": tick.ltq,
                "open": float(tick.open) if tick.open else None,
                "high": float(tick.high) if tick.high else None,
                "low": float(tick.low) if tick.low else None,
                "close": float(tick.close) if tick.close else None,
                "volume": tick.volume,
                "change": float(tick.change) if tick.change else None,
            }
            for tick in ticks
        ],
    }


@router.get("/aggregated")
async def get_aggregated_candles(
    db: AsyncSession = Depends(get_db),
    interval: str = Query("1m", regex="^(1s|5s|10s|1m)$"),
    limit: int = Query(100, le=1000),
    asset: str = Query("silver"),
    market: str = Query("mcx"),
) -> Dict[str, Any]:
    """
    Get aggregated candle data.

    Args:
        interval: Candle interval (1s, 5s, 10s, 1m)
        limit: Number of candles to return (max 1000)
        asset: Asset to filter by
        market: Market to filter by

    Returns:
        List of aggregated OHLCV candles
    """
    result = await db.execute(
        select(TickDataAggregated)
        .where(
            TickDataAggregated.asset == asset,
            TickDataAggregated.market == market,
            TickDataAggregated.interval == interval,
        )
        .order_by(TickDataAggregated.timestamp.desc())
        .limit(limit)
    )
    candles = result.scalars().all()

    return {
        "count": len(candles),
        "interval": interval,
        "asset": asset,
        "market": market,
        "candles": [
            {
                "timestamp": candle.timestamp.isoformat(),
                "open": float(candle.open) if candle.open else None,
                "high": float(candle.high) if candle.high else None,
                "low": float(candle.low) if candle.low else None,
                "close": float(candle.close) if candle.close else None,
                "volume": candle.volume,
                "tick_count": candle.tick_count,
            }
            for candle in candles
        ],
    }


@router.post("/aggregate")
async def trigger_aggregation(
    db: AsyncSession = Depends(get_db),
    interval: str = Query("1m", regex="^(1s|5s|10s|1m)$"),
) -> Dict[str, Any]:
    """
    Manually trigger tick aggregation.

    This is normally done automatically by the tick collector worker,
    but can be triggered manually if needed.

    Args:
        interval: Aggregation interval (1s, 5s, 10s, 1m)

    Returns:
        Number of candles created/updated
    """
    try:
        count = await tick_collector.aggregate_ticks(interval)
        return {
            "status": "success",
            "interval": interval,
            "candles_created": count,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/collector/status")
async def get_collector_status() -> Dict[str, Any]:
    """
    Get the current status of the tick collector.

    Returns:
        - Running state
        - Connection info
        - Error counts
    """
    stats = tick_collector.stats

    return {
        "is_running": tick_collector.is_running,
        "stats": {
            "ticks_received": stats.get("ticks_received", 0),
            "ticks_stored": stats.get("ticks_stored", 0),
            "errors": stats.get("errors", 0),
        },
        "connection": {
            "last_tick_time": stats.get("last_tick_time").isoformat()
                if stats.get("last_tick_time") else None,
            "connected_since": stats.get("connected_since").isoformat()
                if stats.get("connected_since") else None,
        },
    }
