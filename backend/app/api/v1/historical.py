"""
Historical data API endpoints.
Fetches from database or directly from Yahoo Finance as fallback.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.price_data import PriceData
from app.services.yahoo_client import yahoo_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/historical")


@router.get("/{asset}/{market}")
async def get_historical_data(
    asset: str,
    market: str,
    interval: str = Query("30m", description="Candle interval"),
    limit: int = Query(100, le=1000, description="Number of candles"),
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    source: str = Query("auto", description="Data source: auto, db, yahoo"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get historical OHLCV data for an asset/market.

    If database is empty or source=yahoo, fetches directly from Yahoo Finance.
    MCX data will be converted from COMEX using approximate INR conversion.
    """
    candles = []
    data_source = "database"

    # Try database first if source is auto or db
    if source in ["auto", "db"]:
        try:
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

            if rows:
                # Sort by timestamp (oldest first) for charting
                candles = sorted([r.to_dict() for r in rows], key=lambda x: x["timestamp"])
                data_source = "database"
        except Exception as e:
            logger.warning(f"Database query failed: {e}")

    # Fallback to Yahoo Finance if no data in database
    if not candles and source in ["auto", "yahoo"]:
        try:
            logger.info(f"Fetching {asset}/{market} from Yahoo Finance...")

            # Determine period based on limit
            if limit <= 100:
                period = "1mo"
            elif limit <= 500:
                period = "3mo"
            else:
                period = "1y"

            # Fetch from Yahoo Finance
            df = await yahoo_client.get_historical_data(
                symbol=asset,
                interval=interval,
                period=period,
            )

            if not df.empty:
                # Convert to candle format
                raw_candles = yahoo_client.convert_to_candles(df)

                # For MCX, convert USD to INR (approximate rate)
                if market == "mcx":
                    USD_TO_INR = 83.5  # Approximate conversion rate
                    for candle in raw_candles:
                        if candle.get("open"):
                            candle["open"] = round(candle["open"] * USD_TO_INR, 2)
                        if candle.get("high"):
                            candle["high"] = round(candle["high"] * USD_TO_INR, 2)
                        if candle.get("low"):
                            candle["low"] = round(candle["low"] * USD_TO_INR, 2)
                        if candle.get("close"):
                            candle["close"] = round(candle["close"] * USD_TO_INR, 2)

                # Limit and sort
                candles = raw_candles[-limit:] if len(raw_candles) > limit else raw_candles
                data_source = "yahoo_finance"

        except Exception as e:
            logger.error(f"Yahoo Finance fetch failed: {e}")
            return {
                "asset": asset,
                "market": market,
                "interval": interval,
                "count": 0,
                "candles": [],
                "source": "error",
                "message": f"Failed to fetch data: {str(e)}",
            }

    return {
        "asset": asset,
        "market": market,
        "interval": interval,
        "count": len(candles),
        "candles": candles,
        "source": data_source,
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
    Falls back to Yahoo Finance if database is empty.
    """
    # Try database first
    try:
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

        if price_data:
            data = price_data.to_dict()
            data["source"] = "database"
            return data
    except Exception as e:
        logger.warning(f"Database query failed: {e}")

    # Fallback to Yahoo Finance for live price
    try:
        logger.info(f"Fetching latest price for {asset} from Yahoo Finance...")
        price_info = await yahoo_client.get_current_price(asset)

        if price_info and price_info.get("price"):
            price = price_info["price"]
            high = price_info.get("high") or price
            low = price_info.get("low") or price
            open_price = price_info.get("open") or price

            # Convert to INR for MCX
            if market == "mcx":
                USD_TO_INR = 83.5
                price = round(price * USD_TO_INR, 2)
                high = round(high * USD_TO_INR, 2)
                low = round(low * USD_TO_INR, 2)
                open_price = round(open_price * USD_TO_INR, 2)

            return {
                "asset": asset,
                "market": market,
                "interval": interval,
                "timestamp": price_info.get("timestamp", datetime.now()).isoformat(),
                "open": open_price,
                "high": high,
                "low": low,
                "close": price,
                "volume": price_info.get("volume", 0),
                "change": price_info.get("change"),
                "change_percent": price_info.get("change_percent"),
                "source": "yahoo_finance",
            }
    except Exception as e:
        logger.error(f"Yahoo Finance fetch failed: {e}")

    return {
        "status": "no_data",
        "message": "No price data available",
        "asset": asset,
        "market": market,
        "interval": interval,
    }


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
    try:
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
    except Exception as e:
        return {
            "asset": asset,
            "market": market,
            "interval": interval,
            "period_days": period_days,
            "stats": {
                "low": None,
                "high": None,
                "avg_close": None,
                "candle_count": 0,
                "total_volume": None,
                "price_change": None,
                "change_percent": None,
            },
            "message": f"No data available: {str(e)}",
        }


@router.post("/sync/{asset}/{market}")
async def sync_historical_data(
    asset: str,
    market: str,
    interval: str = Query("30m", description="Candle interval"),
    days: int = Query(60, le=365, description="Days of history to sync"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Sync historical data from external sources to database.

    This endpoint fetches data from Yahoo Finance and stores it in the database.
    """
    try:
        from app.services.data_sync import data_sync_service

        if market == "mcx":
            result = await data_sync_service.sync_mcx_data(db, asset, interval, days)
        elif market == "comex":
            result = await data_sync_service.sync_comex_data(db, asset, interval, days)
        else:
            return {
                "status": "error",
                "message": f"Unknown market: {market}",
            }

        return {
            "asset": asset,
            "market": market,
            "interval": interval,
            "sync_result": result,
        }
    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return {
            "status": "error",
            "message": f"Sync failed: {str(e)}",
        }


@router.get("/factors")
async def get_market_factors(
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get correlated market factors data.
    """
    try:
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
    except Exception as e:
        return {
            "period_days": period_days,
            "factors": {},
            "message": f"No data available: {str(e)}",
        }


@router.get("/live/{asset}")
async def get_live_price(
    asset: str,
    market: str = Query("comex", description="Market: mcx or comex"),
) -> Dict[str, Any]:
    """
    Get live/current price from Yahoo Finance.

    This endpoint always fetches fresh data from Yahoo Finance.
    """
    try:
        price_info = await yahoo_client.get_current_price(asset)

        if not price_info or not price_info.get("price"):
            return {
                "status": "error",
                "message": "Could not fetch price",
                "asset": asset,
                "market": market,
            }

        price = price_info["price"]
        change = price_info.get("change")
        change_percent = price_info.get("change_percent")
        high = price_info.get("high") or price
        low = price_info.get("low") or price
        open_price = price_info.get("open") or price
        prev_close = price_info.get("previous_close")

        # Convert to INR for MCX
        if market == "mcx":
            USD_TO_INR = 83.5
            price = round(price * USD_TO_INR, 2)
            high = round(high * USD_TO_INR, 2)
            low = round(low * USD_TO_INR, 2)
            open_price = round(open_price * USD_TO_INR, 2)
            if prev_close:
                prev_close = round(prev_close * USD_TO_INR, 2)
            if change:
                change = round(change * USD_TO_INR, 2)

        return {
            "asset": asset,
            "market": market,
            "symbol": price_info.get("symbol"),
            "price": price,
            "open": open_price,
            "high": high,
            "low": low,
            "previous_close": prev_close,
            "change": change,
            "change_percent": change_percent,
            "volume": price_info.get("volume"),
            "timestamp": datetime.now().isoformat(),
            "source": "yahoo_finance",
        }
    except Exception as e:
        logger.error(f"Live price fetch failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch price: {str(e)}",
            "asset": asset,
            "market": market,
        }
