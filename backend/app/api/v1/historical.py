"""
Historical data API endpoints.
Fetches from database, Upstox (MCX), or Yahoo Finance (COMEX/fallback).
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc, func, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.price_data import PriceData
from app.services.yahoo_client import yahoo_client
from app.services.upstox_client import upstox_client, UpstoxAuthError, UpstoxAPIError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/historical")


@router.get("/live/{asset}")
async def get_live_price(
    asset: str,
    market: str = Query("comex", description="Market: mcx or comex"),
) -> Dict[str, Any]:
    """
    Get live/current price.

    For MCX silver:
    1. First tries Upstox (real MCX data) if authenticated
    2. Falls back to Silver Bees ETF (actual INR prices)
    3. Finally falls back to COMEX silver with USD/INR conversion

    For COMEX: Uses Yahoo Finance directly.
    """
    try:
        # For MCX assets, try specific methods first
        if market == "mcx":
            if asset == "silver":
                # Priority 1: Upstox (real MCX data)
                if upstox_client.is_authenticated:
                    try:
                        quote = await upstox_client.get_live_quote()
                        if quote and quote.get("price"):
                            return {
                                "asset": asset,
                                "market": market,
                                "symbol": quote.get("symbol"),
                                "price": quote["price"],
                                "open": quote.get("open"),
                                "high": quote.get("high"),
                                "low": quote.get("low"),
                                "change": quote.get("change"),
                                "change_percent": quote.get("change_percent"),
                                "volume": quote.get("volume"),
                                "currency": "INR",
                                "timestamp": datetime.now().isoformat(),
                                "source": "upstox",
                                "note": "Real MCX Silver futures data via Upstox API",
                            }
                    except UpstoxAPIError as e:
                        logger.warning(f"Upstox quote fetch failed: {e}")
                    except UpstoxAuthError as e:
                        logger.warning(f"Upstox auth error: {e}")

                # Priority 2: Silver Bees ETF (NSE)
                try:
                    mcx_price = await yahoo_client.get_silver_price_inr()
                    if mcx_price and mcx_price.get("price"):
                        return {
                            "asset": asset,
                            "market": market,
                            "symbol": mcx_price.get("symbol"),
                            "price": mcx_price["price"],
                            "price_per_gram": mcx_price.get("price_per_gram"),
                            "currency": "INR",
                            "change": mcx_price.get("change"),
                            "change_percent": mcx_price.get("change_percent"),
                            "timestamp": datetime.now().isoformat(),
                            "source": mcx_price.get("source", "yahoo_finance"),
                            "note": "Price per kg. Silver Bees ETF is used as MCX proxy.",
                        }
                except Exception as e:
                    logger.warning(f"MCX silver price fetch failed: {e}")

            elif asset == "gold":
                # Try Gold Bees ETF (NSE)
                try:
                    mcx_price = await yahoo_client.get_gold_price_inr()
                    if mcx_price and mcx_price.get("price"):
                        return {
                            "asset": asset,
                            "market": market,
                            "symbol": mcx_price.get("symbol"),
                            "price": mcx_price["price"],
                            "price_per_gram": mcx_price.get("price_per_gram"),
                            "currency": "INR",
                            "change": mcx_price.get("change"),
                            "change_percent": mcx_price.get("change_percent"),
                            "timestamp": datetime.now().isoformat(),
                            "source": mcx_price.get("source", "yahoo_finance"),
                            "note": "Price per 10g. Gold Bees ETF is used as MCX proxy.",
                        }
                except Exception as e:
                    logger.warning(f"MCX gold price fetch failed: {e}")

        # Standard approach for COMEX or fallback
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
        source = "yahoo_finance"

        # Convert to INR for MCX
        if market == "mcx":
            try:
                USD_TO_INR = await yahoo_client.get_usdinr_rate()
            except Exception:
                USD_TO_INR = 83.5
            # Convert per oz to per kg: 1 kg = 32.1507 troy oz
            conversion = USD_TO_INR * 32.1507
            price = round(price * conversion, 2)
            high = round(high * conversion, 2)
            low = round(low * conversion, 2)
            open_price = round(open_price * conversion, 2)
            if prev_close:
                prev_close = round(prev_close * conversion, 2)
            if change:
                change = round(change * conversion, 2)
            source = "comex_converted"

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
            "source": source,
            "currency": "INR" if market == "mcx" else "USD",
        }
    except Exception as e:
        logger.error(f"Live price fetch failed: {e}")
        return {
            "status": "error",
            "message": f"Failed to fetch price: {str(e)}",
            "asset": asset,
            "market": market,
        }


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

            # For MCX, try ETF data first (actual INR prices)
            if market == "mcx":
                if asset == "silver":
                    try:
                        df = await yahoo_client.get_silver_mcx(interval=interval, days=60)
                        if not df.empty:
                            raw_candles = yahoo_client.convert_to_candles(df)
                            # Silver Bees is per gram, convert to per kg for MCX-like prices
                            for candle in raw_candles:
                                if candle.get("open"):
                                    candle["open"] = round(candle["open"] * 1000, 2)
                                if candle.get("high"):
                                    candle["high"] = round(candle["high"] * 1000, 2)
                                if candle.get("low"):
                                    candle["low"] = round(candle["low"] * 1000, 2)
                                if candle.get("close"):
                                    candle["close"] = round(candle["close"] * 1000, 2)
                            candles = raw_candles[-limit:] if len(raw_candles) > limit else raw_candles
                            data_source = "silver_bees_etf"
                    except Exception as e:
                        logger.warning(f"Silver Bees fetch failed, falling back to COMEX conversion: {e}")
                elif asset == "gold":
                    try:
                        df = await yahoo_client.get_gold_mcx(interval=interval, days=60)
                        if not df.empty:
                            raw_candles = yahoo_client.convert_to_candles(df)
                            # Gold Bees is per gram, convert to per 10g for MCX-like prices
                            for candle in raw_candles:
                                if candle.get("open"):
                                    candle["open"] = round(candle["open"] * 10, 2)
                                if candle.get("high"):
                                    candle["high"] = round(candle["high"] * 10, 2)
                                if candle.get("low"):
                                    candle["low"] = round(candle["low"] * 10, 2)
                                if candle.get("close"):
                                    candle["close"] = round(candle["close"] * 10, 2)
                            candles = raw_candles[-limit:] if len(raw_candles) > limit else raw_candles
                            data_source = "gold_bees_etf"
                    except Exception as e:
                        logger.warning(f"Gold Bees fetch failed, falling back to COMEX conversion: {e}")

            # If MCX data not available or for COMEX, use standard approach
            if not candles:
                df = await yahoo_client.get_historical_data(
                    symbol=asset,
                    interval=interval,
                    period=period,
                )

                if not df.empty:
                    # Convert to candle format
                    raw_candles = yahoo_client.convert_to_candles(df)

                    # For MCX, convert USD to INR with live rate
                    if market == "mcx":
                        try:
                            USD_TO_INR = await yahoo_client.get_usdinr_rate()
                        except Exception:
                            USD_TO_INR = 83.5  # Fallback rate

                        # COMEX is per troy ounce, convert to per kg for MCX
                        # 1 kg = 32.1507 troy ounces
                        conversion = USD_TO_INR * 32.1507

                        for candle in raw_candles:
                            if candle.get("open"):
                                candle["open"] = round(candle["open"] * conversion, 2)
                            if candle.get("high"):
                                candle["high"] = round(candle["high"] * conversion, 2)
                            if candle.get("low"):
                                candle["low"] = round(candle["low"] * conversion, 2)
                            if candle.get("close"):
                                candle["close"] = round(candle["close"] * conversion, 2)
                        data_source = "comex_converted"
                    else:
                        data_source = "yahoo_finance"

                    # Limit and sort
                    candles = raw_candles[-limit:] if len(raw_candles) > limit else raw_candles

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

        # For MCX silver, use dedicated method that tries Silver Bees first
        if market == "mcx" and asset == "silver":
            try:
                mcx_price = await yahoo_client.get_silver_price_inr()
                if mcx_price and mcx_price.get("price"):
                    return {
                        "asset": asset,
                        "market": market,
                        "interval": interval,
                        "timestamp": mcx_price.get("timestamp", datetime.now()).isoformat(),
                        "price": mcx_price["price"],
                        "close": mcx_price["price"],
                        "currency": "INR",
                        "change": mcx_price.get("change"),
                        "change_percent": mcx_price.get("change_percent"),
                        "source": mcx_price.get("source", "yahoo_finance"),
                    }
            except Exception as e:
                logger.warning(f"MCX silver price fetch failed: {e}")

        # Standard approach for COMEX or fallback
        price_info = await yahoo_client.get_current_price(asset)

        if price_info and price_info.get("price"):
            price = price_info["price"]
            high = price_info.get("high") or price
            low = price_info.get("low") or price
            open_price = price_info.get("open") or price

            # Convert to INR for MCX
            if market == "mcx":
                try:
                    USD_TO_INR = await yahoo_client.get_usdinr_rate()
                except Exception:
                    USD_TO_INR = 83.5
                # Convert per oz to per kg
                conversion = USD_TO_INR * 32.1507
                price = round(price * conversion, 2)
                high = round(high * conversion, 2)
                low = round(low * conversion, 2)
                open_price = round(open_price * conversion, 2)

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
                "source": "comex_converted" if market == "mcx" else "yahoo_finance",
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
    period_days: int = Query(30, le=730),
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
    days: int = Query(60, le=730, description="Days of history to sync"),
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


@router.post("/sync-all")
async def sync_all_data(
    days: int = Query(60, le=730, description="Days of history to sync (max 2 years)"),
    force: bool = Query(False, description="Force full re-sync by clearing existing data first"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Sync all historical data (silver for both MCX and COMEX, all intervals).

    This is useful for initial data loading.
    Note: Upstox provides ~1 year of 30m data, Yahoo Finance provides more for COMEX.

    Use force=true to clear existing data and re-sync from scratch.
    """
    from app.services.data_sync import data_sync_service

    results = {}
    intervals = ["30m", "1h", "4h", "1d"]

    # If force=true, clear existing data first
    if force:
        try:
            for market in ["comex", "mcx"]:
                for interval in intervals:
                    await db.execute(
                        text("DELETE FROM price_data WHERE asset = :asset AND market = :market AND interval = :interval"),
                        {"asset": "silver", "market": market, "interval": interval}
                    )
            await db.commit()
            results["cleared"] = True
            logger.info("Cleared existing price data for force re-sync")
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            results["cleared"] = False

    for market in ["comex", "mcx"]:
        for interval in intervals:
            key = f"{market}_{interval}"
            try:
                if market == "comex":
                    result = await data_sync_service.sync_comex_data(
                        db, "silver", interval, days
                    )
                else:
                    result = await data_sync_service.sync_mcx_data(
                        db, "silver", interval, days
                    )
                results[key] = result
            except Exception as e:
                logger.error(f"Sync failed for {key}: {e}")
                results[key] = {"status": "error", "error": str(e)}

    return {
        "status": "complete",
        "days": days,
        "force": force,
        "results": results,
    }


@router.post("/sync-mcx-per-contract")
async def sync_mcx_per_contract(
    days: int = Query(60, le=365, description="Days of history to sync"),
    clear_old: bool = Query(True, description="Clear old MCX data before syncing"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Sync MCX silver data per contract (SILVER, SILVERM, SILVERMIC).

    This fetches historical data for each contract separately, allowing
    per-contract model training and accurate predictions.

    Args:
        days: Days of history to sync (default 60, max 365)
        clear_old: If true, clears old MCX data before syncing (recommended)

    Returns:
        Sync results per contract
    """
    from app.services.data_sync import data_sync_service

    results = {}
    intervals = ["30m", "1h", "4h", "1d"]

    # Clear old MCX data if requested
    if clear_old:
        try:
            await db.execute(
                text("DELETE FROM price_data WHERE market = 'mcx'")
            )
            await db.execute(
                text("DELETE FROM predictions WHERE market = 'mcx'")
            )
            await db.commit()
            results["cleared_mcx_data"] = True
            results["cleared_mcx_predictions"] = True
            logger.info("Cleared old MCX data")
        except Exception as e:
            logger.error(f"Failed to clear MCX data: {e}")
            results["clear_error"] = str(e)

    # Sync each interval with per-contract data
    for interval in intervals:
        key = f"mcx_per_contract_{interval}"
        try:
            result = await data_sync_service.sync_mcx_data_per_contract(
                db, "silver", interval, days
            )
            results[key] = result
            logger.info(f"Per-contract sync complete for {interval}: {result.get('total_records', 0)} records")
        except Exception as e:
            logger.error(f"Per-contract sync failed for {interval}: {e}")
            results[key] = {"status": "error", "error": str(e)}

    return {
        "status": "complete",
        "days": days,
        "clear_old": clear_old,
        "results": results,
    }


@router.post("/migrate-price-data")
async def migrate_price_data_schema(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Run database migration to add contract-specific columns to price_data table.

    This adds: instrument_key, contract_type, trading_symbol, expiry columns.
    Must be run before using sync-mcx-per-contract endpoint.
    """
    results = {}

    try:
        # Step 1: Add new columns if they don't exist
        columns_to_add = [
            ("instrument_key", "VARCHAR(50)"),
            ("contract_type", "VARCHAR(20)"),
            ("trading_symbol", "VARCHAR(100)"),
            ("expiry", "TIMESTAMP WITH TIME ZONE"),
        ]

        for col_name, col_type in columns_to_add:
            try:
                await db.execute(text(f"""
                    ALTER TABLE price_data
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type};
                """))
                results[f"column_{col_name}"] = "added"
            except Exception as e:
                results[f"column_{col_name}"] = f"exists or error: {str(e)[:100]}"

        await db.commit()

        # Step 2: Create indexes if they don't exist
        indexes = [
            ("idx_price_data_instrument_key", "instrument_key"),
            ("idx_price_data_contract_type", "asset, market, contract_type"),
        ]

        for idx_name, idx_cols in indexes:
            try:
                await db.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON price_data ({idx_cols});
                """))
                results[f"index_{idx_name}"] = "created"
            except Exception as e:
                results[f"index_{idx_name}"] = f"exists or error: {str(e)[:100]}"

        await db.commit()

        return {
            "status": "success",
            "message": "Migration completed",
            "results": results,
        }

    except Exception as e:
        logger.error(f"Migration failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "results": results,
        }


@router.get("/factors")
async def get_market_factors(
    period_days: int = Query(30, le=730),
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
