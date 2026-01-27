"""
Data synchronization service for fetching and storing market data.
Syncs data from Upstox (MCX) and Yahoo Finance (COMEX).

Updated to support per-contract data for MCX (SILVER, SILVERM, SILVERMIC).
"""

import logging
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select, text, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.price_data import PriceData
from app.services.upstox_client import upstox_client
from app.services.yahoo_client import yahoo_client


def utc_now() -> datetime:
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)

logger = logging.getLogger(__name__)


# Upstox interval mapping
UPSTOX_INTERVAL_MAP = {
    "30m": "30minute",
    "1d": "day",
    "1w": "week",
    "1M": "month",
}


class DataSyncService:
    """
    Service for synchronizing market data from external sources.

    Features:
    - Sync MCX data from Upstox
    - Sync COMEX data from Yahoo Finance
    - Incremental updates (only fetch new data)
    - Batch insert for efficiency
    """

    async def sync_mcx_data(
        self,
        db: AsyncSession,
        asset: str,
        interval: str,
        days: int = 60,
    ) -> Dict[str, Any]:
        """
        Sync MCX data from Upstox.

        Args:
            db: Database session
            asset: Asset symbol (e.g., "silver")
            interval: Candle interval
            days: Days of history to sync

        Returns:
            Dict with sync results
        """
        logger.info(f"Syncing MCX {asset} {interval} data...")

        try:
            # Verify Upstox authentication
            if not upstox_client.is_authenticated:
                logger.warning("Upstox not authenticated - no access token set")
                return {
                    "status": "skipped",
                    "reason": "not_authenticated",
                    "message": "Set UPSTOX_ACCESS_TOKEN in environment or authenticate via OAuth",
                }

            # Verify the token is actually valid
            auth_status = await upstox_client.verify_authentication()
            if not auth_status.get("authenticated"):
                logger.warning(f"Upstox authentication invalid: {auth_status.get('message')}")
                return {
                    "status": "skipped",
                    "reason": auth_status.get("reason", "auth_failed"),
                    "message": auth_status.get("message"),
                }

            # Get the latest timestamp we have
            latest = await self._get_latest_timestamp(db, asset, "mcx", interval)
            now = utc_now()

            # Determine start date
            if latest:
                # Ensure latest is timezone-aware
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                start_date = latest + timedelta(minutes=1)
            else:
                start_date = now - timedelta(days=days)

            # Only proceed if we need data
            if latest and start_date > now:
                logger.info("MCX data is already up to date")
                return {"status": "success", "records": 0, "message": "Already up to date"}

            # Fetch from Upstox using the proper method
            if asset.lower() == "silver":
                df = await upstox_client.get_mcx_silver_data(
                    interval=interval,
                    start_date=start_date,
                    end_date=now,
                )
            else:
                logger.warning(f"Asset {asset} not yet supported for MCX sync")
                return {"status": "skipped", "reason": f"Asset {asset} not supported"}

            if df.empty:
                logger.info("No new MCX data to sync")
                return {"status": "success", "records": 0}

            # Insert into database
            records_inserted = await self._insert_price_data(
                db, df, asset, "mcx", interval
            )

            logger.info(f"Synced {records_inserted} MCX records from Upstox")
            return {
                "status": "success",
                "records": records_inserted,
                "source": "upstox",
            }

        except Exception as e:
            logger.error(f"MCX sync failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def sync_mcx_data_per_contract(
        self,
        db: AsyncSession,
        asset: str,
        interval: str,
        days: int = 60,
    ) -> Dict[str, Any]:
        """
        Sync MCX data for all silver contracts (SILVER, SILVERM, SILVERMIC).

        This fetches historical data for each contract separately and stores
        with the contract's instrument_key, allowing per-contract model training.

        Args:
            db: Database session
            asset: Asset symbol (e.g., "silver")
            interval: Candle interval (30m, 1h, 4h, 1d)
            days: Days of history to sync (max 60 for 30m data)

        Returns:
            Dict with sync results per contract
        """
        logger.info(f"Syncing MCX {asset} {interval} data per contract...")

        results = {}

        try:
            # Verify Upstox authentication
            if not upstox_client.is_authenticated:
                return {
                    "status": "skipped",
                    "reason": "not_authenticated",
                    "message": "Set UPSTOX_ACCESS_TOKEN in environment",
                }

            auth_status = await upstox_client.verify_authentication()
            if not auth_status.get("authenticated"):
                return {
                    "status": "skipped",
                    "reason": auth_status.get("reason", "auth_failed"),
                    "message": auth_status.get("message"),
                }

            # Get all silver contracts
            contracts = await upstox_client.get_all_silver_instrument_keys()

            if not contracts:
                logger.warning("No MCX silver contracts found")
                return {"status": "error", "error": "No contracts found"}

            # Get unique contract types with nearest expiry (max 3)
            seen_types = set()
            contracts_to_sync = []

            for contract in contracts:
                contract_type = contract.get("contract_type")
                if contract_type and contract_type not in seen_types:
                    seen_types.add(contract_type)
                    contracts_to_sync.append(contract)
                    if len(contracts_to_sync) >= 3:
                        break

            logger.info(
                f"Syncing {len(contracts_to_sync)} contracts: "
                f"{[c.get('contract_type') for c in contracts_to_sync]}"
            )

            total_records = 0

            # Sync data for each contract
            for contract in contracts_to_sync:
                contract_type = contract.get("contract_type")
                instrument_key = contract.get("instrument_key")
                trading_symbol = contract.get("trading_symbol")
                expiry = contract.get("expiry")

                try:
                    # Get latest timestamp for this specific contract
                    latest = await self._get_latest_timestamp_for_contract(
                        db, asset, "mcx", interval, instrument_key
                    )

                    now = utc_now()

                    # Determine start date
                    if latest:
                        if latest.tzinfo is None:
                            latest = latest.replace(tzinfo=timezone.utc)
                        start_date = latest + timedelta(minutes=1)
                    else:
                        start_date = now - timedelta(days=days)

                    # Skip if already up to date
                    if latest and start_date > now:
                        logger.info(f"{contract_type} data is already up to date")
                        results[contract_type] = {"status": "success", "records": 0, "message": "Up to date"}
                        continue

                    # Handle intervals that need aggregation (1h, 4h)
                    needs_aggregation = interval in ["1h", "4h"]
                    upstox_interval = "30minute" if needs_aggregation else UPSTOX_INTERVAL_MAP.get(interval, "30minute")

                    # Fetch data for this specific contract
                    calc_days = min((now - start_date).days + 1, days)

                    candles = await upstox_client.fetch_historical_data_chunked(
                        instrument_key=instrument_key,
                        interval=upstox_interval,
                        days=calc_days,
                        chunk_days=30,
                    )

                    if not candles:
                        logger.info(f"No data returned for {contract_type}")
                        results[contract_type] = {"status": "success", "records": 0}
                        continue

                    # Convert to DataFrame
                    df = pd.DataFrame(candles)

                    if "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])

                    # Sort and drop duplicates
                    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])

                    # Aggregate if needed (1h, 4h)
                    if needs_aggregation:
                        df = self._aggregate_candles(df, interval)

                    if df.empty:
                        results[contract_type] = {"status": "success", "records": 0}
                        continue

                    # Insert with contract info
                    records_inserted = await self._insert_price_data_with_contract(
                        db=db,
                        df=df,
                        asset=asset,
                        market="mcx",
                        interval=interval,
                        instrument_key=instrument_key,
                        contract_type=contract_type,
                        trading_symbol=trading_symbol,
                        expiry=expiry,
                    )

                    total_records += records_inserted
                    results[contract_type] = {
                        "status": "success",
                        "records": records_inserted,
                        "instrument_key": instrument_key,
                    }

                    logger.info(f"Synced {records_inserted} records for {contract_type}")

                except Exception as e:
                    logger.error(f"Error syncing {contract_type}: {e}")
                    results[contract_type] = {"status": "error", "error": str(e)}

            return {
                "status": "success",
                "total_records": total_records,
                "contracts": results,
            }

        except Exception as e:
            logger.error(f"MCX per-contract sync failed: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}

    async def _get_latest_timestamp_for_contract(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
        instrument_key: str,
    ) -> Optional[datetime]:
        """Get the latest timestamp for a specific contract."""
        result = await db.execute(
            select(PriceData.timestamp)
            .where(
                and_(
                    PriceData.asset == asset,
                    PriceData.market == market,
                    PriceData.interval == interval,
                    PriceData.instrument_key == instrument_key,
                )
            )
            .order_by(PriceData.timestamp.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row if row else None

    async def _insert_price_data_with_contract(
        self,
        db: AsyncSession,
        df: pd.DataFrame,
        asset: str,
        market: str,
        interval: str,
        instrument_key: str,
        contract_type: str,
        trading_symbol: str,
        expiry: Optional[datetime],
    ) -> int:
        """
        Insert price data with contract information.

        Uses upsert on the new unique constraint (asset, market, interval, instrument_key, timestamp).
        """
        if df.empty:
            return 0

        records = []
        for _, row in df.iterrows():
            timestamp = row.get("timestamp") or row.get("date") or row.get("datetime")
            if timestamp is None:
                continue

            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()

            record = {
                "asset": asset,
                "market": market,
                "interval": interval,
                "instrument_key": instrument_key,
                "contract_type": contract_type,
                "trading_symbol": trading_symbol,
                "expiry": expiry,
                "timestamp": timestamp,
                "open": float(row["open"]) if pd.notna(row.get("open")) else None,
                "high": float(row["high"]) if pd.notna(row.get("high")) else None,
                "low": float(row["low"]) if pd.notna(row.get("low")) else None,
                "close": float(row["close"]) if pd.notna(row.get("close")) else None,
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                "created_at": datetime.now(timezone.utc),
                "source": "upstox",
            }
            records.append(record)

        if not records:
            return 0

        inserted_count = 0
        batch_size = 500

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            stmt = pg_insert(PriceData).values(batch)

            # Use on_conflict_do_update with the new constraint that includes instrument_key
            # This allows different contracts to have data for the same timestamp
            # The constraint is: (asset, market, interval, COALESCE(instrument_key, 'none'), timestamp)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_price_data_asset_market_interval_contract_timestamp",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "contract_type": stmt.excluded.contract_type,
                    "trading_symbol": stmt.excluded.trading_symbol,
                    "expiry": stmt.excluded.expiry,
                    "source": stmt.excluded.source,
                },
            )

            result = await db.execute(stmt)
            inserted_count += len(batch)

        await db.commit()
        return inserted_count

    def _aggregate_candles(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Aggregate 30m candles to 1h or 4h intervals.
        """
        if df.empty:
            return df

        df = df.copy()
        df.set_index("timestamp", inplace=True)

        # Determine resample rule
        if target_interval == "1h":
            rule = "1H"
        elif target_interval == "4h":
            rule = "4H"
        else:
            return df.reset_index()

        # Aggregate OHLCV
        agg_df = df.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return agg_df.reset_index()

    async def sync_comex_data(
        self,
        db: AsyncSession,
        asset: str,
        interval: str,
        days: int = 60,
    ) -> Dict[str, Any]:
        """
        Sync COMEX data from Yahoo Finance.

        Args:
            db: Database session
            asset: Asset symbol (e.g., "silver")
            interval: Candle interval
            days: Days of history to sync

        Returns:
            Dict with sync results
        """
        logger.info(f"Syncing COMEX {asset} {interval} data...")

        try:
            # Get the latest timestamp we have
            latest = await self._get_latest_timestamp(db, asset, "comex", interval)
            now = utc_now()

            # Determine start date
            if latest:
                # Ensure latest is timezone-aware
                if latest.tzinfo is None:
                    latest = latest.replace(tzinfo=timezone.utc)
                start_date = latest + timedelta(minutes=1)
            else:
                start_date = now - timedelta(days=days)

            # Fetch from Yahoo Finance
            df = await yahoo_client.get_historical_data(
                symbol=asset,
                interval=interval,
                start=start_date,
                end=now,
            )

            if df.empty:
                logger.info("No new COMEX data to sync")
                return {"status": "success", "records": 0}

            # Insert into database
            records_inserted = await self._insert_price_data(
                db, df, asset, "comex", interval
            )

            logger.info(f"Synced {records_inserted} COMEX records")
            return {"status": "success", "records": records_inserted}

        except Exception as e:
            logger.error(f"COMEX sync failed: {e}")
            return {"status": "error", "error": str(e)}

    async def sync_correlated_factors(
        self,
        db: AsyncSession,
        interval: str = "1d",
        period: str = "3mo",
    ) -> Dict[str, Any]:
        """
        Sync correlated market factors from Yahoo Finance.

        Returns:
            Dict with sync results per factor
        """
        logger.info("Syncing correlated factors...")

        results = {}

        try:
            factors_data = await yahoo_client.get_correlated_factors(
                interval=interval,
                period=period,
            )

            for factor_name, df in factors_data.items():
                if df.empty:
                    results[factor_name] = {"status": "empty"}
                    continue

                records = await self._insert_price_data(
                    db, df, factor_name, "factor", interval
                )
                results[factor_name] = {"status": "success", "records": records}

        except Exception as e:
            logger.error(f"Factor sync failed: {e}")
            results["error"] = str(e)

        return results

    async def backfill_data(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
        days: int = 365,
    ) -> Dict[str, Any]:
        """
        Backfill historical data for an asset.

        Args:
            db: Database session
            asset: Asset symbol
            market: Market (mcx or comex)
            interval: Candle interval
            days: Days of history

        Returns:
            Dict with backfill results
        """
        logger.info(f"Backfilling {days} days of {asset}/{market}/{interval} data...")

        if market == "mcx":
            return await self.sync_mcx_data(db, asset, interval, days)
        elif market == "comex":
            return await self.sync_comex_data(db, asset, interval, days)
        else:
            return {"status": "error", "error": f"Unknown market: {market}"}

    async def _get_latest_timestamp(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
    ) -> Optional[datetime]:
        """Get the latest timestamp for an asset in the database."""
        result = await db.execute(
            select(PriceData.timestamp)
            .where(
                PriceData.asset == asset,
                PriceData.market == market,
                PriceData.interval == interval,
            )
            .order_by(PriceData.timestamp.desc())
            .limit(1)
        )
        row = result.scalar_one_or_none()
        return row if row else None

    async def _insert_price_data(
        self,
        db: AsyncSession,
        df: pd.DataFrame,
        asset: str,
        market: str,
        interval: str,
    ) -> int:
        """
        Insert price data into the database using upsert (INSERT ... ON CONFLICT DO UPDATE).

        This handles duplicates gracefully by updating existing records.
        """
        if df.empty:
            return 0

        # Prepare records as dicts for bulk upsert
        records = []
        for _, row in df.iterrows():
            timestamp = row.get("timestamp") or row.get("date") or row.get("datetime")
            if timestamp is None:
                continue

            # Convert pandas Timestamp to Python datetime
            if hasattr(timestamp, "to_pydatetime"):
                timestamp = timestamp.to_pydatetime()

            record = {
                "asset": asset,
                "market": market,
                "interval": interval,
                "timestamp": timestamp,
                "open": float(row["open"]) if pd.notna(row.get("open")) else None,
                "high": float(row["high"]) if pd.notna(row.get("high")) else None,
                "low": float(row["low"]) if pd.notna(row.get("low")) else None,
                "close": float(row["close"]) if pd.notna(row.get("close")) else None,
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                "created_at": datetime.now(),
                "source": market,
            }
            records.append(record)

        if not records:
            return 0

        # Use PostgreSQL INSERT ... ON CONFLICT DO UPDATE (upsert)
        # This handles duplicates by updating the existing record
        inserted_count = 0
        batch_size = 500  # Insert in batches to avoid memory issues

        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]

            stmt = pg_insert(PriceData).values(batch)

            # On conflict, update the price data (keeps the latest values)
            # Uses the constraint that includes COALESCE(instrument_key, 'none')
            # For COMEX data without instrument_key, it will use 'none'
            stmt = stmt.on_conflict_do_update(
                constraint="uq_price_data_asset_market_interval_contract_timestamp",
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "created_at": stmt.excluded.created_at,
                    "source": stmt.excluded.source,
                }
            )

            await db.execute(stmt)
            inserted_count += len(batch)

        await db.commit()
        return inserted_count

    async def get_sync_status(
        self,
        db: AsyncSession,
        asset: str,
        market: str,
        interval: str,
    ) -> Dict[str, Any]:
        """
        Get the sync status for an asset.

        Returns:
            Dict with sync status info
        """
        latest = await self._get_latest_timestamp(db, asset, market, interval)

        # Count total records
        from sqlalchemy import func
        result = await db.execute(
            select(func.count(PriceData.id))
            .where(
                PriceData.asset == asset,
                PriceData.market == market,
                PriceData.interval == interval,
            )
        )
        total_records = result.scalar_one()

        return {
            "asset": asset,
            "market": market,
            "interval": interval,
            "latest_timestamp": latest.isoformat() if latest else None,
            "total_records": total_records,
            "is_stale": latest < datetime.now() - timedelta(hours=1) if latest else True,
        }


# Singleton instance
data_sync_service = DataSyncService()
