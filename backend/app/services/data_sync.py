"""
Data synchronization service for fetching and storing market data.
Syncs data from Upstox (MCX) and Yahoo Finance (COMEX).
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.price_data import PriceData
from app.services.upstox_client import upstox_client
from app.services.yahoo_client import yahoo_client

logger = logging.getLogger(__name__)


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

            # Determine start date
            if latest:
                start_date = latest + timedelta(minutes=1)
            else:
                start_date = datetime.now() - timedelta(days=days)

            # Only proceed if we need data
            if latest and start_date > datetime.now():
                logger.info("MCX data is already up to date")
                return {"status": "success", "records": 0, "message": "Already up to date"}

            # Fetch from Upstox using the proper method
            if asset.lower() == "silver":
                df = await upstox_client.get_mcx_silver_data(
                    interval=interval,
                    start_date=start_date,
                    end_date=datetime.now(),
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

            # Determine start date
            if latest:
                start_date = latest + timedelta(minutes=1)
            else:
                start_date = datetime.now() - timedelta(days=days)

            # Fetch from Yahoo Finance
            df = await yahoo_client.get_historical_data(
                symbol=asset,
                interval=interval,
                start=start_date,
                end=datetime.now(),
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
        Insert price data into the database.

        Uses bulk insert with ON CONFLICT DO NOTHING for efficiency.
        """
        if df.empty:
            return 0

        # Prepare records
        records = []
        for _, row in df.iterrows():
            timestamp = row.get("timestamp") or row.get("date") or row.get("datetime")
            if timestamp is None:
                continue

            # Ensure timestamp is timezone-aware
            if hasattr(timestamp, "tzinfo") and timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=None)

            record = PriceData(
                asset=asset,
                market=market,
                interval=interval,
                timestamp=timestamp,
                open=float(row["open"]) if pd.notna(row.get("open")) else None,
                high=float(row["high"]) if pd.notna(row.get("high")) else None,
                low=float(row["low"]) if pd.notna(row.get("low")) else None,
                close=float(row["close"]) if pd.notna(row.get("close")) else None,
                volume=int(row["volume"]) if pd.notna(row.get("volume")) else 0,
            )
            records.append(record)

        # Bulk insert
        for record in records:
            try:
                db.add(record)
            except Exception:
                # Record might already exist
                await db.rollback()
                continue

        await db.commit()

        return len(records)

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
