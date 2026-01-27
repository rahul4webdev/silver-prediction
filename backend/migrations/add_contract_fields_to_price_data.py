"""
Migration: Add contract-specific fields to price_data table.

This migration adds instrument_key, contract_type, trading_symbol, and expiry columns
to support per-contract data storage for MCX silver (SILVER, SILVERM, SILVERMIC).

Usage:
    cd backend
    python -m migrations.add_contract_fields_to_price_data

What this does:
1. Adds new columns (nullable) to price_data table
2. Drops old unique constraint
3. Creates new unique constraint including instrument_key
4. Creates new indexes for contract-based queries
5. Optionally clears old MCX data (recommended for clean start)
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from app.models.database import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def migrate():
    """Run the migration."""
    async with engine.begin() as conn:
        logger.info("Starting migration: add_contract_fields_to_price_data")

        # Step 1: Add new columns (if they don't exist)
        logger.info("Step 1: Adding new columns...")

        columns_to_add = [
            ("instrument_key", "VARCHAR(50)", "Upstox instrument key"),
            ("contract_type", "VARCHAR(20)", "Contract type: SILVER, SILVERM, SILVERMIC"),
            ("trading_symbol", "VARCHAR(100)", "Human-readable symbol"),
            ("expiry", "TIMESTAMP WITH TIME ZONE", "Contract expiry date"),
        ]

        for col_name, col_type, comment in columns_to_add:
            try:
                await conn.execute(text(f"""
                    ALTER TABLE price_data
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type};
                """))
                await conn.execute(text(f"""
                    COMMENT ON COLUMN price_data.{col_name} IS '{comment}';
                """))
                logger.info(f"  Added column: {col_name}")
            except Exception as e:
                logger.warning(f"  Column {col_name} may already exist: {e}")

        # Step 2: Drop old unique constraint
        logger.info("Step 2: Dropping old unique constraint...")
        try:
            await conn.execute(text("""
                ALTER TABLE price_data
                DROP CONSTRAINT IF EXISTS uq_price_data_asset_market_interval_timestamp;
            """))
            logger.info("  Dropped old unique constraint")
        except Exception as e:
            logger.warning(f"  Could not drop old constraint: {e}")

        # Step 3: Create new unique constraint (includes instrument_key)
        logger.info("Step 3: Creating new unique constraint...")
        try:
            await conn.execute(text("""
                ALTER TABLE price_data
                ADD CONSTRAINT uq_price_data_asset_market_interval_contract_timestamp
                UNIQUE (asset, market, interval, instrument_key, timestamp);
            """))
            logger.info("  Created new unique constraint")
        except Exception as e:
            logger.warning(f"  Could not create new constraint (may exist): {e}")

        # Step 4: Create new indexes
        logger.info("Step 4: Creating new indexes...")

        indexes = [
            ("idx_price_data_contract_lookup",
             "asset, market, interval, instrument_key, timestamp"),
            ("idx_price_data_contract_type",
             "asset, market, contract_type"),
            ("idx_price_data_instrument_key",
             "instrument_key"),
        ]

        for idx_name, idx_cols in indexes:
            try:
                await conn.execute(text(f"""
                    CREATE INDEX IF NOT EXISTS {idx_name}
                    ON price_data ({idx_cols});
                """))
                logger.info(f"  Created index: {idx_name}")
            except Exception as e:
                logger.warning(f"  Could not create index {idx_name}: {e}")

        logger.info("Migration completed successfully!")


async def clear_mcx_data():
    """Clear old MCX data to prepare for per-contract re-sync."""
    async with engine.begin() as conn:
        logger.info("Clearing old MCX price_data...")

        result = await conn.execute(text("""
            DELETE FROM price_data WHERE market = 'mcx';
        """))

        logger.info(f"Deleted {result.rowcount} MCX price_data rows")

        # Also clear old MCX predictions (optional, as they used wrong data)
        result = await conn.execute(text("""
            DELETE FROM predictions WHERE market = 'mcx';
        """))

        logger.info(f"Deleted {result.rowcount} MCX predictions")


async def main():
    """Run migration and optionally clear MCX data."""
    import argparse

    parser = argparse.ArgumentParser(description="Add contract fields to price_data")
    parser.add_argument("--clear-mcx", action="store_true",
                        help="Clear old MCX data after migration")
    args = parser.parse_args()

    await migrate()

    if args.clear_mcx:
        await clear_mcx_data()
        logger.info("\nMCX data cleared. Now run data sync to re-fetch with contract info.")


if __name__ == "__main__":
    asyncio.run(main())
