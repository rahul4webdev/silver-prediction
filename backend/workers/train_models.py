#!/usr/bin/env python3
"""
Model Training Worker.
Runs training for all asset/market/interval combinations.

Usage:
    python -m workers.train_models [--asset silver] [--market mcx] [--interval 30m]

Without arguments, trains all combinations:
- Markets: mcx, comex
- Intervals: 30m, 1h, 4h, 1d
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config import settings
from app.models.database import init_db, get_db_session
from app.services.prediction_engine import prediction_engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Training configurations
MARKETS = ["mcx", "comex"]
INTERVALS = ["30m", "1h", "4h", "1d"]


async def train_single(asset: str, market: str, interval: str) -> dict:
    """Train models for a single combination."""
    logger.info(f"Training {asset}/{market}/{interval}...")

    async with get_db_session() as db:
        try:
            result = await prediction_engine.train_models(db, asset, market, interval)
            logger.info(f"✓ Trained {asset}/{market}/{interval}: {result.get('training_samples', 0)} samples")
            return {
                "status": "success",
                "asset": asset,
                "market": market,
                "interval": interval,
                "result": result,
            }
        except ValueError as e:
            logger.warning(f"⚠ Skipped {asset}/{market}/{interval}: {e}")
            return {
                "status": "skipped",
                "asset": asset,
                "market": market,
                "interval": interval,
                "reason": str(e),
            }
        except Exception as e:
            logger.error(f"✗ Failed {asset}/{market}/{interval}: {e}")
            return {
                "status": "error",
                "asset": asset,
                "market": market,
                "interval": interval,
                "error": str(e),
            }


async def train_all(asset: str = "silver", markets: list = None, intervals: list = None):
    """Train models for all combinations."""
    markets = markets or MARKETS
    intervals = intervals or INTERVALS

    logger.info("=" * 60)
    logger.info("Starting Model Training")
    logger.info(f"Asset: {asset}")
    logger.info(f"Markets: {markets}")
    logger.info(f"Intervals: {intervals}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    # Initialize database
    await init_db()

    results = []
    successful = 0
    failed = 0
    skipped = 0

    for market in markets:
        for interval in intervals:
            result = await train_single(asset, market, interval)
            results.append(result)

            if result["status"] == "success":
                successful += 1
            elif result["status"] == "error":
                failed += 1
            else:
                skipped += 1

    # Summary
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Skipped (insufficient data): {skipped}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 60)

    return {
        "total": len(results),
        "successful": successful,
        "failed": failed,
        "skipped": skipped,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Train ML models for price prediction")
    parser.add_argument("--asset", default="silver", help="Asset to train (default: silver)")
    parser.add_argument("--market", help="Specific market to train (mcx or comex)")
    parser.add_argument("--interval", help="Specific interval to train (30m, 1h, 4h, 1d)")

    args = parser.parse_args()

    markets = [args.market] if args.market else MARKETS
    intervals = [args.interval] if args.interval else INTERVALS

    result = asyncio.run(train_all(args.asset, markets, intervals))

    # Exit with error code if any failures
    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
