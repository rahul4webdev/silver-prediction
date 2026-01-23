"""
Confluence and correlation API endpoints.
"""

import logging
from typing import Any, Dict

from fastapi import APIRouter, Query

from app.models.database import get_db_session
from app.services.confluence import confluence_detector, correlation_analyzer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/confluence")


@router.get("/detect", response_model=Dict[str, Any])
async def detect_confluence(
    asset: str = Query(default="silver"),
    market: str = Query(default="mcx"),
):
    """
    Detect multi-timeframe confluence.

    Checks if predictions across 30m, 1h, 4h, and daily intervals
    are aligned in the same direction.

    Strong confluence (3+ timeframes aligned) indicates
    higher probability trade setups.
    """
    try:
        async with get_db_session() as db:
            result = await confluence_detector.detect_confluence(db, asset, market)
            return result
    except Exception as e:
        logger.error(f"Error detecting confluence: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/history", response_model=Dict[str, Any])
async def get_confluence_history(
    asset: str = Query(default="silver"),
    market: str = Query(default="mcx"),
    hours: int = Query(default=24, le=168),
):
    """
    Get confluence detection history.
    """
    try:
        async with get_db_session() as db:
            history = await confluence_detector.get_confluence_history(
                db, asset, market, hours
            )
            return {
                "status": "success",
                "asset": asset,
                "market": market,
                "hours": hours,
                "count": len(history),
                "history": history,
            }
    except Exception as e:
        logger.error(f"Error getting confluence history: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/correlations", response_model=Dict[str, Any])
async def get_correlations(
    lookback_days: int = Query(default=30, ge=7, le=90),
):
    """
    Get correlations between silver and related assets.

    Analyzes correlations with:
    - Gold (typically +0.85)
    - DXY/Dollar Index (typically -0.75)
    - S&P 500 (variable)
    - Copper (industrial indicator)
    - Oil (inflation proxy)

    Also identifies current market regime based on correlation patterns.
    """
    try:
        result = await correlation_analyzer.calculate_correlations(lookback_days)
        return result
    except Exception as e:
        logger.error(f"Error calculating correlations: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
