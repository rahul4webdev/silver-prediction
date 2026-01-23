"""
Macro economic data API endpoints.
Provides access to DXY, Fear & Greed Index, COT data, and more.
"""

import logging
from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Query

from app.services.macro_data import macro_data_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/macro")


@router.get("/all", response_model=Dict[str, Any])
async def get_all_macro_data(
    asset: str = Query(default="silver", description="Asset to get macro data for"),
):
    """
    Get all macro economic indicators relevant to silver/gold.

    Returns DXY, Fear & Greed Index, COT data, Treasury yields,
    and a composite signal indicating overall macro sentiment.
    """
    try:
        data = await macro_data_service.get_all_macro_data(asset=asset)
        return {
            "status": "success",
            **data,
        }
    except Exception as e:
        logger.error(f"Error fetching macro data: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/dxy", response_model=Dict[str, Any])
async def get_dxy():
    """
    Get US Dollar Index (DXY) data.

    DXY has a strong inverse correlation with silver (-0.7 to -0.8).
    A weakening dollar is typically bullish for precious metals.
    """
    try:
        data = await macro_data_service.get_dxy()
        return {
            "status": "success",
            **data,
        }
    except Exception as e:
        logger.error(f"Error fetching DXY: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/fear-greed", response_model=Dict[str, Any])
async def get_fear_greed():
    """
    Get CNN Fear & Greed Index.

    The index ranges from 0 (Extreme Fear) to 100 (Extreme Greed).
    Extreme fear often signals buying opportunities for precious metals.
    """
    try:
        data = await macro_data_service.get_fear_greed_index()
        return {
            "status": "success",
            **data,
        }
    except Exception as e:
        logger.error(f"Error fetching Fear & Greed: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/cot", response_model=Dict[str, Any])
async def get_cot(
    asset: str = Query(default="silver", description="Asset (silver or gold)"),
):
    """
    Get CFTC Commitment of Traders (COT) data.

    Shows positioning of commercials (hedgers) and speculators.
    Extreme speculative positioning often precedes reversals.
    """
    try:
        data = await macro_data_service.get_cot_data(asset=asset)
        return {
            "status": "success",
            **data,
        }
    except Exception as e:
        logger.error(f"Error fetching COT: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/treasury", response_model=Dict[str, Any])
async def get_treasury():
    """
    Get US Treasury yields (10Y, 3M) and yield curve status.

    An inverted yield curve signals recession risk,
    which is typically bullish for precious metals.
    """
    try:
        data = await macro_data_service.get_treasury_yields()
        return {
            "status": "success",
            **data,
        }
    except Exception as e:
        logger.error(f"Error fetching treasury yields: {e}")
        return {
            "status": "error",
            "message": str(e),
        }


@router.get("/features", response_model=Dict[str, Any])
async def get_macro_features(
    asset: str = Query(default="silver", description="Asset to get features for"),
):
    """
    Get macro data formatted as ML features.

    Returns normalized features that can be used in prediction models.
    """
    try:
        macro_data = await macro_data_service.get_all_macro_data(asset=asset)
        features = macro_data_service.macro_to_features(macro_data)
        return {
            "status": "success",
            "asset": asset,
            "features": features,
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error fetching macro features: {e}")
        return {
            "status": "error",
            "message": str(e),
        }
