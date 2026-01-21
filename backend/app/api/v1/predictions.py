"""
Prediction API endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import Asset, Market, PredictionInterval
from app.models.database import get_db
from app.models.predictions import Prediction
from app.services.prediction_engine import prediction_engine

router = APIRouter(prefix="/predictions")


@router.post("/generate")
async def generate_prediction(
    asset: str = Query("silver", description="Asset to predict"),
    market: str = Query("mcx", description="Market (mcx or comex)"),
    interval: str = Query("30m", description="Prediction interval"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Generate a new prediction for the specified asset/market/interval.

    This will:
    1. Fetch recent price data
    2. Calculate technical indicators
    3. Run the ensemble model
    4. Store and return the prediction
    """
    try:
        result = await prediction_engine.generate_prediction(
            db, asset, market, interval
        )
        return {
            "status": "success",
            "prediction": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/latest")
async def get_latest_prediction(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get the most recent prediction.
    """
    try:
        conditions = [Prediction.asset == asset]

        if market:
            conditions.append(Prediction.market == market)
        if interval:
            conditions.append(Prediction.interval == interval)

        query = (
            select(Prediction)
            .where(and_(*conditions))
            .order_by(desc(Prediction.created_at))
            .limit(1)
        )

        result = await db.execute(query)
        prediction = result.scalar_one_or_none()

        if not prediction:
            return {
                "status": "no_predictions",
                "message": "No predictions available yet. Models need to be trained first.",
                "asset": asset,
                "market": market,
                "interval": interval,
            }

        return prediction.to_dict()
    except Exception as e:
        return {
            "status": "error",
            "message": f"Database error: {str(e)}",
            "asset": asset,
        }


@router.get("/history")
async def get_prediction_history(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    verified_only: bool = Query(False),
    limit: int = Query(100, le=500),
    offset: int = Query(0),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get prediction history with optional filters.
    """
    conditions = [Prediction.asset == asset]

    if market:
        conditions.append(Prediction.market == market)
    if interval:
        conditions.append(Prediction.interval == interval)
    if verified_only:
        conditions.append(Prediction.verified_at.isnot(None))

    query = (
        select(Prediction)
        .where(and_(*conditions))
        .order_by(desc(Prediction.created_at))
        .offset(offset)
        .limit(limit)
    )

    result = await db.execute(query)
    predictions = result.scalars().all()

    return {
        "total": len(predictions),
        "offset": offset,
        "limit": limit,
        "predictions": [p.to_dict() for p in predictions],
    }


@router.post("/verify")
async def trigger_verification(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Manually trigger verification of pending predictions.
    """
    try:
        result = await prediction_engine.verify_pending_predictions(db)
        return {
            "status": "success",
            "result": result,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get a specific prediction by ID.
    """
    query = select(Prediction).where(Prediction.id == prediction_id)
    result = await db.execute(query)
    prediction = result.scalar_one_or_none()

    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return prediction.to_dict()


@router.post("/train")
async def train_models(
    asset: str = Query("silver"),
    market: str = Query("mcx"),
    interval: str = Query("30m"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Train ML models on historical data.

    Warning: This is a long-running operation.
    """
    try:
        result = await prediction_engine.train_models(db, asset, market, interval)
        return {
            "status": "success",
            "result": result,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info(
    interval: str = Query("30m"),
) -> Dict[str, Any]:
    """
    Get information about the ensemble model.
    """
    try:
        ensemble = prediction_engine.get_ensemble(interval)
        return ensemble.get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
