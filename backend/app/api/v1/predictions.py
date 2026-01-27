"""
Prediction API endpoints.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.constants import Asset, Market, PredictionInterval
from app.models.database import get_db, get_db_session
from app.models.predictions import Prediction
from app.services.prediction_engine import prediction_engine

logger = logging.getLogger(__name__)

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
    try:
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
    except Exception as e:
        return {
            "total": 0,
            "offset": offset,
            "limit": limit,
            "predictions": [],
            "message": f"No data available: {str(e)}",
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


# Training status tracker
_training_status: Dict[str, Any] = {
    "is_running": False,
    "started_at": None,
    "completed_at": None,
    "current_task": None,
    "progress": [],
    "results": [],
}


async def _train_all_background(asset: str = "silver"):
    """Background task to train all market/interval combinations."""
    global _training_status

    markets = ["mcx", "comex"]
    intervals = ["30m", "1h", "4h", "1d"]

    _training_status["is_running"] = True
    _training_status["started_at"] = datetime.now().isoformat()
    _training_status["completed_at"] = None
    _training_status["progress"] = []
    _training_status["results"] = []

    for market in markets:
        for interval in intervals:
            task_name = f"{asset}/{market}/{interval}"
            _training_status["current_task"] = task_name
            _training_status["progress"].append(f"Training {task_name}...")

            try:
                async with get_db_session() as db:
                    result = await prediction_engine.train_models(db, asset, market, interval)
                    _training_status["results"].append({
                        "task": task_name,
                        "status": "success",
                        "samples": result.get("training_samples", 0),
                    })
                    _training_status["progress"].append(f"Completed {task_name}: {result.get('training_samples', 0)} samples")
                    logger.info(f"Trained {task_name}: {result.get('training_samples', 0)} samples")
            except ValueError as e:
                _training_status["results"].append({
                    "task": task_name,
                    "status": "skipped",
                    "reason": str(e),
                })
                _training_status["progress"].append(f"Skipped {task_name}: {e}")
                logger.warning(f"Skipped {task_name}: {e}")
            except Exception as e:
                _training_status["results"].append({
                    "task": task_name,
                    "status": "error",
                    "error": str(e),
                })
                _training_status["progress"].append(f"Error {task_name}: {e}")
                logger.error(f"Error training {task_name}: {e}")

    _training_status["is_running"] = False
    _training_status["completed_at"] = datetime.now().isoformat()
    _training_status["current_task"] = None
    logger.info("All training complete")


@router.post("/train-all")
async def train_all_models(
    background_tasks: BackgroundTasks,
    asset: str = Query("silver"),
) -> Dict[str, Any]:
    """
    Train all ML models for all markets and intervals.

    This runs as a background task to avoid timeout.
    Use GET /predictions/train-status to check progress.
    """
    global _training_status

    if _training_status["is_running"]:
        return {
            "status": "already_running",
            "message": "Training is already in progress",
            "started_at": _training_status["started_at"],
            "current_task": _training_status["current_task"],
        }

    background_tasks.add_task(_train_all_background, asset)

    return {
        "status": "started",
        "message": "Training started in background. Use GET /predictions/train-status to check progress.",
        "asset": asset,
        "markets": ["mcx", "comex"],
        "intervals": ["30m", "1h", "4h", "1d"],
    }


@router.get("/train-status")
async def get_training_status() -> Dict[str, Any]:
    """
    Get the status of background training.
    """
    try:
        return {
            "is_running": _training_status.get("is_running", False),
            "started_at": _training_status.get("started_at"),
            "completed_at": _training_status.get("completed_at"),
            "current_task": _training_status.get("current_task"),
            "progress": _training_status.get("progress", [])[-10:],  # Last 10 entries
            "results": _training_status.get("results", []),
        }
    except Exception as e:
        return {
            "is_running": False,
            "error": str(e),
        }


@router.delete("/clear-all")
async def clear_all_predictions(
    confirm: bool = Query(False, description="Set to true to confirm deletion"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Delete all predictions from the database.

    WARNING: This is a destructive operation. Set confirm=true to proceed.
    """
    if not confirm:
        # Get count first
        count_result = await db.execute(select(func.count()).select_from(Prediction))
        count = count_result.scalar() or 0
        return {
            "status": "confirmation_required",
            "message": f"This will delete {count} predictions. Set confirm=true to proceed.",
            "predictions_count": count,
        }

    try:
        # Get count before deletion
        count_result = await db.execute(select(func.count()).select_from(Prediction))
        count = count_result.scalar() or 0

        # Delete all predictions
        await db.execute(delete(Prediction))
        await db.commit()

        logger.info(f"Deleted {count} predictions")

        return {
            "status": "success",
            "message": f"Deleted {count} predictions",
            "deleted_count": count,
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to delete predictions: {str(e)}")


# NOTE: This route MUST be last since it matches any path pattern
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
