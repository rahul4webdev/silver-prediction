"""
Health check endpoints.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import get_db

router = APIRouter(prefix="/health")


@router.get("")
async def health_check() -> Dict[str, Any]:
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": settings.environment,
    }


@router.get("/db")
async def database_health(db: AsyncSession = Depends(get_db)) -> Dict[str, Any]:
    """Database health check."""
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        return {
            "status": "healthy",
            "database": "connected",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }


@router.get("/upstox")
async def upstox_health() -> Dict[str, Any]:
    """Upstox API health check."""
    from app.services.upstox_client import upstox_client

    try:
        # Check if token is set
        has_token = upstox_client.access_token is not None
        return {
            "status": "healthy" if has_token else "needs_auth",
            "has_token": has_token,
            "message": "Upstox connected" if has_token else "Token not set, authentication required",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@router.get("/models")
async def models_health() -> Dict[str, Any]:
    """ML models health check."""
    from app.services.prediction_engine import prediction_engine

    model_status = {}
    for interval in ["30m", "1h", "4h", "daily"]:
        try:
            ensemble = prediction_engine.get_ensemble(interval)
            model_status[interval] = {
                "is_trained": ensemble.is_trained,
                "last_trained": ensemble.last_trained.isoformat() if ensemble.last_trained else None,
                "weights": ensemble.weights,
            }
        except Exception as e:
            model_status[interval] = {"error": str(e)}

    trained_count = sum(1 for s in model_status.values() if s.get("is_trained", False))

    return {
        "status": "healthy" if trained_count > 0 else "needs_training",
        "trained_models": trained_count,
        "models": model_status,
    }
