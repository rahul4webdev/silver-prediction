"""
System Status API endpoints.
Provides comprehensive status of all services, models, and scheduled tasks.
"""

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends
from sqlalchemy import text, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import get_db

router = APIRouter(prefix="/status")


@router.get("/")
async def get_system_status(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get comprehensive system status including all services and models.
    """
    status = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": settings.environment,
        "services": {},
        "models": {},
        "scheduler": {},
        "database": {},
        "data": {},
    }

    # ==================== Database Status ====================
    try:
        await db.execute(text("SELECT 1"))
        status["services"]["database"] = {
            "status": "healthy",
            "connected": True,
        }

        # Get prediction counts
        from app.models.predictions import Prediction
        total_result = await db.execute(select(func.count()).select_from(Prediction))
        total_count = total_result.scalar() or 0

        verified_result = await db.execute(
            select(func.count()).select_from(Prediction).where(Prediction.verified_at.isnot(None))
        )
        verified_count = verified_result.scalar() or 0

        pending_result = await db.execute(
            select(func.count()).select_from(Prediction).where(Prediction.verified_at.is_(None))
        )
        pending_count = pending_result.scalar() or 0

        # Get today's predictions count
        today = datetime.now(timezone.utc).date()
        today_result = await db.execute(
            select(func.count()).select_from(Prediction).where(
                func.date(Prediction.created_at) == today
            )
        )
        today_count = today_result.scalar() or 0

        status["database"] = {
            "predictions": {
                "total": total_count,
                "verified": verified_count,
                "pending": pending_count,
                "today": today_count,
            }
        }
    except Exception as e:
        status["services"]["database"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }

    # ==================== Redis Status ====================
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.redis_url)
        await r.ping()
        info = await r.info("memory")
        await r.close()
        status["services"]["redis"] = {
            "status": "healthy",
            "connected": True,
            "memory_used": info.get("used_memory_human", "unknown"),
        }
    except Exception as e:
        status["services"]["redis"] = {
            "status": "unhealthy",
            "connected": False,
            "error": str(e),
        }

    # ==================== Upstox Status ====================
    try:
        from app.services.upstox_client import upstox_client
        # Reload token from .env in case it was updated by OAuth in another process
        upstox_client.reload_token_from_env()
        auth_status = await upstox_client.verify_authentication()
        user_data = auth_status.get("user", {})
        status["services"]["upstox"] = {
            "status": "healthy" if auth_status.get("authenticated") else "needs_auth",
            "authenticated": auth_status.get("authenticated", False),
            "user_id": user_data.get("user_id"),
            "user_name": user_data.get("user_name"),
            "reason": auth_status.get("reason"),
            "reauth_url": "/api/v1/auth/upstox/reauth" if not auth_status.get("authenticated") else None,
        }
    except Exception as e:
        status["services"]["upstox"] = {
            "status": "unhealthy",
            "authenticated": False,
            "error": str(e),
            "reauth_url": "/api/v1/auth/upstox/reauth",
        }

    # ==================== Tick Collector Status ====================
    try:
        from app.services.tick_collector import tick_collector
        stats = tick_collector.stats
        last_tick = stats.get("last_tick_time")
        tick_count = stats.get("ticks_received", 0)
        subscribed = stats.get("contracts_subscribed", 0)

        status["services"]["tick_collector"] = {
            "status": "running" if tick_collector.is_running else "stopped",
            "is_running": tick_collector.is_running,
            "last_tick": last_tick.isoformat() if last_tick else None,
            "tick_count": tick_count,
            "subscribed_instruments": subscribed,
        }
    except Exception as e:
        status["services"]["tick_collector"] = {
            "status": "error",
            "is_running": False,
            "error": str(e),
        }

    # ==================== Notification Worker Status ====================
    try:
        from workers.notification_worker import notification_worker
        worker_status = notification_worker.get_status()
        status["scheduler"] = {
            "status": "running" if worker_status.get("is_running") else "stopped",
            "is_running": worker_status.get("is_running", False),
            "jobs": worker_status.get("jobs", []),
        }
    except Exception as e:
        status["scheduler"] = {
            "status": "unknown",
            "is_running": False,
            "error": str(e),
        }

    # ==================== ML Models Status ====================
    try:
        from app.services.prediction_engine import prediction_engine

        models_status = {}
        for interval in ["30m", "1h", "4h", "1d"]:
            try:
                ensemble = prediction_engine.get_ensemble(interval)
                models_status[interval] = {
                    "is_trained": ensemble.is_trained,
                    "models": {},
                }

                # Check individual models - ensemble has named attributes, not a dict
                model_names = ["prophet", "lstm", "xgboost", "gru", "random_forest"]
                for name in model_names:
                    model = getattr(ensemble, name, None)
                    if model is not None:
                        models_status[interval]["models"][name] = {
                            "is_trained": model.is_trained if hasattr(model, 'is_trained') else False,
                            "weight": ensemble.weights.get(name, 0),
                        }
            except Exception as e:
                models_status[interval] = {
                    "is_trained": False,
                    "error": str(e),
                }

        status["models"] = models_status
    except Exception as e:
        status["models"] = {"error": str(e)}

    # ==================== News/Sentiment Status ====================
    try:
        from app.services.news_sentiment import news_sentiment_service

        # Check cache status
        cache_size = len(news_sentiment_service._cache)
        status["services"]["news_sentiment"] = {
            "status": "available",
            "cache_entries": cache_size,
            "newsapi_configured": bool(news_sentiment_service.newsapi_key),
        }
    except Exception as e:
        status["services"]["news_sentiment"] = {
            "status": "error",
            "error": str(e),
        }

    # ==================== Trading Hours Status ====================
    try:
        import pytz
        IST = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(timezone.utc).astimezone(IST)
    except ImportError:
        ist_offset = timedelta(hours=5, minutes=30)
        now_ist = datetime.now(timezone.utc) + ist_offset

    weekday = now_ist.weekday()
    is_weekend = weekday >= 5
    hour = now_ist.hour
    minute = now_ist.minute
    current_time = hour * 60 + minute
    market_open = 9 * 60  # 9:00 AM
    market_close = 23 * 60 + 30  # 11:30 PM

    is_trading_hours = market_open <= current_time <= market_close and not is_weekend

    status["market"] = {
        "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "is_weekend": is_weekend,
        "is_trading_hours": is_trading_hours,
        "market_status": "open" if is_trading_hours else "closed",
        "hours": {
            "open": "9:00 AM IST",
            "close": "11:30 PM IST",
        },
    }

    # ==================== Overall Health ====================
    # Core services that must be healthy
    core_healthy = (
        status["services"].get("database", {}).get("status") == "healthy" and
        status["services"].get("redis", {}).get("status") == "healthy"
    )

    # Upstox is optional - can use Yahoo Finance proxy
    upstox_ok = status["services"].get("upstox", {}).get("authenticated", False)

    # Determine overall status
    if core_healthy and upstox_ok:
        status["overall_health"] = "healthy"
    elif core_healthy:
        status["overall_health"] = "operational"  # Working but using fallback data source
    else:
        status["overall_health"] = "degraded"

    return status


@router.get("/services")
async def get_services_status(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get status of individual services."""
    services = {}

    # Database
    try:
        await db.execute(text("SELECT 1"))
        services["database"] = {"status": "healthy", "connected": True}
    except Exception as e:
        services["database"] = {"status": "unhealthy", "error": str(e)}

    # Redis
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.redis_url)
        await r.ping()
        await r.close()
        services["redis"] = {"status": "healthy", "connected": True}
    except Exception as e:
        services["redis"] = {"status": "unhealthy", "error": str(e)}

    # Upstox
    try:
        from app.services.upstox_client import upstox_client
        auth = await upstox_client.verify_authentication()
        services["upstox"] = {
            "status": "authenticated" if auth.get("authenticated") else "needs_auth",
            "authenticated": auth.get("authenticated", False),
        }
    except Exception as e:
        services["upstox"] = {"status": "error", "error": str(e)}

    return services


@router.get("/models")
async def get_models_status() -> Dict[str, Any]:
    """Get status of all ML models."""
    try:
        from app.services.prediction_engine import prediction_engine

        models_status = {}
        for interval in ["30m", "1h", "4h", "1d"]:
            try:
                ensemble = prediction_engine.get_ensemble(interval)
                info = ensemble.get_info()
                models_status[interval] = {
                    "is_trained": ensemble.is_trained,
                    "total_models": info.get("total_models", 0),
                    "trained_models": info.get("trained_models", 0),
                    "models": {},
                }

                # Check individual models - ensemble has named attributes, not a dict
                model_names = ["prophet", "lstm", "xgboost", "gru", "random_forest"]
                for name in model_names:
                    model = getattr(ensemble, name, None)
                    if model is not None:
                        models_status[interval]["models"][name] = {
                            "is_trained": model.is_trained if hasattr(model, 'is_trained') else False,
                            "weight": ensemble.weights.get(name, 0),
                        }
            except Exception as e:
                models_status[interval] = {"error": str(e)}

        return {
            "status": "ok",
            "intervals": models_status,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@router.get("/scheduler")
async def get_scheduler_status() -> Dict[str, Any]:
    """Get status of scheduled jobs."""
    try:
        from workers.notification_worker import notification_worker
        return notification_worker.get_status()
    except Exception as e:
        return {
            "is_running": False,
            "error": str(e),
        }


@router.get("/predictions/summary")
async def get_predictions_summary(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Get summary of predictions."""
    try:
        from app.models.predictions import Prediction

        # Total counts
        total_result = await db.execute(select(func.count()).select_from(Prediction))
        total = total_result.scalar() or 0

        verified_result = await db.execute(
            select(func.count()).select_from(Prediction).where(Prediction.verified_at.isnot(None))
        )
        verified = verified_result.scalar() or 0

        correct_result = await db.execute(
            select(func.count()).select_from(Prediction).where(
                Prediction.verified_at.isnot(None),
                Prediction.is_direction_correct == True
            )
        )
        correct = correct_result.scalar() or 0

        # Today's counts
        today = datetime.now(timezone.utc).date()
        today_result = await db.execute(
            select(func.count()).select_from(Prediction).where(
                func.date(Prediction.created_at) == today
            )
        )
        today_count = today_result.scalar() or 0

        # Accuracy
        accuracy = (correct / verified * 100) if verified > 0 else 0

        return {
            "total_predictions": total,
            "verified_predictions": verified,
            "pending_predictions": total - verified,
            "correct_predictions": correct,
            "accuracy_percent": round(accuracy, 2),
            "predictions_today": today_count,
        }
    except Exception as e:
        return {"error": str(e)}
