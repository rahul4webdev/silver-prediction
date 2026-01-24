"""
Notification API endpoints.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.services.telegram_notifier import telegram_notifier
from app.services.prediction_engine import prediction_engine
from app.services.upstox_client import upstox_client
from app.core.config import settings

router = APIRouter(prefix="/notifications")


@router.get("/status")
async def get_notification_status() -> Dict[str, Any]:
    """
    Get notification service status.
    """
    return {
        "telegram_configured": telegram_notifier.is_configured,
        "bot_token_set": bool(settings.telegram_bot_token),
        "chat_id_set": bool(settings.telegram_chat_id),
    }


@router.post("/test")
async def send_test_notification() -> Dict[str, Any]:
    """
    Send a test notification to verify Telegram setup.
    """
    if not telegram_notifier.is_configured:
        raise HTTPException(
            status_code=400,
            detail="Telegram not configured. Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID",
        )

    message = f"""
🧪 <b>Test Notification</b>

This is a test message from the Silver Prediction System.

✅ Telegram integration is working correctly!

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    success = await telegram_notifier.send_message(message.strip())

    if success:
        return {"status": "success", "message": "Test notification sent"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send test notification")


@router.post("/auth-reminder")
async def send_auth_reminder_now() -> Dict[str, Any]:
    """
    Manually trigger the auth reminder notification.
    """
    if not telegram_notifier.is_configured:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    success = await telegram_notifier.send_auth_reminder()
    return {"status": "success" if success else "failed"}


@router.post("/platform-health")
async def send_platform_health_now(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Manually trigger platform health notification.
    """
    if not telegram_notifier.is_configured:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    # Check various health indicators
    api_healthy = True
    db_healthy = True  # If we got here, DB is working

    # Check Redis
    redis_healthy = False
    try:
        import redis.asyncio as redis
        r = redis.from_url(settings.redis_url)
        await r.ping()
        redis_healthy = True
        await r.close()
    except Exception:
        pass

    # Check Upstox
    upstox_authenticated = False
    try:
        auth_status = await upstox_client.verify_authentication()
        upstox_authenticated = auth_status.get("authenticated", False)
    except Exception:
        pass

    # Check models
    models_trained = {}
    for interval in ["30m", "1h", "4h", "1d"]:
        try:
            ensemble = prediction_engine.get_ensemble(interval)
            models_trained[interval] = ensemble.is_trained
        except Exception:
            models_trained[interval] = False

    success = await telegram_notifier.send_platform_health(
        api_healthy=api_healthy,
        db_healthy=db_healthy,
        redis_healthy=redis_healthy,
        upstox_authenticated=upstox_authenticated,
        tick_collector_running=False,  # Can't check from API context
        scheduler_running=True,
        models_trained=models_trained,
    )

    return {"status": "success" if success else "failed"}


@router.post("/predictions/{interval}")
async def send_predictions_now(
    interval: str,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Manually trigger prediction notification for a specific interval.
    """
    if not telegram_notifier.is_configured:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    if interval not in ["30m", "1h", "4h", "1d"]:
        raise HTTPException(status_code=400, detail="Invalid interval")

    predictions = []

    # Get MCX contracts
    try:
        contracts = await upstox_client.get_all_silver_instrument_keys()
    except Exception:
        contracts = [{"contract_type": "SILVER", "instrument_key": None, "trading_symbol": None}]

    # Generate MCX predictions
    for contract in contracts[:3]:
        try:
            prediction = await prediction_engine.generate_prediction(
                db,
                asset="silver",
                market="mcx",
                interval=interval,
                instrument_key=contract.get("instrument_key"),
                contract_type=contract.get("contract_type"),
                trading_symbol=contract.get("trading_symbol"),
            )
            predictions.append(prediction)
        except Exception as e:
            pass

    # Generate COMEX prediction
    try:
        prediction = await prediction_engine.generate_prediction(
            db,
            asset="silver",
            market="comex",
            interval=interval,
        )
        predictions.append(prediction)
    except Exception:
        pass

    if not predictions:
        raise HTTPException(status_code=500, detail="No predictions generated")

    success = await telegram_notifier.send_predictions(interval, predictions)

    return {
        "status": "success" if success else "failed",
        "predictions_count": len(predictions),
    }


@router.post("/alert")
async def send_custom_alert(
    component: str = Query(..., description="Component name"),
    message: str = Query(..., description="Alert message"),
    severity: str = Query("warning", description="Severity: info, warning, error, critical"),
) -> Dict[str, Any]:
    """
    Send a custom alert notification.
    """
    if not telegram_notifier.is_configured:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    if severity not in ["info", "warning", "error", "critical"]:
        raise HTTPException(status_code=400, detail="Invalid severity")

    success = await telegram_notifier.send_error_alert(
        component=component,
        error=message,
        severity=severity,
    )

    return {"status": "success" if success else "failed"}
