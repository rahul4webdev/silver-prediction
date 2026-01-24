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


@router.post("/daily-report")
async def send_daily_report_now(
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Manually trigger the daily performance report.
    Shows prediction accuracy for today.
    """
    if not telegram_notifier.is_configured:
        raise HTTPException(status_code=400, detail="Telegram not configured")

    from sqlalchemy import text
    from datetime import date as date_type

    today = date_type.today()

    # Get all verified predictions for today
    result = await db.execute(text("""
        SELECT
            interval,
            market,
            COUNT(*) as total,
            SUM(CASE WHEN is_direction_correct = true THEN 1 ELSE 0 END) as successful,
            SUM(CASE WHEN is_direction_correct = false THEN 1 ELSE 0 END) as failed
        FROM predictions
        WHERE DATE(prediction_time) = :today
        AND verified_at IS NOT NULL
        GROUP BY interval, market
    """), {"today": today})

    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No verified predictions found for today")

    # Aggregate stats
    total_predictions = 0
    successful_predictions = 0
    failed_predictions = 0
    interval_stats = {}
    market_stats = {}

    for row in rows:
        interval = row[0]
        market = row[1]
        total = row[2]
        success = row[3] or 0
        failed = row[4] or 0

        total_predictions += total
        successful_predictions += success
        failed_predictions += failed

        # Aggregate by interval
        if interval not in interval_stats:
            interval_stats[interval] = {"total": 0, "success": 0, "failed": 0}
        interval_stats[interval]["total"] += total
        interval_stats[interval]["success"] += success
        interval_stats[interval]["failed"] += failed

        # Aggregate by market
        if market not in market_stats:
            market_stats[market] = {"total": 0, "success": 0, "failed": 0}
        market_stats[market]["total"] += total
        market_stats[market]["success"] += success
        market_stats[market]["failed"] += failed

    # Send the report
    success = await telegram_notifier.send_daily_performance_report(
        date=datetime.now(),
        total_predictions=total_predictions,
        successful_predictions=successful_predictions,
        failed_predictions=failed_predictions,
        interval_stats=interval_stats,
        market_stats=market_stats,
    )

    return {
        "status": "success" if success else "failed",
        "total_predictions": total_predictions,
        "successful_predictions": successful_predictions,
        "failed_predictions": failed_predictions,
        "interval_stats": interval_stats,
        "market_stats": market_stats,
    }


@router.get("/trading-status")
async def get_trading_status() -> Dict[str, Any]:
    """
    Get current trading status - whether market is open and notifications will be sent.
    """
    from datetime import timezone, timedelta
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

    status_reason = None
    if is_weekend:
        day_name = now_ist.strftime("%A")
        status_reason = f"Weekend ({day_name}) - Market Closed"
    elif current_time < market_open:
        status_reason = f"Pre-market hours (opens at 9:00 AM IST)"
    elif current_time > market_close:
        status_reason = f"Post-market hours (closed at 11:30 PM IST)"
    else:
        status_reason = "Market is open - Trading hours"

    return {
        "current_time_ist": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
        "is_weekend": is_weekend,
        "is_trading_hours": is_trading_hours,
        "notifications_active": is_trading_hours,
        "status_reason": status_reason,
        "market_hours": {
            "open": "9:00 AM IST",
            "close": "11:30 PM IST",
            "days": "Monday - Friday",
        },
    }
