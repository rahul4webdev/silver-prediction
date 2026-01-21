"""
Accuracy metrics API endpoints.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select, and_, desc, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db
from app.models.predictions import Prediction
from app.services.prediction_engine import prediction_engine

router = APIRouter(prefix="/accuracy")


@router.get("/summary")
async def get_accuracy_summary(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get comprehensive accuracy summary.
    """
    try:
        summary = await prediction_engine.get_accuracy_summary(
            db, asset, market, interval, period_days
        )
        return summary
    except Exception as e:
        # Return a default empty response instead of 500 error
        return {
            "total_predictions": 0,
            "verified_predictions": 0,
            "period_days": period_days,
            "message": f"No data available: {str(e)}",
            "direction_accuracy": {
                "overall": 0,
                "correct": 0,
                "wrong": 0,
            },
            "confidence_interval_coverage": {
                "ci_50": 0,
                "ci_80": 0,
                "ci_95": 0,
            },
            "error_metrics": {
                "mape": 0,
            },
            "by_interval": {},
            "by_market": {},
        }


@router.get("/trend")
async def get_accuracy_trend(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    period_days: int = Query(30, le=365),
    bucket_days: int = Query(1, description="Days per data point"),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get accuracy trend over time.
    """
    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = [
        Prediction.asset == asset,
        Prediction.verified_at.isnot(None),
        Prediction.created_at >= since,
    ]

    if market:
        conditions.append(Prediction.market == market)
    if interval:
        conditions.append(Prediction.interval == interval)

    query = (
        select(Prediction)
        .where(and_(*conditions))
        .order_by(Prediction.created_at)
    )

    result = await db.execute(query)
    predictions = result.scalars().all()

    if not predictions:
        return {"trend": [], "message": "No verified predictions"}

    # Group by bucket
    trend_data = []
    current_bucket_start = predictions[0].created_at.replace(hour=0, minute=0, second=0)
    bucket_predictions = []

    for pred in predictions:
        pred_date = pred.created_at.replace(hour=0, minute=0, second=0)

        if (pred_date - current_bucket_start).days >= bucket_days:
            # Save bucket data
            if bucket_predictions:
                correct = sum(1 for p in bucket_predictions if p.is_direction_correct)
                trend_data.append({
                    "date": current_bucket_start.isoformat(),
                    "accuracy": correct / len(bucket_predictions),
                    "total": len(bucket_predictions),
                    "correct": correct,
                })

            current_bucket_start = pred_date
            bucket_predictions = []

        bucket_predictions.append(pred)

    # Add last bucket
    if bucket_predictions:
        correct = sum(1 for p in bucket_predictions if p.is_direction_correct)
        trend_data.append({
            "date": current_bucket_start.isoformat(),
            "accuracy": correct / len(bucket_predictions),
            "total": len(bucket_predictions),
            "correct": correct,
        })

    return {
        "period_days": period_days,
        "bucket_days": bucket_days,
        "data_points": len(trend_data),
        "trend": trend_data,
    }


@router.get("/by-interval")
async def get_accuracy_by_interval(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get accuracy breakdown by prediction interval.
    """
    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = [
        Prediction.asset == asset,
        Prediction.verified_at.isnot(None),
        Prediction.created_at >= since,
    ]

    if market:
        conditions.append(Prediction.market == market)

    query = (
        select(
            Prediction.interval,
            func.count(Prediction.id).label("total"),
            func.sum(
                func.cast(Prediction.is_direction_correct, Integer)
            ).label("correct"),
        )
        .where(and_(*conditions))
        .group_by(Prediction.interval)
    )

    # Use raw SQL for the cast
    from sqlalchemy import Integer, case

    query = (
        select(
            Prediction.interval,
            func.count(Prediction.id).label("total"),
            func.sum(
                case((Prediction.is_direction_correct == True, 1), else_=0)
            ).label("correct"),
        )
        .where(and_(*conditions))
        .group_by(Prediction.interval)
    )

    result = await db.execute(query)
    rows = result.all()

    intervals = {}
    for row in rows:
        intervals[row.interval] = {
            "total": row.total,
            "correct": row.correct or 0,
            "accuracy": (row.correct or 0) / row.total if row.total > 0 else 0,
        }

    return {
        "period_days": period_days,
        "by_interval": intervals,
    }


@router.get("/by-market")
async def get_accuracy_by_market(
    asset: str = Query("silver"),
    interval: Optional[str] = Query(None),
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get accuracy breakdown by market.
    """
    from sqlalchemy import case

    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = [
        Prediction.asset == asset,
        Prediction.verified_at.isnot(None),
        Prediction.created_at >= since,
    ]

    if interval:
        conditions.append(Prediction.interval == interval)

    query = (
        select(
            Prediction.market,
            func.count(Prediction.id).label("total"),
            func.sum(
                case((Prediction.is_direction_correct == True, 1), else_=0)
            ).label("correct"),
        )
        .where(and_(*conditions))
        .group_by(Prediction.market)
    )

    result = await db.execute(query)
    rows = result.all()

    markets = {}
    for row in rows:
        markets[row.market] = {
            "total": row.total,
            "correct": row.correct or 0,
            "accuracy": (row.correct or 0) / row.total if row.total > 0 else 0,
        }

    return {
        "period_days": period_days,
        "by_market": markets,
    }


@router.get("/confidence-intervals")
async def get_ci_coverage(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get confidence interval coverage statistics.

    This shows how well-calibrated our confidence intervals are.
    Ideally, 50% CI should contain ~50% of prices, 80% CI ~80%, etc.
    """
    from sqlalchemy import case

    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = [
        Prediction.asset == asset,
        Prediction.verified_at.isnot(None),
        Prediction.created_at >= since,
    ]

    if market:
        conditions.append(Prediction.market == market)
    if interval:
        conditions.append(Prediction.interval == interval)

    query = (
        select(
            func.count(Prediction.id).label("total"),
            func.sum(
                case((Prediction.within_ci_50 == True, 1), else_=0)
            ).label("within_50"),
            func.sum(
                case((Prediction.within_ci_80 == True, 1), else_=0)
            ).label("within_80"),
            func.sum(
                case((Prediction.within_ci_95 == True, 1), else_=0)
            ).label("within_95"),
        )
        .where(and_(*conditions))
    )

    result = await db.execute(query)
    row = result.one()

    total = row.total or 1  # Avoid division by zero

    return {
        "period_days": period_days,
        "total_predictions": row.total,
        "coverage": {
            "ci_50": {
                "target": 0.50,
                "actual": (row.within_50 or 0) / total,
                "count": row.within_50 or 0,
                "is_calibrated": abs((row.within_50 or 0) / total - 0.50) < 0.10,
            },
            "ci_80": {
                "target": 0.80,
                "actual": (row.within_80 or 0) / total,
                "count": row.within_80 or 0,
                "is_calibrated": abs((row.within_80 or 0) / total - 0.80) < 0.10,
            },
            "ci_95": {
                "target": 0.95,
                "actual": (row.within_95 or 0) / total,
                "count": row.within_95 or 0,
                "is_calibrated": abs((row.within_95 or 0) / total - 0.95) < 0.05,
            },
        },
    }


@router.get("/streaks")
async def get_prediction_streaks(
    asset: str = Query("silver"),
    market: Optional[str] = Query(None),
    interval: Optional[str] = Query(None),
    period_days: int = Query(30, le=365),
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """
    Get win/loss streak statistics.
    """
    since = datetime.utcnow() - timedelta(days=period_days)

    conditions = [
        Prediction.asset == asset,
        Prediction.verified_at.isnot(None),
        Prediction.created_at >= since,
    ]

    if market:
        conditions.append(Prediction.market == market)
    if interval:
        conditions.append(Prediction.interval == interval)

    query = (
        select(Prediction)
        .where(and_(*conditions))
        .order_by(Prediction.created_at)
    )

    result = await db.execute(query)
    predictions = result.scalars().all()

    if not predictions:
        return {"message": "No verified predictions"}

    # Calculate streaks
    current_streak = {"type": None, "count": 0}
    best_win_streak = 0
    worst_loss_streak = 0
    temp_streak = 0

    for pred in predictions:
        is_win = pred.is_direction_correct

        if is_win:
            if temp_streak >= 0:
                temp_streak += 1
            else:
                temp_streak = 1
            best_win_streak = max(best_win_streak, temp_streak)
        else:
            if temp_streak <= 0:
                temp_streak -= 1
            else:
                temp_streak = -1
            worst_loss_streak = max(worst_loss_streak, abs(temp_streak))

    # Current streak (from most recent)
    recent_results = [p.is_direction_correct for p in reversed(predictions[:20])]
    if recent_results:
        first_result = recent_results[0]
        count = 0
        for r in recent_results:
            if r == first_result:
                count += 1
            else:
                break
        current_streak = {
            "type": "win" if first_result else "loss",
            "count": count,
        }

    return {
        "period_days": period_days,
        "total_predictions": len(predictions),
        "current_streak": current_streak,
        "best_win_streak": best_win_streak,
        "worst_loss_streak": worst_loss_streak,
    }
