"""
Price alerts and trade journal API endpoints.
"""

import logging
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Query, Body, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.database import get_db_session
from app.models.alerts import PriceAlert, TradeJournal, AlertStatus, AlertType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/alerts")


# Pydantic models for request validation
class CreateAlertRequest(BaseModel):
    asset: str = "silver"
    market: str
    alert_type: str  # price_above, price_below, percent_change
    target_price: float
    current_price: float
    note: Optional[str] = None
    expires_at: Optional[datetime] = None
    notify_telegram: bool = True


class CreateTradeRequest(BaseModel):
    asset: str = "silver"
    market: str
    trade_type: str  # buy, sell
    entry_price: float
    entry_time: datetime
    quantity: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    prediction_id: Optional[str] = None
    prediction_direction: Optional[str] = None
    prediction_confidence: Optional[float] = None
    notes: Optional[str] = None
    tags: Optional[List[str]] = None


class CloseTradeRequest(BaseModel):
    exit_price: float
    exit_time: datetime
    followed_prediction: Optional[bool] = None
    notes: Optional[str] = None
    lessons_learned: Optional[str] = None


@router.post("/create", response_model=Dict[str, Any])
async def create_price_alert(request: CreateAlertRequest):
    """
    Create a new price alert.

    Alert types:
    - price_above: Trigger when price goes above target
    - price_below: Trigger when price goes below target
    - percent_change: Trigger on X% change from current price
    """
    try:
        async with get_db_session() as db:
            alert = PriceAlert(
                asset=request.asset,
                market=request.market,
                alert_type=AlertType(request.alert_type),
                target_price=Decimal(str(request.target_price)),
                current_price_at_creation=Decimal(str(request.current_price)),
                note=request.note,
                expires_at=request.expires_at,
                notify_telegram=request.notify_telegram,
            )

            db.add(alert)
            await db.commit()
            await db.refresh(alert)

            return {
                "status": "success",
                "message": "Alert created",
                "alert": alert.to_dict(),
            }

    except Exception as e:
        logger.error(f"Error creating alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=Dict[str, Any])
async def list_alerts(
    asset: str = Query(default="silver"),
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, le=200),
):
    """
    List price alerts.
    """
    try:
        async with get_db_session() as db:
            conditions = [PriceAlert.asset == asset]

            if status:
                conditions.append(PriceAlert.status == AlertStatus(status))

            query = (
                select(PriceAlert)
                .where(and_(*conditions))
                .order_by(PriceAlert.created_at.desc())
                .limit(limit)
            )

            result = await db.execute(query)
            alerts = result.scalars().all()

            return {
                "status": "success",
                "count": len(alerts),
                "alerts": [a.to_dict() for a in alerts],
            }

    except Exception as e:
        logger.error(f"Error listing alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{alert_id}", response_model=Dict[str, Any])
async def cancel_alert(alert_id: str):
    """
    Cancel a price alert.
    """
    try:
        async with get_db_session() as db:
            query = select(PriceAlert).where(PriceAlert.id == UUID(alert_id))
            result = await db.execute(query)
            alert = result.scalar_one_or_none()

            if not alert:
                raise HTTPException(status_code=404, detail="Alert not found")

            alert.status = AlertStatus.CANCELLED
            await db.commit()

            return {
                "status": "success",
                "message": "Alert cancelled",
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Trade Journal endpoints
@router.post("/trades/create", response_model=Dict[str, Any])
async def create_trade(request: CreateTradeRequest):
    """
    Log a new trade in the journal.
    """
    try:
        async with get_db_session() as db:
            trade = TradeJournal(
                asset=request.asset,
                market=request.market,
                trade_type=request.trade_type,
                entry_price=Decimal(str(request.entry_price)),
                entry_time=request.entry_time,
                quantity=Decimal(str(request.quantity)) if request.quantity else None,
                stop_loss=Decimal(str(request.stop_loss)) if request.stop_loss else None,
                take_profit=Decimal(str(request.take_profit)) if request.take_profit else None,
                prediction_id=UUID(request.prediction_id) if request.prediction_id else None,
                prediction_direction=request.prediction_direction,
                prediction_confidence=Decimal(str(request.prediction_confidence)) if request.prediction_confidence else None,
                notes=request.notes,
                tags=request.tags,
            )

            db.add(trade)
            await db.commit()
            await db.refresh(trade)

            return {
                "status": "success",
                "message": "Trade logged",
                "trade": trade.to_dict(),
            }

    except Exception as e:
        logger.error(f"Error creating trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/trades/{trade_id}/close", response_model=Dict[str, Any])
async def close_trade(trade_id: str, request: CloseTradeRequest):
    """
    Close a trade and calculate P&L.
    """
    try:
        async with get_db_session() as db:
            query = select(TradeJournal).where(TradeJournal.id == UUID(trade_id))
            result = await db.execute(query)
            trade = result.scalar_one_or_none()

            if not trade:
                raise HTTPException(status_code=404, detail="Trade not found")

            trade.exit_price = Decimal(str(request.exit_price))
            trade.exit_time = request.exit_time
            trade.followed_prediction = request.followed_prediction
            if request.notes:
                trade.notes = (trade.notes or "") + f"\n\nExit notes: {request.notes}"
            trade.lessons_learned = request.lessons_learned

            # Calculate P&L
            trade.calculate_pnl()

            await db.commit()
            await db.refresh(trade)

            return {
                "status": "success",
                "message": "Trade closed",
                "trade": trade.to_dict(),
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error closing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/list", response_model=Dict[str, Any])
async def list_trades(
    asset: str = Query(default="silver"),
    limit: int = Query(default=50, le=200),
    open_only: bool = Query(default=False),
):
    """
    List trades from journal.
    """
    try:
        async with get_db_session() as db:
            conditions = [TradeJournal.asset == asset]

            if open_only:
                conditions.append(TradeJournal.exit_time.is_(None))

            query = (
                select(TradeJournal)
                .where(and_(*conditions))
                .order_by(TradeJournal.entry_time.desc())
                .limit(limit)
            )

            result = await db.execute(query)
            trades = result.scalars().all()

            # Calculate summary stats
            closed_trades = [t for t in trades if t.pnl_percent is not None]
            if closed_trades:
                total_pnl = sum(float(t.pnl_percent) for t in closed_trades)
                winning = sum(1 for t in closed_trades if float(t.pnl_percent) > 0)
                win_rate = winning / len(closed_trades)
            else:
                total_pnl = 0
                win_rate = 0

            return {
                "status": "success",
                "count": len(trades),
                "trades": [t.to_dict() for t in trades],
                "summary": {
                    "total_trades": len(trades),
                    "closed_trades": len(closed_trades),
                    "open_trades": len(trades) - len(closed_trades),
                    "total_pnl_percent": total_pnl,
                    "win_rate": win_rate,
                },
            }

    except Exception as e:
        logger.error(f"Error listing trades: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trades/performance", response_model=Dict[str, Any])
async def get_trade_performance(
    asset: str = Query(default="silver"),
    days: int = Query(default=30, le=365),
):
    """
    Get trade performance vs system predictions.

    Shows how user trades compare to following system predictions.
    """
    try:
        async with get_db_session() as db:
            since = datetime.utcnow() - timedelta(days=days)

            query = (
                select(TradeJournal)
                .where(
                    and_(
                        TradeJournal.asset == asset,
                        TradeJournal.exit_time.isnot(None),
                        TradeJournal.entry_time >= since,
                    )
                )
            )

            result = await db.execute(query)
            trades = result.scalars().all()

            if not trades:
                return {
                    "status": "success",
                    "message": "No closed trades in period",
                    "performance": {},
                }

            # Analyze trades
            followed = [t for t in trades if t.followed_prediction == True]
            not_followed = [t for t in trades if t.followed_prediction == False]

            def calc_stats(trade_list):
                if not trade_list:
                    return {"count": 0, "avg_pnl": 0, "win_rate": 0}
                pnls = [float(t.pnl_percent) for t in trade_list if t.pnl_percent]
                wins = sum(1 for p in pnls if p > 0)
                return {
                    "count": len(trade_list),
                    "avg_pnl": sum(pnls) / len(pnls) if pnls else 0,
                    "win_rate": wins / len(pnls) if pnls else 0,
                }

            return {
                "status": "success",
                "period_days": days,
                "total_trades": len(trades),
                "followed_predictions": calc_stats(followed),
                "not_followed_predictions": calc_stats(not_followed),
                "recommendation": (
                    "Following predictions shows better results"
                    if calc_stats(followed).get("avg_pnl", 0) > calc_stats(not_followed).get("avg_pnl", 0)
                    else "Your judgment outperforms predictions"
                    if not_followed
                    else "Not enough data to compare"
                ),
            }

    except Exception as e:
        logger.error(f"Error getting performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Import timedelta for the performance endpoint
from datetime import timedelta
