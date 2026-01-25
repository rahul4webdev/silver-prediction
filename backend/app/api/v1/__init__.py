# API v1 module
from fastapi import APIRouter

from app.api.v1 import (
    predictions, historical, accuracy, health, auth, ticks,
    sentiment, macro, alerts, confluence, contracts, notifications, status
)

router = APIRouter(prefix="/api/v1")

router.include_router(health.router, tags=["Health"])
router.include_router(auth.router, tags=["Authentication"])
router.include_router(predictions.router, tags=["Predictions"])
router.include_router(historical.router, tags=["Historical Data"])
router.include_router(accuracy.router, tags=["Accuracy"])
router.include_router(ticks.router, tags=["Tick Data"])
router.include_router(sentiment.router, tags=["Sentiment"])
router.include_router(macro.router, tags=["Macro Data"])
router.include_router(alerts.router, tags=["Alerts & Journal"])
router.include_router(confluence.router, tags=["Confluence & Correlation"])
router.include_router(contracts.router, tags=["Contracts"])
router.include_router(notifications.router, tags=["Notifications"])
router.include_router(status.router, tags=["System Status"])
