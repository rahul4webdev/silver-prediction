# API v1 module
from fastapi import APIRouter

from app.api.v1 import predictions, historical, accuracy, health

router = APIRouter(prefix="/api/v1")

router.include_router(health.router, tags=["Health"])
router.include_router(predictions.router, tags=["Predictions"])
router.include_router(historical.router, tags=["Historical Data"])
router.include_router(accuracy.router, tags=["Accuracy"])
