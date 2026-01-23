"""
Silver Price Prediction System - Main FastAPI Application
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import router as api_router
from app.core.config import settings
from app.models.database import init_db, close_db, create_hypertables

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    Application lifespan handler.
    Initializes database and other services on startup.
    """
    logger.info("Starting Silver Price Prediction System...")

    # Initialize database
    try:
        await init_db()
        logger.info("Database initialized")

        # Create TimescaleDB hypertables
        await create_hypertables()
        logger.info("Hypertables created")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    # Start Redis subscriber for price updates
    try:
        from app.services.price_broadcaster import price_broadcaster
        await price_broadcaster.start_subscriber()
        logger.info("Price broadcaster Redis subscriber started")
    except Exception as e:
        logger.error(f"Failed to start price broadcaster: {e}")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down...")

    # Stop price broadcaster
    try:
        from app.services.price_broadcaster import price_broadcaster
        await price_broadcaster.stop_subscriber()
    except Exception as e:
        logger.error(f"Error stopping price broadcaster: {e}")

    await close_db()


# Create FastAPI application
app = FastAPI(
    title="Silver Price Prediction API",
    description="""
    Real-time silver price prediction system using ensemble ML models.

    ## Features
    - Real-time price predictions for MCX and COMEX silver
    - Multiple prediction intervals (30m, 1h, 4h, daily)
    - Ensemble model combining Prophet, LSTM, and XGBoost
    - Probability-based forecasts with confidence intervals
    - Prediction tracking and accuracy metrics
    """,
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS - allow frontend domains
cors_origins = [
    "https://prediction.gahfaudio.in",
    "http://prediction.gahfaudio.in",
    "https://predictionapi.gahfaudio.in",
    "http://predictionapi.gahfaudio.in",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8024",
    "http://127.0.0.1:8024",
]

# Add settings.frontend_url if configured
if settings.frontend_url and settings.frontend_url not in cors_origins:
    cors_origins.append(settings.frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include API routes
app.include_router(api_router)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Silver Price Prediction API",
        "version": "1.0.0",
        "status": "running",
        "environment": settings.environment,
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


# WebSocket endpoint for real-time updates
from fastapi import WebSocket, WebSocketDisconnect
import json
import asyncio
from app.services.price_broadcaster import price_broadcaster


class ConnectionManager:
    """Manages WebSocket connections."""

    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


@app.websocket("/ws/prices/{asset}")
async def websocket_prices(websocket: WebSocket, asset: str):
    """
    WebSocket endpoint for real-time price updates.

    Clients receive automatic price updates pushed from the tick collector.
    """
    await manager.connect(websocket)

    # Create a callback to send price updates to this client
    async def send_price_update(message: dict):
        # Filter by asset if needed
        if message.get("asset") == asset or asset == "all":
            try:
                await websocket.send_json(message)
            except Exception:
                pass

    # Register callback with the price broadcaster
    price_broadcaster.register_callback(send_price_update)

    try:
        # Send initial connection confirmation
        await websocket.send_json({
            "type": "connected",
            "asset": asset,
            "message": f"Connected to {asset} price stream",
        })

        # Send latest cached price if available (check local cache first, then Redis)
        latest = price_broadcaster.get_latest_price(asset, "mcx")
        if not latest:
            latest = await price_broadcaster.get_latest_price_from_redis(asset, "mcx")
        if latest:
            await websocket.send_json({
                "type": "price_update",
                **latest.to_dict()
            })

        while True:
            # Receive messages from client (for ping/pong or commands)
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=60)

                message = json.loads(data)
                action = message.get("action")

                if action == "ping":
                    await websocket.send_json({"type": "pong"})
                elif action == "subscribe":
                    # Already subscribed, just confirm
                    await websocket.send_json({
                        "type": "subscribed",
                        "asset": asset,
                        "message": f"Subscribed to {asset} updates",
                    })
                elif action == "get_latest":
                    # Send latest cached price (check local cache first, then Redis)
                    market = message.get("market", "mcx")
                    latest = price_broadcaster.get_latest_price(asset, market)
                    if not latest:
                        latest = await price_broadcaster.get_latest_price_from_redis(asset, market)
                    if latest:
                        await websocket.send_json({
                            "type": "price_update",
                            **latest.to_dict()
                        })

            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                try:
                    await websocket.send_json({"type": "heartbeat"})
                except Exception:
                    break
            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        pass
    finally:
        # Unregister callback and disconnect
        price_broadcaster.unregister_callback(send_price_update)
        manager.disconnect(websocket)


@app.websocket("/ws/predictions")
async def websocket_predictions(websocket: WebSocket):
    """
    WebSocket endpoint for prediction updates.
    """
    await manager.connect(websocket)

    try:
        while True:
            data = await websocket.receive_text()

            try:
                message = json.loads(data)

                if message.get("action") == "get_latest":
                    # Return latest prediction
                    from app.models.database import get_db_context
                    from app.services.prediction_engine import prediction_engine

                    async with get_db_context() as db:
                        # This would return the latest prediction
                        pass

            except json.JSONDecodeError:
                pass

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# Background task to broadcast updates
async def broadcast_predictions():
    """
    Background task that broadcasts new predictions to connected clients.
    """
    while True:
        # Check for new predictions every 30 seconds
        await asyncio.sleep(30)

        if manager.active_connections:
            # Broadcast update notification
            await manager.broadcast({
                "type": "heartbeat",
                "timestamp": str(asyncio.get_event_loop().time()),
            })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
    )
