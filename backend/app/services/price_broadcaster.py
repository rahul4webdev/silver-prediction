"""
Price Broadcaster - Manages WebSocket connections and broadcasts price updates.
Uses Redis pub/sub for inter-process communication between tick collector and API.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Redis channel for price updates
PRICE_CHANNEL = "price_updates"


@dataclass
class PriceUpdate:
    """Price update data structure."""
    asset: str
    market: str
    symbol: str  # instrument_key (e.g., MCX_FO|451669)
    price: float
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None
    volume: Optional[int] = None
    timestamp: str = ""
    source: str = "upstox_ws"
    contract_type: Optional[str] = None  # SILVER, SILVERM, SILVERMIC
    trading_symbol: Optional[str] = None  # Human readable (e.g., SILVERM FUT 27 FEB 26)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PriceUpdate":
        """Create PriceUpdate from dictionary."""
        return cls(
            asset=data.get("asset", ""),
            market=data.get("market", ""),
            symbol=data.get("symbol", ""),
            price=data.get("price", 0),
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            close=data.get("close"),
            change=data.get("change"),
            change_percent=data.get("change_percent"),
            volume=data.get("volume"),
            timestamp=data.get("timestamp", ""),
            source=data.get("source", "upstox_ws"),
            contract_type=data.get("contract_type"),
            trading_symbol=data.get("trading_symbol"),
        )


class PriceBroadcaster:
    """
    Manages price update broadcasting to WebSocket clients.

    Uses Redis pub/sub for inter-process communication:
    - Tick collector publishes price updates to Redis
    - API subscribes to Redis and broadcasts to connected WebSocket clients
    """

    def __init__(self):
        self._latest_prices: Dict[str, PriceUpdate] = {}  # key: "asset:market"
        self._callbacks: Set[Callable] = set()
        self._lock = asyncio.Lock()
        self._redis = None
        self._pubsub = None
        self._subscriber_task = None

    def get_key(self, asset: str, market: str) -> str:
        """Generate cache key for asset/market pair."""
        return f"{asset}:{market}"

    async def _get_redis(self):
        """Get or create Redis connection."""
        if self._redis is None:
            try:
                import redis.asyncio as redis
                from app.core.config import settings

                self._redis = redis.Redis(
                    host=settings.redis_host,
                    port=settings.redis_port,
                    db=0,
                    decode_responses=True,
                )
                logger.info(f"Connected to Redis at {settings.redis_host}:{settings.redis_port}")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                return None
        return self._redis

    async def publish_price(self, update: PriceUpdate) -> None:
        """
        Publish price update to Redis channel.
        Called by the tick collector.
        """
        redis = await self._get_redis()
        if redis:
            try:
                message = json.dumps(update.to_dict())
                await redis.publish(PRICE_CHANNEL, message)

                # Also store latest price in Redis for quick access
                key = f"latest_price:{update.asset}:{update.market}"
                await redis.set(key, message, ex=300)  # Expire in 5 minutes
            except Exception as e:
                logger.error(f"Failed to publish price update: {e}")

    async def start_subscriber(self) -> None:
        """
        Start Redis subscriber to receive price updates.
        Called by the API server.
        """
        redis = await self._get_redis()
        if not redis:
            logger.error("Cannot start subscriber - Redis not available")
            return

        try:
            self._pubsub = redis.pubsub()
            await self._pubsub.subscribe(PRICE_CHANNEL)
            logger.info(f"Subscribed to Redis channel: {PRICE_CHANNEL}")

            # Start background task to process messages
            self._subscriber_task = asyncio.create_task(self._process_messages())
        except Exception as e:
            logger.error(f"Failed to start Redis subscriber: {e}")

    async def stop_subscriber(self) -> None:
        """Stop the Redis subscriber."""
        if self._subscriber_task:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe(PRICE_CHANNEL)
            await self._pubsub.close()

        if self._redis:
            await self._redis.close()

    async def _process_messages(self) -> None:
        """Process incoming Redis messages and broadcast to WebSocket clients."""
        try:
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        update = PriceUpdate.from_dict(data)
                        await self._broadcast_to_clients(update)
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Redis subscriber error: {e}")

    async def _broadcast_to_clients(self, update: PriceUpdate) -> None:
        """Broadcast price update to all registered WebSocket clients."""
        key = self.get_key(update.asset, update.market)

        async with self._lock:
            self._latest_prices[key] = update

        # Notify all callbacks (WebSocket handlers)
        message = {
            "type": "price_update",
            **update.to_dict()
        }

        # Run callbacks concurrently
        if self._callbacks:
            await asyncio.gather(
                *[self._safe_callback(cb, message) for cb in list(self._callbacks)],
                return_exceptions=True
            )

    async def update_price(self, update: PriceUpdate) -> None:
        """
        Update price - publishes to Redis if in tick collector mode,
        or broadcasts directly if clients are connected.
        """
        # Publish to Redis for inter-process communication
        await self.publish_price(update)

        # Also update local cache
        key = self.get_key(update.asset, update.market)
        async with self._lock:
            self._latest_prices[key] = update

    async def _safe_callback(self, callback: Callable, message: Dict) -> None:
        """Safely execute a callback, catching any exceptions."""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(message)
            else:
                callback(message)
        except Exception as e:
            logger.debug(f"Callback error (client may have disconnected): {e}")

    def register_callback(self, callback: Callable) -> None:
        """Register a callback to receive price updates."""
        self._callbacks.add(callback)
        logger.debug(f"Registered callback. Total: {len(self._callbacks)}")

    def unregister_callback(self, callback: Callable) -> None:
        """Unregister a callback."""
        self._callbacks.discard(callback)
        logger.debug(f"Unregistered callback. Total: {len(self._callbacks)}")

    def get_latest_price(self, asset: str, market: str) -> Optional[PriceUpdate]:
        """Get the latest cached price for an asset/market."""
        key = self.get_key(asset, market)
        return self._latest_prices.get(key)

    async def get_latest_price_from_redis(self, asset: str, market: str) -> Optional[PriceUpdate]:
        """Get the latest price from Redis cache."""
        redis = await self._get_redis()
        if redis:
            try:
                key = f"latest_price:{asset}:{market}"
                data = await redis.get(key)
                if data:
                    return PriceUpdate.from_dict(json.loads(data))
            except Exception as e:
                logger.error(f"Failed to get price from Redis: {e}")
        return None

    def get_all_latest_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get all latest cached prices."""
        return {k: v.to_dict() for k, v in self._latest_prices.items()}

    @property
    def callback_count(self) -> int:
        """Number of registered callbacks."""
        return len(self._callbacks)


# Singleton instance
price_broadcaster = PriceBroadcaster()
