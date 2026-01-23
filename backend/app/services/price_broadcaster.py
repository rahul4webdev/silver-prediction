"""
Price Broadcaster - Manages WebSocket connections and broadcasts price updates.
This is a singleton that can be imported by both the tick collector and the API.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class PriceUpdate:
    """Price update data structure."""
    asset: str
    market: str
    symbol: str
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

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class PriceBroadcaster:
    """
    Manages price update broadcasting to WebSocket clients.

    This is a singleton that stores the latest prices and notifies
    registered callbacks when new prices arrive.
    """

    def __init__(self):
        self._latest_prices: Dict[str, PriceUpdate] = {}  # key: "asset:market"
        self._callbacks: Set[Callable] = set()
        self._lock = asyncio.Lock()

    def get_key(self, asset: str, market: str) -> str:
        """Generate cache key for asset/market pair."""
        return f"{asset}:{market}"

    async def update_price(self, update: PriceUpdate) -> None:
        """
        Update the latest price and notify all registered callbacks.

        Args:
            update: PriceUpdate object with latest price data
        """
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
                *[self._safe_callback(cb, message) for cb in self._callbacks],
                return_exceptions=True
            )

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

    def get_all_latest_prices(self) -> Dict[str, Dict[str, Any]]:
        """Get all latest cached prices."""
        return {k: v.to_dict() for k, v in self._latest_prices.items()}

    @property
    def callback_count(self) -> int:
        """Number of registered callbacks."""
        return len(self._callbacks)


# Singleton instance
price_broadcaster = PriceBroadcaster()
