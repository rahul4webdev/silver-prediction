"""
Real-time tick data collector using Upstox WebSocket.
Collects and stores every price update for high-frequency model training.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from collections import deque

import websockets
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.database import async_session_factory
from app.models.tick_data import TickData, TickDataAggregated
from app.services.upstox_client import upstox_client
from app.services.price_broadcaster import price_broadcaster, PriceUpdate

# Import protobuf for decoding Upstox messages
try:
    from app.services.proto import MarketDataFeed_pb2 as pb
    PROTOBUF_AVAILABLE = True
except ImportError:
    PROTOBUF_AVAILABLE = False

logger = logging.getLogger(__name__)


class TickCollector:
    """
    Collects real-time tick data from Upstox WebSocket.

    Features:
    - Connects to Upstox WebSocket for MCX silver
    - Stores every tick in the database
    - Aggregates ticks into 1s, 5s, 10s, 1m intervals
    - Auto-reconnects on disconnection
    - Buffers ticks for batch inserts
    """

    UPSTOX_WS_URL = "wss://api.upstox.com/v2/feed/market-data-feed"

    def __init__(self):
        self._running = False
        self._ws = None
        self._instrument_key: Optional[str] = None
        self._tick_buffer: deque = deque(maxlen=1000)  # Buffer for batch inserts
        self._last_insert_time = datetime.now(timezone.utc)
        self._insert_interval = 1  # Insert every 1 second
        self._reconnect_delay = 5  # Seconds between reconnection attempts
        self._stats = {
            "ticks_received": 0,
            "ticks_stored": 0,
            "errors": 0,
            "last_tick_time": None,
            "connected_since": None,
        }

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats.copy()

    async def start(self) -> None:
        """Start the tick collector."""
        if self._running:
            logger.warning("Tick collector is already running")
            return

        self._running = True
        logger.info("Starting tick collector...")

        while self._running:
            try:
                await self._connect_and_collect()
            except Exception as e:
                logger.error(f"Tick collector error: {e}")
                self._stats["errors"] += 1

            if self._running:
                logger.info(f"Reconnecting in {self._reconnect_delay} seconds...")
                await asyncio.sleep(self._reconnect_delay)

    async def stop(self) -> None:
        """Stop the tick collector."""
        logger.info("Stopping tick collector...")
        self._running = False

        if self._ws:
            await self._ws.close()
            self._ws = None

        # Flush remaining buffer
        await self._flush_buffer()

    async def _connect_and_collect(self) -> None:
        """Connect to WebSocket and collect ticks."""
        # Get access token
        if not upstox_client.is_authenticated:
            logger.error("Upstox not authenticated - cannot start tick collector")
            await asyncio.sleep(60)  # Wait before retrying
            return

        # Get instrument key
        self._instrument_key = await upstox_client.get_silver_instrument_key()
        if not self._instrument_key:
            logger.error("Could not get silver instrument key")
            await asyncio.sleep(60)
            return

        # Get authorized WebSocket URL from Upstox
        try:
            ws_auth_data = await upstox_client.get_websocket_auth()
            ws_url = ws_auth_data.get("authorizedRedirectUri")
            if not ws_url:
                logger.error("Could not get authorized WebSocket URL from Upstox")
                await asyncio.sleep(60)
                return
            logger.info(f"Got authorized WebSocket URL")
        except Exception as e:
            logger.error(f"Failed to get WebSocket authorization: {e}")
            await asyncio.sleep(60)
            return

        try:
            async with websockets.connect(
                ws_url,
                ping_interval=30,
                ping_timeout=10,
            ) as ws:
                self._ws = ws
                self._stats["connected_since"] = datetime.now(timezone.utc)
                logger.info(f"Connected to Upstox WebSocket for {self._instrument_key}")

                # Subscribe to instrument
                await self._subscribe(ws)

                # Start background task for periodic buffer flush
                flush_task = asyncio.create_task(self._periodic_flush())

                try:
                    # Receive and process messages
                    async for message in ws:
                        if not self._running:
                            break

                        try:
                            await self._process_message(message)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            self._stats["errors"] += 1

                finally:
                    flush_task.cancel()
                    try:
                        await flush_task
                    except asyncio.CancelledError:
                        pass

        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            raise

    async def _subscribe(self, ws) -> None:
        """Subscribe to market data feed."""
        subscribe_message = {
            "guid": "tick-collector",
            "method": "sub",
            "data": {
                "mode": "full",  # Full mode includes all data
                "instrumentKeys": [self._instrument_key],
            },
        }

        # Send as binary (bytes) as required by Upstox v3 WebSocket
        message_bytes = json.dumps(subscribe_message).encode('utf-8')
        await ws.send(message_bytes)
        logger.info(f"Subscribed to {self._instrument_key} (binary message)")

    async def _process_message(self, message) -> None:
        """Process incoming WebSocket message (protobuf or JSON)."""
        # Handle binary protobuf messages
        if isinstance(message, bytes):
            # Log every 100th message at INFO level
            if self._stats["ticks_received"] % 100 == 0:
                logger.info(f"Processing binary message ({len(message)} bytes), PROTOBUF_AVAILABLE={PROTOBUF_AVAILABLE}, total received: {self._stats['ticks_received']}")
            await self._process_protobuf_message(message)
            return

        # Handle JSON messages (fallback)
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            logger.debug("Received undecodable message, skipping")
            return

        # Check if it's market data
        if data.get("type") == "live_feed":
            feeds = data.get("feeds", {})

            for instrument_key, feed_data in feeds.items():
                if "ff" in feed_data:  # Full feed
                    await self._process_full_feed(instrument_key, feed_data["ff"])
                elif "ltpc" in feed_data:  # LTP change feed
                    await self._process_ltpc_feed(instrument_key, feed_data["ltpc"])

    async def _process_protobuf_message(self, data: bytes) -> None:
        """Process protobuf-encoded message from Upstox."""
        if not PROTOBUF_AVAILABLE:
            logger.warning("Protobuf not available, cannot decode message")
            return

        try:
            feed_response = pb.FeedResponse()
            feed_response.ParseFromString(data)
            # Log parse results periodically
            if self._stats["ticks_received"] % 100 == 0:
                logger.info(f"Parsed protobuf: type={feed_response.type}, feeds={len(feed_response.feeds)}")

            # Check message type
            msg_type = feed_response.type
            # Type 0 = initial_feed, 1 = live_feed, 2 = market_info

            if msg_type == 1:  # live_feed
                for instrument_key, feed in feed_response.feeds.items():
                    await self._process_protobuf_feed(instrument_key, feed)
            elif msg_type == 0:  # initial_feed (snapshot)
                for instrument_key, feed in feed_response.feeds.items():
                    await self._process_protobuf_feed(instrument_key, feed)
            # Type 2 (market_info) contains market status, we can log it
            elif msg_type == 2:
                logger.debug(f"Market info received: {feed_response.marketInfo}")

        except Exception as e:
            logger.error(f"Failed to decode protobuf message: {e}")
            self._stats["errors"] += 1

    async def _process_protobuf_feed(self, instrument_key: str, feed) -> None:
        """Process a single feed from protobuf message."""
        self._stats["ticks_received"] += 1
        self._stats["last_tick_time"] = datetime.now(timezone.utc)

        # Extract data based on feed type
        ltpc = None
        ohlc_data = None
        volume = None
        oi = None

        # Check which feed type we have
        if feed.HasField("fullFeed"):
            full_feed = feed.fullFeed
            if full_feed.HasField("marketFF"):
                market_ff = full_feed.marketFF
                ltpc = market_ff.ltpc
                volume = market_ff.vtt  # Volume traded today
                oi = market_ff.oi  # Open interest

                # Get OHLC from marketOHLC
                if market_ff.marketOHLC and market_ff.marketOHLC.ohlc:
                    for ohlc in market_ff.marketOHLC.ohlc:
                        if ohlc.interval == "I1":  # Intraday
                            ohlc_data = ohlc
                            break
        elif feed.HasField("ltpc"):
            ltpc = feed.ltpc

        if ltpc is None:
            return

        # Create tick record
        tick = {
            "asset": "silver",
            "market": "mcx",
            "symbol": instrument_key,
            "timestamp": datetime.now(timezone.utc),
            "ltp": ltpc.ltp if ltpc.ltp else None,
            "ltq": ltpc.ltq if ltpc.ltq else None,
            "open": ohlc_data.open if ohlc_data else None,
            "high": ohlc_data.high if ohlc_data else None,
            "low": ohlc_data.low if ohlc_data else None,
            "close": ohlc_data.close if ohlc_data else (ltpc.ltp if ltpc.ltp else None),
            "volume": volume,
            "oi": int(oi) if oi else None,
            "bid_price": None,
            "bid_qty": None,
            "ask_price": None,
            "ask_qty": None,
            "change": ltpc.cp if ltpc.cp else None,  # Close price (previous day)
            "change_percent": None,
            "source": "upstox_ws",
            "created_at": datetime.now(timezone.utc),
        }

        self._tick_buffer.append(tick)

        # Broadcast price update to WebSocket clients
        if ltpc.ltp:
            # Calculate change_percent: ((current_price - prev_close) / prev_close) * 100
            # ltpc.cp is the previous day's close price
            change_percent = None
            if ltpc.cp and float(ltpc.cp) > 0:
                change_percent = ((float(ltpc.ltp) - float(ltpc.cp)) / float(ltpc.cp)) * 100

            update = PriceUpdate(
                asset="silver",
                market="mcx",
                symbol=instrument_key,
                price=float(ltpc.ltp),
                open=float(ohlc_data.open) if ohlc_data and ohlc_data.open else None,
                high=float(ohlc_data.high) if ohlc_data and ohlc_data.high else None,
                low=float(ohlc_data.low) if ohlc_data and ohlc_data.low else None,
                close=float(ohlc_data.close) if ohlc_data and ohlc_data.close else float(ltpc.ltp),
                change=float(ltpc.ltp) - float(ltpc.cp) if ltpc.cp else None,  # Actual price change
                change_percent=change_percent,
                volume=int(volume) if volume else None,
            )
            # Fire and forget - don't await to avoid blocking tick processing
            asyncio.create_task(price_broadcaster.update_price(update))

        # Log periodically
        if self._stats["ticks_received"] % 100 == 0:
            logger.info(f"Received {self._stats['ticks_received']} ticks, LTP: {ltpc.ltp}, WS clients: {price_broadcaster.callback_count}")

    async def _process_full_feed(self, instrument_key: str, feed: Dict) -> None:
        """Process full market data feed."""
        self._stats["ticks_received"] += 1
        self._stats["last_tick_time"] = datetime.now(timezone.utc)

        # Extract market data
        market_ff = feed.get("marketFF", {})
        ltpc = market_ff.get("ltpc", {})
        market_ohlc = market_ff.get("marketOHLC", {})

        # Get OHLC data (intraday)
        ohlc_list = market_ohlc.get("ohlc", [])
        intraday_ohlc = None
        for ohlc in ohlc_list:
            if ohlc.get("interval") == "I1":  # Intraday
                intraday_ohlc = ohlc
                break

        # Create tick record
        tick = {
            "asset": "silver",
            "market": "mcx",
            "symbol": instrument_key,
            "timestamp": datetime.now(timezone.utc),
            "ltp": ltpc.get("ltp"),
            "ltq": ltpc.get("ltq"),
            "open": intraday_ohlc.get("open") if intraday_ohlc else None,
            "high": intraday_ohlc.get("high") if intraday_ohlc else None,
            "low": intraday_ohlc.get("low") if intraday_ohlc else None,
            "close": intraday_ohlc.get("close") if intraday_ohlc else ltpc.get("ltp"),
            "volume": intraday_ohlc.get("volume") if intraday_ohlc else None,
            "oi": market_ff.get("eFeedDetails", {}).get("oi"),
            "bid_price": None,  # Would need market depth
            "bid_qty": None,
            "ask_price": None,
            "ask_qty": None,
            "change": ltpc.get("cp"),  # Change points
            "change_percent": None,
            "source": "upstox_ws",
            "created_at": datetime.now(timezone.utc),
        }

        self._tick_buffer.append(tick)

        # Broadcast price update to WebSocket clients
        if ltpc.get("ltp"):
            # Calculate change_percent: ((current_price - prev_close) / prev_close) * 100
            change_percent = None
            if ltpc.get("cp") and float(ltpc["cp"]) > 0:
                change_percent = ((float(ltpc["ltp"]) - float(ltpc["cp"])) / float(ltpc["cp"])) * 100

            update = PriceUpdate(
                asset="silver",
                market="mcx",
                symbol=instrument_key,
                price=float(ltpc["ltp"]),
                open=float(intraday_ohlc["open"]) if intraday_ohlc and intraday_ohlc.get("open") else None,
                high=float(intraday_ohlc["high"]) if intraday_ohlc and intraday_ohlc.get("high") else None,
                low=float(intraday_ohlc["low"]) if intraday_ohlc and intraday_ohlc.get("low") else None,
                close=float(intraday_ohlc["close"]) if intraday_ohlc and intraday_ohlc.get("close") else float(ltpc["ltp"]),
                change=float(ltpc["ltp"]) - float(ltpc["cp"]) if ltpc.get("cp") else None,  # Actual price change
                change_percent=change_percent,
                volume=int(intraday_ohlc["volume"]) if intraday_ohlc and intraday_ohlc.get("volume") else None,
            )
            asyncio.create_task(price_broadcaster.update_price(update))

    async def _process_ltpc_feed(self, instrument_key: str, ltpc: Dict) -> None:
        """Process LTP change feed (lighter than full feed)."""
        self._stats["ticks_received"] += 1
        self._stats["last_tick_time"] = datetime.now(timezone.utc)

        tick = {
            "asset": "silver",
            "market": "mcx",
            "symbol": instrument_key,
            "timestamp": datetime.now(timezone.utc),
            "ltp": ltpc.get("ltp"),
            "ltq": ltpc.get("ltq"),
            "change": ltpc.get("cp"),
            "source": "upstox_ws",
            "created_at": datetime.now(timezone.utc),
        }

        self._tick_buffer.append(tick)

        # Broadcast price update to WebSocket clients
        if ltpc.get("ltp"):
            # Calculate change_percent: ((current_price - prev_close) / prev_close) * 100
            change_percent = None
            if ltpc.get("cp") and float(ltpc["cp"]) > 0:
                change_percent = ((float(ltpc["ltp"]) - float(ltpc["cp"])) / float(ltpc["cp"])) * 100

            update = PriceUpdate(
                asset="silver",
                market="mcx",
                symbol=instrument_key,
                price=float(ltpc["ltp"]),
                change=float(ltpc["ltp"]) - float(ltpc["cp"]) if ltpc.get("cp") else None,  # Actual price change
                change_percent=change_percent,
            )
            asyncio.create_task(price_broadcaster.update_price(update))

    async def _periodic_flush(self) -> None:
        """Periodically flush tick buffer to database."""
        while self._running:
            await asyncio.sleep(self._insert_interval)
            await self._flush_buffer()

    async def _flush_buffer(self) -> None:
        """Flush tick buffer to database."""
        if not self._tick_buffer:
            return

        # Get all ticks from buffer
        ticks = []
        while self._tick_buffer:
            try:
                ticks.append(self._tick_buffer.popleft())
            except IndexError:
                break

        if not ticks:
            return

        try:
            async with async_session_factory() as session:
                # Batch insert ticks
                stmt = pg_insert(TickData).values(ticks)
                await session.execute(stmt)
                await session.commit()

                self._stats["ticks_stored"] += len(ticks)
                logger.debug(f"Stored {len(ticks)} ticks")

        except Exception as e:
            logger.error(f"Failed to store ticks: {e}")
            self._stats["errors"] += 1
            # Put ticks back in buffer for retry
            for tick in reversed(ticks):
                self._tick_buffer.appendleft(tick)

    async def aggregate_ticks(self, interval: str = "1m") -> int:
        """
        Aggregate raw ticks into OHLCV candles.

        Args:
            interval: Aggregation interval (1s, 5s, 10s, 1m)

        Returns:
            Number of candles created
        """
        interval_seconds = {
            "1s": 1,
            "5s": 5,
            "10s": 10,
            "1m": 60,
        }

        seconds = interval_seconds.get(interval, 60)

        try:
            async with async_session_factory() as session:
                # Get ticks from the last hour that haven't been aggregated
                one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)

                # SQL to aggregate ticks into candles
                # Using PostgreSQL date_trunc for time bucketing
                sql = f"""
                INSERT INTO tick_data_aggregated (asset, market, interval, timestamp, open, high, low, close, volume, tick_count, created_at)
                SELECT
                    asset,
                    market,
                    '{interval}' as interval,
                    date_trunc('second', timestamp) -
                        (EXTRACT(SECOND FROM timestamp)::int % {seconds}) * interval '1 second' as bucket,
                    (array_agg(ltp ORDER BY timestamp))[1] as open,
                    MAX(ltp) as high,
                    MIN(ltp) as low,
                    (array_agg(ltp ORDER BY timestamp DESC))[1] as close,
                    SUM(COALESCE(volume, 0)) as volume,
                    COUNT(*) as tick_count,
                    NOW() as created_at
                FROM tick_data
                WHERE timestamp > :since
                GROUP BY asset, market, bucket
                ON CONFLICT (asset, market, interval, timestamp)
                DO UPDATE SET
                    high = GREATEST(tick_data_aggregated.high, EXCLUDED.high),
                    low = LEAST(tick_data_aggregated.low, EXCLUDED.low),
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume,
                    tick_count = EXCLUDED.tick_count
                RETURNING id
                """

                result = await session.execute(
                    text(sql),
                    {"since": one_hour_ago}
                )
                await session.commit()

                count = len(result.fetchall())
                logger.info(f"Aggregated {count} {interval} candles")
                return count

        except Exception as e:
            logger.error(f"Tick aggregation failed: {e}")
            return 0

    async def get_tick_stats(self) -> Dict[str, Any]:
        """Get tick collection statistics."""
        try:
            async with async_session_factory() as session:
                # Count total ticks
                total_result = await session.execute(
                    select(func.count(TickData.id))
                )
                total_ticks = total_result.scalar() or 0

                # Get latest tick
                latest_result = await session.execute(
                    select(TickData)
                    .order_by(TickData.timestamp.desc())
                    .limit(1)
                )
                latest_tick = latest_result.scalar_one_or_none()

                # Get oldest tick
                oldest_result = await session.execute(
                    select(TickData)
                    .order_by(TickData.timestamp.asc())
                    .limit(1)
                )
                oldest_tick = oldest_result.scalar_one_or_none()

                # Count today's ticks
                today_start = datetime.now(timezone.utc).replace(
                    hour=0, minute=0, second=0, microsecond=0
                )
                today_result = await session.execute(
                    select(func.count(TickData.id))
                    .where(TickData.timestamp >= today_start)
                )
                today_ticks = today_result.scalar() or 0

                return {
                    "collector_stats": self._stats,
                    "database_stats": {
                        "total_ticks": total_ticks,
                        "today_ticks": today_ticks,
                        "oldest_tick": oldest_tick.timestamp.isoformat() if oldest_tick else None,
                        "latest_tick": latest_tick.timestamp.isoformat() if latest_tick else None,
                        "latest_ltp": float(latest_tick.ltp) if latest_tick and latest_tick.ltp else None,
                    },
                }

        except Exception as e:
            logger.error(f"Failed to get tick stats: {e}")
            return {
                "collector_stats": self._stats,
                "database_stats": {"error": str(e)},
            }


# Need to import text for raw SQL
from sqlalchemy import text

# Singleton instance
tick_collector = TickCollector()


async def run_tick_collector():
    """Run the tick collector (for use as a standalone script)."""
    logger.info("Starting tick collector service...")

    try:
        await tick_collector.start()
    except KeyboardInterrupt:
        logger.info("Received shutdown signal")
    finally:
        await tick_collector.stop()


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    asyncio.run(run_tick_collector())
