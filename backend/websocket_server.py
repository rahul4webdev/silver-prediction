"""
Dedicated WebSocket Server for Price Streaming
Runs on a separate port with SSL support.
"""

import asyncio
import json
import logging
import ssl
from pathlib import Path
from typing import Set

import websockets
from websockets.server import serve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Redis configuration
REDIS_HOST = "localhost"
REDIS_PORT = 6379
PRICE_CHANNEL = "price_updates"

# WebSocket server configuration
WS_HOST = "0.0.0.0"
WS_PORT = 8025

# SSL certificates (Let's Encrypt)
SSL_CERT = "/etc/letsencrypt/live/predictionapi.gahfaudio.in/fullchain.pem"
SSL_KEY = "/etc/letsencrypt/live/predictionapi.gahfaudio.in/privkey.pem"


class WebSocketServer:
    """Dedicated WebSocket server for price streaming."""

    def __init__(self):
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.redis = None
        self.pubsub = None
        self.latest_prices = {}

    async def register(self, websocket: websockets.WebSocketServerProtocol, asset: str):
        """Register a new client connection."""
        self.clients.add(websocket)
        logger.info(f"Client connected for {asset}. Total: {len(self.clients)}")

        # Send connection confirmation
        await websocket.send(json.dumps({
            "type": "connected",
            "asset": asset,
            "message": f"Connected to {asset} price stream"
        }))

        # Send latest cached price if available
        key = f"{asset}:mcx"
        if key in self.latest_prices:
            await websocket.send(json.dumps({
                "type": "price_update",
                **self.latest_prices[key]
            }))

    def unregister(self, websocket: websockets.WebSocketServerProtocol):
        """Unregister a client connection."""
        self.clients.discard(websocket)
        logger.info(f"Client disconnected. Total: {len(self.clients)}")

    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return

        msg_str = json.dumps(message)
        disconnected = set()

        for client in self.clients:
            try:
                await client.send(msg_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
            except Exception as e:
                logger.error(f"Error broadcasting to client: {e}")
                disconnected.add(client)

        # Clean up disconnected clients
        for client in disconnected:
            self.clients.discard(client)

    async def redis_subscriber(self):
        """Subscribe to Redis price updates and broadcast to clients."""
        import redis.asyncio as redis

        while True:
            try:
                self.redis = redis.Redis(
                    host=REDIS_HOST,
                    port=REDIS_PORT,
                    db=0,
                    decode_responses=True
                )
                self.pubsub = self.redis.pubsub()
                await self.pubsub.subscribe(PRICE_CHANNEL)
                logger.info(f"Subscribed to Redis channel: {PRICE_CHANNEL}")

                async for message in self.pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            # Cache the latest price
                            key = f"{data.get('asset', 'silver')}:{data.get('market', 'mcx')}"
                            self.latest_prices[key] = data

                            # Broadcast to all clients
                            await self.broadcast({
                                "type": "price_update",
                                **data
                            })
                        except Exception as e:
                            logger.error(f"Error processing Redis message: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Redis subscriber error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            finally:
                if self.pubsub:
                    try:
                        await self.pubsub.unsubscribe(PRICE_CHANNEL)
                        await self.pubsub.aclose()
                    except:
                        pass
                if self.redis:
                    try:
                        await self.redis.aclose()
                    except:
                        pass

    async def handle_client(self, websocket: websockets.WebSocketServerProtocol, path: str):
        """Handle individual client connections."""
        # Extract asset from path (e.g., /ws/prices/silver -> silver)
        parts = path.strip("/").split("/")
        if len(parts) >= 3 and parts[0] == "ws" and parts[1] == "prices":
            asset = parts[2]
        else:
            asset = "silver"  # Default

        await self.register(websocket, asset)

        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    action = data.get("action")

                    if action == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                    elif action == "subscribe":
                        await websocket.send(json.dumps({
                            "type": "subscribed",
                            "asset": asset,
                            "message": f"Subscribed to {asset} updates"
                        }))
                    elif action == "get_latest":
                        market = data.get("market", "mcx")
                        key = f"{asset}:{market}"
                        if key in self.latest_prices:
                            await websocket.send(json.dumps({
                                "type": "price_update",
                                **self.latest_prices[key]
                            }))
                except json.JSONDecodeError:
                    pass

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.unregister(websocket)

    async def heartbeat(self):
        """Send periodic heartbeats to keep connections alive."""
        while True:
            await asyncio.sleep(30)
            if self.clients:
                await self.broadcast({"type": "heartbeat"})

    async def start(self, use_ssl: bool = True):
        """Start the WebSocket server."""
        ssl_context = None
        if use_ssl and Path(SSL_CERT).exists() and Path(SSL_KEY).exists():
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_context.load_cert_chain(SSL_CERT, SSL_KEY)
            logger.info("SSL enabled")
        else:
            logger.warning("SSL certificates not found, running without SSL")

        # Start Redis subscriber
        redis_task = asyncio.create_task(self.redis_subscriber())

        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.heartbeat())

        # Start WebSocket server
        async with serve(
            self.handle_client,
            WS_HOST,
            WS_PORT,
            ssl=ssl_context,
            ping_interval=20,
            ping_timeout=60,
        ):
            logger.info(f"WebSocket server started on {'wss' if ssl_context else 'ws'}://{WS_HOST}:{WS_PORT}")
            try:
                await asyncio.Future()  # Run forever
            except asyncio.CancelledError:
                pass
            finally:
                redis_task.cancel()
                heartbeat_task.cancel()


if __name__ == "__main__":
    server = WebSocketServer()
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("WebSocket server stopped")
