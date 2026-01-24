"""
Telegram notification service for the Silver Prediction System.

Sends notifications for:
- Daily Upstox token re-authentication reminder (8:45 AM IST)
- Tick collector status
- Platform health status
- Predictions for all intervals and contracts
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from decimal import Decimal

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification service using Bot API.
    """

    def __init__(self):
        self.bot_token = settings.telegram_bot_token
        self.chat_id = settings.telegram_chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return bool(self.bot_token and self.chat_id)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    async def send_message(
        self,
        text: str,
        parse_mode: str = "HTML",
        disable_notification: bool = False,
    ) -> bool:
        """
        Send a message via Telegram Bot API.

        Args:
            text: Message text (supports HTML formatting)
            parse_mode: Message parse mode (HTML or Markdown)
            disable_notification: Send silently

        Returns:
            True if message sent successfully
        """
        if not self.is_configured:
            logger.warning("Telegram not configured - skipping notification")
            return False

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                    "disable_notification": disable_notification,
                },
            )

            if response.status_code == 200:
                logger.info("Telegram notification sent successfully")
                return True
            else:
                logger.error(f"Telegram API error: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Failed to send Telegram notification: {e}")
            return False

    async def close(self):
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ==================== Notification Templates ====================

    async def send_auth_reminder(self) -> bool:
        """
        Send daily Upstox token re-authentication reminder.
        Should be scheduled at 8:45 AM IST.
        """
        message = """
🔔 <b>Daily Authentication Reminder</b>

⏰ Time to re-authenticate Upstox token!

The Upstox access token expires daily at midnight IST.
Please authenticate before market opens at 9:00 AM.

🔗 <b>Authentication URL:</b>
https://predictionapi.gahfaudio.in/api/v1/auth/login

📊 Market opens in <b>15 minutes</b>
"""
        return await self.send_message(message.strip())

    async def send_tick_collector_status(
        self,
        is_running: bool,
        contracts_subscribed: int = 0,
        last_tick_time: Optional[datetime] = None,
        ticks_per_minute: float = 0,
        error_message: Optional[str] = None,
    ) -> bool:
        """
        Send tick collector health status notification.
        """
        status_emoji = "✅" if is_running and not error_message else "❌"
        status_text = "Running" if is_running else "Stopped"

        if error_message:
            status_text = f"Error: {error_message}"

        last_tick_str = "N/A"
        if last_tick_time:
            last_tick_str = last_tick_time.strftime("%H:%M:%S IST")

        message = f"""
📡 <b>Tick Collector Status</b>

{status_emoji} Status: <b>{status_text}</b>
📊 Contracts Subscribed: <b>{contracts_subscribed}</b>
🕐 Last Tick: <b>{last_tick_str}</b>
📈 Ticks/min: <b>{ticks_per_minute:.1f}</b>
"""
        return await self.send_message(message.strip())

    async def send_platform_health(
        self,
        api_healthy: bool,
        db_healthy: bool,
        redis_healthy: bool,
        upstox_authenticated: bool,
        tick_collector_running: bool,
        scheduler_running: bool,
        models_trained: Dict[str, bool] = None,
    ) -> bool:
        """
        Send comprehensive platform health status.
        """
        def status_icon(healthy: bool) -> str:
            return "✅" if healthy else "❌"

        models_status = ""
        if models_trained:
            models_status = "\n<b>Model Status:</b>\n"
            for interval, trained in models_trained.items():
                models_status += f"  {status_icon(trained)} {interval}\n"

        message = f"""
🏥 <b>Platform Health Status</b>

<b>Core Services:</b>
{status_icon(api_healthy)} API Server
{status_icon(db_healthy)} Database
{status_icon(redis_healthy)} Redis Cache

<b>Data Collection:</b>
{status_icon(upstox_authenticated)} Upstox Auth
{status_icon(tick_collector_running)} Tick Collector
{status_icon(scheduler_running)} Scheduler
{models_status}
⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}
"""
        return await self.send_message(message.strip())

    async def send_predictions(
        self,
        interval: str,
        predictions: List[Dict[str, Any]],
    ) -> bool:
        """
        Send prediction notifications for a specific interval.

        Args:
            interval: Prediction interval (30m, 1h, 4h, 1d)
            predictions: List of prediction dictionaries
        """
        interval_labels = {
            "30m": "30 Minute",
            "1h": "1 Hour",
            "4h": "4 Hour",
            "1d": "Daily",
        }

        interval_label = interval_labels.get(interval, interval)

        # Group by market
        mcx_predictions = [p for p in predictions if p.get("market") == "mcx"]
        comex_predictions = [p for p in predictions if p.get("market") == "comex"]

        def format_prediction(p: Dict) -> str:
            direction = p.get("predicted_direction", "neutral")
            direction_emoji = "🟢" if direction == "bullish" else "🔴" if direction == "bearish" else "⚪"
            confidence = p.get("direction_confidence", 0) * 100

            current = p.get("current_price", 0)
            predicted = p.get("predicted_price", 0)

            # Format price based on market
            market = p.get("market", "mcx")
            if market == "mcx":
                current_str = f"₹{current:,.0f}"
                predicted_str = f"₹{predicted:,.0f}"
            else:
                current_str = f"${current:.2f}"
                predicted_str = f"${predicted:.2f}"

            # Contract info
            contract = p.get("contract_type", "")
            trading_symbol = p.get("trading_symbol", "")

            # Extract expiry from trading symbol
            expiry = ""
            if trading_symbol:
                parts = trading_symbol.split()
                if len(parts) >= 5:
                    expiry = f" ({parts[2]} {parts[3]} {parts[4]})"

            change_pct = ((predicted - current) / current * 100) if current else 0
            change_str = f"+{change_pct:.2f}%" if change_pct >= 0 else f"{change_pct:.2f}%"

            return f"{direction_emoji} <b>{contract}{expiry}</b>\n   {current_str} → {predicted_str} ({change_str}) [{confidence:.0f}%]"

        message = f"""
🔮 <b>{interval_label} Predictions</b>
━━━━━━━━━━━━━━━━━━━━━━
"""

        if mcx_predictions:
            message += "\n<b>🇮🇳 MCX Silver</b>\n"
            for p in mcx_predictions:
                message += format_prediction(p) + "\n"

        if comex_predictions:
            message += "\n<b>🇺🇸 COMEX Silver</b>\n"
            for p in comex_predictions:
                message += format_prediction(p) + "\n"

        message += f"\n⏰ {datetime.now().strftime('%H:%M IST')}"

        return await self.send_message(message.strip())

    async def send_prediction_verification(
        self,
        prediction: Dict[str, Any],
        verification: Dict[str, Any],
    ) -> bool:
        """
        Send notification when a prediction is verified.
        """
        direction = prediction.get("predicted_direction", "neutral")
        is_correct = verification.get("is_direction_correct", False)

        result_emoji = "✅" if is_correct else "❌"
        result_text = "Correct" if is_correct else "Wrong"

        market = prediction.get("market", "mcx")
        interval = prediction.get("interval", "30m")
        contract = prediction.get("contract_type", "SILVER")

        current = prediction.get("current_price", 0)
        predicted = prediction.get("predicted_price", 0)
        actual = verification.get("actual_price", 0)

        # Format prices
        if market == "mcx":
            current_str = f"₹{current:,.0f}"
            predicted_str = f"₹{predicted:,.0f}"
            actual_str = f"₹{actual:,.0f}"
        else:
            current_str = f"${current:.2f}"
            predicted_str = f"${predicted:.2f}"
            actual_str = f"${actual:.2f}"

        error_pct = verification.get("price_error_percent", 0)

        message = f"""
{result_emoji} <b>Prediction Verified</b>

📊 {market.upper()} {contract} ({interval})

Predicted: {direction.upper()}
Result: <b>{result_text}</b>

💰 Prices:
  Start: {current_str}
  Predicted: {predicted_str}
  Actual: {actual_str}

📉 Error: {error_pct:+.2f}%
"""
        return await self.send_message(message.strip())

    async def send_error_alert(
        self,
        component: str,
        error: str,
        severity: str = "warning",
    ) -> bool:
        """
        Send error alert notification.
        """
        severity_emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "error": "❌",
            "critical": "🚨",
        }

        emoji = severity_emoji.get(severity, "⚠️")

        message = f"""
{emoji} <b>Alert: {component}</b>

{error}

⏰ {datetime.now().strftime("%Y-%m-%d %H:%M:%S IST")}
"""
        return await self.send_message(message.strip())


# Singleton instance
telegram_notifier = TelegramNotifier()
