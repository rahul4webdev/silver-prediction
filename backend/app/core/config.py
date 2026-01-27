"""
Application configuration using Pydantic Settings.
Loads environment variables and provides typed configuration.

Configuration is loaded from environment variables and the parent .env file.
The parent .env file is at /project-root/.env (one level above backend/).

IMPORTANT: UPSTOX_ACCESS_TOKEN should NOT be in GitHub secrets.
It's obtained via OAuth and saved to .env by the /auth/callback endpoint.
Tokens expire daily at midnight IST and need to be refreshed via OAuth.

Version: 1.0.1 - Added flexible training thresholds
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Determine the path to the main .env file
# Backend runs from /path/to/project/backend, .env is at /path/to/project/.env
BACKEND_DIR = Path(__file__).parent.parent.parent  # backend/
PROJECT_ROOT = BACKEND_DIR.parent  # project root
MAIN_ENV_FILE = PROJECT_ROOT / ".env"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        # Load from parent .env file only (where GitHub Actions writes other secrets)
        # UPSTOX_ACCESS_TOKEN is managed separately via OAuth, not GitHub secrets
        env_file=str(MAIN_ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ==========================================================================
    # Application
    # ==========================================================================
    environment: str = Field(default="development", description="Environment name")
    debug: bool = Field(default=False, description="Debug mode")
    secret_key: str = Field(
        default="change-me-in-production-min-32-characters",
        description="Secret key for JWT signing",
    )

    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_domain: str = Field(default="localhost", description="API domain")

    # Frontend
    frontend_url: str = Field(
        default="http://localhost:3000", description="Frontend URL for CORS"
    )
    dashboard_domain: str = Field(
        default="localhost:3000", description="Dashboard domain"
    )

    # ==========================================================================
    # Upstox API (MCX Data)
    # ==========================================================================
    upstox_api_key: str = Field(default="", description="Upstox API key")
    upstox_api_secret: str = Field(default="", description="Upstox API secret")
    upstox_redirect_uri: str = Field(
        default="http://localhost:8000/api/v1/auth/callback",
        description="Upstox OAuth redirect URI",
    )
    upstox_access_token: Optional[str] = Field(
        default=None, description="Upstox access token (if already obtained)"
    )

    # ==========================================================================
    # Database
    # ==========================================================================
    postgres_host: str = Field(default="127.0.0.1", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(
        default="silver_prediction", description="PostgreSQL database name"
    )
    postgres_user: str = Field(
        default="prediction_user", description="PostgreSQL username"
    )
    postgres_password: str = Field(default="", description="PostgreSQL password")

    @property
    def database_url(self) -> str:
        """Construct async database URL with URL-encoded password."""
        # Password may contain special chars like @ that need URL encoding
        encoded_password = quote_plus(self.postgres_password)
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def sync_database_url(self) -> str:
        """Construct sync database URL for Alembic."""
        encoded_password = quote_plus(self.postgres_password)
        return (
            f"postgresql://{self.postgres_user}:{encoded_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ==========================================================================
    # Redis
    # ==========================================================================
    redis_host: str = Field(default="127.0.0.1", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: Optional[str] = Field(default=None, description="Redis password")

    @property
    def redis_url(self) -> str:
        """Construct Redis URL."""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"

    @property
    def celery_broker_url(self) -> str:
        """Celery broker URL."""
        return self.redis_url

    @property
    def celery_result_backend(self) -> str:
        """Celery result backend URL."""
        return self.redis_url

    # ==========================================================================
    # Backup Data Sources
    # ==========================================================================
    alpha_vantage_api_key: str = Field(
        default="", description="Alpha Vantage API key (backup)"
    )

    # ==========================================================================
    # News Sentiment - Multiple Sources
    # ==========================================================================
    newsapi_key: Optional[str] = Field(
        default=None, description="NewsAPI.org API key (optional, for enhanced sentiment)"
    )
    finnhub_api_key: Optional[str] = Field(
        default=None, description="Finnhub API key (free tier available)"
    )
    reddit_client_id: Optional[str] = Field(
        default=None, description="Reddit API client ID (optional, for social sentiment)"
    )
    reddit_client_secret: Optional[str] = Field(
        default=None, description="Reddit API client secret"
    )

    # ==========================================================================
    # Notifications
    # ==========================================================================
    telegram_bot_token: Optional[str] = Field(
        default=None, description="Telegram bot token"
    )
    telegram_chat_id: Optional[str] = Field(
        default=None, description="Telegram chat ID"
    )
    smtp_host: Optional[str] = Field(default=None, description="SMTP host")
    smtp_port: int = Field(default=587, description="SMTP port")
    smtp_user: Optional[str] = Field(default=None, description="SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="SMTP password")
    notification_email: Optional[str] = Field(
        default=None, description="Notification email"
    )

    # ==========================================================================
    # ML Model Settings
    # ==========================================================================
    model_retrain_hour: int = Field(
        default=2, description="Hour to retrain models (0-23)"
    )
    model_weights_path: str = Field(
        default="./data/models", description="Path to store model weights"
    )
    prediction_intervals: str = Field(
        default="30m,1h,4h,daily", description="Comma-separated prediction intervals"
    )

    @property
    def prediction_intervals_list(self) -> List[str]:
        """Get prediction intervals as a list."""
        return [i.strip() for i in self.prediction_intervals.split(",")]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience instance
settings = get_settings()
