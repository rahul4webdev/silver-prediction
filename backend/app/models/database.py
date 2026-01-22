"""
Database configuration and session management.
Uses SQLAlchemy async with PostgreSQL + TimescaleDB extension.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


# Create async engine with SSL disabled for local connections
# asyncpg has issues with SSL hostname resolution on localhost
connect_args = {}
if settings.postgres_host in ("127.0.0.1", "localhost"):
    connect_args["ssl"] = False

engine = create_async_engine(
    settings.database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    connect_args=connect_args,
)

# Create session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager for database session (for use outside of FastAPI)."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initialize database - create tables and enable TimescaleDB extension.
    Should be called once at application startup.
    """
    async with engine.begin() as conn:
        # Enable TimescaleDB extension
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)


async def create_hypertables() -> None:
    """
    Convert tables to TimescaleDB hypertables for time-series optimization.
    Should be called after init_db().
    """
    async with engine.begin() as conn:
        # Convert price_data to hypertable
        try:
            await conn.execute(text("""
                SELECT create_hypertable(
                    'price_data',
                    'timestamp',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """))
        except Exception:
            pass  # Table might already be a hypertable

        # Convert predictions to hypertable
        try:
            await conn.execute(text("""
                SELECT create_hypertable(
                    'predictions',
                    'created_at',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """))
        except Exception:
            pass

        # Convert market_factors to hypertable
        try:
            await conn.execute(text("""
                SELECT create_hypertable(
                    'market_factors',
                    'timestamp',
                    if_not_exists => TRUE,
                    migrate_data => TRUE
                );
            """))
        except Exception:
            pass


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
