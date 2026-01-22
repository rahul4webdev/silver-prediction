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
    Initialize database - create tables and optionally enable TimescaleDB extension.
    Should be called once at application startup.
    TimescaleDB is optional - system works without it (just without time-series optimizations).
    """
    import logging
    logger = logging.getLogger(__name__)

    # Try to enable TimescaleDB extension in a separate connection (optional)
    # This needs to be in its own transaction since it may fail
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;"))
            logger.info("TimescaleDB extension enabled")
    except Exception as e:
        logger.warning(f"TimescaleDB not available (optional): {e}")
        # Continue without TimescaleDB - system will work with regular PostgreSQL

    # Create all tables in a new transaction
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")


async def create_hypertables() -> None:
    """
    Convert tables to TimescaleDB hypertables for time-series optimization.
    Should be called after init_db().
    This is optional - if TimescaleDB is not installed, regular PostgreSQL tables are used.
    """
    import logging
    logger = logging.getLogger(__name__)

    async with engine.begin() as conn:
        # Check if TimescaleDB is available
        try:
            result = await conn.execute(text(
                "SELECT 1 FROM pg_extension WHERE extname = 'timescaledb'"
            ))
            if not result.scalar():
                logger.info("TimescaleDB not installed - using regular PostgreSQL tables")
                return
        except Exception:
            logger.info("TimescaleDB not available - using regular PostgreSQL tables")
            return

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
            logger.info("price_data converted to hypertable")
        except Exception as e:
            logger.debug(f"price_data hypertable: {e}")

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
            logger.info("predictions converted to hypertable")
        except Exception as e:
            logger.debug(f"predictions hypertable: {e}")

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
            logger.info("market_factors converted to hypertable")
        except Exception as e:
            logger.debug(f"market_factors hypertable: {e}")


async def close_db() -> None:
    """Close database connections."""
    await engine.dispose()
