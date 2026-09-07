from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.core.config import get_settings
import threading

# Base class for models
Base = declarative_base()

# We'll create the engine and sessionmaker lazily, but allow reset
_engine = None
_AsyncSessionLocal = None
_lock = threading.Lock()

def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_async_engine(
            settings.DATABASE_URL,
            echo=False,  # Set to True for SQL logging
            future=True,
            pool_pre_ping=True,  # Validate connections before use — Neon's
                                 # pooler closes idle connections server-side,
                                 # and without this SQLAlchemy hands out a
                                 # dead connection, causing
                                 # "InterfaceError: connection is closed".
            pool_recycle=300,   # Proactively recycle connections every 5
                                 # minutes, before Neon's idle timeout can
                                 # close them out from under us.
        )
    return _engine

def get_async_session_local():
    global _AsyncSessionLocal
    if _AsyncSessionLocal is None:
        engine = get_engine()
        _AsyncSessionLocal = sessionmaker(
            engine, class_=AsyncSession, expire_on_commit=False
        )
    return _AsyncSessionLocal

async def reset_engine():
    global _engine, _AsyncSessionLocal

    if _engine is not None:
        await _engine.dispose()

    _engine = None
    _AsyncSessionLocal = None

# Dependency to get DB session
async def get_db() -> AsyncSession:
    async with get_async_session_local()() as session:
        try:
            yield session
        finally:
            await session.close()
