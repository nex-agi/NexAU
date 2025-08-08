"""Database connection and session management."""

from sqlmodel import create_engine, Session as SQLSession, SQLModel
from sqlalchemy.engine import Engine
from contextlib import contextmanager
import os
from typing import Generator, Optional


def get_database_url(env: str = "development") -> str:
    """Get database URL based on environment."""
    if env == "development" or env == "test":
        # SQLite for local development
        db_path = os.path.join("data", "autotuning.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return f"sqlite:///{db_path}"
    elif env == "production":
        # PostgreSQL for production
        user = os.getenv("DATABASE_USER", "autotuning")
        password = os.getenv("DATABASE_PASSWORD")
        host = os.getenv("DATABASE_HOST", "localhost")
        port = os.getenv("DATABASE_PORT", "5432")
        database = os.getenv("DATABASE_NAME", "autotuning")
        
        if not password:
            raise ValueError("DATABASE_PASSWORD environment variable required for production")
            
        return f"postgresql://{user}:{password}@{host}:{port}/{database}"
    else:
        raise ValueError(f"Unknown environment: {env}")


def create_database_engine(env: str = "development") -> Engine:
    """Create database engine with appropriate settings."""
    database_url = get_database_url(env)
    
    if env == "development" or env == "test":
        # SQLite settings
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},  # Allow SQLite threading
            echo=False  # Disable SQL logging by default
        )
    else:
        # PostgreSQL settings
        return create_engine(
            database_url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True  # Validate connections
        )


def init_database(engine: Engine) -> None:
    """Initialize database tables."""
    SQLModel.metadata.create_all(engine)


@contextmanager
def get_db_session(engine: Engine) -> Generator[SQLSession, None, None]:
    """Get database session with automatic cleanup."""
    session = SQLSession(engine)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Global engine instance (initialized once per application)
_engine: Optional[Engine] = None


def get_engine(env: str = "development") -> Engine:
    """Get or create the global database engine."""
    global _engine
    if _engine is None:
        _engine = create_database_engine(env)
        init_database(_engine)
    return _engine