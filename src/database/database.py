from functools import lru_cache
import logging
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session


from config import Settings

# Configure logging with a more specific format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@lru_cache()
def get_settings() -> Settings:
    """Retrieve application settings with caching."""
    return Settings()


def get_database_url() -> str:
    """
    Generate database URL from settings.

    Returns:
        str: Formatted database URL
    """
    settings = get_settings()
    return (
        f"postgresql://{settings.POSTGRES_USER}:"
        f"{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:"
        f"{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )


def create_db_engine() -> Engine:
    """Create and configure database engine."""
    return create_engine(
        get_database_url(),
        pool_pre_ping=True,  # Enable connection health checks
        pool_size=5,  # Set reasonable pool size
        max_overflow=10,  # Set maximum overflow connections
    )


engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Create database session.

    Yields:
        Session: Database session

    Raises:
        Exception: If database connection fails
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        raise
    finally:
        db.close()
