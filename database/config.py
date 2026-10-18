"""
Database configuration and session management
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

load_dotenv()

# Database URL - configure according to your environment
# For PostgreSQL: postgresql://username:password@localhost:5432/database_name
# For SQLite (development): sqlite:///./health_app.db
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./health_app.db")

# Validate PostgreSQL connection string
if DATABASE_URL.startswith("postgresql"):
    print(f"Using PostgreSQL database")
else:
    print(f"Using SQLite database: {DATABASE_URL}")

# Create engine with appropriate settings
if "sqlite" in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        pool_pre_ping=True,
        echo=False  # Set to True for debugging SQL queries
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        pool_size=10,
        max_overflow=20,
        pool_timeout=30,
        pool_recycle=3600,
        pool_pre_ping=True,
        echo=False  # Set to True for debugging SQL queries
    )

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db():
    """
    Dependency for FastAPI routes to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Apply versioned database migrations.
    """
    from alembic import command
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.script import ScriptDirectory

    alembic_config = Config("alembic.ini")
    with engine.connect() as connection:
        current_revision = MigrationContext.configure(connection).get_current_revision()
    script = ScriptDirectory.from_config(alembic_config)
    head_revisions = set(script.get_heads())
    if current_revision in head_revisions:
        return
    command.upgrade(alembic_config, "head")
