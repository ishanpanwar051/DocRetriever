from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager
from config.settings import settings

# WHY: pool_pre_ping=True prevents the application from using stale connections 
# by issuing a lightweight ping before checking out a connection.
# WHY: pool_size=5 and max_overflow=10 balance database resources. 5 persistent 
# connections cover baseline traffic, with 10 extra temporary connections for bursts.
engine = create_engine(
    settings.database_url,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

# WHY: autoflush=False gives us manual control over when changes are flushed to DB,
# preventing unintended writes during complex read/modify logic.
# WHY: autocommit=False ensures transactions are explicit, adhering to ACID principles.
SessionLocal = sessionmaker(
    autocommit=False, 
    autoflush=False, 
    bind=engine
)

@contextmanager
def get_db():
    """
    Context manager for database sessions.
    WHY: Using a context manager ensures that database connections are properly 
    closed (yielded back to the pool) even if exceptions occur, preventing resource leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_connection():
    """
    Test database connection and verify pgvector extension is enabled.
    WHY: Fails fast on startup if the critical pgvector extension is missing.
    """
    with engine.connect() as conn:
        try:
            # Check if pgvector is available
            res = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector';")).fetchone()
            if not res:
                print("WARNING: pgvector extension not found!")
                return False
            print("Successfully connected to database and pgvector is enabled.")
            return True
        except Exception as e:
            print(f"Database connection failed: {e}")
            return False
