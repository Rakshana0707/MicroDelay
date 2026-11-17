import os
import enum
import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey, Text, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database file URL - resolve absolute path relative to this backend module
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("DATABASE_PATH", os.path.join(BASE_DIR, "microdelay.db"))
DATABASE_URL = f"sqlite:///{DB_PATH}"

# Create database engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()


class DataProvenance(str, enum.Enum):
    SYNTHETIC = "synthetic"
    REAL = "real"
    UNKNOWN = "unknown"


class DataSource(str, enum.Enum):
    GENERATOR = "generator"
    SIMULATOR_CHECKOUT = "simulator_checkout"
    MANUAL = "manual"
    LEGACY_IMPORT = "legacy_import"
    UNKNOWN = "unknown"


class ObservationStatus(str, enum.Enum):
    VALID = "valid"
    FLAGGED = "flagged"
    ARCHIVED = "archived"


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    profile_type = Column(String, nullable=False) # student, teacher, professor, it_professional
    consent_to_global_training = Column(Boolean, default=True, nullable=False)

    events = relationship("Event", back_populates="user", cascade="all, delete-orphan")
    departure_logs = relationship("DepartureLog", back_populates="user", cascade="all, delete-orphan")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    day_of_week = Column(String, nullable=False)
    start_time = Column(String, nullable=False) # e.g. "08:30:00"
    estimated_prep_duration = Column(Integer, nullable=False) # min
    estimated_travel_duration = Column(Integer, nullable=False) # min
    importance = Column(String, nullable=False) # high, medium, low

    user = relationship("User", back_populates="events")
    logs = relationship("DepartureLog", back_populates="event", cascade="all, delete-orphan")


class DepartureLog(Base):
    __tablename__ = "departure_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    event_id = Column(Integer, ForeignKey("events.id"), nullable=True)
    event_name = Column(String, nullable=False)
    date = Column(String, nullable=False) # e.g. "2026-08-17"
    day_of_week = Column(String, nullable=False)

    # Planned / Estimated
    planned_prep_start = Column(String, nullable=False)
    planned_departure = Column(String, nullable=False)
    estimated_prep_duration_min = Column(Float, nullable=False)
    estimated_travel_duration_min = Column(Float, nullable=False)

    # Actuals
    actual_prep_start = Column(String, nullable=False)
    actual_departure = Column(String, nullable=False)
    actual_arrival = Column(String, nullable=False)
    actual_prep_duration_min = Column(Float, nullable=False)
    actual_travel_duration_min = Column(Float, nullable=False)

    # Lateness metric
    total_delay_min = Column(Float, default=0.0)

    # Pre-departure state variables
    bag_prepared = Column(Boolean, default=False)

    # Binary delay tags
    tag_one_last_task = Column(Boolean, default=False)
    tag_finding_items = Column(Boolean, default=False)
    tag_phone_usage = Column(Boolean, default=False)
    tag_waiting = Column(Boolean, default=False)

    # Computed/Allocated delay durations in minutes
    delay_one_last_task_min = Column(Float, default=0.0)
    delay_finding_items_min = Column(Float, default=0.0)
    delay_phone_usage_min = Column(Float, default=0.0)
    delay_waiting_min = Column(Float, default=0.0)
    delay_prep_bias_min = Column(Float, default=0.0)

    # Data Provenance and Integrity Metadata
    provenance = Column(String, default=DataProvenance.UNKNOWN.value, nullable=False) # synthetic, real, unknown
    data_source = Column(String, default=DataSource.UNKNOWN.value, nullable=False) # generator, simulator_checkout, etc.
    is_synthetic = Column(Boolean, default=False, nullable=False)
    recorded_at = Column(String, default=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat(), nullable=False)
    observation_status = Column(String, default=ObservationStatus.VALID.value, nullable=False)
    attribution_metadata = Column(Text, nullable=True)

    user = relationship("User", back_populates="departure_logs")
    event = relationship("Event", back_populates="logs")


class ModelVersionRecord(Base):
    __tablename__ = "model_metadata"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    model_version = Column(String, nullable=False)
    model_type = Column(String, nullable=False) # RandomForestRegressor, etc.
    training_data_source = Column(String, nullable=False) # real, synthetic, synthetic_assisted
    training_samples = Column(Integer, nullable=False)
    training_started_at = Column(String, nullable=False)
    training_completed_at = Column(String, nullable=True)
    validation_mae = Column(Float, nullable=True)
    test_mae = Column(Float, nullable=True)
    feature_schema_version = Column(String, default="v1_standard", nullable=False)
    artifact_path = Column(String, nullable=False)
    model_status = Column(String, default="training", nullable=False) # training, trained, validated, approved, active, rejected, archived, failed
    is_active = Column(Boolean, default=False, nullable=False)
    status_reason = Column(Text, nullable=True)
    metrics_json = Column(Text, nullable=True)

    user = relationship("User", backref="model_versions")


class UserClusterProfile(Base):
    __tablename__ = "user_cluster_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    cluster_id = Column(Integer, nullable=False)
    cluster_name = Column(String, nullable=False)
    cluster_description = Column(Text, nullable=False)
    dominant_characteristics = Column(Text, nullable=False)  # JSON list
    user_features = Column(Text, nullable=False)  # JSON dict
    cluster_centroid = Column(Text, nullable=False)  # JSON dict
    population_mean = Column(Text, nullable=False)  # JSON dict
    data_sufficiency_status = Column(String, nullable=False)  # sufficient, gathering_history, insufficient_data
    sample_count = Column(Integer, nullable=False)
    provenance = Column(String, default="real", nullable=False)  # real, synthetic
    cluster_model_version = Column(String, nullable=False)
    generated_at = Column(String, nullable=False)

    user = relationship("User", backref="cluster_profiles")


def migrate_db(db_engine=None):
    """Safely migrate SQLite database schema to include provenance fields and backfill legacy records."""
    target_engine = db_engine or engine
    with target_engine.connect() as conn:
        result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='departure_logs'")).fetchone()
        if not result:
            return

        columns = [row[1] for row in conn.execute(text("PRAGMA table_info(departure_logs)")).fetchall()]

        if "provenance" not in columns:
            conn.execute(text("ALTER TABLE departure_logs ADD COLUMN provenance VARCHAR DEFAULT 'unknown'"))
        if "data_source" not in columns:
            conn.execute(text("ALTER TABLE departure_logs ADD COLUMN data_source VARCHAR DEFAULT 'unknown'"))
        if "is_synthetic" not in columns:
            conn.execute(text("ALTER TABLE departure_logs ADD COLUMN is_synthetic BOOLEAN DEFAULT 0"))
        if "recorded_at" not in columns:
            conn.execute(text("ALTER TABLE departure_logs ADD COLUMN recorded_at VARCHAR DEFAULT ''"))
        if "observation_status" not in columns:
            conn.execute(text("ALTER TABLE departure_logs ADD COLUMN observation_status VARCHAR DEFAULT 'valid'"))
        if "attribution_metadata" not in columns:
            conn.execute(text("ALTER TABLE departure_logs ADD COLUMN attribution_metadata TEXT"))

        # Ensure consent_to_global_training column exists in users
        user_table = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")).fetchone()
        if user_table:
            user_columns = [row[1] for row in conn.execute(text("PRAGMA table_info(users)")).fetchall()]
            if "consent_to_global_training" not in user_columns:
                conn.execute(text("ALTER TABLE users ADD COLUMN consent_to_global_training BOOLEAN DEFAULT 1"))

        # Backfill existing legacy demo records: demo user logs were generated by generator.py
        conn.execute(text("""
            UPDATE departure_logs
            SET provenance = 'synthetic',
                data_source = 'generator',
                is_synthetic = 1
            WHERE (provenance = 'unknown' OR provenance IS NULL OR provenance = '')
              AND user_id IN (
                  SELECT id FROM users WHERE username IN ('student_demo', 'teacher_demo', 'professor_demo', 'engineer_demo')
              )
        """))

        # Backfill recorded_at if missing
        conn.execute(text("""
            UPDATE departure_logs
            SET recorded_at = date || 'T12:00:00Z'
            WHERE recorded_at IS NULL OR recorded_at = ''
        """))

        # Create model_metadata table if not present
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id VARCHAR UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                model_version VARCHAR NOT NULL,
                model_type VARCHAR NOT NULL,
                training_data_source VARCHAR NOT NULL,
                training_samples INTEGER NOT NULL,
                training_started_at VARCHAR NOT NULL,
                training_completed_at VARCHAR,
                validation_mae FLOAT,
                test_mae FLOAT,
                feature_schema_version VARCHAR NOT NULL,
                artifact_path VARCHAR NOT NULL,
                model_status VARCHAR NOT NULL,
                is_active BOOLEAN NOT NULL DEFAULT 0,
                status_reason TEXT,
                metrics_json TEXT
            )
        """))

        # Create user_cluster_profiles table if not present
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_cluster_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                cluster_id INTEGER NOT NULL,
                cluster_name VARCHAR NOT NULL,
                cluster_description TEXT NOT NULL,
                dominant_characteristics TEXT NOT NULL,
                user_features TEXT NOT NULL,
                cluster_centroid TEXT NOT NULL,
                population_mean TEXT NOT NULL,
                data_sufficiency_status VARCHAR NOT NULL,
                sample_count INTEGER NOT NULL,
                provenance VARCHAR NOT NULL DEFAULT 'real',
                cluster_model_version VARCHAR NOT NULL,
                generated_at VARCHAR NOT NULL
            )
        """))

        conn.commit()


def init_db():
    Base.metadata.create_all(bind=engine)
    migrate_db(engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
