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


class BehavioralProfileSnapshot(Base):
    __tablename__ = "behavioral_profiles"

    id = Column(Integer, primary_key=True, index=True)
    profile_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    profile_version = Column(String, nullable=False)
    generated_at = Column(String, nullable=False)
    observation_window = Column(String, nullable=False)  # "all_time", "30d", "7d"
    observation_count = Column(Integer, nullable=False)
    data_provenance = Column(String, nullable=False)  # "real", "synthetic"
    behavioral_metrics = Column(Text, nullable=False)  # JSON dict covering dimensions A-F
    cluster_reference = Column(String, nullable=True)  # e.g. "Phone-Interrupted Preparation"
    confidence_or_sufficiency_status = Column(String, nullable=False)  # "sufficient", "gathering_history", "insufficient_data"
    insights = Column(Text, nullable=False)  # JSON list of evidence-backed strings

    user = relationship("User", backref="behavioral_profiles")


class PredictionExplanationRecord(Base):
    __tablename__ = "prediction_explanations"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_version = Column(String, nullable=False)
    feature_schema_version = Column(String, default="v1_leakage_safe", nullable=False)
    explanation_method = Column(String, nullable=False)  # "tree_shap", "tree_feature_contribution", "linear_coefficients", "behavioral_rule_reasoning"
    predicted_delay_min = Column(Float, nullable=True)
    late_probability = Column(Float, nullable=True)
    top_contributing_features = Column(Text, nullable=False)  # JSON list of positive contributors
    mitigating_factors = Column(Text, nullable=True)  # JSON list of mitigating contributors
    historical_comparisons = Column(Text, nullable=True)  # JSON list of comparison statements
    limitations = Column(Text, nullable=True)  # JSON list of caveats
    causation_disclaimer = Column(String, nullable=False)
    feature_audit_values = Column(Text, nullable=False)  # JSON dict of input values
    explanation_timestamp = Column(String, nullable=False)

    user = relationship("User", backref="prediction_explanations")


class UserInterventionRecord(Base):
    __tablename__ = "user_interventions"

    id = Column(Integer, primary_key=True, index=True)
    intervention_id = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recommendation_id = Column(String, index=True, nullable=False)
    category = Column(String, nullable=False)  # "preparation", "distraction", "timing", "routine_consistency"
    action = Column(String, nullable=False)
    description = Column(String, nullable=False)
    implementation_steps = Column(Text, nullable=False)  # JSON list of steps
    behavioral_problem = Column(String, nullable=False)
    supporting_evidence = Column(Text, nullable=False)  # JSON dict
    status = Column(String, default="suggested", index=True, nullable=False)  # "suggested", "accepted", "started", "completed", "dismissed", "snoozed", "expired"
    expected_delay_reduction_min = Column(Float, default=0.0)
    expected_outcome = Column(Text, nullable=False)  # JSON dict
    measured_outcome = Column(Text, nullable=True)  # JSON dict
    baseline_metrics = Column(Text, nullable=True)  # JSON dict
    post_metrics = Column(Text, nullable=True)  # JSON dict
    confounding_factors = Column(Text, nullable=True)  # JSON list
    causation_disclaimer = Column(String, nullable=False)
    generated_at = Column(String, nullable=False)
    accepted_at = Column(String, nullable=True)
    started_at = Column(String, nullable=True)
    completed_at = Column(String, nullable=True)
    dismissed_at = Column(String, nullable=True)
    snoozed_until = Column(String, nullable=True)
    user_rating = Column(Integer, nullable=True)  # 1 to 5
    user_feedback_notes = Column(Text, nullable=True)
    user_helpful = Column(Boolean, nullable=True)
    completion_status = Column(String, nullable=True)  # "success", "neutral", "unsuccessful", "abandoned"

    user = relationship("User", backref="interventions")


class BehavioralDriftEventRecord(Base):
    """Stores diagnosed behavioral, data, and concept drift events."""
    __tablename__ = "behavioral_drift_events"

    id = Column(Integer, primary_key=True, index=True)
    drift_event_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    feature_name = Column(String, nullable=False)
    drift_type = Column(String, nullable=False)  # "data_drift", "concept_drift", "behavioral_improvement", "behavioral_deterioration"
    detection_method = Column(String, nullable=False)  # "kolmogorov_smirnov", "population_stability_index", "rolling_mean_shift", "page_hinkley"
    baseline_value = Column(Float, nullable=False)
    current_value = Column(Float, nullable=False)
    drift_score = Column(Float, nullable=False)
    p_value = Column(Float, nullable=True)
    severity = Column(String, nullable=False)  # "informational", "moderate", "significant"
    valence = Column(String, nullable=False)  # "positive", "negative", "neutral"
    temporal_window = Column(String, nullable=False)  # "7d_vs_baseline", "30d_vs_baseline", "sequential_stream"
    sample_count_baseline = Column(Integer, nullable=False)
    sample_count_current = Column(Integer, nullable=False)
    detected_at = Column(String, nullable=False)
    status = Column(String, nullable=False, default="active")  # "active", "acknowledged", "resolved", "dismissed"
    humanized_summary = Column(Text, nullable=False)
    is_statistically_significant = Column(Boolean, nullable=False, default=False)

    user = relationship("User", backref="drift_events")


class UserBehavioralBaselineRecord(Base):
    """Stores versioned historical behavioral baseline distributions for a user."""
    __tablename__ = "user_behavioral_baselines"

    id = Column(Integer, primary_key=True, index=True)
    baseline_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    baseline_version = Column(String, nullable=False, default="v1.0")
    generated_at = Column(String, nullable=False)
    observation_count = Column(Integer, nullable=False)
    historical_window = Column(String, nullable=False, default="all_time")
    data_provenance = Column(String, nullable=False, default="genuine")  # "genuine", "synthetic", "hybrid"
    metrics = Column(Text, nullable=False)  # JSON dict of metric -> summary stats
    is_active = Column(Boolean, nullable=False, default=True)

    user = relationship("User", backref="behavioral_baselines")


class ModelMonitoringLogRecord(Base):
    """Tracks sequential model prediction errors, rolling MAE, bias, and performance degradation."""
    __tablename__ = "model_monitoring_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_version = Column(String, nullable=False)
    evaluated_at = Column(String, nullable=False)
    rolling_window_size = Column(Integer, nullable=False)
    rolling_mae = Column(Float, nullable=False)
    rolling_bias = Column(Float, nullable=False)
    baseline_mae = Column(Float, nullable=False)
    baseline_bias = Column(Float, nullable=False)
    performance_degraded = Column(Boolean, nullable=False, default=False)
    degradation_pct = Column(Float, nullable=False, default=0.0)
    error_distribution_shift = Column(Boolean, nullable=False, default=False)
    details = Column(Text, nullable=True)  # JSON

    user = relationship("User", backref="model_monitoring_logs")


class ModelReliabilityRecord(Base):
    """Tracks empirical prediction intervals, calibration metrics, data sufficiency, and reliability states (Prompt 2.9)."""
    __tablename__ = "model_reliability_records"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_version = Column(String, nullable=False)
    evaluated_at = Column(String, nullable=False)
    conformal_quantile_80 = Column(Float, nullable=True)
    conformal_quantile_90 = Column(Float, nullable=True)
    empirical_coverage_80 = Column(Float, nullable=True)
    validation_mae = Column(Float, nullable=True)
    validation_rmse = Column(Float, nullable=True)
    brier_score = Column(Float, nullable=True)
    expected_calibration_error = Column(Float, nullable=True)
    calibration_status = Column(String, nullable=False)  # "calibrated", "uncalibrated", "insufficient_samples"
    data_sufficiency = Column(String, nullable=False)    # "insufficient", "minimal", "moderate", "robust"
    reliability_status = Column(String, nullable=False)  # "insufficient_history", "low_reliability", "moderate_reliability", "high_reliability", "degraded_model"
    sample_count_evaluated = Column(Integer, nullable=False)
    is_abstained = Column(Boolean, nullable=False, default=False)
    details = Column(Text, nullable=True)  # JSON: limitations, calibration bins, error distribution

    user = relationship("User", backref="reliability_records")


class CalendarConnection(Base):
    """Stores external calendar provider authorization and sync tokens (Prompt 3.2)."""
    __tablename__ = "calendar_connections"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    provider = Column(String, default="google", nullable=False)
    provider_account_reference = Column(String, nullable=True)  # e.g. email or account ID
    encrypted_tokens = Column(Text, nullable=False)  # JSON dict encrypted via Fernet
    connection_status = Column(String, default="connected", nullable=False)  # connected, expired, revoked, disconnected
    connected_at = Column(String, nullable=False)
    last_synced_at = Column(String, nullable=True)
    sync_token = Column(String, nullable=True)
    granted_scopes = Column(Text, nullable=True)  # JSON list of scopes

    user = relationship("User", backref="calendar_connections")
    events = relationship("CalendarEventRecord", back_populates="connection", cascade="all, delete-orphan")


class CalendarEventRecord(Base):
    """Synchronized event records distinct from routine templates (Prompt 3.2)."""
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    connection_id = Column(Integer, ForeignKey("calendar_connections.id"), nullable=False, index=True)
    provider_event_id = Column(String, nullable=False, index=True)
    title = Column(String, nullable=False)
    start_time = Column(String, nullable=False, index=True)  # ISO format
    end_time = Column(String, nullable=False)  # ISO format
    location = Column(String, nullable=True)
    calendar_timezone = Column(String, nullable=True)
    is_all_day = Column(Boolean, default=False, nullable=False)
    status = Column(String, default="confirmed", nullable=False)  # confirmed, tentative, cancelled
    recurrence_rule = Column(String, nullable=True)
    recurring_event_id = Column(String, nullable=True)
    travel_requirement = Column(String, default="unknown", nullable=False)  # physical, online, personal, excluded, unknown
    requires_departure_planning = Column(Boolean, default=False, nullable=False)
    missing_location_warning = Column(Boolean, default=False, nullable=False)
    online_meeting_url = Column(String, nullable=True)
    associated_routine_id = Column(Integer, ForeignKey("events.id"), nullable=True)
    prep_buffer_override_min = Column(Float, nullable=True)
    travel_mode_override = Column(String, nullable=True)  # driving, transit, walking, bicycling
    synced_at = Column(String, nullable=False)
    updated_at = Column(String, nullable=False)

    user = relationship("User", backref="calendar_events")
    connection = relationship("CalendarConnection", back_populates="events")
    associated_routine = relationship("Event", backref="calendar_events")



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

        # Create behavioral_profiles table if not present (Prompt 2.5)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS behavioral_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                profile_id VARCHAR UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                profile_version VARCHAR NOT NULL,
                generated_at VARCHAR NOT NULL,
                observation_window VARCHAR NOT NULL,
                observation_count INTEGER NOT NULL,
                data_provenance VARCHAR NOT NULL,
                behavioral_metrics TEXT NOT NULL,
                cluster_reference VARCHAR,
                confidence_or_sufficiency_status VARCHAR NOT NULL,
                insights TEXT NOT NULL
            )
        """))

        # Create prediction_explanations table if not present (Prompt 2.6)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS prediction_explanations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id VARCHAR UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                model_version VARCHAR NOT NULL,
                feature_schema_version VARCHAR NOT NULL,
                explanation_method VARCHAR NOT NULL,
                predicted_delay_min FLOAT,
                late_probability FLOAT,
                top_contributing_features TEXT NOT NULL,
                mitigating_factors TEXT,
                historical_comparisons TEXT,
                limitations TEXT,
                causation_disclaimer VARCHAR NOT NULL,
                feature_audit_values TEXT NOT NULL,
                explanation_timestamp VARCHAR NOT NULL
            )
        """))

        # Create user_interventions table if not present (Prompt 2.7)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                intervention_id VARCHAR UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                recommendation_id VARCHAR NOT NULL,
                category VARCHAR NOT NULL,
                action VARCHAR NOT NULL,
                description TEXT NOT NULL,
                implementation_steps TEXT NOT NULL,
                behavioral_problem VARCHAR NOT NULL,
                supporting_evidence TEXT NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'suggested',
                expected_delay_reduction_min FLOAT DEFAULT 0.0,
                expected_outcome TEXT NOT NULL,
                measured_outcome TEXT,
                baseline_metrics TEXT,
                post_metrics TEXT,
                confounding_factors TEXT,
                causation_disclaimer VARCHAR NOT NULL,
                generated_at VARCHAR NOT NULL,
                accepted_at VARCHAR,
                started_at VARCHAR,
                completed_at VARCHAR,
                dismissed_at VARCHAR,
                snoozed_until VARCHAR,
                user_rating INTEGER,
                user_feedback_notes TEXT,
                user_helpful BOOLEAN,
                completion_status VARCHAR
            )
        """))

        # Create behavioral_drift_events table if not exists (Prompt 2.8)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS behavioral_drift_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                drift_event_id VARCHAR UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                feature_name VARCHAR NOT NULL,
                drift_type VARCHAR NOT NULL,
                detection_method VARCHAR NOT NULL,
                baseline_value FLOAT NOT NULL,
                current_value FLOAT NOT NULL,
                drift_score FLOAT NOT NULL,
                p_value FLOAT,
                severity VARCHAR NOT NULL,
                valence VARCHAR NOT NULL,
                temporal_window VARCHAR NOT NULL,
                sample_count_baseline INTEGER NOT NULL,
                sample_count_current INTEGER NOT NULL,
                detected_at VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'active',
                humanized_summary TEXT NOT NULL,
                is_statistically_significant BOOLEAN DEFAULT 0
            )
        """))

        # Create user_behavioral_baselines table if not exists (Prompt 2.8)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_behavioral_baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_id VARCHAR UNIQUE NOT NULL,
                user_id INTEGER NOT NULL REFERENCES users(id),
                baseline_version VARCHAR NOT NULL DEFAULT 'v1.0',
                generated_at VARCHAR NOT NULL,
                observation_count INTEGER NOT NULL,
                historical_window VARCHAR NOT NULL DEFAULT 'all_time',
                data_provenance VARCHAR NOT NULL DEFAULT 'genuine',
                metrics TEXT NOT NULL,
                is_active BOOLEAN DEFAULT 1
            )
        """))

        # Create model_monitoring_logs table if not exists (Prompt 2.8)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_monitoring_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                model_version VARCHAR NOT NULL,
                evaluated_at VARCHAR NOT NULL,
                rolling_window_size INTEGER NOT NULL,
                rolling_mae FLOAT NOT NULL,
                rolling_bias FLOAT NOT NULL,
                baseline_mae FLOAT NOT NULL,
                baseline_bias FLOAT NOT NULL,
                performance_degraded BOOLEAN DEFAULT 0,
                degradation_pct FLOAT DEFAULT 0.0,
                error_distribution_shift BOOLEAN DEFAULT 0,
                details TEXT
            )
        """))

        # Create model_reliability_records table if not exists (Prompt 2.9)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS model_reliability_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                model_version VARCHAR NOT NULL,
                evaluated_at VARCHAR NOT NULL,
                conformal_quantile_80 FLOAT,
                conformal_quantile_90 FLOAT,
                empirical_coverage_80 FLOAT,
                validation_mae FLOAT,
                validation_rmse FLOAT,
                brier_score FLOAT,
                expected_calibration_error FLOAT,
                calibration_status VARCHAR NOT NULL,
                data_sufficiency VARCHAR NOT NULL,
                reliability_status VARCHAR NOT NULL,
                sample_count_evaluated INTEGER NOT NULL,
                is_abstained BOOLEAN DEFAULT 0,
                details TEXT
            )
        """))

        # Create calendar_connections table if not present (Prompt 3.2)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS calendar_connections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                provider VARCHAR NOT NULL DEFAULT 'google',
                provider_account_reference VARCHAR,
                encrypted_tokens TEXT NOT NULL,
                connection_status VARCHAR NOT NULL DEFAULT 'connected',
                connected_at VARCHAR NOT NULL,
                last_synced_at VARCHAR,
                sync_token VARCHAR,
                granted_scopes TEXT
            )
        """))

        # Create calendar_events table if not present (Prompt 3.2)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS calendar_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id),
                connection_id INTEGER NOT NULL REFERENCES calendar_connections(id),
                provider_event_id VARCHAR NOT NULL,
                title VARCHAR NOT NULL,
                start_time VARCHAR NOT NULL,
                end_time VARCHAR NOT NULL,
                location VARCHAR,
                calendar_timezone VARCHAR,
                is_all_day BOOLEAN DEFAULT 0,
                status VARCHAR NOT NULL DEFAULT 'confirmed',
                recurrence_rule VARCHAR,
                recurring_event_id VARCHAR,
                travel_requirement VARCHAR NOT NULL DEFAULT 'unknown',
                requires_departure_planning BOOLEAN DEFAULT 0,
                missing_location_warning BOOLEAN DEFAULT 0,
                online_meeting_url VARCHAR,
                associated_routine_id INTEGER REFERENCES events(id),
                prep_buffer_override_min FLOAT,
                travel_mode_override VARCHAR,
                synced_at VARCHAR NOT NULL,
                updated_at VARCHAR NOT NULL
            )
        """))

        # Unique index preventing duplicate events per connection and provider event ID (Prompt 3.3)
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_calendar_events_conn_provider_id
            ON calendar_events (connection_id, provider_event_id)
        """))

        # Provider index per user
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_calendar_connections_user_provider
            ON calendar_connections (user_id, provider)
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
