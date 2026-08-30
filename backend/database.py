from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

# Database file URL
DATABASE_URL = "sqlite:///./microdelay.db"

# Create database engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for declarative models
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    salt = Column(String, nullable=False)
    profile_type = Column(String, nullable=False) # student, teacher, professor, it_professional

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

    user = relationship("User", back_populates="departure_logs")
    event = relationship("Event", back_populates="logs")

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
