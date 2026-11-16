import os
import sys
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi.testclient import TestClient

# Ensure backend directory is in path
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import database
from database import Base, User, Event, DepartureLog, get_db, init_db, migrate_db
from security import hash_password, create_access_token
import main
from main import app

# In-memory SQLite engine for tests
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="session")
def test_engine():
    engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    migrate_db(engine)
    return engine

@pytest.fixture
def db_session(test_engine):
    connection = test_engine.connect()
    transaction = connection.begin()
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=connection)
    session = TestingSessionLocal()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()

@pytest.fixture
def test_user(db_session):
    pwd_hash, pwd_salt = hash_password("testpassword123")
    user = User(
        username="test_provenance_user",
        hashed_password=pwd_hash,
        salt=pwd_salt,
        profile_type="student"
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)

    # Seed an event
    event = Event(
        user_id=user.id,
        name="Test Morning Lab",
        day_of_week="Monday",
        start_time="09:00:00",
        estimated_prep_duration=25,
        estimated_travel_duration=20,
        importance="high"
    )
    db_session.add(event)
    db_session.commit()
    db_session.refresh(event)

    return user

@pytest.fixture
def auth_headers(test_user):
    token = create_access_token({"sub": test_user.username})
    return {"Authorization": f"Bearer {token}"}
