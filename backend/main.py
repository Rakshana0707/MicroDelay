from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import os
import json
import datetime
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

from database import get_db, init_db, User, Event, DepartureLog, ModelVersionRecord
import generator
import models
import security

app = FastAPI(title="MicroDelay Multi-User API")

import os
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication token extraction scheme
security_scheme = HTTPBearer()

@app.on_event("startup")
def startup_event():
    # Ensure SQLite tables are created
    init_db()

# Dependency to secure API endpoints
def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security_scheme), db: Session = Depends(get_db)) -> User:
    token = credentials.credentials
    payload = security.decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    username: str = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session payload missing credentials",
        )
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user

# Pydantic Schemas
class UserRegisterSchema(BaseModel):
    username: str
    password: str
    profile_type: str # student, teacher, professor, it_professional

class UserLoginSchema(BaseModel):
    username: str
    password: str

class AuthTokenSchema(BaseModel):
    token: str
    username: str
    profile_type: str

class EventSchema(BaseModel):
    id: int
    name: str
    day_of_week: str
    start_time: str
    estimated_prep_duration: int
    estimated_travel_duration: int
    importance: str

    class Config:
        from_attributes = True

class EventCreateSchema(BaseModel):
    name: str
    day_of_week: str
    start_time: str
    estimated_prep_duration: int
    estimated_travel_duration: int
    importance: str


class LogCreateSchema(BaseModel):
    event_id: Optional[int] = None
    event_name: str
    date: str
    day_of_week: str
    planned_prep_start: str
    planned_departure: str
    estimated_prep_duration_min: float
    estimated_travel_duration_min: float
    actual_prep_start: str
    actual_departure: str
    actual_arrival: str
    actual_prep_duration_min: float
    actual_travel_duration_min: float
    bag_prepared: bool
    tag_one_last_task: bool
    tag_finding_items: bool
    tag_phone_usage: bool
    tag_waiting: bool
    # Optional directly observed/timed minutes
    observed_one_last_task_min: Optional[float] = None
    observed_finding_items_min: Optional[float] = None
    observed_phone_usage_min: Optional[float] = None
    observed_waiting_min: Optional[float] = None

class LogSchema(BaseModel):
    id: int
    event_name: str
    date: str
    day_of_week: str
    actual_departure: str
    actual_arrival: str
    estimated_prep_duration_min: float
    actual_prep_duration_min: float
    total_delay_min: float
    bag_prepared: bool
    tag_one_last_task: bool
    tag_finding_items: bool
    tag_phone_usage: bool
    tag_waiting: bool
    # Optional directly observed/timed minutes
    observed_one_last_task_min: Optional[float] = None
    observed_finding_items_min: Optional[float] = None
    observed_phone_usage_min: Optional[float] = None
    observed_waiting_min: Optional[float] = None
    provenance: Optional[str] = "unknown"
    data_source: Optional[str] = "unknown"
    is_synthetic: Optional[bool] = False
    recorded_at: Optional[str] = None
    observation_status: Optional[str] = "valid"

    class Config:
        from_attributes = True

class IntelligencePredictRequestSchema(BaseModel):
    event_name: str
    day_of_week: str
    importance: str = "medium"
    bag_prepared: bool = False
    estimated_prep_duration: float = 30.0
    estimated_travel_duration: float = 20.0
    event_id: Optional[int] = None

class UserConsentUpdateSchema(BaseModel):
    consent: bool

class GlobalTrainRequestSchema(BaseModel):
    min_cohort_users: Optional[int] = 2
    min_cohort_samples: Optional[int] = 10

# Authentication Endpoints
@app.post("/api/auth/register", response_model=AuthTokenSchema)
def register_user(user_in: UserRegisterSchema, db: Session = Depends(get_db)):
    # 1. Validate profile type
    valid_profiles = ["student", "teacher", "professor", "it_professional"]
    if user_in.profile_type not in valid_profiles:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid profile type. Must be one of {valid_profiles}"
        )
        
    # 2. Check if username exists
    existing = db.query(User).filter(User.username == user_in.username).first()
    if existing:
        raise HTTPException(
            status_code=400,
            detail="Username already registered. Please choose another one."
        )
        
    # 3. Create User record
    hashed_pwd, salt = security.hash_password(user_in.password)
    user = User(
        username=user_in.username,
        hashed_password=hashed_pwd,
        salt=salt,
        profile_type=user_in.profile_type
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    # 4. Dynamically seed events & logs for this user's profile
    try:
        # Seeds 60 days of historical logs linked to user.id
        generator.generate_user_logs(db, user_id=user.id, num_days=60)
        # Pre-train their custom Random Forest model
        models.train_and_evaluate(db, user_id=user.id)
    except Exception as e:
        print(f"Error seeding user records on register: {e}")
        
    # 5. Issue login session token
    token = security.create_access_token({"sub": user.username})
    return {
        "token": token,
        "username": user.username,
        "profile_type": user.profile_type
    }

@app.post("/api/auth/login", response_model=AuthTokenSchema)
def login_user(user_in: UserLoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == user_in.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
        
    if not security.verify_password(user_in.password, user.hashed_password, user.salt):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password"
        )
        
    token = security.create_access_token({"sub": user.username})
    
    # Check if they have an active trained model, if not train it on the fly
    if not models.get_model_status(user.id)["trained"]:
        models.train_and_evaluate(db, user_id=user.id)
        
    return {
        "token": token,
        "username": user.username,
        "profile_type": user.profile_type
    }

# Protected Business logic endpoints
@app.get("/api/events", response_model=List[EventSchema])
def get_events(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Event).filter(Event.user_id == current_user.id).all()

@app.post("/api/events", response_model=EventSchema)
def create_event(event_in: EventCreateSchema, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    new_event = Event(
        user_id=current_user.id,
        name=event_in.name,
        day_of_week=event_in.day_of_week,
        start_time=event_in.start_time,
        estimated_prep_duration=event_in.estimated_prep_duration,
        estimated_travel_duration=event_in.estimated_travel_duration,
        importance=event_in.importance
    )
    db.add(new_event)
    db.commit()
    db.refresh(new_event)
    return new_event

@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    event = db.query(Event).filter(Event.id == event_id, Event.user_id == current_user.id).first()
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    db.delete(event)
    db.commit()
    return {"message": "Event deleted successfully"}

@app.get("/api/logs", response_model=List[LogSchema])
def get_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(DepartureLog).filter(DepartureLog.user_id == current_user.id).order_by(DepartureLog.date.desc(), DepartureLog.id.desc()).all()

@app.get("/api/metrics")
def get_metrics(
    training_mode: Optional[str] = "auto",
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    return models.train_and_evaluate(db, current_user.id, training_mode=training_mode)

@app.get("/api/benchmarks")
def get_benchmarks(
    training_mode: Optional[str] = "auto",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run reproducible multi-algorithm benchmark for authenticated user."""
    return models.benchmark_user_models(db, current_user.id, training_mode=training_mode)

@app.get("/api/reports/latest")
def get_latest_report(
    provenance: Optional[str] = "synthetic",
    current_user: User = Depends(get_current_user)
):
    """Fetch the latest evaluation report (JSON format) for the authenticated user."""
    from ml.reporting import REAL_REPORTS_DIR, SYNTHETIC_REPORTS_DIR
    target_dir = REAL_REPORTS_DIR if provenance == "real" else SYNTHETIC_REPORTS_DIR
    latest_file = os.path.join(target_dir, f"latest_user_{current_user.id}.json")
    if not os.path.exists(latest_file):
        raise HTTPException(status_code=404, detail="No evaluation report found for user.")
    with open(latest_file, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/data-summary")
def get_data_summary(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Return data provenance statistics and model status for the authenticated user."""
    counts = models.get_user_data_counts(db, current_user.id)
    status_info = models.get_model_status(current_user.id, db=db)
    return {
        **counts,
        "active_model": status_info
    }

@app.get("/api/models/versions")
def get_user_model_versions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve all model versions and statuses for the authenticated user."""
    records = db.query(ModelVersionRecord).filter(
        ModelVersionRecord.user_id == current_user.id
    ).order_by(ModelVersionRecord.id.desc()).all()
    
    return [
        {
            "id": r.id,
            "model_id": r.model_id,
            "model_version": r.model_version,
            "model_type": r.model_type,
            "training_data_source": r.training_data_source,
            "training_samples": r.training_samples,
            "training_started_at": r.training_started_at,
            "training_completed_at": r.training_completed_at,
            "validation_mae": r.validation_mae,
            "test_mae": r.test_mae,
            "model_status": r.model_status,
            "is_active": r.is_active,
            "status_reason": r.status_reason,
            "created_at": r.training_started_at
        }
        for r in records
    ]

@app.post("/api/models/rollback")
def rollback_model_version(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Rollback to the most recent archived model version for the authenticated user."""
    res = models.rollback_user_model(db, current_user.id)
    if not res["success"]:
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@app.get("/api/predict")
def predict_delay(
    event_name: str,
    day_of_week: str,
    importance: str,
    bag_prepared: bool,
    estimated_prep: float,
    estimated_travel: float,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return models.predict_micro_delay(
        user_id=current_user.id,
        event_name=event_name,
        day_of_week=day_of_week,
        importance=importance,
        bag_prepared=bag_prepared,
        estimated_prep_duration_min=estimated_prep,
        estimated_travel_duration_min=estimated_travel,
        db=db
    )

@app.get("/api/intelligence/predict")
def predict_adaptive_intelligence_get(
    event_name: str,
    day_of_week: str,
    importance: str = "medium",
    bag_prepared: bool = False,
    estimated_prep: float = 30.0,
    estimated_travel: float = 20.0,
    event_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unified Adaptive Intelligence Prediction endpoint (GET)."""
    try:
        from ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator
    except ModuleNotFoundError:
        from backend.ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator

    ctx = PredictionContext(
        user_id=current_user.id,
        event_name=event_name,
        day_of_week=day_of_week,
        event_importance=importance,
        estimated_prep_duration=estimated_prep,
        estimated_travel_duration=estimated_travel,
        bag_prepared=bag_prepared,
        event_id=event_id
    )
    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=db)
    return output.to_dict()


@app.get("/api/intelligence/risk")
def predict_lateness_risk_get(
    event_name: str,
    day_of_week: str,
    importance: str = "medium",
    bag_prepared: bool = False,
    estimated_prep: float = 30.0,
    estimated_travel: float = 20.0,
    event_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dedicated Lateness Risk Classification prediction endpoint (GET)."""
    try:
        from ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator
    except ModuleNotFoundError:
        from backend.ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator

    ctx = PredictionContext(
        user_id=current_user.id,
        event_name=event_name,
        day_of_week=day_of_week,
        event_importance=importance,
        estimated_prep_duration=estimated_prep,
        estimated_travel_duration=estimated_travel,
        bag_prepared=bag_prepared,
        event_id=event_id
    )
    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=db)
    
    return {
        "late_probability": output.late_probability,
        "risk_category": output.risk_category,
        "predicted_delay_minutes": output.predicted_delay_minutes,
        "baseline_predicted_delay_minutes": output.baseline_predicted_delay_minutes,
        "model_version": output.model_version,
        "model_type": output.model_type,
        "model_availability": output.model_availability,
        "training_data_provenance": output.data_quality_status.get("provenance", "unknown"),
        "prediction_timestamp": output.execution_timestamp,
        "confidence_status": output.confidence_status,
        "contributing_factors": output.contributing_factors,
        "recommended_interventions": output.recommended_interventions,
        "personalization_info": output.personalization_info,
        "explanation": output.explanation
    }

@app.post("/api/intelligence/risk")
def predict_lateness_risk_post(
    req: IntelligencePredictRequestSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Dedicated Lateness Risk Classification prediction endpoint (POST)."""
    try:
        from ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator
    except ModuleNotFoundError:
        from backend.ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator

    ctx = PredictionContext(
        user_id=current_user.id,
        event_name=req.event_name,
        day_of_week=req.day_of_week,
        event_importance=req.importance,
        estimated_prep_duration=req.estimated_prep_duration,
        estimated_travel_duration=req.estimated_travel_duration,
        bag_prepared=req.bag_prepared,
        event_id=req.event_id
    )
    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=db)
    
    return {
        "late_probability": output.late_probability,
        "risk_category": output.risk_category,
        "predicted_delay_minutes": output.predicted_delay_minutes,
        "baseline_predicted_delay_minutes": output.baseline_predicted_delay_minutes,
        "model_version": output.model_version,
        "model_type": output.model_type,
        "model_availability": output.model_availability,
        "training_data_provenance": output.data_quality_status.get("provenance", "unknown"),
        "prediction_timestamp": output.execution_timestamp,
        "confidence_status": output.confidence_status,
        "contributing_factors": output.contributing_factors,
        "recommended_interventions": output.recommended_interventions,
        "personalization_info": output.personalization_info,
        "explanation": output.explanation
    }

@app.get("/api/models/classifier/metrics")
def get_classifier_metrics(current_user: User = Depends(get_current_user)):
    """Fetch current user's trained classifier benchmarking and calibration metrics."""
    return models.get_user_classifier_metrics(current_user.id)

# Global-to-Personalized Learning Endpoints (Prompt 2.3)
@app.get("/api/users/consent")
def get_user_consent_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch current user's consent preferences and eligibility for shared global model training."""
    return models.get_user_consent(db, current_user.id)

@app.post("/api/users/consent")
def update_user_consent_endpoint(
    req: UserConsentUpdateSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update current user's consent for shared global model training (opt-in/opt-out)."""
    return models.update_user_consent(db, current_user.id, req.consent)

@app.get("/api/models/global/status")
def get_global_model_status_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch the current operational status, version, and validation metrics of the shared global model."""
    return models.get_global_model_status(db)

@app.post("/api/models/global/train")
def train_global_model_endpoint(
    req: Optional[GlobalTrainRequestSchema] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Train or retrain the global shared model across all consented genuine user observations."""
    min_users = req.min_cohort_users if req and req.min_cohort_users is not None else 2
    min_samples = req.min_cohort_samples if req and req.min_cohort_samples is not None else 10
    return models.train_global_model(db, min_cohort_users=min_users, min_cohort_samples=min_samples)

@app.get("/api/benchmarks/leave-users-out")
def get_leave_users_out_benchmark_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run and return the leave-users-out evaluation experiment across all consented genuine users."""
    return models.get_leave_users_out_benchmark(db)

# Behavioral Habit Discovery & Clustering Endpoints (Prompt 2.4)
@app.get("/api/intelligence/clusters/profile")
def get_user_cluster_profile_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch current user's behavioral cluster profile, habit characteristics, and historical comparison."""
    return models.get_user_cluster_profile(db, current_user.id)

@app.post("/api/intelligence/clusters/train")
def train_cluster_models_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Train unsupervised clustering tournament across genuine history and update all user profiles."""
    return models.train_cluster_models(db)

@app.get("/api/intelligence/clusters/benchmark")
def get_clustering_benchmark_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch comparative benchmarking metrics across K-Means, Agglomerative, and DBSCAN algorithms."""
    return models.get_clustering_benchmark(db)

# Personalized Behavioral Profiling Engine Endpoints (Prompt 2.5)
@app.get("/api/intelligence/profile")
def get_user_behavioral_profile_endpoint(
    provenance: str = "real",
    force_refresh: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch current user's personalized multi-window behavioral profile across 6 dimensions."""
    return models.get_user_behavioral_profile(
        db=db,
        user_id=current_user.id,
        provenance=provenance,
        force_refresh=force_refresh
    )

@app.get("/api/intelligence/profile/history")
def get_user_profile_history_endpoint(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve history of persisted behavioral profile snapshots for the authenticated user."""
    return models.get_user_profile_history(
        db=db,
        user_id=current_user.id,
        limit=limit
    )

@app.get("/api/intelligence/profile/compare")
def compare_user_profiles_endpoint(
    profile_id_a: Optional[str] = None,
    profile_id_b: Optional[str] = None,
    window_a: Optional[str] = None,
    window_b: Optional[str] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare two profile snapshots or rolling windows for the authenticated user."""
    res = models.compare_user_profiles(
        db=db,
        user_id=current_user.id,
        profile_id_a=profile_id_a,
        profile_id_b=profile_id_b,
        window_a=window_a,
        window_b=window_b
    )
    if isinstance(res, dict) and "error" in res:
        raise HTTPException(status_code=400, detail=res["error"])
    return res

@app.post("/api/intelligence/profile/refresh")
def refresh_user_profile_endpoint(
    provenance: str = "real",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Force recalculation and snapshot generation of user's behavioral profile."""
    return models.get_user_behavioral_profile(
        db=db,
        user_id=current_user.id,
        provenance=provenance,
        force_refresh=True
    )


@app.post("/api/intelligence/predict")
def predict_adaptive_intelligence_post(
    req: IntelligencePredictRequestSchema,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Unified Adaptive Intelligence Prediction endpoint (POST)."""
    try:
        from ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator
    except ModuleNotFoundError:
        from backend.ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator

    ctx = PredictionContext(
        user_id=current_user.id,
        event_name=req.event_name,
        day_of_week=req.day_of_week,
        event_importance=req.importance,
        estimated_prep_duration=req.estimated_prep_duration,
        estimated_travel_duration=req.estimated_travel_duration,
        bag_prepared=req.bag_prepared,
        event_id=req.event_id
    )
    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=db)
    return output.to_dict()

# Prediction Explainability Endpoints (Prompt 2.6)
@app.get("/api/intelligence/explain")
def get_prediction_explanation_endpoint(
    prediction_id: Optional[str] = None,
    event_name: Optional[str] = None,
    day_of_week: Optional[str] = None,
    importance: str = "medium",
    bag_prepared: bool = False,
    estimated_prep: float = 30.0,
    estimated_travel: float = 20.0,
    event_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve an existing prediction explanation by ID or generate on-demand explanation for context."""
    if prediction_id:
        exp = models.get_user_prediction_explanation(db, current_user.id, prediction_id)
        if not exp:
            raise HTTPException(status_code=404, detail="Prediction explanation not found or access denied.")
        return exp

    if not event_name or not day_of_week:
        raise HTTPException(status_code=400, detail="Provide either prediction_id or event_name and day_of_week.")

    try:
        from ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator
    except ModuleNotFoundError:
        from backend.ml.adaptive import PredictionContext, AdaptiveIntelligenceOrchestrator

    ctx = PredictionContext(
        user_id=current_user.id,
        event_name=event_name,
        day_of_week=day_of_week,
        event_importance=importance,
        estimated_prep_duration=estimated_prep,
        estimated_travel_duration=estimated_travel,
        bag_prepared=bag_prepared,
        event_id=event_id
    )
    orchestrator = AdaptiveIntelligenceOrchestrator()
    output = orchestrator.predict(ctx, db=db)
    return output.explanation or {
        "prediction_id": f"pred_adhoc_{current_user.id}",
        "top_positive_contributors": output.contributing_factors,
        "top_mitigating_factors": [],
        "historical_comparisons": [],
        "limitations": ["Generated via fallback adaptive pipeline."],
        "causation_disclaimer": "Contributing factors reflect statistical correlation, not direct causation.",
        "explanation_method": "behavioral_rule_reasoning"
    }

@app.get("/api/intelligence/explanations/history")
def get_explanation_history_endpoint(
    limit: int = 10,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Fetch recent prediction explanation audit records for the authenticated user."""
    return models.get_user_explanation_history(db, current_user.id, limit=limit)

@app.get("/api/intelligence/features/importance")
def get_feature_importance_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieve global feature importance ranking for the user's active model."""
    return models.get_user_feature_importance(db, current_user.id)


@app.post("/api/checkout")
def create_checkout_log(
    log_in: LogCreateSchema, 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    try:
        fmt = "%H:%M:%S"
        t_planned_dep = datetime.datetime.strptime(log_in.planned_departure, fmt)
        t_actual_dep = datetime.datetime.strptime(log_in.actual_departure, fmt)
        
        if t_actual_dep < t_planned_dep and (t_planned_dep - t_actual_dep).seconds > 43200:
            delay_sec = (t_actual_dep + datetime.timedelta(days=1) - t_planned_dep).seconds
        else:
            delay_sec = (t_actual_dep - t_planned_dep).total_seconds()
            
        total_delay_min = round(delay_sec / 60.0, 2)
        pos_delay = max(0.0, total_delay_min)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid time format. Error: {e}")

    # Attribute delay using explainable DelayAttributionEngine
    observed_durations = {}
    if getattr(log_in, "observed_one_last_task_min", None) is not None:
        observed_durations["one_last_task"] = float(log_in.observed_one_last_task_min)
    if getattr(log_in, "observed_finding_items_min", None) is not None:
        observed_durations["finding_items"] = float(log_in.observed_finding_items_min)
    if getattr(log_in, "observed_phone_usage_min", None) is not None:
        observed_durations["phone_usage"] = float(log_in.observed_phone_usage_min)
    if getattr(log_in, "observed_waiting_min", None) is not None:
        observed_durations["waiting"] = float(log_in.observed_waiting_min)

    tags = {
        "one_last_task": bool(log_in.tag_one_last_task),
        "finding_items": bool(log_in.tag_finding_items),
        "phone_usage": bool(log_in.tag_phone_usage),
        "waiting": bool(log_in.tag_waiting)
    }

    try:
        from ml.attribution import DelayAttributionEngine
    except ModuleNotFoundError:
        from backend.ml.attribution import DelayAttributionEngine

    attribution_result = DelayAttributionEngine.attribute_departure(
        total_delay_min=total_delay_min,
        actual_prep_duration_min=log_in.actual_prep_duration_min,
        estimated_prep_duration_min=log_in.estimated_prep_duration_min,
        tags=tags,
        observed_durations=observed_durations
    )
    legacy_allocated = attribution_result.get_legacy_allocated_dict()
    allocated_one_last = legacy_allocated["delay_one_last_task_min"]
    allocated_finding = legacy_allocated["delay_finding_items_min"]
    allocated_phone = legacy_allocated["delay_phone_usage_min"]
    allocated_waiting = legacy_allocated["delay_waiting_min"]
    allocated_prep_bias = legacy_allocated["delay_prep_bias_min"]

    db_log = DepartureLog(
        user_id=current_user.id,
        event_id=log_in.event_id,
        event_name=log_in.event_name,
        date=log_in.date,
        day_of_week=log_in.day_of_week,
        planned_prep_start=log_in.planned_prep_start,
        planned_departure=log_in.planned_departure,
        estimated_prep_duration_min=log_in.estimated_prep_duration_min,
        estimated_travel_duration_min=log_in.estimated_travel_duration_min,
        actual_prep_start=log_in.actual_prep_start,
        actual_departure=log_in.actual_departure,
        actual_arrival=log_in.actual_arrival,
        actual_prep_duration_min=log_in.actual_prep_duration_min,
        actual_travel_duration_min=log_in.actual_travel_duration_min,
        total_delay_min=total_delay_min,
        bag_prepared=log_in.bag_prepared,
        tag_one_last_task=log_in.tag_one_last_task,
        tag_finding_items=log_in.tag_finding_items,
        tag_phone_usage=log_in.tag_phone_usage,
        tag_waiting=log_in.tag_waiting,
        delay_one_last_task_min=allocated_one_last,
        delay_finding_items_min=allocated_finding,
        delay_phone_usage_min=allocated_phone,
        delay_waiting_min=allocated_waiting,
        delay_prep_bias_min=allocated_prep_bias,
        attribution_metadata=attribution_result.to_json(),
        provenance="real",
        data_source="simulator_checkout",
        is_synthetic=False,
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        observation_status="valid"
    )
    
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    
    # Retrain this user's custom ML model using auto provenance resolution
    retrain_res = models.train_and_evaluate(db, current_user.id, training_mode="auto")

    # Conditionally recalculate behavioral profile snapshot if >= 3 logs have accumulated
    try:
        models.get_user_behavioral_profile(db, current_user.id, provenance="real")
    except Exception as prof_err:
        print(f"Profile auto-update notice: {prof_err}")

    
    return {
        "success": True,
        "log_id": db_log.id,
        "calculated_delay_min": total_delay_min,
        "provenance": "real",
        "is_synthetic": False,
        "allocations": {
            "one_last_task": allocated_one_last,
            "finding_items": allocated_finding,
            "phone_usage": allocated_phone,
            "waiting": allocated_waiting,
            "prep_bias": allocated_prep_bias
        },
        "model_retrained": retrain_res.get("success", False),
        "training_mode": retrain_res.get("training_mode", "auto"),
        "is_personalized_real": retrain_res.get("is_personalized_real", False)
    }

@app.get("/api/insights")
def get_insights(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Calculate advanced user-profile behavioral insights."""
    logs = db.query(DepartureLog).filter(DepartureLog.user_id == current_user.id).all()
    if not logs:
        return {"has_data": False}
        
    df = pd.DataFrame([{
        "event_name": log.event_name,
        "day_of_week": log.day_of_week,
        "estimated_prep": log.estimated_prep_duration_min,
        "actual_prep": log.actual_prep_duration_min,
        "total_delay": log.total_delay_min,
        "bag_prepared": log.bag_prepared,
        "one_last_task": log.tag_one_last_task,
        "finding_items": log.tag_finding_items,
        "phone_usage": log.tag_phone_usage,
        "waiting": log.tag_waiting,
        "delay_one_last_task_min": log.delay_one_last_task_min,
        "delay_finding_items_min": log.delay_finding_items_min,
        "delay_phone_usage_min": log.delay_phone_usage_min,
        "delay_waiting_min": log.delay_waiting_min,
        "delay_prep_bias_min": log.delay_prep_bias_min
    } for log in logs])

    profile = current_user.profile_type

    # Personal Time Bias
    avg_estimated_prep = df["estimated_prep"].mean()
    avg_actual_prep = df["actual_prep"].mean()
    personal_prep_bias = avg_actual_prep - avg_estimated_prep
    
    late_logs = df[df["total_delay"] > 0]
    lateness_rate = len(late_logs) / len(df) if len(df) > 0 else 0
    total_minutes_lost = df[df["total_delay"] > 0]["total_delay"].sum()
    
    # Delay sums & percentages
    delay_sums = {
        "one_last_task": float(df["delay_one_last_task_min"].sum()),
        "finding_items": float(df["delay_finding_items_min"].sum()),
        "phone_usage": float(df["delay_phone_usage_min"].sum()),
        "waiting": float(df["delay_waiting_min"].sum()),
        "prep_bias": float(df["delay_prep_bias_min"].sum())
    }
    
    total_allocated_delay = sum(delay_sums.values())
    delay_percentages = {}
    for key, val in delay_sums.items():
        delay_percentages[key] = round((val / total_allocated_delay) * 100, 1) if total_allocated_delay > 0 else 0.0

    # Structured Observed vs. Estimated Attribution
    observed_contributions = {}
    estimated_contributions = {}
    total_observed_all = 0.0
    total_estimated_all = 0.0
    total_unexplained_all = 0.0

    try:
        from ml.attribution import DelayAttributionEngine
    except ModuleNotFoundError:
        from backend.ml.attribution import DelayAttributionEngine

    for log_obj in logs:
        attr_res = DelayAttributionEngine.parse_legacy_record(log_obj)
        total_observed_all += attr_res.total_observed_min
        total_estimated_all += attr_res.total_estimated_min
        total_unexplained_all += attr_res.unexplained_residual_min
        for cat in attr_res.categories:
            if cat.attribution_type == "observed":
                observed_contributions[cat.category] = observed_contributions.get(cat.category, 0.0) + cat.departure_impact_min
            else:
                estimated_contributions[cat.category] = estimated_contributions.get(cat.category, 0.0) + cat.departure_impact_min

    pos_lateness = float(total_minutes_lost)
    completeness = round(min(100.0, ((total_observed_all + total_estimated_all) / pos_lateness * 100.0)), 1) if pos_lateness > 0 else 100.0

    attribution_summary = {
        "total_actual_lateness_min": round(pos_lateness, 1),
        "observed_delay_contributions": {k: round(v, 1) for k, v in observed_contributions.items()},
        "estimated_contributions": {k: round(v, 1) for k, v in estimated_contributions.items()},
        "unexplained_delay_min": round(total_unexplained_all, 1),
        "total_observed_min": round(total_observed_all, 1),
        "total_estimated_min": round(total_estimated_all, 1),
        "attribution_completeness_pct": completeness,
        "scientific_disclaimer": "Observed contributions represent directly timed or measured intervals. Estimated contributions are model/heuristic allocations and do not represent isolated causal experiments."
    }

    # Segregate calculations by profile for targeted recommendations
    recommendations = []
    
    if profile == "student":
        monday_logs = df[df["day_of_week"] == "Monday"]
        monday_late_rate = len(monday_logs[monday_logs["total_delay"] > 0]) / len(monday_logs) if len(monday_logs) > 0 else 0
        unprepared_delay = monday_logs[monday_logs["bag_prepared"] == False]["total_delay"].mean() if len(monday_logs[monday_logs["bag_prepared"] == False]) > 0 else 0
        prepared_delay = monday_logs[monday_logs["bag_prepared"] == True]["total_delay"].mean() if len(monday_logs[monday_logs["bag_prepared"] == True]) > 0 else 0
        
        if monday_late_rate > 0.50:
            recommendations.append({
                "title": "Monday Lateness Pattern Detected",
                "description": f"Your Monday lateness rate is {round(monday_late_rate*100)}% during early morning ML labs.",
                "action": "Pack your student notebook and lab record materials the previous night.",
                "impact": f"Predicted delay reduction: {round(max(3.0, unprepared_delay - prepared_delay), 1)} minutes on Mondays."
            })
            
        olt_delay = df[df["one_last_task"] == True]["delay_one_last_task_min"].mean() if len(df[df["one_last_task"] == True]) > 0 else 0
        if olt_delay > 3.0:
            recommendations.append({
                "title": "Social Media / WhatsApp Distraction Bias",
                "description": "You routinely check notifications before walking out.",
                "action": "Implement a strict 'no phone checks 10 mins before departure' rule.",
                "impact": f"Saves {round(olt_delay, 1)} minutes of delays."
            })

    elif profile == "teacher":
        school_mornings = df[df["day_of_week"].isin(["Monday", "Thursday"])]
        late_rate = len(school_mornings[school_mornings["total_delay"] > 0]) / len(school_mornings) if len(school_mornings) > 0 else 0
        unprepared = school_mornings[school_mornings["bag_prepared"] == False]["total_delay"].mean() if len(school_mornings[school_mornings["bag_prepared"] == False]) > 0 else 0
        prepared = school_mornings[school_mornings["bag_prepared"] == True]["total_delay"].mean() if len(school_mornings[school_mornings["bag_prepared"] == True]) > 0 else 0

        if late_rate > 0.45:
            recommendations.append({
                "title": "School Materials Prep Bottleneck",
                "description": "Handout printing and classroom planning adds heavy morning overhead.",
                "action": "Complete worksheet photocopying and print lesson materials the day before.",
                "impact": f"Saves {round(max(4.0, unprepared - prepared), 1)} minutes of teacher preparation rush."
            })
            
        keys_delay = df[df["finding_items"] == True]["delay_finding_items_min"].mean() if len(df[df["finding_items"] == True]) > 0 else 0
        if keys_delay > 3.0:
            recommendations.append({
                "title": "Classroom Keys Overhead",
                "description": "Time lost searching for school classroom keys and staff badges.",
                "action": "Designate a dedicated hook near your home door for classroom keys only.",
                "impact": f"Reduces item search delays by {round(keys_delay, 1)} minutes."
            })

    elif profile == "professor":
        lecture_mornings = df[df["event_name"].isin(["Research Lecture", "Symposium"])]
        late_rate = len(lecture_mornings[lecture_mornings["total_delay"] > 0]) / len(lecture_mornings) if len(lecture_mornings) > 0 else 0
        unprepared = lecture_mornings[lecture_mornings["bag_prepared"] == False]["total_delay"].mean() if len(lecture_mornings[lecture_mornings["bag_prepared"] == False]) > 0 else 0
        prepared = lecture_mornings[lecture_mornings["bag_prepared"] == True]["total_delay"].mean() if len(lecture_mornings[lecture_mornings["bag_prepared"] == True]) > 0 else 0

        if late_rate > 0.40:
            recommendations.append({
                "title": "Presentation Slides & Adapter Bottleneck",
                "description": "Double checking slide printouts and finding cables creates departure delays.",
                "action": "Pack your presentation slides, laptop, and HDMI projector adapters in your briefcase the night before.",
                "impact": f"Predicted delay reduction: {round(max(4.0, unprepared - prepared), 1)} minutes."
            })
            
        email_delay = df[df["one_last_task"] == True]["delay_one_last_task_min"].mean() if len(df[df["one_last_task"] == True]) > 0 else 0
        if email_delay > 4.0:
            recommendations.append({
                "title": "Research Draft Reading Bias",
                "description": "You routinely check student thesis drafts or faculty emails right before leaving.",
                "action": "Defer checking student email inbox until you arrive at your office campus.",
                "impact": f"Eliminates {round(email_delay, 1)} minutes of pre-commute delay."
            })

    else: # it_professional
        standup_mornings = df[df["day_of_week"].isin(["Monday", "Wednesday", "Friday"])]
        late_rate = len(standup_mornings[standup_mornings["total_delay"] > 0]) / len(standup_mornings) if len(standup_mornings) > 0 else 0
        unprepared = standup_mornings[standup_mornings["bag_prepared"] == False]["total_delay"].mean() if len(standup_mornings[standup_mornings["bag_prepared"] == False]) > 0 else 0
        prepared = standup_mornings[standup_mornings["bag_prepared"] == True]["total_delay"].mean() if len(standup_mornings[standup_mornings["bag_prepared"] == True]) > 0 else 0

        if late_rate > 0.50:
            recommendations.append({
                "title": "Standup Build Cycle Delay",
                "description": "Git pull conflicts and waiting for local builds before leaving causes lateness.",
                "action": "Sync your local git repository branches and trigger initial builds the night before.",
                "impact": f"Predicted delay reduction: {round(max(4.0, unprepared - prepared), 1)} minutes."
            })
            
        slack_delay = df[df["one_last_task"] == True]["delay_one_last_task_min"].mean() if len(df[df["one_last_task"] == True]) > 0 else 0
        if slack_delay > 3.0:
            recommendations.append({
                "title": "Slack / Teams Notification Distraction",
                "description": "Replying to peer tags right before departure delays standup attendance.",
                "action": "Close your work messaging clients 15 minutes before your planned departure.",
                "impact": f"Saves {round(slack_delay, 1)} minutes."
            })

    # Default fallback recommendation if profile-specific ones are sparse
    if len(recommendations) < 2:
        recommendations.append({
            "title": "Calibrate Preparation Estimation Bias",
            "description": f"You take {round(personal_prep_bias, 1)} minutes longer to get ready than you estimate.",
            "action": "Adjust your app's default prep timer templates to reflect your true 30-day average.",
            "impact": f"Saves {round(personal_prep_bias, 1)} minutes of commute stress."
        })

    counts = models.get_user_data_counts(db, current_user.id)

    return {
        "has_data": True,
        "total_logs": len(df),
        "is_synthetic_demonstration": counts["real_samples"] == 0,
        "real_samples": counts["real_samples"],
        "synthetic_samples": counts["synthetic_samples"],
        "total_samples": counts["total_samples"],
        "data_provenance": "real" if counts["real_samples"] >= 8 else ("synthetic" if counts["real_samples"] == 0 else "mixed_insufficient_real"),
        "provenance_notice": (
            "Verified Personal Profile: Derived from your genuine departure routines."
            if counts["real_samples"] >= 8
            else (
                "Demonstration Profile: Based on 60 days of pre-seeded persona data. Track actual morning departures to calibrate your personal profile."
                if counts["real_samples"] == 0
                else f"Early Calibration Phase: {counts['real_samples']}/8 genuine routines recorded. Personal model will activate after 8 real departures."
            )
        ),
        "lateness_rate_pct": round(lateness_rate * 100, 1),
        "total_minutes_lost": round(float(total_minutes_lost), 1),
        "personal_prep_bias_min": round(float(personal_prep_bias), 1),
        "avg_estimated_prep_min": round(float(avg_estimated_prep), 1),
        "avg_actual_prep_min": round(float(avg_actual_prep), 1),
        "delay_sums": delay_sums,
        "delay_percentages": delay_percentages,
        "attribution_summary": attribution_summary,
        "recommendations": recommendations
    }

import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")
    
    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        file_path = os.path.join(frontend_dist, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
