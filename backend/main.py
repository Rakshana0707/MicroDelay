from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import datetime
import pandas as pd
from typing import List, Optional
from pydantic import BaseModel

from database import get_db, init_db, User, Event, DepartureLog
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

    class Config:
        from_attributes = True

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

@app.get("/api/logs", response_model=List[LogSchema])
def get_logs(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(DepartureLog).filter(DepartureLog.user_id == current_user.id).order_by(DepartureLog.date.desc(), DepartureLog.id.desc()).all()

@app.get("/api/metrics")
def get_metrics(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return models.train_and_evaluate(db, current_user.id)

@app.get("/api/predict")
def predict_delay(
    event_name: str,
    day_of_week: str,
    importance: str,
    bag_prepared: bool,
    estimated_prep: float,
    estimated_travel: float,
    current_user: User = Depends(get_current_user)
):
    return models.predict_micro_delay(
        user_id=current_user.id,
        event_name=event_name,
        day_of_week=day_of_week,
        importance=importance,
        bag_prepared=bag_prepared,
        estimated_prep_duration_min=estimated_prep,
        estimated_travel_duration_min=estimated_travel
    )

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

    # Checked delay tags
    checked_tags = []
    if log_in.tag_one_last_task:
        checked_tags.append("one_last_task")
    if log_in.tag_finding_items:
        checked_tags.append("finding_items")
    if log_in.tag_phone_usage:
        checked_tags.append("phone_usage")
    if log_in.tag_waiting:
        checked_tags.append("waiting")

    allocated_one_last = 0.0
    allocated_finding = 0.0
    allocated_phone = 0.0
    allocated_waiting = 0.0
    allocated_prep_bias = 0.0

    if pos_delay > 0:
        if not checked_tags:
            allocated_prep_bias = pos_delay
        else:
            # Custom default allocation weight ratios
            weights = {
                "one_last_task": 6.0,
                "finding_items": 5.0,
                "phone_usage": 3.0,
                "waiting": 4.0
            }
            active_weights = {tag: weights[tag] for tag in checked_tags}
            total_weight = sum(active_weights.values())
            
            for tag in checked_tags:
                allocated = (active_weights[tag] / total_weight) * pos_delay
                if tag == "one_last_task":
                    allocated_one_last = round(allocated, 2)
                elif tag == "finding_items":
                    allocated_finding = round(allocated, 2)
                elif tag == "phone_usage":
                    allocated_phone = round(allocated, 2)
                elif tag == "waiting":
                    allocated_waiting = round(allocated, 2)
            
            total_allocated = allocated_one_last + allocated_finding + allocated_phone + allocated_waiting
            remainder = pos_delay - total_allocated
            allocated_prep_bias = round(max(0.0, remainder), 2)
            
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
        delay_prep_bias_min=allocated_prep_bias
    )
    
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    
    # Retrain this user's custom ML model
    models.train_and_evaluate(db, current_user.id)
    
    return {
        "success": True,
        "log_id": db_log.id,
        "calculated_delay_min": total_delay_min,
        "allocations": {
            "one_last_task": allocated_one_last,
            "finding_items": allocated_finding,
            "phone_usage": allocated_phone,
            "waiting": allocated_waiting,
            "prep_bias": allocated_prep_bias
        }
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

    return {
        "has_data": True,
        "total_logs": len(df),
        "lateness_rate_pct": round(lateness_rate * 100, 1),
        "total_minutes_lost": round(float(total_minutes_lost), 1),
        "personal_prep_bias_min": round(float(personal_prep_bias), 1),
        "avg_estimated_prep_min": round(float(avg_estimated_prep), 1),
        "avg_actual_prep_min": round(float(avg_actual_prep), 1),
        "delay_sums": delay_sums,
        "delay_percentages": delay_percentages,
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
