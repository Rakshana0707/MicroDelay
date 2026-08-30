import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from database import DepartureLog, Event
import joblib
import os

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
EVENT_NAMES = [
    # General templates across all profiles
    "ML Lab", "Seminar", "Math Lecture", "Library Study", "Hardware Lab", "Theory Lecture", "Weekend Kickoff",
    "Morning Assembly", "Homeroom Class", "Staff Meeting", "Science Lab", "Grading Session",
    "Research Lecture", "Thesis Review", "Faculty Council", "Office Hours", "Symposium",
    "Daily Standup", "Code Review", "Sprint Planning", "Design Sync", "Deploy & Retro"
]
IMPORTANCE_MAPPING = {"low": 0, "medium": 1, "high": 2}

def get_model_paths(user_id: int):
    """Generate dynamic model file paths per user_id."""
    os.makedirs("backend/models", exist_ok=True)
    return {
        "model": f"backend/models/rf_model_user_{user_id}.pkl",
        "features": f"backend/models/feats_user_{user_id}.pkl",
        "baseline": f"backend/models/baseline_user_{user_id}.pkl"
    }

def get_logs_df(db: Session, user_id: int) -> pd.DataFrame:
    """Fetch departure logs from database for a specific user."""
    logs = db.query(DepartureLog).filter(DepartureLog.user_id == user_id).all()
    if not logs:
        return pd.DataFrame()
        
    data = []
    for log in logs:
        # Find importance from the event template, fallback to medium
        importance = "medium"
        if log.event_id:
            event = db.query(Event).filter(Event.id == log.event_id).first()
            if event:
                importance = event.importance
                
        data.append({
            "event_name": log.event_name,
            "day_of_week": log.day_of_week,
            "importance": importance,
            "bag_prepared": 1 if log.bag_prepared else 0,
            "estimated_prep_duration_min": log.estimated_prep_duration_min,
            "estimated_travel_duration_min": log.estimated_travel_duration_min,
            "total_delay_min": log.total_delay_min
        })
        
    return pd.DataFrame(data)

def preprocess_features(df: pd.DataFrame, fit_columns=None):
    """Encode categorical features consistently."""
    df["importance_code"] = df["importance"].map(IMPORTANCE_MAPPING).fillna(1)
    
    for day in DAYS_OF_WEEK:
        df[f"day_{day}"] = (df["day_of_week"] == day).astype(int)
        
    for event in EVENT_NAMES:
        clean_evt = event.lower().replace(" ", "_").replace("&", "and")
        df[f"event_{clean_evt}"] = (df["event_name"] == event).astype(int)
        
    X = df.drop(columns=["event_name", "day_of_week", "importance"], errors="ignore")
    if "total_delay_min" in X.columns:
        X = X.drop(columns=["total_delay_min"])
        
    if fit_columns is not None:
        for col in fit_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[fit_columns]
        
    return X

def train_and_evaluate(db: Session, user_id: int):
    """Train per-user statistical baseline and Random Forest models."""
    df = get_logs_df(db, user_id)
    paths = get_model_paths(user_id)
    
    if df.empty or len(df) < 8:
        # Delete stale models if they exist
        for p in paths.values():
            if os.path.exists(p):
                os.remove(p)
        return {
            "success": False,
            "message": "Not enough log data in database for this user (minimum 8 logs required)."
        }
        
    try:
        # Train-Test Split (use smaller test size if dataset is small)
        test_size = 0.2 if len(df) >= 10 else 0.1
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
        
        # --- 1. Event Averages Baseline ---
        event_means = train_df.groupby("event_name")["total_delay_min"].mean().to_dict()
        global_mean = train_df["total_delay_min"].mean()
        
        baseline_predictions = test_df["event_name"].map(event_means).fillna(global_mean).values
        baseline_mae = mean_absolute_error(test_df["total_delay_min"], baseline_predictions)
        
        # --- 2. Random Forest ML Model ---
        X_train = preprocess_features(train_df.copy())
        y_train = train_df["total_delay_min"].values
        feature_columns = list(X_train.columns)
        
        X_test = preprocess_features(test_df.copy(), fit_columns=feature_columns)
        y_test = test_df["total_delay_min"].values
        
        rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)
        
        ml_predictions = rf.predict(X_test)
        ml_mae = mean_absolute_error(y_test, ml_predictions)
        
        # Final Fit on all User Data
        X_all = preprocess_features(df.copy())
        y_all = df["total_delay_min"].values
        final_rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        final_rf.fit(X_all, y_all)
        
        # Save per-user models
        joblib.dump(final_rf, paths["model"])
        joblib.dump(feature_columns, paths["features"])
        joblib.dump({"event_means": event_means, "global_mean": global_mean}, paths["baseline"])
        
        # Extract Top Features
        importances = final_rf.feature_importances_
        feat_imp = sorted(zip(feature_columns, importances), key=lambda x: x[1], reverse=True)[:5]
        feat_imp_list = [{"feature": f, "importance": round(float(imp), 4)} for f, imp in feat_imp]
        
        return {
            "success": True,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "baseline_mae": round(float(baseline_mae), 2),
            "ml_mae": round(float(ml_mae), 2),
            "error_reduction_pct": round(float((baseline_mae - ml_mae) / baseline_mae) * 100, 1) if baseline_mae > 0 else 0,
            "top_features": feat_imp_list
        }
    except Exception as e:
        print(f"Error training models for user {user_id}: {e}")
        return {
            "success": False,
            "message": f"Model training failed: {e}"
        }

def get_model_status(user_id: int):
    """Check if model is trained for the specific user."""
    paths = get_model_paths(user_id)
    return {"trained": os.path.exists(paths["model"]) and os.path.exists(paths["features"])}

def predict_micro_delay(
    user_id: int,
    event_name: str,
    day_of_week: str,
    importance: str,
    bag_prepared: bool,
    estimated_prep_duration_min: float,
    estimated_travel_duration_min: float
) -> dict:
    """Predict expected delay using the user's personal ML model."""
    paths = get_model_paths(user_id)
    
    if not (os.path.exists(paths["model"]) and os.path.exists(paths["features"])):
        # Default fallback
        return {"predicted_delay_min": 4.5, "baseline_predicted_delay_min": 4.5, "model_used": "default_fallback"}
        
    try:
        rf = joblib.load(paths["model"])
        feature_columns = joblib.load(paths["features"])
        
        baseline = joblib.load(paths["baseline"])
        event_means = baseline.get("event_means", {})
        global_mean = baseline.get("global_mean", 4.5)
        
        single_row = pd.DataFrame([{
            "event_name": event_name,
            "day_of_week": day_of_week,
            "importance": importance,
            "bag_prepared": 1 if bag_prepared else 0,
            "estimated_prep_duration_min": estimated_prep_duration_min,
            "estimated_travel_duration_min": estimated_travel_duration_min
        }])
        
        X_pred = preprocess_features(single_row, fit_columns=feature_columns)
        
        predicted_ml = rf.predict(X_pred)[0]
        predicted_baseline = event_means.get(event_name, global_mean)
        
        return {
            "predicted_delay_min": round(float(predicted_ml), 1),
            "baseline_predicted_delay_min": round(float(predicted_baseline), 1),
            "model_used": "random_forest"
        }
    except Exception as e:
        print(f"Error predicting delay for user {user_id}: {e}")
        return {"predicted_delay_min": 4.5, "baseline_predicted_delay_min": 4.5, "model_used": "error_fallback"}
