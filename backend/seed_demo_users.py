import os
import sys
from sqlalchemy.orm import Session

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database import SessionLocal, User, engine, Base
from security import hash_password
from generator import generate_user_logs
from models import train_and_evaluate

def seed_demo_users():
    # Initialize all database tables fresh
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if demo users already exist
        existing = db.query(User).filter(User.username.in_([
            'student_demo', 'teacher_demo', 'professor_demo', 'engineer_demo'
        ])).first()
        
        if existing:
            print("Demo users already exist.")
            return

        demo_accounts = [
            {"username": "student_demo", "profile": "student", "desc": "College Student"},
            {"username": "teacher_demo", "profile": "teacher", "desc": "School Teacher"},
            {"username": "professor_demo", "profile": "professor", "desc": "University Professor"},
            {"username": "engineer_demo", "profile": "it_professional", "desc": "IT Software Engineer"}
        ]
        
        for acc in demo_accounts:
            print(f"Creating user: {acc['username']} ({acc['desc']})...")
            pwd_hash, pwd_salt = hash_password("password")
            new_user = User(
                username=acc['username'],
                hashed_password=pwd_hash,
                salt=pwd_salt,
                profile_type=acc['profile']
            )
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            # Seed 60 days of historical logs
            print(f"  -> Seeding 60-day historical logs...")
            generate_user_logs(db, new_user.id)
            
            # Pre-train the machine learning model
            print(f"  -> Training personalized AI model...")
            train_and_evaluate(db, new_user.id)
            
        print("All demo users registered, seeded, and trained successfully!")
        
    finally:
        db.close()

if __name__ == "__main__":
    seed_demo_users()
