import datetime
import random
from sqlalchemy.orm import Session
from database import Event, DepartureLog, User, DataProvenance, DataSource, ObservationStatus

# Profile-specific schedules
PROFILE_SCHEDULES = {
    "student": [
        {"name": "ML Lab", "day_of_week": "Monday", "start_time": "08:30:00", "estimated_prep_duration": 20, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "Seminar", "day_of_week": "Monday", "start_time": "14:00:00", "estimated_prep_duration": 10, "estimated_travel_duration": 20, "importance": "medium"},
        {"name": "Math Lecture", "day_of_week": "Tuesday", "start_time": "09:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "ML Lab", "day_of_week": "Wednesday", "start_time": "08:30:00", "estimated_prep_duration": 20, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "Library Study", "day_of_week": "Wednesday", "start_time": "15:00:00", "estimated_prep_duration": 10, "estimated_travel_duration": 10, "importance": "low"},
        {"name": "Hardware Lab", "day_of_week": "Thursday", "start_time": "10:00:00", "estimated_prep_duration": 20, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "Theory Lecture", "day_of_week": "Friday", "start_time": "09:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 20, "importance": "medium"},
        {"name": "Weekend Kickoff", "day_of_week": "Friday", "start_time": "17:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 15, "importance": "low"}
    ],
    "teacher": [
        {"name": "Morning Assembly", "day_of_week": "Monday", "start_time": "08:00:00", "estimated_prep_duration": 25, "estimated_travel_duration": 15, "importance": "high"},
        {"name": "Homeroom Class", "day_of_week": "Tuesday", "start_time": "08:30:00", "estimated_prep_duration": 20, "estimated_travel_duration": 15, "importance": "high"},
        {"name": "Staff Meeting", "day_of_week": "Wednesday", "start_time": "13:30:00", "estimated_prep_duration": 10, "estimated_travel_duration": 15, "importance": "medium"},
        {"name": "Science Lab", "day_of_week": "Thursday", "start_time": "08:30:00", "estimated_prep_duration": 25, "estimated_travel_duration": 15, "importance": "high"},
        {"name": "Grading Session", "day_of_week": "Friday", "start_time": "15:30:00", "estimated_prep_duration": 15, "estimated_travel_duration": 15, "importance": "low"}
    ],
    "professor": [
        {"name": "Research Lecture", "day_of_week": "Monday", "start_time": "09:30:00", "estimated_prep_duration": 30, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "Thesis Review", "day_of_week": "Tuesday", "start_time": "11:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 20, "importance": "medium"},
        {"name": "Research Lecture", "day_of_week": "Wednesday", "start_time": "09:30:00", "estimated_prep_duration": 30, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "Faculty Council", "day_of_week": "Wednesday", "start_time": "14:30:00", "estimated_prep_duration": 10, "estimated_travel_duration": 20, "importance": "high"},
        {"name": "Office Hours", "day_of_week": "Thursday", "start_time": "13:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 10, "importance": "low"},
        {"name": "Symposium", "day_of_week": "Friday", "start_time": "10:00:00", "estimated_prep_duration": 20, "estimated_travel_duration": 20, "importance": "medium"}
    ],
    "it_professional": [
        {"name": "Daily Standup", "day_of_week": "Monday", "start_time": "09:30:00", "estimated_prep_duration": 20, "estimated_travel_duration": 25, "importance": "high"},
        {"name": "Code Review", "day_of_week": "Tuesday", "start_time": "11:00:00", "estimated_prep_duration": 10, "estimated_travel_duration": 25, "importance": "medium"},
        {"name": "Daily Standup", "day_of_week": "Wednesday", "start_time": "09:30:00", "estimated_prep_duration": 20, "estimated_travel_duration": 25, "importance": "high"},
        {"name": "Sprint Planning", "day_of_week": "Wednesday", "start_time": "14:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 25, "importance": "high"},
        {"name": "Design Sync", "day_of_week": "Thursday", "start_time": "10:00:00", "estimated_prep_duration": 15, "estimated_travel_duration": 25, "importance": "medium"},
        {"name": "Daily Standup", "day_of_week": "Friday", "start_time": "09:30:00", "estimated_prep_duration": 20, "estimated_travel_duration": 25, "importance": "high"},
        {"name": "Deploy & Retro", "day_of_week": "Friday", "start_time": "16:30:00", "estimated_prep_duration": 10, "estimated_travel_duration": 25, "importance": "low"}
    ]
}

def seed_user_events(db: Session, user_id: int, profile_type: str):
    """Seed user events based on their selected profile type."""
    templates = PROFILE_SCHEDULES.get(profile_type, PROFILE_SCHEDULES["student"])
    
    events_created = []
    for t in templates:
        # Check if already exists
        existing = db.query(Event).filter(
            Event.user_id == user_id,
            Event.name == t["name"],
            Event.day_of_week == t["day_of_week"]
        ).first()
        
        if not existing:
            evt = Event(
                user_id=user_id,
                name=t["name"],
                day_of_week=t["day_of_week"],
                start_time=t["start_time"],
                estimated_prep_duration=t["estimated_prep_duration"],
                estimated_travel_duration=t["estimated_travel_duration"],
                importance=t["importance"]
            )
            db.add(evt)
            events_created.append(evt)
            
    db.commit()
    return events_created

def generate_user_logs(db: Session, user_id: int, num_days: int = 60):
    """Generate realistic 60-day historical logs linked to user_id, customized by profile."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        print(f"User ID {user_id} not found. Cannot generate logs.")
        return
        
    profile = user.profile_type
    
    # 1. Seed events for this user
    seed_user_events(db, user_id, profile)
    
    # Clean previous synthetic demonstration logs for this user (preserve real observations!)
    db.query(DepartureLog).filter(
        DepartureLog.user_id == user_id,
        DepartureLog.is_synthetic == True
    ).delete()
    db.commit()
    
    events = db.query(Event).filter(Event.user_id == user_id).all()
    if not events:
        print(f"No events found for user {user_id}.")
        return
        
    start_date = datetime.date.today() - datetime.timedelta(days=num_days)
    logged_count = 0
    
    for i in range(num_days):
        current_date = start_date + datetime.timedelta(days=i)
        day_name = current_date.strftime("%A")
        
        # Get events for this day
        day_events = [e for e in events if e.day_of_week == day_name]
        
        for event in day_events:
            est_prep = float(event.estimated_prep_duration)
            est_travel = float(event.estimated_travel_duration)
            
            # Event start time
            start_h, start_m, start_s = map(int, event.start_time.split(":"))
            event_start_dt = datetime.datetime.combine(current_date, datetime.time(start_h, start_m, start_s))
            
            # Planned departure and prep start
            planned_departure_dt = event_start_dt - datetime.timedelta(minutes=est_travel)
            planned_prep_start_dt = planned_departure_dt - datetime.timedelta(minutes=est_prep)
            
            # Simulate Actual Prep Start Time
            prep_start_noise = random.normalvariate(0, 1.8)
            actual_prep_start_dt = planned_prep_start_dt + datetime.timedelta(minutes=prep_start_noise)
            
            actual_prep_dur = est_prep
            bag_prepared = True
            bag_prep_delay = 0.0
            
            # 2. Seed profile-specific behaviors
            if profile == "student":
                # Monday bag packing bottleneck
                if day_name == "Monday" and event.name == "ML Lab":
                    bag_prepared = random.random() > 0.70 # 70% chance not prepped
                    if not bag_prepared:
                        bag_prep_delay = random.uniform(5.0, 9.0)
                        actual_prep_dur += bag_prep_delay
                else:
                    bag_prepared = random.random() > 0.40
                
                # WhatsApp/Social distraction
                one_last_task = (event.importance == "high" or start_h <= 9) and random.random() < 0.45
                one_last_task_delay = random.uniform(4.0, 8.5) if one_last_task else 0.0
                actual_prep_dur += one_last_task_delay
                
                finding_items = random.random() < (0.30 if start_h <= 9 else 0.15)
                finding_items_delay = random.uniform(3.0, 6.5) if finding_items else 0.0
                actual_prep_dur += finding_items_delay
                
                phone_usage = random.random() < 0.38
                phone_delay = random.uniform(2.0, 5.0) if phone_usage else 0.0
                actual_prep_dur += phone_delay
                
                waiting = random.random() < 0.12
                waiting_delay = random.uniform(2.0, 5.0) if waiting else 0.0
                actual_prep_dur += waiting_delay
                
            elif profile == "teacher":
                # Lesson plans/handouts prep bottleneck
                if day_name in ["Monday", "Thursday"] and event.importance == "high":
                    bag_prepared = random.random() > 0.65
                    if not bag_prepared:
                        bag_prep_delay = random.uniform(6.0, 10.0) # Grading worksheets search
                        actual_prep_dur += bag_prep_delay
                else:
                    bag_prepared = random.random() > 0.35
                
                # "Double-checking worksheets/printouts" (One Last Task)
                one_last_task = start_h <= 9 and random.random() < 0.40
                one_last_task_delay = random.uniform(5.0, 9.0) if one_last_task else 0.0
                actual_prep_dur += one_last_task_delay
                
                # Searching for classroom keys
                finding_items = random.random() < 0.35
                finding_items_delay = random.uniform(3.5, 7.0) if finding_items else 0.0
                actual_prep_dur += finding_items_delay
                
                phone_usage = random.random() < 0.20 # Teachers scroll less in the morning
                phone_delay = random.uniform(2.0, 4.0) if phone_usage else 0.0
                actual_prep_dur += phone_delay
                
                waiting = random.random() < 0.15
                waiting_delay = random.uniform(2.0, 6.0) if waiting else 0.0
                actual_prep_dur += waiting_delay
                
            elif profile == "professor":
                # Projector adapter / lecture handouts search bottleneck
                if event.name in ["Research Lecture", "Symposium"]:
                    bag_prepared = random.random() > 0.60
                    if not bag_prepared:
                        bag_prep_delay = random.uniform(7.0, 12.0) # Projector adapter / slide check
                        actual_prep_dur += bag_prep_delay
                else:
                    bag_prepared = random.random() > 0.30
                
                # Checking student emails before leaving (One Last Task)
                one_last_task = random.random() < 0.50
                one_last_task_delay = random.uniform(5.0, 11.0) if one_last_task else 0.0
                actual_prep_dur += one_last_task_delay
                
                finding_items = random.random() < 0.25
                finding_items_delay = random.uniform(3.0, 6.0) if finding_items else 0.0
                actual_prep_dur += finding_items_delay
                
                phone_usage = random.random() < 0.25
                phone_delay = random.uniform(2.0, 4.5) if phone_usage else 0.0
                actual_prep_dur += phone_delay
                
                waiting = random.random() < 0.10
                waiting_delay = random.uniform(2.0, 5.0) if waiting else 0.0
                actual_prep_dur += waiting_delay
                
            else: # it_professional
                # "One Last Commit / Server Sync" bottleneck
                if day_name in ["Monday", "Wednesday", "Friday"] and start_h <= 10:
                    bag_prepared = random.random() > 0.70
                    if not bag_prepared:
                        bag_prep_delay = random.uniform(6.0, 13.0) # Git sync / build cycle wait
                        actual_prep_dur += bag_prep_delay
                else:
                    bag_prepared = random.random() > 0.40
                
                # Reviewing Slack / Teams tags before leaving (One Last Task)
                one_last_task = event.importance == "high" and random.random() < 0.55
                one_last_task_delay = random.uniform(4.5, 9.5) if one_last_task else 0.0
                actual_prep_dur += one_last_task_delay
                
                # Finding laptop charger / entry badge
                finding_items = random.random() < 0.30
                finding_items_delay = random.uniform(3.0, 6.5) if finding_items else 0.0
                actual_prep_dur += finding_items_delay
                
                # Message distraction
                phone_usage = random.random() < 0.35
                phone_delay = random.uniform(2.0, 5.5) if phone_usage else 0.0
                actual_prep_dur += phone_delay
                
                waiting = random.random() < 0.18
                waiting_delay = random.uniform(2.0, 6.0) if waiting else 0.0
                actual_prep_dur += waiting_delay
            
            # Common baseline noise
            prep_noise = random.normalvariate(1.2, 1.8)
            actual_prep_dur += max(-3.0, prep_noise)
            
            # Departure
            actual_departure_dt = actual_prep_start_dt + datetime.timedelta(minutes=actual_prep_dur)
            departure_delay_min = (actual_departure_dt - planned_departure_dt).total_seconds() / 60.0
            
            # Allocation
            if departure_delay_min > 0:
                allocated_one_last = one_last_task_delay
                allocated_finding = finding_items_delay
                allocated_phone = phone_delay
                allocated_waiting = waiting_delay
                
                total_tagged = allocated_one_last + allocated_finding + allocated_phone + allocated_waiting + bag_prep_delay
                remainder = departure_delay_min - total_tagged
                allocated_prep_bias = max(0.0, remainder)
                
                if bag_prep_delay > 0:
                    allocated_prep_bias += bag_prep_delay
            else:
                allocated_one_last = 0.0
                allocated_finding = 0.0
                allocated_phone = 0.0
                allocated_waiting = 0.0
                allocated_prep_bias = 0.0
            
            # Travel noise
            traffic_bias = 3.0 if (profile == "it_professional" and day_name == "Monday" and start_h <= 10) else 0.0
            travel_noise = random.normalvariate(0.5, 1.2)
            actual_travel_dur = est_travel + traffic_bias + max(-2.0, travel_noise)
            
            actual_arrival_dt = actual_departure_dt + datetime.timedelta(minutes=actual_travel_dur)
            
            # Create DB Log
            log = DepartureLog(
                user_id=user_id,
                event_id=event.id,
                event_name=event.name,
                date=current_date.strftime("%Y-%m-%d"),
                day_of_week=day_name,
                planned_prep_start=planned_prep_start_dt.strftime("%H:%M:%S"),
                planned_departure=planned_departure_dt.strftime("%H:%M:%S"),
                estimated_prep_duration_min=est_prep,
                estimated_travel_duration_min=est_travel,
                actual_prep_start=actual_prep_start_dt.strftime("%H:%M:%S"),
                actual_departure=actual_departure_dt.strftime("%H:%M:%S"),
                actual_arrival=actual_arrival_dt.strftime("%H:%M:%S"),
                actual_prep_duration_min=round(actual_prep_dur, 2),
                actual_travel_duration_min=round(actual_travel_dur, 2),
                total_delay_min=round(departure_delay_min, 2),
                bag_prepared=bag_prepared,
                tag_one_last_task=one_last_task,
                tag_finding_items=finding_items,
                tag_phone_usage=phone_usage,
                tag_waiting=waiting,
                delay_one_last_task_min=round(allocated_one_last, 2),
                delay_finding_items_min=round(allocated_finding, 2),
                delay_phone_usage_min=round(allocated_phone, 2),
                delay_waiting_min=round(allocated_waiting, 2),
                delay_prep_bias_min=round(allocated_prep_bias, 2),
                provenance=DataProvenance.SYNTHETIC.value,
                data_source=DataSource.GENERATOR.value,
                is_synthetic=True,
                recorded_at=actual_departure_dt.isoformat() + "Z",
                observation_status=ObservationStatus.VALID.value
            )
            db.add(log)
            logged_count += 1
            
    db.commit()
    print(f"Generated {logged_count} logs for user {user_id} ({profile}).")
