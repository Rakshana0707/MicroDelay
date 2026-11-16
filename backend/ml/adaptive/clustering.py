"""Behavioral Clustering Foundation for MicroDelay.

Identifies behavioral archetypes and context clusters from departure parameters.
"""

from typing import Dict, Any, List
import logging
from .context import PredictionContext

logger = logging.getLogger("microdelay.ml.adaptive.clustering")


class BehavioralClusteringEngine:
    """Classifies routine contexts into behavioral archetypes."""

    ARCHETYPES = {
        "early_morning_rush": {
            "name": "Early Morning Rush",
            "description": "High urgency with compressed preparation window (<25 min). Prone to forgetting items.",
            "risk_multiplier": 1.35
        },
        "high_stakes_departure": {
            "name": "High-Stakes Departure",
            "description": "Critical or high-importance events where delays have elevated consequences.",
            "risk_multiplier": 1.20
        },
        "deliberate_preparation": {
            "name": "Deliberate Preparation",
            "description": "Longer preparation buffer (>= 35 min). Lower baseline lateness unless distracted.",
            "risk_multiplier": 0.85
        },
        "unprepared_commute": {
            "name": "Unprepared Commute",
            "description": "Bag and essentials not organized the night before. Higher morning item search delay.",
            "risk_multiplier": 1.40
        },
        "routine_commute": {
            "name": "Standard Routine Commute",
            "description": "Regular recurring event with standard parameters.",
            "risk_multiplier": 1.00
        }
    }

    def assign_archetype(self, context: PredictionContext) -> Dict[str, Any]:
        """Assign context to the most representative behavioral cluster archetype."""
        try:
            prep_dur = float(context.estimated_prep_duration)
            importance = str(context.event_importance).lower()
            bag_ready = bool(context.bag_prepared)
            day = str(context.day_of_week).capitalize()

            if not bag_ready:
                key = "unprepared_commute"
            elif importance in ("high", "critical"):
                key = "high_stakes_departure"
            elif prep_dur < 25.0 and day in ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday"):
                key = "early_morning_rush"
            elif prep_dur >= 35.0:
                key = "deliberate_preparation"
            else:
                key = "routine_commute"

            archetype_data = self.ARCHETYPES.get(key, self.ARCHETYPES["routine_commute"])
            return {
                "archetype_id": key,
                "name": archetype_data["name"],
                "description": archetype_data["description"],
                "risk_multiplier": archetype_data["risk_multiplier"]
            }
        except Exception as e:
            logger.warning("Error assigning behavioral archetype: %s. Using default.", e)
            default = self.ARCHETYPES["routine_commute"]
            return {
                "archetype_id": "routine_commute",
                "name": default["name"],
                "description": default["description"],
                "risk_multiplier": default["risk_multiplier"]
            }
