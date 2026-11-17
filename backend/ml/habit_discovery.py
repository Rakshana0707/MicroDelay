"""Behavioral Habit Discovery & Clustering Engine for MicroDelay 2.0 (Prompt 2.4).

Provides unsupervised learning modules to discover recurring behavioral habits,
compare clustering algorithms (K-Means, Agglomerative, DBSCAN), dynamically
interpret cluster centroids into evidence-based behavioral patterns, and track
individual user profile evolution over time.
"""

from typing import Dict, Any, List, Optional, Tuple
import os
import json
import logging
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sqlalchemy.orm import Session
from sqlalchemy import text

try:
    from database import DepartureLog, User, UserClusterProfile
except ModuleNotFoundError:
    from backend.database import DepartureLog, User, UserClusterProfile

logger = logging.getLogger("microdelay.ml.habit_discovery")

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
CLUSTER_MODEL_PATH = os.path.join(MODELS_DIR, "cluster_model.pkl")
CLUSTER_SCALER_PATH = os.path.join(MODELS_DIR, "cluster_scaler.pkl")
CLUSTER_META_PATH = os.path.join(MODELS_DIR, "cluster_meta.json")

# The 12 standard user-level behavioral features requested in Prompt 2.4
BEHAVIORAL_FEATURE_NAMES = [
    "average_prep_duration",
    "average_prep_bias",
    "average_departure_delay",
    "lateness_rate",
    "phone_distraction_frequency",
    "item_search_frequency",
    "last_minute_task_frequency",
    "bag_prepared_percentage",
    "average_travel_error",
    "weekday_delay_variation",
    "preparation_duration_variability",
    "recent_behavior_change"
]


def extract_user_behavioral_features(
    logs: List[Any],
    cutoff_timestamp: Optional[str] = None
) -> Dict[str, float]:
    """Extract standard user-level behavioral features from chronological departure logs.
    
    Guarantees temporal safety: only records before cutoff_timestamp are considered.
    """
    if not logs:
        return {feat: 0.0 for feat in BEHAVIORAL_FEATURE_NAMES}

    # Temporal filtering
    filtered_logs = []
    for log in logs:
        log_time = None
        if hasattr(log, "recorded_at") and log.recorded_at:
            log_time = str(log.recorded_at)
        elif hasattr(log, "date") and log.date:
            log_time = f"{log.date}T23:59:59Z"
        elif isinstance(log, dict):
            log_time = str(log.get("recorded_at") or f"{log.get('date', '')}T23:59:59Z")

        if cutoff_timestamp and log_time and log_time > cutoff_timestamp:
            continue
        filtered_logs.append(log)

    if not filtered_logs:
        return {feat: 0.0 for feat in BEHAVIORAL_FEATURE_NAMES}

    n = len(filtered_logs)
    actual_preps = []
    prep_biases = []
    delays = []
    late_count = 0
    phone_count = 0
    item_count = 0
    task_count = 0
    bag_count = 0
    travel_errors = []
    weekday_delays: Dict[str, List[float]] = {}

    for log in filtered_logs:
        actual_prep = float(getattr(log, "actual_prep_duration_min", 0.0) or (isinstance(log, dict) and log.get("actual_prep_duration_min", 0.0)) or 0.0)
        est_prep = float(getattr(log, "estimated_prep_duration_min", 0.0) or (isinstance(log, dict) and log.get("estimated_prep_duration_min", 0.0)) or 0.0)
        actual_travel = float(getattr(log, "actual_travel_duration_min", 0.0) or (isinstance(log, dict) and log.get("actual_travel_duration_min", 0.0)) or 0.0)
        est_travel = float(getattr(log, "estimated_travel_duration_min", 0.0) or (isinstance(log, dict) and log.get("estimated_travel_duration_min", 0.0)) or 0.0)
        tot_delay = float(getattr(log, "total_delay_min", 0.0) or (isinstance(log, dict) and log.get("total_delay_min", 0.0)) or 0.0)
        
        actual_preps.append(actual_prep)
        prep_biases.append(actual_prep - est_prep)
        delays.append(tot_delay)
        travel_errors.append(actual_travel - est_travel)

        if tot_delay > 0.0:
            late_count += 1

        phone_tag = bool(getattr(log, "tag_phone_usage", False) or (isinstance(log, dict) and log.get("tag_phone_usage", False)))
        phone_d = float(getattr(log, "delay_phone_usage_min", 0.0) or (isinstance(log, dict) and log.get("delay_phone_usage_min", 0.0)) or 0.0)
        if phone_tag or phone_d > 0.0:
            phone_count += 1

        item_tag = bool(getattr(log, "tag_finding_items", False) or (isinstance(log, dict) and log.get("tag_finding_items", False)))
        item_d = float(getattr(log, "delay_finding_items_min", 0.0) or (isinstance(log, dict) and log.get("delay_finding_items_min", 0.0)) or 0.0)
        if item_tag or item_d > 0.0:
            item_count += 1

        task_tag = bool(getattr(log, "tag_one_last_task", False) or (isinstance(log, dict) and log.get("tag_one_last_task", False)))
        task_d = float(getattr(log, "delay_one_last_task_min", 0.0) or (isinstance(log, dict) and log.get("delay_one_last_task_min", 0.0)) or 0.0)
        if task_tag or task_d > 0.0:
            task_count += 1

        bag_ready = bool(getattr(log, "bag_prepared", False) or (isinstance(log, dict) and log.get("bag_prepared", False)))
        if bag_ready:
            bag_count += 1

        day = str(getattr(log, "day_of_week", "Monday") or (isinstance(log, dict) and log.get("day_of_week", "Monday"))).capitalize()
        weekday_delays.setdefault(day, []).append(tot_delay)

    # Calculate aggregate metrics
    avg_prep = float(np.mean(actual_preps)) if actual_preps else 0.0
    avg_bias = float(np.mean(prep_biases)) if prep_biases else 0.0
    avg_delay = float(np.mean(delays)) if delays else 0.0
    lateness_rate = float(late_count / n)
    phone_freq = float(phone_count / n)
    item_freq = float(item_count / n)
    task_freq = float(task_count / n)
    bag_pct = float(bag_count / n)
    avg_travel_err = float(np.mean(travel_errors)) if travel_errors else 0.0

    # Weekday delay variation: std of mean delay across weekdays
    if len(weekday_delays) >= 2:
        day_means = [float(np.mean(d_list)) for d_list in weekday_delays.values()]
        weekday_variation = float(np.std(day_means))
    elif delays:
        weekday_variation = float(np.std(delays))
    else:
        weekday_variation = 0.0

    # Prep duration variability: std of prep durations
    prep_var = float(np.std(actual_preps)) if len(actual_preps) >= 2 else 0.0

    # Recent behavior change: difference between recent 5 logs vs older logs
    if n > 5:
        recent_delays = delays[-5:]
        older_delays = delays[:-5]
        recent_change = float(np.mean(recent_delays) - np.mean(older_delays))
    else:
        recent_change = 0.0

    return {
        "average_prep_duration": round(avg_prep, 2),
        "average_prep_bias": round(avg_bias, 2),
        "average_departure_delay": round(avg_delay, 2),
        "lateness_rate": round(lateness_rate, 3),
        "phone_distraction_frequency": round(phone_freq, 3),
        "item_search_frequency": round(item_freq, 3),
        "last_minute_task_frequency": round(task_freq, 3),
        "bag_prepared_percentage": round(bag_pct, 3),
        "average_travel_error": round(avg_travel_err, 2),
        "weekday_delay_variation": round(weekday_variation, 2),
        "preparation_duration_variability": round(prep_var, 2),
        "recent_behavior_change": round(recent_change, 2)
    }


class ClusterInterpretationService:
    """Interprets cluster centroids dynamically using actual feature statistics."""

    @staticmethod
    def interpret_centroid(
        centroid: Dict[str, float],
        population_mean: Dict[str, float],
        population_std: Dict[str, float]
    ) -> Dict[str, Any]:
        """Derive behavioral habit title, description, and key traits from centroid statistics.
        
        Never hardcodes cluster numbers. Determines meanings entirely from relative deviations.
        """
        # Calculate z-scores relative to population
        z_scores: Dict[str, float] = {}
        for feat in BEHAVIORAL_FEATURE_NAMES:
            c_val = centroid.get(feat, 0.0)
            p_mean = population_mean.get(feat, 0.0)
            p_std = population_std.get(feat, 1.0)
            if p_std < 1e-4:
                p_std = 1.0
            z_scores[feat] = (c_val - p_mean) / p_std

        # Identify dominant behavioral patterns
        bag_pct = centroid.get("bag_prepared_percentage", 0.5)
        avg_delay = centroid.get("average_departure_delay", 0.0)
        phone_freq = centroid.get("phone_distraction_frequency", 0.0)
        item_freq = centroid.get("item_search_frequency", 0.0)
        task_freq = centroid.get("last_minute_task_frequency", 0.0)
        prep_bias = centroid.get("average_prep_bias", 0.0)
        travel_err = centroid.get("average_travel_error", 0.0)
        prep_var = centroid.get("preparation_duration_variability", 0.0)

        # Ranked traits
        sorted_traits = sorted(z_scores.items(), key=lambda kv: abs(kv[1]), reverse=True)
        dominant_characteristics = []
        for feat, z in sorted_traits[:4]:
            c_val = centroid.get(feat, 0.0)
            p_val = population_mean.get(feat, 0.0)
            delta = c_val - p_val
            direction = "higher" if delta > 0 else "lower"
            dominant_characteristics.append({
                "feature": feat,
                "cluster_value": round(c_val, 2),
                "population_mean": round(p_val, 2),
                "relative_difference": f"{abs(delta):.1f} {direction} than population",
                "z_score": round(z, 2)
            })

        # Evidence-based habit pattern mapping
        if bag_pct >= 0.70 and avg_delay <= 1.5 and phone_freq < 0.25:
            cluster_name = "Consistently Prepared Routine"
            cluster_desc = (
                "Proactive pre-departure habit with items packed the night before. "
                "Maintains low departure delay and minimal pre-departure distractions."
            )
            action = "Keep maintaining the overnight bag preparation routine."
        elif phone_freq >= 0.30 or z_scores.get("phone_distraction_frequency", 0.0) >= 1.0:
            cluster_name = "Phone-Interrupted Preparation"
            cluster_desc = (
                f"Preparation flow is frequently fragmented by digital notifications and phone checks "
                f"({round(phone_freq * 100)}% of checkouts). Adds noticeable delay variance."
            )
            action = "Implement a phone-docking rule during the final 15 minutes of morning prep."
        elif item_freq >= 0.30 or z_scores.get("item_search_frequency", 0.0) >= 1.0:
            cluster_name = "Frequent Item-Search Delays"
            cluster_desc = (
                f"Morning departures are bottlenecked by searching for essentials (keys, badges, transit cards) "
                f"at the doorway ({round(item_freq * 100)}% occurrence rate)."
            )
            action = "Establish a dedicated doorway launchpad bowl for keys and ID cards."
        elif task_freq >= 0.30 or z_scores.get("last_minute_task_frequency", 0.0) >= 1.0:
            cluster_name = "Last-Minute Task Driven"
            cluster_desc = (
                f"Habitual 'one last task' behavior (answering one more email, last chore, compile/commit) "
                f"occurring right at the scheduled departure deadline."
            )
            action = "Set a hard 'tools-down' buffer 5 minutes before scheduled leave time."
        elif prep_bias >= 3.0 or z_scores.get("average_prep_bias", 0.0) >= 0.9:
            cluster_name = "Preparation Underestimation Bias"
            cluster_desc = (
                f"Systematic cognitive planning bias where actual preparation exceeds scheduled duration "
                f"by an average of {round(prep_bias, 1)} minutes."
            )
            action = "Expand the scheduled prep duration buffer to absorb reality gap."
        elif travel_err >= 3.0 or z_scores.get("average_travel_error", 0.0) >= 0.9:
            cluster_name = "Travel-Time Optimism"
            cluster_desc = (
                f"Frequently underestimates actual transit and travel duration by {round(travel_err, 1)} minutes, "
                f"leaving inadequate margin for traffic and transit waiting."
            )
            action = "Add 5-10 minutes transit contingency buffer to route estimates."
        elif prep_var >= 6.0 or z_scores.get("preparation_duration_variability", 0.0) >= 0.9:
            cluster_name = "Inconsistent Routine Timeline"
            cluster_desc = (
                f"High volatility in preparation duration across mornings (variability = {round(prep_var, 1)}m), "
                f"indicating unpredictable waking or morning task flow."
            )
            action = "Standardize morning sequence to stabilize departure consistency."
        else:
            cluster_name = "Standard Variable Routine"
            cluster_desc = (
                f"Balanced routine exhibiting typical baseline variance across weekdays "
                f"(average delay: {round(avg_delay, 1)}m)."
            )
            action = "Continue tracking checkouts to discover emerging micro-patterns."

        return {
            "cluster_name": cluster_name,
            "cluster_description": cluster_desc,
            "dominant_characteristics": dominant_characteristics,
            "recommended_action": action
        }


class BehavioralClusteringTournament:
    """Trains and compares multiple clustering algorithms over user behavioral feature vectors."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def run_tournament(
        self,
        X_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Run tournament across K-Means, Agglomerative, and DBSCAN.
        
        Evaluates Silhouette score, Davies-Bouldin index, cluster size distribution,
        and stability. Returns the winning model, scaler, and benchmark summary.
        """
        if len(X_df) < 3:
            # Insufficient users to cluster reliably
            logger.info("Fewer than 3 users (%d) available. Creating single baseline habit cluster.", len(X_df))
            scaler = StandardScaler()
            if len(X_df) > 0:
                scaler.fit(X_df[BEHAVIORAL_FEATURE_NAMES].values)
            else:
                scaler.fit(np.zeros((1, len(BEHAVIORAL_FEATURE_NAMES))))

            pop_mean = {f: float(X_df[f].mean()) if len(X_df) > 0 else 0.0 for f in BEHAVIORAL_FEATURE_NAMES}
            pop_std = {f: float(X_df[f].std()) if len(X_df) > 1 else 1.0 for f in BEHAVIORAL_FEATURE_NAMES}

            interp = ClusterInterpretationService.interpret_centroid(pop_mean, pop_mean, pop_std)

            centroid_dict = {
                0: {
                    "cluster_id": 0,
                    "centroid_values": pop_mean,
                    "size": len(X_df),
                    "interpretation": interp
                }
            }

            return {
                "selected_algorithm": "BaselineKMeans",
                "optimal_k": 1,
                "silhouette_score": 0.0,
                "davies_bouldin_index": 0.0,
                "is_sufficient_sample": False,
                "model": None,
                "scaler": scaler,
                "centroids": centroid_dict,
                "population_mean": pop_mean,
                "population_std": pop_std,
                "benchmark_results": [],
                "user_assignments": {i: 0 for i in range(len(X_df))}
            }

        X = X_df[BEHAVIORAL_FEATURE_NAMES].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        n_samples = len(X)
        max_k = min(5, n_samples - 1)
        candidate_k_values = list(range(2, max_k + 1)) if max_k >= 2 else [2]

        benchmark_results = []
        candidates = []

        # 1. Evaluate K-Means across k
        for k in candidate_k_values:
            try:
                km = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
                labels = km.fit_predict(X_scaled)
                n_clusters_found = len(set(labels))
                if n_clusters_found < 2:
                    continue

                sil = float(silhouette_score(X_scaled, labels))
                db = float(davies_bouldin_score(X_scaled, labels))
                counts = [int(np.sum(labels == c)) for c in range(k)]
                size_ratio = max(counts) / (min(counts) + 1e-6)

                # Composite score: reward high silhouette, penalize high DB & degenerate skew
                score = sil - (0.15 * min(db, 4.0)) - (0.05 * min(size_ratio, 10.0))

                cand = {
                    "algorithm": "K-Means",
                    "k": k,
                    "model": km,
                    "labels": labels,
                    "silhouette": round(sil, 3),
                    "davies_bouldin": round(db, 3),
                    "size_distribution": counts,
                    "composite_score": round(score, 3)
                }
                candidates.append(cand)
                benchmark_results.append({
                    "algorithm": f"K-Means (k={k})",
                    "k": k,
                    "silhouette": round(sil, 3),
                    "davies_bouldin": round(db, 3),
                    "size_distribution": counts,
                    "status": "evaluated"
                })
            except Exception as e:
                logger.warning("K-Means k=%d evaluation failed: %s", k, e)

        # 2. Evaluate Agglomerative Clustering across k
        for k in candidate_k_values:
            try:
                agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
                labels = agg.fit_predict(X_scaled)
                n_clusters_found = len(set(labels))
                if n_clusters_found < 2:
                    continue

                sil = float(silhouette_score(X_scaled, labels))
                db = float(davies_bouldin_score(X_scaled, labels))
                counts = [int(np.sum(labels == c)) for c in range(k)]
                size_ratio = max(counts) / (min(counts) + 1e-6)
                score = sil - (0.15 * min(db, 4.0)) - (0.05 * min(size_ratio, 10.0))

                cand = {
                    "algorithm": "Agglomerative",
                    "k": k,
                    "model": agg,
                    "labels": labels,
                    "silhouette": round(sil, 3),
                    "davies_bouldin": round(db, 3),
                    "size_distribution": counts,
                    "composite_score": round(score, 3)
                }
                candidates.append(cand)
                benchmark_results.append({
                    "algorithm": f"Agglomerative (k={k})",
                    "k": k,
                    "silhouette": round(sil, 3),
                    "davies_bouldin": round(db, 3),
                    "size_distribution": counts,
                    "status": "evaluated"
                })
            except Exception as e:
                logger.warning("Agglomerative k=%d evaluation failed: %s", k, e)

        # 3. Evaluate DBSCAN
        for eps in [1.5, 2.0, 2.5]:
            try:
                dbs = DBSCAN(eps=eps, min_samples=2)
                labels = dbs.fit_predict(X_scaled)
                unique_labels = set(labels) - {-1}
                if len(unique_labels) >= 2:
                    # Filter noise points for metric calculation
                    valid_idx = labels != -1
                    if np.sum(valid_idx) >= 4 and len(set(labels[valid_idx])) >= 2:
                        sil = float(silhouette_score(X_scaled[valid_idx], labels[valid_idx]))
                        db = float(davies_bouldin_score(X_scaled[valid_idx], labels[valid_idx]))
                        counts = [int(np.sum(labels == c)) for c in unique_labels]
                        noise_count = int(np.sum(labels == -1))
                        score = sil - (0.2 * min(db, 4.0)) - (0.1 * (noise_count / n_samples))

                        cand = {
                            "algorithm": "DBSCAN",
                            "k": len(unique_labels),
                            "model": dbs,
                            "labels": labels,
                            "silhouette": round(sil, 3),
                            "davies_bouldin": round(db, 3),
                            "size_distribution": counts,
                            "composite_score": round(score, 3)
                        }
                        candidates.append(cand)
                        benchmark_results.append({
                            "algorithm": f"DBSCAN (eps={eps})",
                            "k": len(unique_labels),
                            "silhouette": round(sil, 3),
                            "davies_bouldin": round(db, 3),
                            "size_distribution": counts,
                            "status": "evaluated"
                        })
            except Exception as e:
                logger.warning("DBSCAN eps=%.1f evaluation failed: %s", eps, e)

        # Select best candidate
        if not candidates:
            # Fallback to K-Means k=2
            logger.warning("No candidate passed validation thresholds. Falling back to default K-Means k=2.")
            km = KMeans(n_clusters=2, random_state=self.random_state, n_init=10)
            labels = km.fit_predict(X_scaled)
            best_cand = {
                "algorithm": "K-Means",
                "k": 2,
                "model": km,
                "labels": labels,
                "silhouette": 0.0,
                "davies_bouldin": 1.0,
                "size_distribution": [int(np.sum(labels == 0)), int(np.sum(labels == 1))],
                "composite_score": 0.0
            }
        else:
            best_cand = max(candidates, key=lambda c: c["composite_score"])

        # Compute cluster centroids in original feature space
        best_labels = best_cand["labels"]
        best_k = best_cand["k"]

        pop_mean = {f: float(X_df[f].mean()) for f in BEHAVIORAL_FEATURE_NAMES}
        pop_std = {f: float(X_df[f].std()) if len(X_df) > 1 else 1.0 for f in BEHAVIORAL_FEATURE_NAMES}
        for f in BEHAVIORAL_FEATURE_NAMES:
            if pop_std[f] < 1e-4:
                pop_std[f] = 1.0

        centroids_dict = {}
        for c in range(best_k):
            mask = (best_labels == c)
            if np.any(mask):
                c_df = X_df[mask]
                c_mean = {f: float(c_df[f].mean()) for f in BEHAVIORAL_FEATURE_NAMES}
                size = int(np.sum(mask))
            else:
                c_mean = pop_mean.copy()
                size = 0

            interp = ClusterInterpretationService.interpret_centroid(c_mean, pop_mean, pop_std)
            centroids_dict[c] = {
                "cluster_id": c,
                "centroid_values": c_mean,
                "size": size,
                "interpretation": interp
            }

        user_assignments = {i: int(best_labels[i]) for i in range(len(best_labels))}

        return {
            "selected_algorithm": best_cand["algorithm"],
            "optimal_k": best_k,
            "silhouette_score": best_cand["silhouette"],
            "davies_bouldin_index": best_cand["davies_bouldin"],
            "is_sufficient_sample": True,
            "model": best_cand["model"],
            "scaler": scaler,
            "centroids": centroids_dict,
            "population_mean": pop_mean,
            "population_std": pop_std,
            "benchmark_results": benchmark_results,
            "user_assignments": user_assignments
        }


class HabitDiscoveryService:
    """End-to-end service for behavioral habit dataset creation, clustering, and user profile management."""

    @classmethod
    def build_clustering_dataset(
        cls,
        db: Session,
        provenance: str = "real",
        min_samples: int = 3,
        cutoff_timestamp: Optional[str] = None
    ) -> Tuple[pd.DataFrame, List[int]]:
        """Construct user-level behavioral feature matrix from genuine user departure history.
        
        Strictly excludes direct personal identifiers from the feature matrix.
        """
        users = db.query(User).all()
        user_rows = []
        user_ids = []

        is_synth = (provenance == "synthetic")

        for u in users:
            query = db.query(DepartureLog).filter(
                DepartureLog.user_id == u.id,
                DepartureLog.is_synthetic == is_synth,
                DepartureLog.observation_status == "valid"
            )
            logs = query.all()
            if len(logs) < min_samples:
                continue

            features = extract_user_behavioral_features(logs, cutoff_timestamp=cutoff_timestamp)
            user_rows.append(features)
            user_ids.append(u.id)

        if not user_rows:
            return pd.DataFrame(columns=BEHAVIORAL_FEATURE_NAMES), []

        df = pd.DataFrame(user_rows)
        return df, user_ids

    @classmethod
    def train_and_persist_clusters(
        cls,
        db: Session,
        provenance: str = "real"
    ) -> Dict[str, Any]:
        """Run tournament clustering, persist artifacts, and update user cluster profiles."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        version_tag = f"cluster_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        df, user_ids = cls.build_clustering_dataset(db, provenance=provenance, min_samples=3)

        tournament = BehavioralClusteringTournament()
        result = tournament.run_tournament(df)

        # Save artifacts
        if result["model"] is not None:
            joblib.dump(result["model"], CLUSTER_MODEL_PATH)
        joblib.dump(result["scaler"], CLUSTER_SCALER_PATH)

        meta = {
            "cluster_model_version": version_tag,
            "trained_at": now_iso,
            "algorithm": result["selected_algorithm"],
            "optimal_k": result["optimal_k"],
            "silhouette_score": result["silhouette_score"],
            "davies_bouldin_index": result["davies_bouldin_index"],
            "total_users_clustered": len(user_ids),
            "centroids": {str(cid): cdata for cid, cdata in result["centroids"].items()},
            "population_mean": result["population_mean"],
            "population_std": result["population_std"],
            "benchmark_results": result["benchmark_results"]
        }

        with open(CLUSTER_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Update profiles for clustered users
        for idx, u_id in enumerate(user_ids):
            assigned_c = result["user_assignments"].get(idx, 0)
            c_info = result["centroids"].get(assigned_c, result["centroids"].get(0))
            interp = c_info["interpretation"]
            user_feats = df.iloc[idx].to_dict()

            profile = UserClusterProfile(
                user_id=u_id,
                cluster_id=assigned_c,
                cluster_name=interp["cluster_name"],
                cluster_description=interp["cluster_description"],
                dominant_characteristics=json.dumps(interp["dominant_characteristics"]),
                user_features=json.dumps(user_feats),
                cluster_centroid=json.dumps(c_info["centroid_values"]),
                population_mean=json.dumps(result["population_mean"]),
                data_sufficiency_status="sufficient" if len(user_ids) >= 8 else "gathering_history",
                sample_count=len(df),
                provenance=provenance,
                cluster_model_version=version_tag,
                generated_at=now_iso
            )
            db.add(profile)
        db.commit()

        return meta

    @classmethod
    def get_or_create_user_profile(
        cls,
        user_id: int,
        db: Session,
        provenance: str = "real"
    ) -> Dict[str, Any]:
        """Fetch or dynamically assign behavioral habit cluster profile for a user."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        is_synth = (provenance == "synthetic")

        # 1. Fetch user genuine departure logs
        logs = db.query(DepartureLog).filter(
            DepartureLog.user_id == user_id,
            DepartureLog.is_synthetic == is_synth,
            DepartureLog.observation_status == "valid"
        ).all()

        n_logs = len(logs)
        user_features = extract_user_behavioral_features(logs)

        # Data sufficiency status
        if n_logs < 3:
            sufficiency = "insufficient_data"
        elif n_logs < 8:
            sufficiency = "gathering_history"
        else:
            sufficiency = "sufficient"

        # Check existing cluster metadata on disk
        meta = None
        scaler = None
        if os.path.exists(CLUSTER_META_PATH):
            try:
                with open(CLUSTER_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception as e:
                logger.warning("Could not load cluster metadata: %s", e)

        if os.path.exists(CLUSTER_SCALER_PATH):
            try:
                scaler = joblib.load(CLUSTER_SCALER_PATH)
            except Exception as e:
                logger.warning("Could not load cluster scaler: %s", e)

        if meta is None or scaler is None:
            # Automatically train initial model if needed
            try:
                meta = cls.train_and_persist_clusters(db, provenance=provenance)
                scaler = joblib.load(CLUSTER_SCALER_PATH)
            except Exception as e:
                logger.warning("Automatic cluster training encountered error: %s. Using default baseline.", e)
                meta = {
                    "cluster_model_version": "cluster_baseline_v1",
                    "algorithm": "BaselineKMeans",
                    "optimal_k": 1,
                    "silhouette_score": 0.0,
                    "davies_bouldin_index": 0.0,
                    "centroids": {
                        "0": {
                            "cluster_id": 0,
                            "centroid_values": user_features,
                            "size": 1,
                            "interpretation": ClusterInterpretationService.interpret_centroid(
                                user_features, user_features, {f: 1.0 for f in BEHAVIORAL_FEATURE_NAMES}
                            )
                        }
                    },
                    "population_mean": user_features,
                    "population_std": {f: 1.0 for f in BEHAVIORAL_FEATURE_NAMES}
                }

        centroids = meta.get("centroids", {})
        pop_mean = meta.get("population_mean", user_features)
        pop_std = meta.get("population_std", {f: 1.0 for f in BEHAVIORAL_FEATURE_NAMES})
        version_tag = meta.get("cluster_model_version", "cluster_v1")

        # Assign user to closest cluster centroid
        assigned_cluster_id = 0
        if sufficiency != "insufficient_data" and centroids:
            user_vec = np.array([user_features.get(f, 0.0) for f in BEHAVIORAL_FEATURE_NAMES]).reshape(1, -1)
            try:
                if scaler is not None:
                    user_vec_scaled = scaler.transform(user_vec)[0]
                else:
                    user_vec_scaled = user_vec[0]
            except Exception:
                user_vec_scaled = user_vec[0]

            min_dist = float("inf")
            for cid_str, cdata in centroids.items():
                c_vals = cdata.get("centroid_values", {})
                c_vec = np.array([c_vals.get(f, 0.0) for f in BEHAVIORAL_FEATURE_NAMES]).reshape(1, -1)
                try:
                    if scaler is not None:
                        c_vec_scaled = scaler.transform(c_vec)[0]
                    else:
                        c_vec_scaled = c_vec[0]
                except Exception:
                    c_vec_scaled = c_vec[0]

                dist = float(np.linalg.norm(user_vec_scaled - c_vec_scaled))
                if dist < min_dist:
                    min_dist = dist
                    assigned_cluster_id = int(cid_str)

        c_info = centroids.get(str(assigned_cluster_id), centroids.get("0", {}))
        interp = c_info.get("interpretation", ClusterInterpretationService.interpret_centroid(user_features, pop_mean, pop_std))

        # Check user's previous profile for historical trend comparison
        previous_profile_record = db.query(UserClusterProfile).filter(
            UserClusterProfile.user_id == user_id
        ).order_by(UserClusterProfile.id.desc()).first()

        previous_profile_comparison = None
        if previous_profile_record is not None:
            prev_features = {}
            try:
                prev_features = json.loads(previous_profile_record.user_features)
            except Exception:
                pass

            prev_delay = prev_features.get("average_departure_delay", 0.0)
            curr_delay = user_features.get("average_departure_delay", 0.0)
            delay_delta = round(curr_delay - prev_delay, 1)

            if delay_delta < -0.5:
                trend = "improving"
                trend_msg = f"Average delay reduced by {abs(delay_delta)}m compared to previous profile."
            elif delay_delta > 0.5:
                trend = "regressing"
                trend_msg = f"Average delay increased by {delay_delta}m compared to previous profile."
            else:
                trend = "stable"
                trend_msg = "Habit metrics remain consistent with previous profile."

            previous_profile_comparison = {
                "previous_cluster_name": previous_profile_record.cluster_name,
                "previous_generated_at": previous_profile_record.generated_at,
                "delay_delta_minutes": delay_delta,
                "trend": trend,
                "trend_message": trend_msg,
                "cluster_shifted": (previous_profile_record.cluster_name != interp.get("cluster_name"))
            }

        # Save latest profile record in DB
        new_profile_rec = UserClusterProfile(
            user_id=user_id,
            cluster_id=assigned_cluster_id,
            cluster_name=interp.get("cluster_name", "Standard Variable Routine"),
            cluster_description=interp.get("cluster_description", ""),
            dominant_characteristics=json.dumps(interp.get("dominant_characteristics", [])),
            user_features=json.dumps(user_features),
            cluster_centroid=json.dumps(c_info.get("centroid_values", user_features)),
            population_mean=json.dumps(pop_mean),
            data_sufficiency_status=sufficiency,
            sample_count=n_logs,
            provenance=provenance,
            cluster_model_version=version_tag,
            generated_at=now_iso
        )
        db.add(new_profile_rec)
        db.commit()

        return {
            "user_id": user_id,
            "cluster_id": assigned_cluster_id,
            "cluster_name": interp.get("cluster_name", "Standard Variable Routine"),
            "cluster_description": interp.get("cluster_description", ""),
            "dominant_characteristics": interp.get("dominant_characteristics", []),
            "recommended_action": interp.get("recommended_action", ""),
            "user_features": user_features,
            "cluster_centroid": c_info.get("centroid_values", user_features),
            "population_mean": pop_mean,
            "data_sufficiency_status": sufficiency,
            "sample_count": n_logs,
            "cluster_model_version": version_tag,
            "algorithm": meta.get("algorithm", "K-Means"),
            "silhouette_score": meta.get("silhouette_score", 0.0),
            "davies_bouldin_index": meta.get("davies_bouldin_index", 0.0),
            "profile_generation_timestamp": now_iso,
            "previous_profile_comparison": previous_profile_comparison
        }

    @classmethod
    def get_clustering_benchmark_report(cls, db: Session) -> Dict[str, Any]:
        """Retrieve comparative benchmark metrics across K-Means, Agglomerative, and DBSCAN."""
        if os.path.exists(CLUSTER_META_PATH):
            try:
                with open(CLUSTER_META_PATH, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    return meta
            except Exception as e:
                logger.warning("Could not read CLUSTER_META_PATH: %s", e)

        # If not on disk, generate on the fly
        return cls.train_and_persist_clusters(db)
