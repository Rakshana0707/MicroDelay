export type TravelMode = 'driving' | 'transit' | 'walking' | 'bicycling';

export interface UserSavedLocation {
  id: number;
  user_id: number;
  name: string;
  category: 'home' | 'work' | 'education' | 'frequent' | 'custom';
  address: string;
  latitude?: number;
  longitude?: number;
  is_primary: boolean;
  created_at: string;
  updated_at: string;
}

export interface RouteOption {
  route_id: string;
  summary: string;
  origin: string;
  destination: string;
  travel_mode: TravelMode;
  duration_min: number;
  duration_in_traffic_min?: number;
  distance_km: number;
  retrieval_timestamp: string;
  provider: string;
  is_recommended: boolean;
  warnings?: string[];
  polyline?: string;
}

export interface UserSavedRoute {
  id: number;
  user_id: number;
  name: string;
  origin_address: string;
  origin_lat?: number;
  origin_lng?: number;
  destination_address: string;
  destination_lat?: number;
  destination_lng?: number;
  travel_mode: TravelMode;
  preferred_route_identifier?: string;
  distance_km?: number;
  estimated_duration_min?: number;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface UserTravelPreferences {
  user_id: number;
  default_mode: TravelMode;
  avoid_tolls: boolean;
  avoid_highways: boolean;
  buffer_multiplier: number;
  updated_at: string;
}

export interface ModeComparisonItem {
  available: boolean;
  duration_min?: number;
  duration_in_traffic_min?: number;
  distance_km?: number;
  summary?: string;
  reason?: string;
}

export interface ModeComparisonResult {
  origin: string;
  destination: string;
  comparison: Record<TravelMode, ModeComparisonItem>;
}

export interface TravelReliabilityMetrics {
  total_journeys: number;
  mean_estimated_min: number;
  mean_actual_min: number;
  mean_travel_bias_min: number;
  travel_bias_ratio: number;
  mae_min: number;
  on_time_travel_pct: number;
  mode_breakdown: Record<string, number>;
  insights: string[];
}
