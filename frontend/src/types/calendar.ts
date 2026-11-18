// TypeScript interfaces for Google Calendar Integration (Prompt 3.2)

export interface CalendarConnectionData {
  id: number;
  provider: string;
  provider_account_reference?: string;
  connection_status: 'connected' | 'expired' | 'revoked' | 'disconnected';
  connected_at: string;
  last_synced_at?: string;
  synced_event_count: number;
  departure_event_count: number;
}

export interface SynchronizedCalendarEvent {
  id: number;
  provider_event_id: string;
  connection_id: number;
  provider?: string;
  title: string;
  start_time: string;
  end_time: string;
  location?: string | null;
  calendar_timezone?: string | null;
  is_all_day: boolean;
  status: string;
  travel_requirement: 'physical' | 'online' | 'personal' | 'excluded' | 'unknown';
  requires_departure_planning: boolean;
  missing_location_warning: boolean;
  online_meeting_url?: string | null;
  associated_routine_id?: number | null;
  prep_buffer_override_min?: number | null;
  travel_mode_override?: string | null;
  synced_at: string;
}

export interface PlanDepartureResult {
  status: string;
  event_id: number;
  event_name: string;
  context: any;
  optimization: {
    recommended_prep_start_time: string;
    recommended_departure_time: string;
    target_arrival_time: string;
    event_start_time: string;
    total_lead_time_min: number;
    prep_duration_min: number;
    travel_duration_min: number;
    safety_buffer_min: number;
    timeline: Array<{
      phase: string;
      start_time: string;
      end_time: string;
      duration_min: number;
      description: string;
    }>;
    optimization_notes: string[];
  };
}
