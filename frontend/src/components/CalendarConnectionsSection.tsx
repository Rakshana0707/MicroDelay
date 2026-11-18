import React, { useState, useEffect } from 'react';
import {
  Calendar,
  RefreshCw,
  Link2,
  Unlink,
  AlertTriangle,
  Clock,
  MapPin,
  Video,
  Compass,
  Sparkles,
  ShieldCheck
} from 'lucide-react';
import type { CalendarConnectionData, SynchronizedCalendarEvent, PlanDepartureResult } from '../types/calendar';

interface RoutineOption {
  id: number;
  name: string;
  day_of_week: string;
  start_time: string;
}

interface CalendarConnectionsSectionProps {
  token: string;
  routines: RoutineOption[];
  onOpenMorningTracker?: (eventName: string) => void;
}

export const CalendarConnectionsSection: React.FC<CalendarConnectionsSectionProps> = ({
  token,
  routines,
  onOpenMorningTracker
}) => {
  const [connections, setConnections] = useState<CalendarConnectionData[]>([]);
  const [events, setEvents] = useState<SynchronizedCalendarEvent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [syncingConnectionId, setSyncingConnectionId] = useState<number | null>(null);
  const [filterMode, setFilterMode] = useState<'departure' | 'all' | 'online' | 'missing_location'>('departure');
  const [selectedPlan, setSelectedPlan] = useState<PlanDepartureResult | null>(null);
  const [planningEventId, setPlanningEventId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [editingLocationId, setEditingLocationId] = useState<number | null>(null);
  const [locationInput, setLocationInput] = useState<string>('');

  const fetchConnections = async () => {
    try {
      const resp = await fetch('/api/calendar/connections', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        const data = await resp.json();
        setConnections(data.connections || []);
      }
    } catch (err) {
      console.error('Error fetching calendar connections:', err);
    }
  };

  const fetchEvents = async () => {
    try {
      const resp = await fetch('/api/calendar/events?limit=100', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        const data = await resp.json();
        setEvents(data.events || []);
      }
    } catch (err) {
      console.error('Error fetching calendar events:', err);
    }
  };

  const loadData = async () => {
    setLoading(true);
    await Promise.all([fetchConnections(), fetchEvents()]);
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, [token]);

  const handleConnectGoogle = async () => {
    setActionError(null);
    try {
      sessionStorage.setItem('microdelay_oauth_provider', 'google');
      const resp = await fetch('/api/calendar/google/auth-url', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.auth_url) {
          window.location.href = data.auth_url;
        }
      } else {
        setActionError('Failed to initiate Google OAuth. Please check client credentials in backend.');
      }
    } catch (err: any) {
      setActionError(err.message || 'Network error initiating Google OAuth.');
    }
  };

  const handleConnectOutlook = async () => {
    setActionError(null);
    try {
      sessionStorage.setItem('microdelay_oauth_provider', 'outlook');
      const resp = await fetch('/api/calendar/outlook/auth-url', {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        const data = await resp.json();
        if (data.auth_url) {
          window.location.href = data.auth_url;
        }
      } else {
        setActionError('Failed to initiate Microsoft Outlook OAuth. Please check client credentials in backend.');
      }
    } catch (err: any) {
      setActionError(err.message || 'Network error initiating Outlook OAuth.');
    }
  };

  const handleSync = async (connectionId: number) => {
    setSyncingConnectionId(connectionId);
    setActionError(null);
    try {
      const resp = await fetch(`/api/calendar/connections/${connectionId}/sync`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        await loadData();
      } else {
        const data = await resp.json();
        setActionError(data.detail || 'Synchronization failed.');
      }
    } catch (err: any) {
      setActionError(err.message || 'Error communicating with synchronization service.');
    } finally {
      setSyncingConnectionId(null);
    }
  };

  const handleDisconnect = async (connectionId: number, providerName: string) => {
    if (!window.confirm(`Disconnect ${providerName} Calendar? All synchronized events and authorization tokens will be revoked.`)) {
      return;
    }
    setActionError(null);
    try {
      const resp = await fetch(`/api/calendar/connections/${connectionId}/disconnect`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        await loadData();
      } else {
        setActionError(`Failed to disconnect ${providerName} Calendar.`);
      }
    } catch (err: any) {
      setActionError(err.message || 'Error disconnecting calendar.');
    }
  };

  const handleClassify = async (eventId: number, req: string) => {
    try {
      const resp = await fetch(`/api/calendar/events/${eventId}/classify`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ travel_requirement: req })
      });
      if (resp.ok) {
        await fetchEvents();
      }
    } catch (err) {
      console.error('Classification error:', err);
    }
  };

  const handleAssociateRoutine = async (eventId: number, routineId: number | null) => {
    try {
      const resp = await fetch(`/api/calendar/events/${eventId}/classify`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ associated_routine_id: routineId || 0 })
      });
      if (resp.ok) {
        await fetchEvents();
      }
    } catch (err) {
      console.error('Routine association error:', err);
    }
  };

  const handleSaveLocation = async (eventId: number) => {
    try {
      const resp = await fetch(`/api/calendar/events/${eventId}/classify`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${token}`
        },
        body: JSON.stringify({ location: locationInput })
      });
      if (resp.ok) {
        setEditingLocationId(null);
        setLocationInput('');
        await fetchEvents();
      }
    } catch (err) {
      console.error('Location update error:', err);
    }
  };

  const handlePlanDeparture = async (eventId: number) => {
    setPlanningEventId(eventId);
    setActionError(null);
    try {
      const resp = await fetch(`/api/calendar/events/${eventId}/plan-departure`, {
        method: 'POST',
        headers: { Authorization: `Bearer ${token}` }
      });
      if (resp.ok) {
        const data = await resp.json();
        setSelectedPlan(data);
      } else {
        const err = await resp.json();
        setActionError(err.detail || 'Could not optimize departure.');
      }
    } catch (err: any) {
      setActionError(err.message || 'Departure optimization failed.');
    } finally {
      setPlanningEventId(null);
    }
  };

  // Filter events according to active tab
  const filteredEvents = events.filter((ev) => {
    if (filterMode === 'departure') return ev.requires_departure_planning;
    if (filterMode === 'online') return ev.travel_requirement === 'online';
    if (filterMode === 'missing_location') return ev.missing_location_warning;
    return true;
  });

  const googleConn = connections.find((c) => c.provider === 'google' && c.connection_status === 'connected');
  const outlookConn = connections.find((c) => c.provider === 'outlook' && c.connection_status === 'connected');

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      {/* Header & Unified Calendar Connection Cards */}
      <div className="card" style={{ padding: '1.5rem' }}>
        <div style={{ marginBottom: '1.25rem' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
            <Calendar size={26} style={{ color: 'var(--accent-primary)' }} />
            <h2 style={{ margin: 0, fontSize: '1.4rem', fontWeight: 800 }}>Calendar Connections</h2>
          </div>
          <p style={{ margin: '0.4rem 0 0 0', color: 'var(--text-muted)', fontSize: '0.92rem' }}>
            Connect Google Calendar and Microsoft Outlook to automatically identify departure requirements. All tokens are encrypted at rest with minimum read-only permissions.
          </p>
        </div>

        {/* Dual Provider Status Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1rem' }}>
          {/* Google Calendar Card */}
          <div
            style={{
              padding: '1.15rem',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '1rem'
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '26px',
                      height: '26px',
                      borderRadius: '6px',
                      backgroundColor: 'rgba(234, 67, 53, 0.15)',
                      color: '#EA4335',
                      fontWeight: 800,
                      fontSize: '0.85rem'
                    }}
                  >
                    G
                  </span>
                  <span style={{ fontWeight: 700, fontSize: '1rem' }}>Google Calendar</span>
                </div>

                {googleConn ? (
                  <span
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '12px',
                      backgroundColor: 'rgba(16, 185, 129, 0.15)',
                      color: 'var(--accent-success)'
                    }}
                  >
                    ● Connected
                  </span>
                ) : (
                  <span
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '12px',
                      backgroundColor: 'rgba(107, 114, 128, 0.15)',
                      color: 'var(--text-muted)'
                    }}
                  >
                    Not Connected
                  </span>
                )}
              </div>

              {googleConn ? (
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                    {googleConn.provider_account_reference || 'Connected Google Account'}
                  </div>
                  <div>
                    Last Synced:{' '}
                    {googleConn.last_synced_at
                      ? new Date(googleConn.last_synced_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                      : 'Never'}
                  </div>
                </div>
              ) : (
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Connect your Google workspace or personal account with OAuth 2.0 PKCE.
                </p>
              )}
            </div>

            <div>
              {googleConn ? (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="btn btn-outline btn-sm"
                    onClick={() => handleSync(googleConn.id)}
                    disabled={syncingConnectionId === googleConn.id}
                    style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flex: 1, justifyContent: 'center' }}
                  >
                    <RefreshCw size={14} className={syncingConnectionId === googleConn.id ? 'spin' : ''} />
                    {syncingConnectionId === googleConn.id ? 'Syncing...' : 'Sync Now'}
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDisconnect(googleConn.id, 'Google')}
                    style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}
                  >
                    <Unlink size={14} />
                    Disconnect
                  </button>
                </div>
              ) : (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={handleConnectGoogle}
                  style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', fontWeight: 600 }}
                >
                  <Link2 size={15} />
                  Connect Google Calendar
                </button>
              )}
            </div>
          </div>

          {/* Microsoft Outlook Calendar Card */}
          <div
            style={{
              padding: '1.15rem',
              backgroundColor: 'var(--bg-secondary)',
              borderRadius: '8px',
              border: '1px solid var(--border-color)',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'space-between',
              gap: '1rem'
            }}
          >
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <span
                    style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      width: '26px',
                      height: '26px',
                      borderRadius: '6px',
                      backgroundColor: 'rgba(0, 120, 212, 0.15)',
                      color: '#0078D4',
                      fontWeight: 800,
                      fontSize: '0.85rem'
                    }}
                  >
                    O
                  </span>
                  <span style={{ fontWeight: 700, fontSize: '1rem' }}>Microsoft Outlook</span>
                </div>

                {outlookConn ? (
                  <span
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '12px',
                      backgroundColor: 'rgba(16, 185, 129, 0.15)',
                      color: 'var(--accent-success)'
                    }}
                  >
                    ● Connected
                  </span>
                ) : (
                  <span
                    style={{
                      fontSize: '0.75rem',
                      fontWeight: 600,
                      padding: '2px 8px',
                      borderRadius: '12px',
                      backgroundColor: 'rgba(107, 114, 128, 0.15)',
                      color: 'var(--text-muted)'
                    }}
                  >
                    Not Connected
                  </span>
                )}
              </div>

              {outlookConn ? (
                <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem' }}>
                    {outlookConn.provider_account_reference || 'Connected Outlook Account'}
                  </div>
                  <div>
                    Last Synced:{' '}
                    {outlookConn.last_synced_at
                      ? new Date(outlookConn.last_synced_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                      : 'Never'}
                  </div>
                </div>
              ) : (
                <p style={{ margin: 0, fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                  Connect your Microsoft 365 or Outlook account via Microsoft Graph API with delta queries.
                </p>
              )}
            </div>

            <div>
              {outlookConn ? (
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                  <button
                    className="btn btn-outline btn-sm"
                    onClick={() => handleSync(outlookConn.id)}
                    disabled={syncingConnectionId === outlookConn.id}
                    style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', flex: 1, justifyContent: 'center' }}
                  >
                    <RefreshCw size={14} className={syncingConnectionId === outlookConn.id ? 'spin' : ''} />
                    {syncingConnectionId === outlookConn.id ? 'Syncing...' : 'Sync Now'}
                  </button>
                  <button
                    className="btn btn-danger btn-sm"
                    onClick={() => handleDisconnect(outlookConn.id, 'Outlook')}
                    style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}
                  >
                    <Unlink size={14} />
                    Disconnect
                  </button>
                </div>
              ) : (
                <button
                  className="btn btn-primary btn-sm"
                  onClick={handleConnectOutlook}
                  style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.4rem', fontWeight: 600, backgroundColor: '#0078D4', borderColor: '#0078D4' }}
                >
                  <Link2 size={15} />
                  Connect Outlook Calendar
                </button>
              )}
            </div>
          </div>
        </div>

        {/* Global Security Pill */}
        <div style={{ marginTop: '1rem', display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.82rem', color: 'var(--accent-success)' }}>
          <ShieldCheck size={15} />
          <span>AES-128-CBC + HMAC-SHA256 authenticated encryption at rest. Scopes restricted to delegated read-only access.</span>
        </div>

        {actionError && (
          <div
            style={{
              marginTop: '1rem',
              padding: '0.75rem 1rem',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              borderLeft: '4px solid var(--accent-danger)',
              borderRadius: '4px',
              color: 'var(--accent-danger)',
              fontSize: '0.9rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem'
            }}
          >
            <AlertTriangle size={16} />
            {actionError}
          </div>
        )}
      </div>

      {/* Summary KPI Cards */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
        <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Synchronized Events
          </div>
          <div style={{ fontSize: '1.7rem', fontWeight: 800, marginTop: '0.25rem', color: 'var(--text-primary)' }}>
            {events.length}
          </div>
        </div>

        <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Requires Departure Planning
          </div>
          <div style={{ fontSize: '1.7rem', fontWeight: 800, marginTop: '0.25rem', color: 'var(--accent-primary)' }}>
            {events.filter((e) => e.requires_departure_planning).length}
          </div>
        </div>

        <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Online Meetings
          </div>
          <div style={{ fontSize: '1.7rem', fontWeight: 800, marginTop: '0.25rem', color: 'var(--accent-secondary, #8b5cf6)' }}>
            {events.filter((e) => e.travel_requirement === 'online').length}
          </div>
        </div>

        <div className="card" style={{ padding: '1rem', textAlign: 'center' }}>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
            Missing Location
          </div>
          <div
            style={{
              fontSize: '1.7rem',
              fontWeight: 800,
              marginTop: '0.25rem',
              color: events.filter((e) => e.missing_location_warning).length > 0 ? 'var(--accent-warning)' : 'var(--text-muted)'
            }}
          >
            {events.filter((e) => e.missing_location_warning).length}
          </div>
        </div>
      </div>

      {/* Unified Synchronized Events Schedule */}
      <div className="card" style={{ padding: '1.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.75rem' }}>
          <h3 style={{ margin: 0, fontSize: '1.15rem', fontWeight: 700 }}>Upcoming Synchronized Schedule</h3>

          {/* Filter Pills */}
          <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
            <button
              className={`btn btn-sm ${filterMode === 'departure' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setFilterMode('departure')}
            >
              Departure Required ({events.filter((e) => e.requires_departure_planning).length})
            </button>
            <button
              className={`btn btn-sm ${filterMode === 'all' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setFilterMode('all')}
            >
              All Events ({events.length})
            </button>
            <button
              className={`btn btn-sm ${filterMode === 'online' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setFilterMode('online')}
            >
              Online ({events.filter((e) => e.travel_requirement === 'online').length})
            </button>
            <button
              className={`btn btn-sm ${filterMode === 'missing_location' ? 'btn-primary' : 'btn-outline'}`}
              onClick={() => setFilterMode('missing_location')}
            >
              Needs Location ({events.filter((e) => e.missing_location_warning).length})
            </button>
          </div>
        </div>

        {loading ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <RefreshCw size={28} className="spin" style={{ marginBottom: '0.5rem' }} />
            <p>Loading calendar events...</p>
          </div>
        ) : filteredEvents.length === 0 ? (
          <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
            <Calendar size={36} style={{ opacity: 0.4, marginBottom: '0.75rem' }} />
            <p style={{ margin: 0, fontWeight: 500 }}>No events matching the active filter.</p>
            {(googleConn || outlookConn) && (
              <p style={{ fontSize: '0.85rem', marginTop: '0.4rem' }}>
                Try clicking "Sync Now" on your connected calendars to refresh.
              </p>
            )}
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {filteredEvents.map((ev) => {
              const startDt = new Date(ev.start_time);
              const endDt = new Date(ev.end_time);
              const dateStr = startDt.toLocaleDateString([], { weekday: 'short', month: 'short', day: 'numeric' });
              const timeStr = `${startDt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} - ${endDt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
              const isOutlook = ev.provider === 'outlook';

              return (
                <div
                  key={ev.id}
                  style={{
                    padding: '1.1rem',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: '8px',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.75rem'
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '0.75rem' }}>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {/* Provider Brand Badge */}
                        <span
                          style={{
                            fontSize: '0.72rem',
                            padding: '2px 7px',
                            borderRadius: '10px',
                            backgroundColor: isOutlook ? 'rgba(0, 120, 212, 0.15)' : 'rgba(234, 67, 53, 0.12)',
                            color: isOutlook ? '#0078D4' : '#EA4335',
                            fontWeight: 700,
                            border: `1px solid ${isOutlook ? 'rgba(0, 120, 212, 0.3)' : 'rgba(234, 67, 53, 0.3)'}`
                          }}
                        >
                          {isOutlook ? 'Outlook' : 'Google'}
                        </span>

                        <span style={{ fontWeight: 700, fontSize: '1.05rem', color: 'var(--text-primary)' }}>
                          {ev.title}
                        </span>

                        {/* Status Badges */}
                        {ev.travel_requirement === 'physical' && (
                          <span
                            style={{
                              fontSize: '0.75rem',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              backgroundColor: 'rgba(37, 99, 235, 0.15)',
                              color: 'var(--accent-primary)',
                              fontWeight: 600,
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '3px'
                            }}
                          >
                            <Compass size={12} /> Physical Attendance
                          </span>
                        )}

                        {ev.travel_requirement === 'online' && (
                          <span
                            style={{
                              fontSize: '0.75rem',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              backgroundColor: 'rgba(139, 92, 246, 0.15)',
                              color: '#8b5cf6',
                              fontWeight: 600,
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '3px'
                            }}
                          >
                            <Video size={12} /> Online Meeting
                          </span>
                        )}

                        {ev.travel_requirement === 'personal' && (
                          <span
                            style={{
                              fontSize: '0.75rem',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              backgroundColor: 'rgba(107, 114, 128, 0.15)',
                              color: 'var(--text-muted)',
                              fontWeight: 600
                            }}
                          >
                            Personal Block
                          </span>
                        )}

                        {ev.missing_location_warning && (
                          <span
                            style={{
                              fontSize: '0.75rem',
                              padding: '2px 8px',
                              borderRadius: '12px',
                              backgroundColor: 'rgba(245, 158, 11, 0.15)',
                              color: 'var(--accent-warning)',
                              fontWeight: 600,
                              display: 'inline-flex',
                              alignItems: 'center',
                              gap: '3px'
                            }}
                          >
                            <AlertTriangle size={12} /> Missing Location
                          </span>
                        )}
                      </div>

                      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                          <Clock size={14} />
                          {dateStr} • {timeStr}
                        </div>

                        {ev.location && (
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: 'var(--text-primary)' }}>
                            <MapPin size={14} style={{ color: 'var(--accent-primary)' }} />
                            {ev.location}
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Actions & Plan Button */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      {ev.requires_departure_planning && (
                        <button
                          className="btn btn-primary btn-sm"
                          onClick={() => handlePlanDeparture(ev.id)}
                          disabled={planningEventId === ev.id}
                          style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontWeight: 600 }}
                        >
                          <Sparkles size={14} />
                          {planningEventId === ev.id ? 'Optimizing...' : 'Plan Departure'}
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Inline Location Prompt for Missing Location */}
                  {ev.missing_location_warning && (
                    <div
                      style={{
                        padding: '0.6rem 0.8rem',
                        backgroundColor: 'rgba(245, 158, 11, 0.08)',
                        borderRadius: '6px',
                        border: '1px dashed var(--accent-warning)',
                        fontSize: '0.85rem',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'space-between',
                        flexWrap: 'wrap',
                        gap: '0.5rem'
                      }}
                    >
                      {editingLocationId === ev.id ? (
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%' }}>
                          <input
                            type="text"
                            placeholder="Enter physical destination address..."
                            value={locationInput}
                            onChange={(e) => setLocationInput(e.target.value)}
                            style={{
                              flex: 1,
                              padding: '0.35rem 0.6rem',
                              borderRadius: '4px',
                              border: '1px solid var(--border-color)',
                              background: 'var(--bg-primary)',
                              color: 'var(--text-primary)',
                              fontSize: '0.85rem'
                            }}
                          />
                          <button className="btn btn-primary btn-sm" onClick={() => handleSaveLocation(ev.id)}>
                            Save
                          </button>
                          <button className="btn btn-outline btn-sm" onClick={() => setEditingLocationId(null)}>
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--accent-warning)' }}>
                            <AlertTriangle size={15} />
                            <span>Destination location missing. Travel duration cannot be calculated without an address.</span>
                          </div>
                          <button
                            className="btn btn-outline btn-sm"
                            onClick={() => {
                              setEditingLocationId(ev.id);
                              setLocationInput('');
                            }}
                          >
                            + Set Destination
                          </button>
                        </>
                      )}
                    </div>
                  )}

                  {/* Classification Controls & Routine Association */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      flexWrap: 'wrap',
                      gap: '0.75rem',
                      fontSize: '0.82rem',
                      paddingTop: '0.5rem',
                      borderTop: '1px solid rgba(255, 255, 255, 0.05)'
                    }}
                  >
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
                      <span style={{ color: 'var(--text-muted)' }}>Classify As:</span>
                      <select
                        value={ev.travel_requirement}
                        onChange={(e) => handleClassify(ev.id, e.target.value)}
                        style={{
                          padding: '0.25rem 0.5rem',
                          borderRadius: '4px',
                          border: '1px solid var(--border-color)',
                          background: 'var(--bg-primary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.82rem'
                        }}
                      >
                        <option value="physical">Physical Attendance</option>
                        <option value="online">Online Meeting</option>
                        <option value="personal">Personal Block</option>
                        <option value="excluded">Excluded from MicroDelay</option>
                      </select>

                      <span style={{ color: 'var(--text-muted)' }}>Associate Routine:</span>
                      <select
                        value={ev.associated_routine_id || 0}
                        onChange={(e) => handleAssociateRoutine(ev.id, parseInt(e.target.value, 10))}
                        style={{
                          padding: '0.25rem 0.5rem',
                          borderRadius: '4px',
                          border: '1px solid var(--border-color)',
                          background: 'var(--bg-primary)',
                          color: 'var(--text-primary)',
                          fontSize: '0.82rem'
                        }}
                      >
                        <option value={0}>None (Use Event Times)</option>
                        {routines.map((r) => (
                          <option key={r.id} value={r.id}>
                            {r.name} ({r.day_of_week} at {r.start_time})
                          </option>
                        ))}
                      </select>
                    </div>

                    {ev.online_meeting_url && (
                      <a
                        href={ev.online_meeting_url}
                        target="_blank"
                        rel="noreferrer"
                        style={{ display: 'flex', alignItems: 'center', gap: '0.3rem', color: '#8b5cf6', textDecoration: 'none' }}
                      >
                        <Video size={13} />
                        Join Virtual Meeting
                      </a>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* Optimized Departure Plan Modal */}
      {selectedPlan && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '1.5rem'
          }}
        >
          <div
            className="card"
            style={{
              width: '100%',
              maxWidth: '650px',
              maxHeight: '90vh',
              overflowY: 'auto',
              padding: '1.75rem',
              backgroundColor: 'var(--bg-primary)',
              borderRadius: '12px',
              border: '1px solid var(--border-color)'
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem' }}>
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Sparkles size={22} style={{ color: 'var(--accent-primary)' }} />
                  <h3 style={{ margin: 0, fontSize: '1.25rem', fontWeight: 800 }}>
                    Optimized Departure Plan
                  </h3>
                </div>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.88rem', marginTop: '0.2rem' }}>
                  {selectedPlan.event_name}
                </div>
              </div>

              <button
                className="btn btn-outline btn-sm"
                onClick={() => setSelectedPlan(null)}
                style={{ borderRadius: '50%', width: '32px', height: '32px', padding: 0 }}
              >
                ✕
              </button>
            </div>

            {/* Timeline Milestones Highlight */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                gap: '0.75rem',
                marginBottom: '1.5rem',
                backgroundColor: 'var(--bg-secondary)',
                padding: '1rem',
                borderRadius: '8px'
              }}
            >
              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>PREP START</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--accent-primary)' }}>
                  {new Date(selectedPlan.optimization.recommended_prep_start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>DEPARTURE</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--accent-warning)' }}>
                  {new Date(selectedPlan.optimization.recommended_departure_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>TARGET ARRIVAL</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--accent-success)' }}>
                  {new Date(selectedPlan.optimization.target_arrival_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>

              <div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>EVENT START</div>
                <div style={{ fontSize: '1.15rem', fontWeight: 800, color: 'var(--text-primary)' }}>
                  {new Date(selectedPlan.optimization.event_start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>

            {/* Timeline Breakdown */}
            <h4 style={{ margin: '0 0 0.75rem 0', fontSize: '0.95rem', fontWeight: 700 }}>Departure Timeline Breakdown</h4>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginBottom: '1.5rem' }}>
              {selectedPlan.optimization.timeline.map((step, idx) => (
                <div
                  key={idx}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    padding: '0.65rem 0.9rem',
                    backgroundColor: 'var(--bg-secondary)',
                    borderRadius: '6px',
                    fontSize: '0.85rem'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                    <div
                      style={{
                        width: '8px',
                        height: '8px',
                        borderRadius: '50%',
                        backgroundColor: step.phase.includes('prep')
                          ? 'var(--accent-primary)'
                          : step.phase.includes('travel')
                          ? 'var(--accent-warning)'
                          : 'var(--accent-success)'
                      }}
                    />
                    <span style={{ fontWeight: 600 }}>{step.description}</span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', color: 'var(--text-muted)' }}>
                    <span>{step.duration_min} min</span>
                    <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>
                      {new Date(step.start_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })} -{' '}
                      {new Date(step.end_time).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Context & Optimization Notes */}
            {selectedPlan.optimization.optimization_notes && selectedPlan.optimization.optimization_notes.length > 0 && (
              <div
                style={{
                  padding: '0.75rem 1rem',
                  backgroundColor: 'rgba(59, 130, 246, 0.08)',
                  borderRadius: '6px',
                  border: '1px solid rgba(59, 130, 246, 0.2)',
                  fontSize: '0.82rem',
                  color: 'var(--text-muted)',
                  marginBottom: '1.25rem'
                }}
              >
                <div style={{ fontWeight: 700, color: 'var(--accent-primary)', marginBottom: '0.3rem' }}>
                  Intelligence Context Notes
                </div>
                <ul style={{ margin: 0, paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                  {selectedPlan.optimization.optimization_notes.map((note, idx) => (
                    <li key={idx}>{note}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Actions */}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '0.75rem' }}>
              <button className="btn btn-outline" onClick={() => setSelectedPlan(null)}>
                Dismiss
              </button>
              {onOpenMorningTracker && (
                <button
                  className="btn btn-primary"
                  onClick={() => {
                    onOpenMorningTracker(selectedPlan.event_name);
                    setSelectedPlan(null);
                  }}
                >
                  Track in Morning Simulator →
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
