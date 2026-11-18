import React, { useState, useEffect } from 'react';
import type {
  UserSavedLocation,
  UserSavedRoute,
  UserTravelPreferences,
  RouteOption,
  TravelReliabilityMetrics,
  ModeComparisonResult,
  TravelMode
} from '../types/travel';

interface TravelIntelligenceSectionProps {
  token: string;
}

export const TravelIntelligenceSection: React.FC<TravelIntelligenceSectionProps> = ({ token }) => {
  // State: Saved Locations
  const [locations, setLocations] = useState<UserSavedLocation[]>([]);
  const [loadingLocations, setLoadingLocations] = useState<boolean>(false);
  const [newLocName, setNewLocName] = useState<string>('');
  const [newLocCategory, setNewLocCategory] = useState<'home' | 'work' | 'education' | 'frequent' | 'custom'>('home');
  const [newLocAddress, setNewLocAddress] = useState<string>('');
  const [newLocPrimary, setNewLocPrimary] = useState<boolean>(false);

  // State: Route Explorer
  const [origin, setOrigin] = useState<string>('');
  const [destination, setDestination] = useState<string>('');
  const [selectedMode, setSelectedMode] = useState<TravelMode>('driving');
  const [calculatedRoutes, setCalculatedRoutes] = useState<RouteOption[]>([]);
  const [personalizedInfo, setPersonalizedInfo] = useState<any>(null);
  const [loadingRoutes, setLoadingRoutes] = useState<boolean>(false);
  const [routeError, setRouteError] = useState<string | null>(null);

  // State: Mode Comparison
  const [modeComparison, setModeComparison] = useState<ModeComparisonResult | null>(null);
  const [loadingComparison, setLoadingComparison] = useState<boolean>(false);

  // State: Saved Routes
  const [savedRoutes, setSavedRoutes] = useState<UserSavedRoute[]>([]);
  const [loadingSavedRoutes, setLoadingSavedRoutes] = useState<boolean>(false);

  // State: Preferences
  const [preferences, setPreferences] = useState<UserTravelPreferences>({
    user_id: 0,
    default_mode: 'driving',
    avoid_tolls: false,
    avoid_highways: false,
    buffer_multiplier: 1.0,
    updated_at: ''
  });
  const [savingPrefs, setSavingPrefs] = useState<boolean>(false);
  const [prefNotice, setPrefNotice] = useState<string | null>(null);

  // State: Analytics
  const [analytics, setAnalytics] = useState<TravelReliabilityMetrics | null>(null);
  const [, setLoadingAnalytics] = useState<boolean>(false);

  const authHeaders = {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${token}`
  };

  // Fetch initial data
  useEffect(() => {
    fetchLocations();
    fetchSavedRoutes();
    fetchPreferences();
    fetchAnalytics();
  }, [token]);

  const fetchLocations = async () => {
    setLoadingLocations(true);
    try {
      const res = await fetch('/api/travel/locations', { headers: authHeaders });
      if (res.ok) {
        const data: UserSavedLocation[] = await res.json();
        setLocations(data);
        // Pre-fill origin with primary location if available
        const primary = data.find((l) => l.is_primary);
        if (primary && !origin) {
          setOrigin(primary.address);
        }
      }
    } catch (err) {
      console.error('Failed to load locations', err);
    } finally {
      setLoadingLocations(false);
    }
  };

  const fetchSavedRoutes = async () => {
    setLoadingSavedRoutes(true);
    try {
      const res = await fetch('/api/travel/saved-routes', { headers: authHeaders });
      if (res.ok) {
        const data: UserSavedRoute[] = await res.json();
        setSavedRoutes(data);
      }
    } catch (err) {
      console.error('Failed to load saved routes', err);
    } finally {
      setLoadingSavedRoutes(false);
    }
  };

  const fetchPreferences = async () => {
    try {
      const res = await fetch('/api/travel/preferences', { headers: authHeaders });
      if (res.ok) {
        const data: UserTravelPreferences = await res.json();
        setPreferences(data);
        setSelectedMode(data.default_mode);
      }
    } catch (err) {
      console.error('Failed to load preferences', err);
    }
  };

  const fetchAnalytics = async () => {
    setLoadingAnalytics(true);
    try {
      const res = await fetch('/api/travel/analytics', { headers: authHeaders });
      if (res.ok) {
        const data: TravelReliabilityMetrics = await res.json();
        setAnalytics(data);
      }
    } catch (err) {
      console.error('Failed to load analytics', err);
    } finally {
      setLoadingAnalytics(false);
    }
  };

  // Add location
  const handleAddLocation = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newLocName.trim() || !newLocAddress.trim()) return;

    try {
      const res = await fetch('/api/travel/locations', {
        method: 'POST',
        headers: authHeaders,
        body: JSON.stringify({
          name: newLocName.trim(),
          category: newLocCategory,
          address: newLocAddress.trim(),
          is_primary: newLocPrimary
        })
      });
      if (res.ok) {
        setNewLocName('');
        setNewLocAddress('');
        setNewLocPrimary(false);
        fetchLocations();
      }
    } catch (err) {
      console.error('Failed to add location', err);
    }
  };

  // Delete location
  const handleDeleteLocation = async (id: number) => {
    try {
      const res = await fetch(`/api/travel/locations/${id}`, {
        method: 'DELETE',
        headers: authHeaders
      });
      if (res.ok) {
        fetchLocations();
      }
    } catch (err) {
      console.error('Failed to delete location', err);
    }
  };

  // Calculate routes
  const handleCalculateRoutes = async () => {
    if (!origin.trim() || !destination.trim()) {
      setRouteError('Please enter both origin and destination addresses.');
      return;
    }
    setRouteError(null);
    setLoadingRoutes(true);

    try {
      const res = await fetch('/api/travel/routes', {
        method: 'POST',
        headers: authHeaders,
        body: JSON.stringify({
          origin: origin.trim(),
          destination: destination.trim(),
          mode: selectedMode,
          alternatives: true
        })
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setCalculatedRoutes(data.routes || []);
        setPersonalizedInfo(data.personalized_estimation || null);
      } else {
        setRouteError(data.detail || 'Failed to calculate routes.');
      }
    } catch (err: any) {
      setRouteError(err.message || 'Error communicating with routing engine.');
    } finally {
      setLoadingRoutes(false);
    }
  };

  // Compare modes
  const handleCompareModes = async () => {
    if (!origin.trim() || !destination.trim()) {
      setRouteError('Please enter both origin and destination addresses to compare travel modes.');
      return;
    }
    setRouteError(null);
    setLoadingComparison(true);

    try {
      const res = await fetch('/api/travel/compare-routes', {
        method: 'POST',
        headers: authHeaders,
        body: JSON.stringify({
          origin: origin.trim(),
          destination: destination.trim()
        })
      });
      const data = await res.json();
      if (res.ok && data.status === 'success') {
        setModeComparison(data);
      } else {
        setRouteError(data.detail || 'Mode comparison failed.');
      }
    } catch (err: any) {
      setRouteError(err.message || 'Error comparing modes.');
    } finally {
      setLoadingComparison(false);
    }
  };

  // Save calculated route as preferred
  const handleSaveRoute = async (route: RouteOption) => {
    try {
      const res = await fetch('/api/travel/saved-routes', {
        method: 'POST',
        headers: authHeaders,
        body: JSON.stringify({
          name: `${route.summary} (${route.travel_mode})`,
          origin_address: route.origin,
          destination_address: route.destination,
          travel_mode: route.travel_mode,
          preferred_route_identifier: route.summary,
          distance_km: route.distance_km,
          estimated_duration_min: route.duration_in_traffic_min || route.duration_min,
          notes: 'Saved from route explorer'
        })
      });
      if (res.ok) {
        fetchSavedRoutes();
      }
    } catch (err) {
      console.error('Failed to save route', err);
    }
  };

  // Delete saved route
  const handleDeleteSavedRoute = async (id: number) => {
    try {
      const res = await fetch(`/api/travel/saved-routes/${id}`, {
        method: 'DELETE',
        headers: authHeaders
      });
      if (res.ok) {
        fetchSavedRoutes();
      }
    } catch (err) {
      console.error('Failed to delete route', err);
    }
  };

  // Save preferences
  const handleSavePreferences = async () => {
    setSavingPrefs(true);
    setPrefNotice(null);
    try {
      const res = await fetch('/api/travel/preferences', {
        method: 'PUT',
        headers: authHeaders,
        body: JSON.stringify(preferences)
      });
      if (res.ok) {
        setPrefNotice('Travel preferences updated successfully!');
        setTimeout(() => setPrefNotice(null), 3500);
      }
    } catch (err) {
      console.error('Failed to save preferences', err);
    } finally {
      setSavingPrefs(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* SECTION 1: HEADER & ANALYTICS BANNER */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-md">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold text-white flex items-center gap-2">
              <span className="text-emerald-400">📍</span> Travel & Route Intelligence Engine
            </h2>
            <p className="text-slate-400 text-sm mt-1">
              Context-aware multi-modal routing, geocoded saved locations, and personal travel behavioral calibration.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="px-3 py-1 bg-emerald-500/10 text-emerald-400 text-xs font-semibold rounded-full border border-emerald-500/20">
              Prompt 3.4 Live
            </span>
            <span className="px-3 py-1 bg-cyan-500/10 text-cyan-400 text-xs font-semibold rounded-full border border-cyan-500/20">
              Phase 2 ML Integrated
            </span>
          </div>
        </div>

        {/* Travel Reliability Metrics */}
        {analytics && (
          <div className="mt-6 grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-slate-800">
            <div className="bg-slate-800/60 p-3 rounded-lg border border-slate-700/50">
              <div className="text-xs text-slate-400 font-medium">Total Journeys</div>
              <div className="text-xl font-bold text-white mt-1">{analytics.total_journeys}</div>
              <div className="text-xs text-slate-500 mt-0.5">Logged trips</div>
            </div>
            <div className="bg-slate-800/60 p-3 rounded-lg border border-slate-700/50">
              <div className="text-xs text-slate-400 font-medium">On-Time Arrival</div>
              <div className="text-xl font-bold text-emerald-400 mt-1">{analytics.on_time_travel_pct}%</div>
              <div className="text-xs text-slate-500 mt-0.5">Within planned duration</div>
            </div>
            <div className="bg-slate-800/60 p-3 rounded-lg border border-slate-700/50">
              <div className="text-xs text-slate-400 font-medium">Mean Travel Bias</div>
              <div className={`text-xl font-bold mt-1 ${analytics.mean_travel_bias_min > 0 ? 'text-amber-400' : 'text-emerald-400'}`}>
                {analytics.mean_travel_bias_min > 0 ? `+${analytics.mean_travel_bias_min}` : analytics.mean_travel_bias_min} min
              </div>
              <div className="text-xs text-slate-500 mt-0.5">Actual vs planned error</div>
            </div>
            <div className="bg-slate-800/60 p-3 rounded-lg border border-slate-700/50">
              <div className="text-xs text-slate-400 font-medium">Bias Ratio</div>
              <div className="text-xl font-bold text-cyan-400 mt-1">{analytics.travel_bias_ratio}x</div>
              <div className="text-xs text-slate-500 mt-0.5">Personal travel multiplier</div>
            </div>
          </div>
        )}

        {analytics?.insights && analytics.insights.length > 0 && (
          <div className="mt-4 p-3 bg-cyan-950/30 border border-cyan-800/40 rounded-lg text-xs text-cyan-300">
            <span className="font-semibold text-cyan-200">Behavioral Insight: </span>
            {analytics.insights.join(' ')}
          </div>
        )}
      </div>

      {/* SECTION 2: SAVED LOCATIONS */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold text-white flex items-center justify-between">
          <span>Saved Frequent Locations</span>
          <span className="text-xs text-slate-400 font-normal">Explicitly authorized addresses only</span>
        </h3>

        {/* Existing Locations Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 mt-4">
          {locations.map((loc) => (
            <div
              key={loc.id}
              className={`p-3.5 rounded-lg border ${
                loc.is_primary ? 'bg-emerald-950/20 border-emerald-500/40' : 'bg-slate-800/60 border-slate-700'
              } flex flex-col justify-between`}
            >
              <div>
                <div className="flex items-center justify-between">
                  <div className="font-semibold text-white text-sm flex items-center gap-1.5">
                    <span>
                      {loc.category === 'home' && '🏠'}
                      {loc.category === 'work' && '🏢'}
                      {loc.category === 'education' && '🎓'}
                      {loc.category === 'frequent' && '⭐'}
                      {loc.category === 'custom' && '📌'}
                    </span>
                    {loc.name}
                  </div>
                  {loc.is_primary && (
                    <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-400 text-2xs font-semibold rounded-full border border-emerald-500/30">
                      Primary
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-300 mt-1.5 truncate" title={loc.address}>
                  {loc.address}
                </div>
                {loc.latitude && loc.longitude && (
                  <div className="text-2xs text-slate-500 mt-1 font-mono">
                    {loc.latitude.toFixed(4)}, {loc.longitude.toFixed(4)}
                  </div>
                )}
              </div>
              <div className="mt-3 flex items-center justify-between pt-2 border-t border-slate-700/50">
                <button
                  type="button"
                  onClick={() => setOrigin(loc.address)}
                  className="text-xs text-cyan-400 hover:text-cyan-300 font-medium"
                >
                  Use as Origin
                </button>
                <button
                  type="button"
                  onClick={() => handleDeleteLocation(loc.id)}
                  className="text-xs text-rose-400 hover:text-rose-300"
                >
                  Delete
                </button>
              </div>
            </div>
          ))}
          {locations.length === 0 && !loadingLocations && (
            <div className="col-span-full py-4 text-center text-sm text-slate-500">
              No saved locations yet. Add Home or Workplace below for automatic departure context.
            </div>
          )}
        </div>

        {/* Add Location Form */}
        <form onSubmit={handleAddLocation} className="mt-5 pt-4 border-t border-slate-800 flex flex-wrap gap-3 items-end">
          <div className="flex-1 min-w-[140px]">
            <label className="block text-xs text-slate-400 mb-1">Name</label>
            <input
              type="text"
              placeholder="e.g. Home, Office"
              value={newLocName}
              onChange={(e) => setNewLocName(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div className="w-[120px]">
            <label className="block text-xs text-slate-400 mb-1">Category</label>
            <select
              value={newLocCategory}
              onChange={(e) => setNewLocCategory(e.target.value as any)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="home">Home</option>
              <option value="work">Workplace</option>
              <option value="education">College / Univ</option>
              <option value="frequent">Frequent</option>
              <option value="custom">Custom</option>
            </select>
          </div>
          <div className="flex-2 min-w-[220px]">
            <label className="block text-xs text-slate-400 mb-1">Address</label>
            <input
              type="text"
              placeholder="Street Address, City, State"
              value={newLocAddress}
              onChange={(e) => setNewLocAddress(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div className="flex items-center gap-2 pb-1.5">
            <input
              type="checkbox"
              id="is_primary"
              checked={newLocPrimary}
              onChange={(e) => setNewLocPrimary(e.target.checked)}
              className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
            />
            <label htmlFor="is_primary" className="text-xs text-slate-300 cursor-pointer">
              Set Primary
            </label>
          </div>
          <button
            type="submit"
            className="px-4 py-1.5 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium rounded-lg transition"
          >
            Add Location
          </button>
        </form>
      </div>

      {/* SECTION 3: ROUTE EXPLORER & MULTI-MODAL COMPARISON */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold text-white">Route Intelligence & Mode Comparison</h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Calculate traffic-aware travel times and alternative routing options across all supported transport modes.
        </p>

        {/* Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Origin</label>
            <input
              type="text"
              placeholder="e.g. 123 Elm St (or select saved location)"
              value={origin}
              onChange={(e) => setOrigin(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Destination</label>
            <input
              type="text"
              placeholder="e.g. 450 Serra Mall, Stanford"
              value={destination}
              onChange={(e) => setDestination(e.target.value)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
            />
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Travel Mode</label>
            <select
              value={selectedMode}
              onChange={(e) => setSelectedMode(e.target.value as TravelMode)}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500"
            >
              <option value="driving">🚗 Driving</option>
              <option value="transit">🚆 Public Transit</option>
              <option value="bicycling">🚴 Bicycling</option>
              <option value="walking">🚶 Walking</option>
            </select>
          </div>
        </div>

        {routeError && (
          <div className="mt-3 p-2.5 bg-rose-950/30 border border-rose-800 text-rose-300 text-xs rounded-lg">
            {routeError}
          </div>
        )}

        <div className="flex flex-wrap gap-3 mt-4">
          <button
            type="button"
            onClick={handleCalculateRoutes}
            disabled={loadingRoutes}
            className="px-4 py-2 bg-cyan-600 hover:bg-cyan-500 text-white text-xs font-semibold rounded-lg transition disabled:opacity-50"
          >
            {loadingRoutes ? 'Calculating Routes...' : 'Calculate Routes'}
          </button>
          <button
            type="button"
            onClick={handleCompareModes}
            disabled={loadingComparison}
            className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold rounded-lg border border-slate-700 transition disabled:opacity-50"
          >
            {loadingComparison ? 'Comparing Modes...' : 'Compare All 4 Modes'}
          </button>
        </div>

        {/* Personalized Adjustment Banner */}
        {personalizedInfo && (
          <div className="mt-4 p-3 bg-indigo-950/30 border border-indigo-800/50 rounded-lg flex items-start gap-3">
            <span className="text-indigo-400 text-base">🧠</span>
            <div className="text-xs">
              <span className="font-semibold text-indigo-300">Personalized ML Travel Estimate: </span>
              <span className="text-white font-bold">{personalizedInfo.personalized_duration_min} min </span>
              <span className="text-slate-400">
                (Provider baseline: {personalizedInfo.provider_duration_min} min, Adjustment: {personalizedInfo.adjustment_min > 0 ? `+${personalizedInfo.adjustment_min}` : personalizedInfo.adjustment_min} min).
              </span>
              <p className="text-slate-400 mt-0.5">{personalizedInfo.explanation}</p>
            </div>
          </div>
        )}

        {/* Calculated Routes Cards */}
        {calculatedRoutes.length > 0 && (
          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3">
            {calculatedRoutes.map((r, idx) => (
              <div
                key={r.route_id || idx}
                className={`p-4 rounded-xl border ${
                  r.is_recommended ? 'bg-cyan-950/20 border-cyan-500/50' : 'bg-slate-800/50 border-slate-700/60'
                } flex flex-col justify-between`}
              >
                <div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-white truncate max-w-[170px]" title={r.summary}>
                      {r.summary}
                    </span>
                    {r.is_recommended && (
                      <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 text-2xs font-semibold rounded-full border border-cyan-500/30">
                        Recommended
                      </span>
                    )}
                  </div>
                  <div className="mt-3 flex items-baseline gap-2">
                    <span className="text-2xl font-black text-white">
                      {r.duration_in_traffic_min || r.duration_min}
                    </span>
                    <span className="text-xs text-slate-400">min</span>
                    {r.duration_in_traffic_min && (
                      <span className="text-2xs text-amber-400 font-medium">
                        (Live traffic: +{Math.max(0, Math.round(r.duration_in_traffic_min - r.duration_min))}m)
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-slate-400 mt-1 flex items-center gap-3">
                    <span>📏 {r.distance_km} km</span>
                    <span className="capitalize">🚘 {r.travel_mode}</span>
                  </div>
                  {r.warnings && r.warnings.length > 0 && (
                    <div className="text-2xs text-amber-300 mt-2 bg-amber-950/30 p-1.5 rounded border border-amber-800/40">
                      ⚠️ {r.warnings.join(', ')}
                    </div>
                  )}
                </div>
                <button
                  type="button"
                  onClick={() => handleSaveRoute(r)}
                  className="mt-3 w-full py-1.5 bg-slate-700 hover:bg-slate-600 text-slate-200 text-xs font-medium rounded-lg transition"
                >
                  Save as Preferred Route
                </button>
              </div>
            ))}
          </div>
        )}

        {/* Mode Comparison Matrix */}
        {modeComparison && (
          <div className="mt-5 pt-4 border-t border-slate-800">
            <h4 className="text-sm font-semibold text-white mb-2.5">Side-by-Side Mode Comparison</h4>
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {(['driving', 'transit', 'bicycling', 'walking'] as TravelMode[]).map((mode) => {
                const item = modeComparison.comparison[mode];
                const icons: Record<TravelMode, string> = {
                  driving: '🚗 Driving',
                  transit: '🚆 Transit',
                  bicycling: '🚴 Bicycling',
                  walking: '🚶 Walking'
                };
                return (
                  <div key={mode} className="bg-slate-800/60 p-3 rounded-lg border border-slate-700/60">
                    <div className="text-xs font-semibold text-slate-300">{icons[mode]}</div>
                    {item && item.available ? (
                      <div className="mt-2">
                        <div className="text-xl font-bold text-white">
                          {item.duration_in_traffic_min || item.duration_min} <span className="text-xs font-normal text-slate-400">min</span>
                        </div>
                        <div className="text-xs text-slate-400 mt-0.5">{item.distance_km} km</div>
                      </div>
                    ) : (
                      <div className="text-xs text-slate-500 mt-2">Unavailable</div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* SECTION 4: SAVED PREFERRED ROUTES */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold text-white">Saved Preferred Corridors</h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Your saved route corridors used as anchors for departure time optimization.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
          {savedRoutes.map((r) => (
            <div key={r.id} className="p-3.5 bg-slate-800/60 border border-slate-700 rounded-lg flex justify-between items-start">
              <div>
                <div className="font-semibold text-white text-sm">{r.name}</div>
                <div className="text-xs text-slate-300 mt-1">
                  <span className="text-slate-500">From:</span> {r.origin_address}
                </div>
                <div className="text-xs text-slate-300 mt-0.5">
                  <span className="text-slate-500">To:</span> {r.destination_address}
                </div>
                <div className="text-2xs text-cyan-400 mt-1.5 font-medium flex gap-3">
                  <span>⏱️ ~{r.estimated_duration_min} min</span>
                  <span>📏 {r.distance_km} km</span>
                  <span className="capitalize">🚘 {r.travel_mode}</span>
                </div>
              </div>
              <button
                type="button"
                onClick={() => handleDeleteSavedRoute(r.id)}
                className="text-xs text-rose-400 hover:text-rose-300 ml-2"
              >
                Delete
              </button>
            </div>
          ))}
          {savedRoutes.length === 0 && !loadingSavedRoutes && (
            <div className="col-span-full py-4 text-center text-sm text-slate-500">
              No saved corridors yet. Calculate a route above and click "Save as Preferred Route".
            </div>
          )}
        </div>
      </div>

      {/* SECTION 5: USER TRAVEL PREFERENCES */}
      <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 shadow-md">
        <h3 className="text-lg font-semibold text-white">Travel & Departure Buffer Preferences</h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Configure default transport mode and safety margins applied across automated calendar planning.
        </p>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Default Transport Mode</label>
            <select
              value={preferences.default_mode}
              onChange={(e) => setPreferences({ ...preferences, default_mode: e.target.value as TravelMode })}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-emerald-500"
            >
              <option value="driving">Driving</option>
              <option value="transit">Public Transit</option>
              <option value="bicycling">Bicycling</option>
              <option value="walking">Walking</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">
              Safety Buffer Multiplier: <span className="text-white font-bold">{preferences.buffer_multiplier}x</span>
            </label>
            <input
              type="range"
              min="0.8"
              max="2.0"
              step="0.1"
              value={preferences.buffer_multiplier}
              onChange={(e) => setPreferences({ ...preferences, buffer_multiplier: parseFloat(e.target.value) })}
              className="w-full accent-emerald-500 mt-2"
            />
          </div>
          <div className="flex flex-col justify-center space-y-2 pt-2">
            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={preferences.avoid_tolls}
                onChange={(e) => setPreferences({ ...preferences, avoid_tolls: e.target.checked })}
                className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
              />
              Avoid Toll Roads
            </label>
            <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={preferences.avoid_highways}
                onChange={(e) => setPreferences({ ...preferences, avoid_highways: e.target.checked })}
                className="rounded bg-slate-800 border-slate-700 text-emerald-500 focus:ring-0"
              />
              Avoid Expressways / Highways
            </label>
          </div>
        </div>

        <div className="mt-5 flex items-center justify-between pt-4 border-t border-slate-800">
          <span className="text-xs text-emerald-400 font-medium">{prefNotice}</span>
          <button
            type="button"
            onClick={handleSavePreferences}
            disabled={savingPrefs}
            className="px-4 py-2 bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-semibold rounded-lg transition disabled:opacity-50"
          >
            {savingPrefs ? 'Saving...' : 'Save Preferences'}
          </button>
        </div>
      </div>
    </div>
  );
};
