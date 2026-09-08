import { useState, useEffect, useRef } from 'react';
import { 
  Clock, 
  CheckCircle, 
  AlertTriangle, 
  TrendingUp, 
  History, 
  BrainCircuit, 
  Zap, 
  UserCheck, 
  RefreshCw,
  Check,
  Lock,
  User as UserIcon,
  LogOut,
  Briefcase,
  Mic,
  MicOff
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  PieChart,
  Pie
} from 'recharts';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || (import.meta.env.PROD ? '' : 'http://localhost:8000');

interface EventTemplate {
  id: number;
  name: string;
  day_of_week: string;
  start_time: string;
  estimated_prep_duration: number;
  estimated_travel_duration: number;
  importance: string;
}

interface DepartureLog {
  id: number;
  event_name: string;
  date: string;
  day_of_week: string;
  actual_departure: string;
  actual_arrival: string;
  estimated_prep_duration_min: number;
  actual_prep_duration_min: number;
  total_delay_min: number;
  bag_prepared: boolean;
  tag_one_last_task: boolean;
  tag_finding_items: boolean;
  tag_phone_usage: boolean;
  tag_waiting: boolean;
}

interface Insights {
  has_data: boolean;
  total_logs: number;
  lateness_rate_pct: number;
  total_minutes_lost: number;
  personal_prep_bias_min: number;
  avg_estimated_prep_min: number;
  avg_actual_prep_min: number;
  delay_sums: Record<string, number>;
  delay_percentages: Record<string, number>;
  recommendations: Array<{
    title: string;
    description: string;
    action: string;
    impact: string;
  }>;
}

interface ModelMetrics {
  success: boolean;
  message?: string;
  train_size: number;
  test_size: number;
  baseline_mae: number;
  ml_mae: number;
  error_reduction_pct: number;
  top_features: Array<{
    feature: string;
    importance: number;
  }>;
}

// Profile-specific checkout prompt tags mapping
const PROFILE_TAGS: Record<string, { oneLast: string; finding: string; phone: string; waiting: string }> = {
  student: {
    oneLast: "One Last Task (lock door, pack snack)",
    finding: "Finding Student ID / Keys",
    phone: "Checking WhatsApp / Socials",
    waiting: "Waiting (elevator / lift)"
  },
  teacher: {
    oneLast: "Lesson resource checks / Photocopying",
    finding: "Finding Classroom Keys / Badge",
    phone: "Replying to urgent parent emails",
    waiting: "Waiting (carpool / traffic delay)"
  },
  professor: {
    oneLast: "Formatting lecture slides / handouts",
    finding: "Finding Projector HDMI Adapter",
    phone: "Answering urgent thesis draft reviews",
    waiting: "Waiting (campus shuttle / lift)"
  },
  it_professional: {
    oneLast: "One Last Git Commit / Local compile",
    finding: "Finding security entry badge / charger",
    phone: "Replying to urgent Slack / Teams tags",
    waiting: "Waiting (software update / compile)"
  }
};

export default function App() {
  // Authentication States
  const [token, setToken] = useState<string | null>(() => localStorage.getItem('microdelay_token'));
  const [username, setUsername] = useState<string | null>(() => localStorage.getItem('microdelay_username'));
  const [profileType, setProfileType] = useState<string | null>(() => localStorage.getItem('microdelay_profile_type'));
  
  // Auth Form Inputs
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login');
  const [usernameInput, setUsernameInput] = useState<string>('');
  const [passwordInput, setPasswordInput] = useState<string>('');
  const [profileInput, setProfileInput] = useState<string>('student');
  const [authError, setAuthError] = useState<string | null>(null);

  // Tab State
  const [activeTab, setActiveTab] = useState<'dashboard' | 'simulator' | 'comparison' | 'history'>('dashboard');
  
  // Data States
  const [events, setEvents] = useState<EventTemplate[]>([]);
  const [logs, setLogs] = useState<DepartureLog[]>([]);
  const [insights, setInsights] = useState<Insights | null>(null);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  
  // App Utility States
  const [loading, setLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [isRefreshing, setIsRefreshing] = useState<boolean>(false);

  // Simulator States
  const [selectedEventId, setSelectedEventId] = useState<number | null>(null);
  const [bagPrepared, setBagPrepared] = useState<boolean>(false);
  const [simState, setSimState] = useState<'idle' | 'preparing' | 'checkout' | 'success_modal'>('idle');
  const [simDate, setSimDate] = useState<string>(() => {
    const today = new Date();
    return today.toISOString().split('T')[0];
  });
  
  // Simulator Running variables
  const [simClock, setSimClock] = useState<string>('07:50:00');
  const [elapsedPrepMin, setElapsedPrepMin] = useState<number>(0);
  
  // Checkout Logging options
  const [checkoutOneLast, setCheckoutOneLast] = useState<boolean>(false);
  const [checkoutFinding, setCheckoutFinding] = useState<boolean>(false);
  const [checkoutPhone, setCheckoutPhone] = useState<boolean>(false);
  const [checkoutWaiting, setCheckoutWaiting] = useState<boolean>(false);
  const [checkoutResult, setCheckoutResult] = useState<any>(null);
  const [predictedDelay, setPredictedDelay] = useState<number | null>(null);

  // References for Simulator Clock
  const clockIntervalRef = useRef<any>(null);
  const currentPrepTimeRef = useRef<number>(0);
  const simStartRealTimeRef = useRef<Date | null>(null);

  // Voice Assistant States
  const [isListening, setIsListening] = useState<boolean>(false);
  const [voiceText, setVoiceText] = useState<string>('');
  const [voiceFeedback, setVoiceFeedback] = useState<string>('');

  const startVoiceAssistant = (stage: 'idle' | 'preparing' | 'checkout') => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert("Voice speech recognition is not supported in this browser. Please try Chrome or Edge.");
      return;
    }

    if (isListening) {
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    let promptMsg = "";
    if (stage === 'idle') {
      const scheduleNames = events.map(e => e.name).join(", ");
      promptMsg = `Which routine are you starting today? Your options are: ${scheduleNames}. Tell me the routine name and if your materials are prepared.`;
    } else if (stage === 'preparing') {
      promptMsg = "What distraction happened? You can say: commits, slide check, whatsapp, or elevator wait.";
    } else {
      promptMsg = "Please say which checkout delays occurred. For example, did you check the phone or search for keys?";
      if (profileType === 'it_professional') {
        promptMsg = "Rushing out? Were you delayed by commits, slack tags, badge search, or elevator wait? Tell me what happened.";
      } else if (profileType === 'teacher') {
        promptMsg = "Were you delayed by printing handouts, replies to parent email, traffic, or classroom keys?";
      } else if (profileType === 'professor') {
        promptMsg = "Were you delayed by slides check, projector adapters, thesis drafts, or faculty reviews?";
      } else if (profileType === 'student') {
        promptMsg = "Were you delayed by checking socials, student ID keys search, elevator wait, or packing snacks?";
      }
    }

    const utterance = new SpeechSynthesisUtterance(promptMsg);
    utterance.rate = 1.05;
    
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);

    utterance.onend = () => {
      setIsListening(true);
      setVoiceText('');
      setVoiceFeedback('Listening now...');
      try {
        recognition.start();
      } catch (e) {
        console.error(e);
      }
    };

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript.toLowerCase();
      setVoiceText(event.results[0][0].transcript);
      setIsListening(false);

      if (stage === 'idle') {
        let matchedEvt = null;
        for (const evt of events) {
          if (transcript.includes(evt.name.toLowerCase())) {
            matchedEvt = evt;
            break;
          }
        }

        if (matchedEvt) {
          const isBag = transcript.includes('bag') || transcript.includes('ready') || transcript.includes('prepared') || transcript.includes('packed') || transcript.includes('yes') || transcript.includes('done') || transcript.includes('built') || transcript.includes('repo');
          setBagPrepared(isBag);
          setSelectedEventId(matchedEvt.id);
          
          const feedback = `Starting routine: ${matchedEvt.name} with bag ${isBag ? 'prepared' : 'unprepared'}. Safe travels!`;
          setVoiceFeedback(feedback);
          
          const confirmUtterance = new SpeechSynthesisUtterance(feedback);
          window.speechSynthesis.speak(confirmUtterance);
          
          setTimeout(() => {
            startPrep(matchedEvt.id, isBag);
          }, 2500);
        } else {
          const feedback = "Could not find matching routine. Please try again.";
          setVoiceFeedback(feedback);
          const confirmUtterance = new SpeechSynthesisUtterance(feedback);
          window.speechSynthesis.speak(confirmUtterance);
        }
      } 
      else if (stage === 'preparing') {
        let detectedDist = "";
        
        if (transcript.includes('task') || transcript.includes('commit') || transcript.includes('chore') || 
            transcript.includes('lock') || transcript.includes('pack') || transcript.includes('slide') || 
            transcript.includes('photo') || transcript.includes('grading') || transcript.includes('resource') ||
            transcript.includes('snack') || transcript.includes('compile') || transcript.includes('sync')) {
          detectedDist = "one_last_task";
          addDistraction('one_last_task');
        }
        else if (transcript.includes('find') || transcript.includes('search') || transcript.includes('key') || 
            transcript.includes('badge') || transcript.includes('wallet') || transcript.includes('charger') || 
            transcript.includes('adapter') || transcript.includes('id') || transcript.includes('card') ||
            transcript.includes('bag') || transcript.includes('item')) {
          detectedDist = "finding_items";
          addDistraction('finding_items');
        }
        else if (transcript.includes('phone') || transcript.includes('slack') || transcript.includes('whatsapp') || 
            transcript.includes('email') || transcript.includes('social') || transcript.includes('message') || 
            transcript.includes('notification') || transcript.includes('text') || transcript.includes('chat') ||
            transcript.includes('teams') || transcript.includes('reply')) {
          detectedDist = "phone_usage";
          addDistraction('phone_usage');
        }
        else if (transcript.includes('wait') || transcript.includes('elevator') || transcript.includes('lift') || 
            transcript.includes('traffic') || transcript.includes('carpool')) {
          detectedDist = "waiting";
          addDistraction('waiting');
        }

        if (detectedDist) {
          const distClean = detectedDist.replace(/_/g, ' ');
          const feedback = `Logged distraction: ${distClean}. Prep timer updated.`;
          setVoiceFeedback(feedback);
          const confirmUtterance = new SpeechSynthesisUtterance(feedback);
          window.speechSynthesis.speak(confirmUtterance);
        } else {
          const feedback = "Did not match distraction. Try saying git commit or search keys.";
          setVoiceFeedback(feedback);
          const confirmUtterance = new SpeechSynthesisUtterance(feedback);
          window.speechSynthesis.speak(confirmUtterance);
        }
      }
      else {
        // Stage: checkout
        let hasOneLast = false;
        let hasFinding = false;
        let hasPhone = false;
        let hasWaiting = false;

        // 1. One Last Task
        if (transcript.includes('task') || transcript.includes('commit') || transcript.includes('chore') || 
            transcript.includes('lock') || transcript.includes('pack') || transcript.includes('slide') || 
            transcript.includes('photo') || transcript.includes('grading') || transcript.includes('resource') ||
            transcript.includes('snack') || transcript.includes('compile') || transcript.includes('sync')) {
          hasOneLast = true;
          setCheckoutOneLast(true);
        }

        // 2. Finding Items
        if (transcript.includes('find') || transcript.includes('search') || transcript.includes('key') || 
            transcript.includes('badge') || transcript.includes('wallet') || transcript.includes('charger') || 
            transcript.includes('adapter') || transcript.includes('id') || transcript.includes('card') ||
            transcript.includes('bag') || transcript.includes('item') || transcript.includes('phone locator')) {
          hasFinding = true;
          setCheckoutFinding(true);
        }

        // 3. Phone Usage
        if (transcript.includes('phone') || transcript.includes('slack') || transcript.includes('whatsapp') || 
            transcript.includes('email') || transcript.includes('social') || transcript.includes('message') || 
            transcript.includes('notification') || transcript.includes('text') || transcript.includes('chat') ||
            transcript.includes('teams') || transcript.includes('reply')) {
          hasPhone = true;
          setCheckoutPhone(true);
        }

        // 4. Waiting
        if (transcript.includes('wait') || transcript.includes('elevator') || transcript.includes('lift') || 
            transcript.includes('traffic') || transcript.includes('carpool') || transcript.includes('traffic delay') || 
            transcript.includes('bus') || transcript.includes('train')) {
          hasWaiting = true;
          setCheckoutWaiting(true);
        }

        const matched: string[] = [];
        if (hasOneLast) matched.push("one last task");
        if (hasFinding) matched.push("finding items");
        if (hasPhone) matched.push("phone usage");
        if (hasWaiting) matched.push("waiting delay");

        if (matched.length > 0) {
          const feedbackText = `Detected: ${matched.join(', ')}. Logging now.`;
          setVoiceFeedback(feedbackText);
          
          const confirmUtterance = new SpeechSynthesisUtterance(feedbackText + " Have a safe trip.");
          window.speechSynthesis.speak(confirmUtterance);

          setTimeout(() => {
            submitCheckout({
              one_last_task: hasOneLast,
              finding_items: hasFinding,
              phone_usage: hasPhone,
              waiting: hasWaiting
            });
          }, 2500);
        } else {
          const feedbackText = "Did not match delay triggers. Please say it again or check boxes manually.";
          setVoiceFeedback(feedbackText);
          const confirmUtterance = new SpeechSynthesisUtterance(feedbackText);
          window.speechSynthesis.speak(confirmUtterance);
        }
      }
    };

    recognition.onerror = (event: any) => {
      console.error(event);
      setIsListening(false);
      setVoiceFeedback("Speech recognition error. Please check mic permissions.");
    };

    recognition.onend = () => {
      setIsListening(false);
    };
  };

  // Load Data Wrapper (Secured with Token)
  const fetchData = async () => {
    if (!token) return;
    try {
      setErrorMsg(null);
      const headers = { 'Authorization': `Bearer ${token}` };
      
      const eventsRes = await fetch(`${BACKEND_URL}/api/events`, { headers });
      if (eventsRes.status === 401) { handleLogout(); return; }
      if (!eventsRes.ok) throw new Error('Failed to fetch events');
      const eventsData = await eventsRes.json();
      setEvents(eventsData);
      
      // Auto select first event template if none is selected
      if (eventsData.length > 0) {
        setSelectedEventId(prev => (prev === null || !eventsData.find((evt: EventTemplate) => evt.id === prev)) ? eventsData[0].id : prev);
      } else {
        setSelectedEventId(null);
      }

      const logsRes = await fetch(`${BACKEND_URL}/api/logs`, { headers });
      if (!logsRes.ok) throw new Error('Failed to fetch logs');
      const logsData = await logsRes.json();
      setLogs(logsData);

      const insightsRes = await fetch(`${BACKEND_URL}/api/insights`, { headers });
      if (!insightsRes.ok) throw new Error('Failed to fetch insights');
      const insightsData = await insightsRes.json();
      setInsights(insightsData);

      const metricsRes = await fetch(`${BACKEND_URL}/api/metrics`, { headers });
      if (!metricsRes.ok) throw new Error('Failed to fetch model metrics');
      const metricsData = await metricsRes.json();
      setMetrics(metricsData);

    } catch (e: any) {
      console.error(e);
      setErrorMsg('Failed to sync data with backend server. Check backend logs.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      setLoading(true);
      fetchData();
    }
  }, [token]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await fetchData();
    setIsRefreshing(false);
  };

  // Auth Handlers
  const handleAuthSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setAuthError(null);
    setLoading(true);

    const payload = authMode === 'register' 
      ? { username: usernameInput, password: passwordInput, profile_type: profileInput }
      : { username: usernameInput, password: passwordInput };

    const endpoint = authMode === 'register' ? '/api/auth/register' : '/api/auth/login';

    try {
      const res = await fetch(`${BACKEND_URL}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || 'Authentication failed');
      }

      // Save credentials
      localStorage.setItem('microdelay_token', data.token);
      localStorage.setItem('microdelay_username', data.username);
      localStorage.setItem('microdelay_profile_type', data.profile_type);

      setToken(data.token);
      setUsername(data.username);
      setProfileType(data.profile_type);
      setAuthError(null);
      setActiveTab('dashboard');
    } catch (err: any) {
      setAuthError(err.message || 'Connection failed.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('microdelay_token');
    localStorage.removeItem('microdelay_username');
    localStorage.removeItem('microdelay_profile_type');
    
    setToken(null);
    setUsername(null);
    setProfileType(null);
    
    // Clear data states
    setEvents([]);
    setLogs([]);
    setInsights(null);
    setMetrics(null);
    setSimState('idle');
    
    if (clockIntervalRef.current) {
      clearInterval(clockIntervalRef.current);
    }
  };

  // Get active Event Template
  const activeEvent = events.find(e => e.id === selectedEventId) || null;

  const fetchPredictionForEvent = async (evt: EventTemplate, isBag: boolean) => {
    if (!token) return;
    try {
      const url = `${BACKEND_URL}/api/predict?event_name=${encodeURIComponent(evt.name)}&day_of_week=${evt.day_of_week}&importance=${evt.importance}&bag_prepared=${isBag}&estimated_prep=${evt.estimated_prep_duration}&estimated_travel=${evt.estimated_travel_duration}`;
      const res = await fetch(url, {
        headers: { 'Authorization': `Bearer ${token}` }
      });
      const data = await res.json();
      setPredictedDelay(data.predicted_delay_min);
    } catch (err) {
      console.error(err);
    }
  };


  // Simulator Start
  const startPrep = (overrideEventId?: number, overrideBagPrepared?: boolean) => {
    const eventId = overrideEventId !== undefined ? overrideEventId : selectedEventId;
    const isBag = overrideBagPrepared !== undefined ? overrideBagPrepared : bagPrepared;
    const targetEvent = events.find(e => e.id === eventId);
    if (!targetEvent) return;
    
    setElapsedPrepMin(0);
    currentPrepTimeRef.current = 0;
    setCheckoutOneLast(false);
    setCheckoutFinding(false);
    setCheckoutPhone(false);
    setCheckoutWaiting(false);
    setCheckoutResult(null);

    fetchPredictionForEvent(targetEvent, isBag);

    const [startH, startM, startS] = targetEvent.start_time.split(':').map(Number);
    const eventTime = new Date();
    eventTime.setHours(startH, startM, startS, 0);
    
    const plannedDepTime = new Date(eventTime.getTime() - targetEvent.estimated_travel_duration * 60 * 1000);
    const plannedPrepStart = new Date(plannedDepTime.getTime() - targetEvent.estimated_prep_duration * 60 * 1000);
    
    const noiseMin = Math.floor(Math.random() * 5) - 2; 
    const actualPrepStart = new Date(plannedPrepStart.getTime() + noiseMin * 60 * 1000);
    
    setSimClock(actualPrepStart.toTimeString().split(' ')[0]);
    simStartRealTimeRef.current = actualPrepStart;
    setSimState('preparing');

    // Start Clock ticking (1 min in-app = 300ms real time)
    if (clockIntervalRef.current) clearInterval(clockIntervalRef.current);
    clockIntervalRef.current = setInterval(() => {
      currentPrepTimeRef.current += 1;
      setElapsedPrepMin(currentPrepTimeRef.current);
      
      const newClockTime = new Date(actualPrepStart.getTime() + currentPrepTimeRef.current * 60 * 1000);
      setSimClock(newClockTime.toTimeString().split(' ')[0]);
    }, 400);
  };

  // Add random simulated distraction
  const addDistraction = (type: string) => {
    if (simState !== 'preparing') return;
    
    let addedMins = 0;
    if (type === 'one_last_task') {
      addedMins = Math.floor(Math.random() * 5) + 4; 
      setCheckoutOneLast(true);
    } else if (type === 'finding_items') {
      addedMins = Math.floor(Math.random() * 4) + 3; 
      setCheckoutFinding(true);
    } else if (type === 'phone_usage') {
      addedMins = Math.floor(Math.random() * 3) + 2; 
      setCheckoutPhone(true);
    } else if (type === 'waiting') {
      addedMins = Math.floor(Math.random() * 3) + 3; 
      setCheckoutWaiting(true);
    }

    currentPrepTimeRef.current += addedMins;
    setElapsedPrepMin(currentPrepTimeRef.current);
    
    if (simStartRealTimeRef.current) {
      const newClockTime = new Date(simStartRealTimeRef.current.getTime() + currentPrepTimeRef.current * 60 * 1000);
      setSimClock(newClockTime.toTimeString().split(' ')[0]);
    }
  };

  // Stop Simulator and Leave
  const leaveNow = () => {
    if (clockIntervalRef.current) {
      clearInterval(clockIntervalRef.current);
    }
    
    if (!activeEvent || !simStartRealTimeRef.current) return;
    
    const [startH, startM, startS] = activeEvent.start_time.split(':').map(Number);
    const eventTime = new Date();
    eventTime.setHours(startH, startM, startS, 0);
    const plannedDepTime = new Date(eventTime.getTime() - activeEvent.estimated_travel_duration * 60 * 1000);
    
    const actualDepTimeStr = simClock;
    
    const actualDepDt = new Date();
    const [actH, actM, actS] = actualDepTimeStr.split(':').map(Number);
    actualDepDt.setHours(actH, actM, actS, 0);
    
    const diffMs = actualDepDt.getTime() - plannedDepTime.getTime();
    const latenessMins = diffMs / 60 / 1000;
    
    setLatenessMinutes(latenessMins);
    
    if (latenessMins > 0) {
      setSimState('checkout');
    } else {
      submitCheckout({
        one_last_task: false,
        finding_items: false,
        phone_usage: false,
        waiting: false
      });
    }
  };

  const [latenessMinutes, setLatenessMinutes] = useState<number>(0);

  const submitCheckout = async (overrideTags?: Record<string, boolean>) => {
    if (!activeEvent || !simStartRealTimeRef.current || !token) return;
    
    setLoading(true);
    
    const [startH, startM, startS] = activeEvent.start_time.split(':').map(Number);
    const eventTime = new Date();
    eventTime.setHours(startH, startM, startS, 0);
    
    const plannedDepTime = new Date(eventTime.getTime() - activeEvent.estimated_travel_duration * 60 * 1000);
    const plannedPrepStart = new Date(plannedDepTime.getTime() - activeEvent.estimated_prep_duration * 60 * 1000);

    const actualDepDt = new Date();
    const [actH, actM, actS] = simClock.split(':').map(Number);
    actualDepDt.setHours(actH, actM, actS, 0);
    
    const actualTravelDuration = activeEvent.estimated_travel_duration + (Math.random() * 4 - 1.5);
    const actualArrivalDt = new Date(actualDepDt.getTime() + actualTravelDuration * 60 * 1000);

    const isOneLast = overrideTags ? overrideTags.one_last_task : checkoutOneLast;
    const isFinding = overrideTags ? overrideTags.finding_items : checkoutFinding;
    const isPhone = overrideTags ? overrideTags.phone_usage : checkoutPhone;
    const isWaiting = overrideTags ? overrideTags.waiting : checkoutWaiting;

    const payload = {
      event_id: activeEvent.id,
      event_name: activeEvent.name,
      date: simDate,
      day_of_week: activeEvent.day_of_week,
      planned_prep_start: plannedPrepStart.toTimeString().split(' ')[0],
      planned_departure: plannedDepTime.toTimeString().split(' ')[0],
      estimated_prep_duration_min: parseFloat(activeEvent.estimated_prep_duration.toFixed(1)),
      estimated_travel_duration_min: parseFloat(activeEvent.estimated_travel_duration.toFixed(1)),
      actual_prep_start: simStartRealTimeRef.current.toTimeString().split(' ')[0],
      actual_departure: simClock,
      actual_arrival: actualArrivalDt.toTimeString().split(' ')[0],
      actual_prep_duration_min: parseFloat(elapsedPrepMin.toFixed(1)),
      actual_travel_duration_min: parseFloat(actualTravelDuration.toFixed(1)),
      bag_prepared: bagPrepared,
      tag_one_last_task: isOneLast,
      tag_finding_items: isFinding,
      tag_phone_usage: isPhone,
      tag_waiting: isWaiting
    };

    try {
      const res = await fetch(`${BACKEND_URL}/api/checkout`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      setCheckoutResult(data);
      setSimState('success_modal');
      
      await fetchData();
    } catch (err) {
      console.error(err);
      setErrorMsg('Failed to log departure to backend database.');
    } finally {
      setLoading(false);
    }
  };

  const cleanFeatureName = (featName: string) => {
    if (featName === 'bag_prepared') return 'Preparing materials previous night';
    if (featName === 'importance_code') return 'Task Priority (High/Medium/Low)';
    
    return featName
      .replace('day_', 'Day of week: ')
      .replace('event_', 'Task: ')
      .replace('_', ' ')
      .replace(/\b\w/g, c => c.toUpperCase()); // Convert to Title Case
  };

  // Profile-specific label configuration
  const customTags = PROFILE_TAGS[profileType || 'student'] || PROFILE_TAGS.student;

  // Formulate Area Chart Data
  const biasChartData = logs
    .slice(0, 15) 
    .reverse()
    .map(log => ({
      date: log.date.substring(5), 
      event: log.event_name,
      estimated: log.estimated_prep_duration_min,
      actual: log.actual_prep_duration_min,
      delay: log.total_delay_min
    }));

  // Formulate Pie Chart Data
  const pieData = insights && insights.delay_sums ? [
    { name: profileType === 'it_professional' ? 'One Last Commit' : profileType === 'professor' ? 'Lecture Slides' : 'One Last Task', value: insights.delay_sums.one_last_task, color: '#a855f7' },
    { name: profileType === 'it_professional' ? 'Security Badge/Charger' : profileType === 'professor' ? 'Projector HDMI Adapter' : 'Finding Items', value: insights.delay_sums.finding_items, color: '#f59e0b' },
    { name: profileType === 'it_professional' ? 'Slack Messages' : profileType === 'professor' ? 'Reading Thesis Draft' : 'Checking Phone', value: insights.delay_sums.phone_usage, color: '#ef4444' },
    { name: 'Waiting Delay', value: insights.delay_sums.waiting, color: '#6366f1' },
    { name: 'Estimation Prep Bias', value: insights.delay_sums.prep_bias, color: '#10b981' }
  ].filter(d => d.value > 0) : [];

  // ==================== RENDERING LOGIN / REGISTER PAGE ====================
  if (!token) {
    return (
      <div style={{ display: 'flex', minHeight: '100vh', width: '100%', alignItems: 'center', justifyContent: 'center', background: 'var(--bg-primary)', padding: '1rem' }}>
        <div className="card" style={{ width: '100%', maxWidth: '420px', padding: '2rem', boxShadow: 'var(--card-hover-shadow)', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          
          <div style={{ textAlign: 'center', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
            <h1 style={{ fontSize: '2rem', fontWeight: 800, background: 'linear-gradient(135deg, var(--accent-primary) 0%, var(--accent-secondary) 100%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', margin: 0 }}>MicroDelay</h1>
            <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Personal Time Estimation Intelligence System</span>
          </div>

          <div style={{ display: 'flex', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem', gap: '1rem' }}>
            <button 
              style={{ flex: 1, background: 'none', border: 'none', padding: '0.5rem', fontWeight: 600, color: authMode === 'login' ? 'var(--accent-primary)' : 'var(--text-muted)', borderBottom: authMode === 'login' ? '2px solid var(--accent-primary)' : 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: '0.95rem' }}
              onClick={() => { setAuthMode('login'); setAuthError(null); }}
            >
              Log In
            </button>
            <button 
              style={{ flex: 1, background: 'none', border: 'none', padding: '0.5rem', fontWeight: 600, color: authMode === 'register' ? 'var(--accent-primary)' : 'var(--text-muted)', borderBottom: authMode === 'register' ? '2px solid var(--accent-primary)' : 'none', cursor: 'pointer', fontFamily: 'var(--font-sans)', fontSize: '0.95rem' }}
              onClick={() => { setAuthMode('register'); setAuthError(null); }}
            >
              Create Account
            </button>
          </div>

          {authError && (
            <div className="rec-card danger" style={{ padding: '0.75rem', fontSize: '0.85rem' }}>
              <div className="rec-title" style={{ color: 'var(--accent-danger)' }}>
                <AlertTriangle size={16} />
                Failed authentication
              </div>
              <p style={{ marginTop: '0.15rem' }}>{authError}</p>
            </div>
          )}

          <form onSubmit={handleAuthSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <UserIcon size={14} />
                Username
              </label>
              <input 
                type="text" 
                required
                className="form-input" 
                placeholder="Enter username" 
                value={usernameInput}
                onChange={(e) => setUsernameInput(e.target.value)}
              />
            </div>

            <div className="form-group">
              <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                <Lock size={14} />
                Password
              </label>
              <input 
                type="password" 
                required
                className="form-input" 
                placeholder="Enter password" 
                value={passwordInput}
                onChange={(e) => setPasswordInput(e.target.value)}
              />
            </div>

            {authMode === 'register' && (
              <div className="form-group">
                <label className="form-label" style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                  <Briefcase size={14} />
                  Your Background/Profession
                </label>
                <select 
                  className="form-select" 
                  value={profileInput}
                  onChange={(e) => setProfileInput(e.target.value)}
                >
                  <option value="student">College Student</option>
                  <option value="teacher">School Teacher</option>
                  <option value="professor">University Professor</option>
                  <option value="it_professional">IT Software Engineer</option>
                </select>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                  This will pre-seed your dashboard with custom schedule templates and profession-specific lateness behaviors (e.g. build cycles for IT, worksheets printing for teachers).
                </span>
              </div>
            )}

            <button type="submit" className="btn-primary" disabled={loading} style={{ marginTop: '0.5rem' }}>
              {loading ? (
                <RefreshCw size={18} className="spin" />
              ) : authMode === 'register' ? (
                'Register & Auto-Seed Logs'
              ) : (
                'Log In'
              )}
            </button>
          </form>

        </div>
      </div>
    );
  }

  // ==================== RENDERING SECURED DASHBOARD APP ====================
  return (
    <div className="app-container">
      {/* Header with Navigation and Profile Logout info */}
      <div className="header">
        <div className="header-title">
          <h1>MicroDelay</h1>
          <span className="header-subtitle" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <span>Personal Time Estimation Intelligence System</span>
            <span style={{ color: 'var(--text-muted)' }}>|</span>
            <span style={{ textTransform: 'capitalize', color: 'var(--accent-primary)', fontWeight: 600 }}>
              {profileType?.replace('_', ' ')} Profile
            </span>
          </span>
        </div>
        
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginRight: '0.5rem', fontSize: '0.9rem', padding: '0.4rem 0.8rem', backgroundColor: 'var(--bg-secondary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
            <UserIcon size={14} style={{ color: 'var(--text-muted)' }} />
            <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>{username}</span>
            <button 
              onClick={handleLogout} 
              style={{ background: 'none', border: 'none', marginLeft: '0.5rem', cursor: 'pointer', display: 'flex', alignItems: 'center', color: 'var(--accent-danger)' }}
              title="Log Out"
            >
              <LogOut size={16} />
            </button>
          </div>

          <button 
            className="btn-secondary" 
            style={{ width: 'auto', padding: '0.5rem 0.75rem', fontSize: '0.85rem' }} 
            onClick={handleRefresh}
            disabled={isRefreshing}
          >
            <RefreshCw size={14} className={isRefreshing ? 'spin' : ''} />
            Sync
          </button>
          
          <div className="tabs-container">
            <button 
              className={`tab-btn ${activeTab === 'dashboard' ? 'active' : ''}`}
              onClick={() => setActiveTab('dashboard')}
            >
              <TrendingUp size={16} />
              Dashboard
            </button>
            <button 
              className={`tab-btn ${activeTab === 'simulator' ? 'active' : ''}`}
              onClick={() => setActiveTab('simulator')}
            >
              <Clock size={16} />
              Morning Tracker
            </button>
            <button 
              className={`tab-btn ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => setActiveTab('history')}
            >
              <History size={16} />
              Log History
            </button>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="rec-card danger" style={{ padding: '1rem', borderLeftWidth: '6px' }}>
          <div className="rec-title" style={{ color: 'var(--accent-danger)' }}>
            <AlertTriangle size={20} />
            Sync Error
          </div>
          <div className="rec-desc">{errorMsg}</div>
        </div>
      )}

      {loading && (
        <div style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <RefreshCw size={36} className="spin" style={{ margin: '0 auto 1rem', display: 'block' }} />
          Processing model compilation & loading user profile...
        </div>
      )}

      {!loading && (
        <>
          {/* TAB 1: DASHBOARD */}
          {activeTab === 'dashboard' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
              {/* Quick Start Guide Banner */}
              <div className="card" style={{ 
                borderLeft: '4px solid var(--accent-primary)', 
                backgroundColor: 'rgba(99, 102, 241, 0.03)',
                padding: '1.25rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.5rem'
              }}>
                <h3 style={{ fontSize: '1.1rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--accent-primary)', margin: 0 }}>
                  <Zap size={18} /> Quick Start Guide
                </h3>
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', margin: 0 }}>
                  Welcome to MicroDelay! Here's how to experience the app's time-intelligence features in 3 quick steps:
                </p>
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', 
                  gap: '1rem', 
                  marginTop: '0.5rem',
                  fontSize: '0.85rem',
                  color: 'var(--text-secondary)'
                }}>
                  <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>1. Track a Morning Routine</strong>
                    Go to the <strong>Morning Tracker</strong> tab, choose an event, toggle "Bag Prepared" state, and click <em>Start Getting Ready</em>.
                  </div>
                  <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>2. Simulate Delays</strong>
                    Use the distraction buttons (like "Git Commit" or "WhatsApp check") to add realistic time delays to your prep clock.
                  </div>
                  <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                    <strong style={{ color: 'var(--text-primary)', display: 'block', marginBottom: '0.25rem' }}>3. Try Voice Checkout</strong>
                    Click <em>I'm Leaving Now</em>, tap the microphone button, and speak your delay reasons (e.g. "I was searching for keys") to auto-save!
                  </div>
                </div>
              </div>

              {/* Stat Cards */}
              <div className="grid-3">
                <div className="card">
                  <div className="stat-card">
                    <div className="stat-icon" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: 'var(--accent-danger)' }}>
                      <AlertTriangle size={24} />
                    </div>
                    <div className="stat-info" style={{ display: 'flex', flexDirection: 'column' }}>
                      <span className="stat-value">{insights?.lateness_rate_pct}%</span>
                      <span className="stat-label">Lateness Rate</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>% of departures that were late</span>
                    </div>
                  </div>
                </div>
                
                <div className="card">
                  <div className="stat-card">
                    <div className="stat-icon" style={{ backgroundColor: 'rgba(99, 102, 241, 0.1)', color: 'var(--accent-primary)' }}>
                      <Clock size={24} />
                    </div>
                    <div className="stat-info" style={{ display: 'flex', flexDirection: 'column' }}>
                      <span className="stat-value">+{insights?.personal_prep_bias_min}m</span>
                      <span className="stat-label">Average Prep Underestimation</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>How much slower you prep than you think</span>
                    </div>
                  </div>
                </div>

                <div className="card">
                  <div className="stat-card">
                    <div className="stat-icon" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: 'var(--accent-success)' }}>
                      <Zap size={24} />
                    </div>
                    <div className="stat-info" style={{ display: 'flex', flexDirection: 'column' }}>
                      <span className="stat-value">{insights ? (insights.total_minutes_lost / 60).toFixed(1) : 0} hrs</span>
                      <span className="stat-label">Total Lost Time (60 Days)</span>
                      <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>Prep delays accumulated over 60 days</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Graphical Insights */}
              <div className="grid-3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))" }}>
                {/* Delay breakdown pie chart */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Where did your extra time go?</h3>
                  </div>
                  <div style={{ height: '240px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    {pieData.length > 0 ? (
                      <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                        <div style={{ width: '50%', height: '200px' }}>
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie
                                data={pieData}
                                cx="50%"
                                cy="50%"
                                innerRadius={60}
                                outerRadius={80}
                                paddingAngle={4}
                                dataKey="value"
                              >
                                {pieData.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                              </Pie>
                            </PieChart>
                          </ResponsiveContainer>
                        </div>
                        <div style={{ width: '50%', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                          {pieData.map((d, index) => (
                            <div key={index} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.85rem' }}>
                              <div style={{ width: '12px', height: '12px', borderRadius: '3px', backgroundColor: d.color }}></div>
                              <span style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{insights?.delay_percentages[Object.keys(insights.delay_sums)[index]]}%</span>
                              <span style={{ color: 'var(--text-secondary)' }}>{d.name}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>No statistics logged yet. Complete tracker sessions to see allocations.</div>
                    )}
                  </div>
                </div>

                {/* AI Behavioral Lateness Triggers Card */}
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Your Top Lateness Triggers (AI Analysis)</h3>
                  </div>
                  <div style={{ height: '240px', display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                    {metrics?.success ? (
                      <div className="feature-list">
                        {metrics.top_features.map((feat, index) => (
                          <div key={index} className="feature-item">
                            <div className="feature-info">
                              <span style={{ color: 'var(--text-primary)', fontSize: '0.85rem' }}>{cleanFeatureName(feat.feature)}</span>
                              <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>{(feat.importance * 100).toFixed(0)}% impact</span>
                            </div>
                            <div className="feature-bar-bg">
                              <div 
                                className="feature-bar-fill" 
                                style={{ width: `${feat.importance * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
                        AI is currently analyzing your routines. Logs needed to compile lateness triggers: <strong>{Math.max(0, 8 - (logs?.length || 0))}</strong> more checkout logs.
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Personal Prep Bias history timeline (Full Width) */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Time Estimation Accuracy History</h3>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Gap between your estimated preparation time (dotted) and actual prep time (solid)</span>
                </div>
                <div style={{ height: '240px' }}>
                  {biasChartData.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={biasChartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                        <defs>
                          <linearGradient id="colorActual" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="var(--accent-primary)" stopOpacity={0.2}/>
                            <stop offset="95%" stopColor="var(--accent-primary)" stopOpacity={0}/>
                          </linearGradient>
                          <linearGradient id="colorEstimated" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor="var(--text-muted)" stopOpacity={0.1}/>
                            <stop offset="95%" stopColor="var(--text-muted)" stopOpacity={0}/>
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--border-color)" />
                        <XAxis dataKey="date" stroke="var(--text-muted)" style={{ fontSize: '11px' }} />
                        <YAxis stroke="var(--text-muted)" style={{ fontSize: '11px' }} unit="m" />
                        <Tooltip 
                          contentStyle={{ backgroundColor: 'var(--bg-secondary)', borderColor: 'var(--border-color)', borderRadius: '8px', color: 'var(--text-primary)' }}
                          labelStyle={{ fontWeight: 600 }}
                        />
                        <Legend wrapperStyle={{ fontSize: '12px' }} />
                        <Area type="monotone" dataKey="actual" name="Actual Prep Time" stroke="var(--accent-primary)" fillOpacity={1} fill="url(#colorActual)" strokeWidth={2} />
                        <Area type="monotone" dataKey="estimated" name="Estimated Prep Time" stroke="var(--text-muted)" fillOpacity={1} fill="url(#colorEstimated)" strokeDasharray="5 5" strokeWidth={1.5} />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div style={{ textAlign: 'center', color: 'var(--text-muted)', paddingTop: '4rem' }}>No data to display.</div>
                  )}
                </div>
              </div>

              {/* ML Behavioral Insight Recommendations */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">
                    <BrainCircuit style={{ color: 'var(--accent-primary)' }} />
                    MicroDelay Pattern Insights
                  </h3>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Machine Learning analysis of your time behavior patterns</span>
                </div>
                
                <div className="recs-list">
                  {insights && insights.recommendations && insights.recommendations.map((rec, i) => (
                    <div 
                      key={i} 
                      className={`rec-card ${
                        rec.title.includes('Lateness') || rec.title.includes('Bottleneck') ? 'danger' : 
                        rec.title.includes('Underestimation') || rec.title.includes('Bias') ? 'warning' : ''
                      }`}
                    >
                      <div className="rec-title">
                        {rec.title.includes('Lateness') || rec.title.includes('Bottleneck') ? <AlertTriangle size={18} style={{ color: 'var(--accent-danger)' }} /> : 
                         rec.title.includes('Underestimation') || rec.title.includes('Bias') ? <Clock size={18} style={{ color: 'var(--accent-warning)' }} /> : 
                         <UserCheck size={18} style={{ color: 'var(--accent-success)' }} />}
                        {rec.title}
                      </div>
                      <div className="rec-desc">{rec.description}</div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
                        <div className="rec-action">
                          <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>Suggested change: </span>
                          {rec.action}
                        </div>
                        <div className="rec-impact">{rec.impact}</div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* TAB 2: MORNING SIMULATOR */}
          {activeTab === 'simulator' && (
            <div className="grid-2-3">
              {/* Event Setup Form */}
              <div className="card">
                <div className="card-header">
                  <h3 className="card-title">Simulate Morning Routine</h3>
                </div>
                
                {simState === 'idle' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {events.length > 0 ? (
                      <>
                        <div className="form-group">
                          <label className="form-label">Select Scheduled Class/Event</label>
                          <select 
                            className="form-select"
                            value={selectedEventId || ''} 
                            onChange={(e) => setSelectedEventId(Number(e.target.value))}
                          >
                            {events.map(evt => (
                              <option key={evt.id} value={evt.id}>
                                {evt.day_of_week} - {evt.name} ({evt.start_time.substring(0, 5)})
                              </option>
                            ))}
                          </select>
                        </div>

                        <div className="form-group">
                          <label className="form-label">Simulation Date</label>
                          <input 
                            type="date" 
                            className="form-input" 
                            value={simDate} 
                            onChange={(e) => setSimDate(e.target.value)} 
                          />
                        </div>

                        {activeEvent && (
                          <div style={{ padding: '1rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span style={{ color: 'var(--text-secondary)' }}>Event Start:</span>
                              <span style={{ fontWeight: 600 }}>{activeEvent.start_time}</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span style={{ color: 'var(--text-secondary)' }}>Expected Travel:</span>
                              <span style={{ fontWeight: 600 }}>{activeEvent.estimated_travel_duration} minutes</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                              <span style={{ color: 'var(--text-secondary)' }}>Planned Prep:</span>
                              <span style={{ fontWeight: 600 }}>{activeEvent.estimated_prep_duration} minutes</span>
                            </div>
                            <div style={{ display: 'flex', justifyContent: 'space-between', borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem', marginTop: '0.25rem' }}>
                              <span style={{ color: 'var(--text-secondary)', fontWeight: 500 }}>Target Departure:</span>
                              <span style={{ fontWeight: 600, color: 'var(--accent-primary)' }}>
                                {(() => {
                                  const [h, m, s] = activeEvent.start_time.split(':').map(Number);
                                  const t = new Date();
                                  t.setHours(h, m, s, 0);
                                  const dep = new Date(t.getTime() - activeEvent.estimated_travel_duration * 60 * 1000);
                                  return dep.toTimeString().split(' ')[0].substring(0, 5);
                                })()} AM
                              </span>
                            </div>
                          </div>
                        )}

                        <div className="switch-group" onClick={() => setBagPrepared(!bagPrepared)}>
                          <span className="switch-label">
                            {profileType === 'it_professional' ? 'Did you pull repo & build code last night?' : 
                             profileType === 'professor' ? 'Did you pack slides and brief last night?' : 
                             profileType === 'teacher' ? 'Did you print handouts/grading last night?' : 
                             'Did you pack your bag/materials last night?'}
                          </span>
                          <div className={`switch-outer ${bagPrepared ? 'checked' : ''}`}>
                            <div className="switch-inner"></div>
                          </div>
                        </div>

                        <button className="btn-primary" onClick={() => startPrep()}>
                          <Zap size={18} />
                          Start Getting Ready
                        </button>

                        {/* Hands-Free Routine Voice Start Button */}
                        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginTop: '0.5rem', padding: '0.75rem', backgroundColor: 'rgba(99, 102, 241, 0.05)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)' }}>
                          <button 
                            type="button"
                            className="btn-secondary"
                            onClick={() => startVoiceAssistant('idle')}
                            style={{ padding: '0.4rem 0.8rem', fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '0.35rem', fontWeight: 600 }}
                          >
                            <Mic size={14} style={{ color: 'var(--accent-primary)' }} /> Voice Start Routine
                          </button>
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                            {isListening ? "Listening..." : "Tap to speak your schedule name!"}
                          </span>
                        </div>
                      </>
                    ) : (
                      <div style={{ textAlign: 'center', color: 'var(--text-muted)', padding: '2rem' }}>
                        No event templates found for your user type. Contact support.
                      </div>
                    )}
                  </div>
                )}

                {simState === 'preparing' && activeEvent && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', textAlign: 'center' }}>
                    <span style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Preparing for: <strong>{activeEvent.name}</strong>
                    </span>
                    
                    <div className="timer-display">{simClock}</div>
                    
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', padding: '0 0.5rem', color: 'var(--text-secondary)' }}>
                      <span>Prep Duration: <strong>{elapsedPrepMin} minutes</strong></span>
                      <span>Target Departure: <strong>{(() => {
                        const [h, m, s] = activeEvent.start_time.split(':').map(Number);
                        const t = new Date();
                        t.setHours(h, m, s, 0);
                        return new Date(t.getTime() - activeEvent.estimated_travel_duration * 60 * 1000).toTimeString().split(' ')[0];
                      })()}</strong></span>
                    </div>

                    {predictedDelay !== null && (
                      <div style={{ padding: '0.75rem', backgroundColor: 'rgba(99, 102, 241, 0.08)', border: '1px solid rgba(99, 102, 241, 0.2)', borderRadius: 'var(--radius-md)', fontSize: '0.85rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                        <BrainCircuit size={16} style={{ color: 'var(--accent-primary)' }} />
                        <span>Personal ML Model predicts delay: <strong>{predictedDelay.toFixed(1)}m</strong> in this scenario.</span>
                      </div>
                    )}

                    <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem', marginTop: '0.5rem' }}>
                      <span className="form-label" style={{ display: 'block', marginBottom: '0.75rem' }}>Simulate Dynamic Distractions (Accelerate time)</span>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                        <button 
                          className="btn-secondary" 
                          style={{ padding: '0.5rem', fontSize: '0.8rem' }}
                          onClick={() => addDistraction('one_last_task')}
                        >
                          + {profileType === 'it_professional' ? 'Git Commit' : profileType === 'professor' ? 'Check Slides' : 'Quick Chore'}
                        </button>
                        <button 
                          className="btn-secondary" 
                          style={{ padding: '0.5rem', fontSize: '0.8rem' }}
                          onClick={() => addDistraction('finding_items')}
                        >
                          + {profileType === 'it_professional' ? 'Badge Search' : profileType === 'professor' ? 'Adapter Search' : 'Search Keys/ID'}
                        </button>
                        <button 
                          className="btn-secondary" 
                          style={{ padding: '0.5rem', fontSize: '0.8rem' }}
                          onClick={() => addDistraction('phone_usage')}
                        >
                          + {profileType === 'it_professional' ? 'Slack message' : profileType === 'professor' ? 'Check draft' : 'Check WhatsApp'}
                        </button>
                        <button 
                          className="btn-secondary" 
                          style={{ padding: '0.5rem', fontSize: '0.8rem' }}
                          onClick={() => addDistraction('waiting')}
                        >
                          + Elevator/Wait
                        </button>
                      </div>

                      {/* Log Voice Distraction Helper */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', marginTop: '0.75rem', padding: '0.5rem', backgroundColor: 'rgba(99, 102, 241, 0.05)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-color)' }}>
                        <button 
                          type="button"
                          className="btn-secondary"
                          onClick={() => startVoiceAssistant('preparing')}
                          style={{ padding: '0.35rem 0.75rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem', fontWeight: 600 }}
                        >
                          <Mic size={12} style={{ color: 'var(--accent-primary)' }} /> Voice Delay Logger
                        </button>
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
                          {isListening ? "Listening..." : 'Say "checking WhatsApp"'}
                        </span>
                      </div>
                    </div>

                    <button 
                      className="btn-primary" 
                      style={{ backgroundColor: 'var(--accent-danger)', background: 'linear-gradient(135deg, var(--accent-danger) 0%, #dc2626 100%)', marginTop: '1rem' }} 
                      onClick={leaveNow}
                    >
                      <CheckCircle size={18} />
                      I'm Leaving Now
                    </button>
                  </div>
                )}

                {simState === 'checkout' && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
                    <div style={{ textAlign: 'center' }}>
                      <AlertTriangle size={48} style={{ color: 'var(--accent-warning)', margin: '0 auto 0.5rem' }} />
                      <h3 style={{ fontSize: '1.4rem', fontWeight: 700 }}>Leaving {Math.ceil(latenessMinutes)} minutes late</h3>
                      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>MicroDelay has intercepted this late departure. Please complete your 3-second checkout:</span>
                    </div>

                    <div style={{ borderTop: '1px solid var(--border-color)', borderBottom: '1px solid var(--border-color)', padding: '1rem 0' }}>
                      <span className="form-label" style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600 }}>Where did the extra time go?</span>
                      
                      <div className="tags-grid">
                        <div 
                          className={`tag-chip ${checkoutOneLast ? 'active' : ''}`}
                          onClick={() => setCheckoutOneLast(!checkoutOneLast)}
                        >
                          <div className="checkbox-circle"></div>
                          {customTags.oneLast}
                        </div>

                        <div 
                          className={`tag-chip ${checkoutFinding ? 'active' : ''}`}
                          onClick={() => setCheckoutFinding(!checkoutFinding)}
                        >
                          <div className="checkbox-circle"></div>
                          {customTags.finding}
                        </div>

                        <div 
                          className={`tag-chip ${checkoutPhone ? 'active' : ''}`}
                          onClick={() => setCheckoutPhone(!checkoutPhone)}
                        >
                          <div className="checkbox-circle"></div>
                          {customTags.phone}
                        </div>

                        <div 
                          className={`tag-chip ${checkoutWaiting ? 'active' : ''}`}
                          onClick={() => setCheckoutWaiting(!checkoutWaiting)}
                        >
                          <div className="checkbox-circle"></div>
                          {customTags.waiting}
                        </div>
                      </div>
                    </div>

                    {/* Hands-Free Voice Assistant Widget */}
                    <div style={{
                      padding: '1rem',
                      backgroundColor: 'rgba(99, 102, 241, 0.05)',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)',
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.75rem',
                      alignItems: 'center',
                      marginTop: '0.25rem',
                      marginBottom: '0.25rem'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', width: '100%', justifyContent: 'space-between' }}>
                        <span style={{ fontSize: '0.85rem', fontWeight: 600, color: 'var(--accent-primary)', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                          <Mic size={16} /> Hands-Free Voice Checkout
                        </span>
                        {isListening && <div className="mic-pulse"></div>}
                      </div>

                      <button 
                        type="button"
                        className={`btn-secondary ${isListening ? 'listening' : ''}`}
                        onClick={() => startVoiceAssistant('checkout')}
                        style={{ 
                          width: '54px', 
                          height: '54px', 
                          borderRadius: '50%', 
                          display: 'flex', 
                          alignItems: 'center', 
                          justifyContent: 'center', 
                          padding: 0,
                          backgroundColor: isListening ? 'var(--accent-danger)' : 'var(--bg-tertiary)',
                          borderColor: isListening ? 'var(--accent-danger)' : 'var(--border-color)',
                          color: '#fff',
                          cursor: 'pointer',
                          boxShadow: isListening ? '0 0 15px rgba(244, 63, 94, 0.4)' : 'none',
                          transition: 'all 0.2s ease-in-out'
                        }}
                      >
                        {isListening ? <MicOff size={22} style={{ color: '#fff' }} /> : <Mic size={22} style={{ color: 'var(--accent-primary)' }} />}
                      </button>

                      <div style={{ textAlign: 'center', width: '100%' }}>
                        {isListening ? (
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-primary)', fontStyle: 'italic' }}>
                            Listening... speak your delay reasons
                          </span>
                        ) : (
                          <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                            Tap the microphone to speak your checkout list.
                          </span>
                        )}

                        {voiceText && (
                          <div style={{ 
                            marginTop: '0.5rem', 
                            padding: '0.5rem', 
                            backgroundColor: 'var(--bg-primary)', 
                            borderRadius: 'var(--radius-sm)', 
                            fontSize: '0.8rem', 
                            fontFamily: 'var(--font-mono)',
                            color: 'var(--text-secondary)',
                            border: '1px solid var(--border-color)',
                            textAlign: 'left'
                          }}>
                            <strong>Transcribed Speech:</strong> "{voiceText}"
                          </div>
                        )}

                        {voiceFeedback && (
                          <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--accent-success)', marginTop: '0.35rem' }}>
                            {voiceFeedback}
                          </div>
                        )}
                      </div>
                    </div>

                    <button className="btn-primary" onClick={() => submitCheckout()}>
                      <Check size={18} />
                      Log Checkout & Retrain ML
                    </button>
                  </div>
                )}

                {simState === 'success_modal' && checkoutResult && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', textAlign: 'center' }}>
                    <CheckCircle size={54} style={{ color: 'var(--accent-success)', margin: '0 auto' }} />
                    <h3 style={{ fontSize: '1.5rem', fontWeight: 700 }}>Logged Successfully</h3>
                    
                    <div style={{ padding: '1rem', backgroundColor: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-color)', fontSize: '0.9rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', textAlign: 'left' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span style={{ color: 'var(--text-secondary)' }}>Departure Delay:</span>
                        <span style={{ fontWeight: 700, color: checkoutResult.calculated_delay_min > 0 ? 'var(--accent-danger)' : 'var(--accent-success)' }}>
                          {checkoutResult.calculated_delay_min > 0 ? `+${checkoutResult.calculated_delay_min} minutes` : 'On Time!'}
                        </span>
                      </div>
                      
                      {checkoutResult.calculated_delay_min > 0 && (
                        <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.5rem', marginTop: '0.25rem' }}>
                          <span className="form-label" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '0.35rem' }}>Smart Delay Allocation:</span>
                          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem', fontSize: '0.85rem' }}>
                            {checkoutResult.allocations.one_last_task > 0 && (
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>{profileType === 'it_professional' ? 'Git Commit Delay:' : profileType === 'professor' ? 'Slide/Brief Check:' : 'One Last Task:'}</span>
                                <span>{checkoutResult.allocations.one_last_task} mins</span>
                              </div>
                            )}
                            {checkoutResult.allocations.finding_items > 0 && (
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>{profileType === 'it_professional' ? 'Badge/Charger Search:' : profileType === 'professor' ? 'HDMI Adapter Search:' : 'Finding Items:'}</span>
                                <span>{checkoutResult.allocations.finding_items} mins</span>
                              </div>
                            )}
                            {checkoutResult.allocations.phone_usage > 0 && (
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>{profileType === 'it_professional' ? 'Slack Notification Delay:' : profileType === 'professor' ? 'Thesis Reading Bias:' : 'Phone checking:'}</span>
                                <span>{checkoutResult.allocations.phone_usage} mins</span>
                              </div>
                            )}
                            {checkoutResult.allocations.waiting > 0 && (
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Waiting:</span>
                                <span>{checkoutResult.allocations.waiting} mins</span>
                              </div>
                            )}
                            {checkoutResult.allocations.prep_bias > 0 && (
                              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)' }}>Time Estimation Prep Bias:</span>
                                <span>{checkoutResult.allocations.prep_bias} mins</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}
                    </div>

                    <button className="btn-secondary" onClick={() => setSimState('idle')}>
                      Back to Start
                    </button>
                  </div>
                )}
              </div>

              {/* Explanatory Sidebar */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">Segregated Data Seeding</h3>
                  </div>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Your account has been pre-seeded with 60 days of historical logs tailored specifically for a <strong>{profileType?.replace('_', ' ')}</strong>. 
                  </p>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', marginTop: '0.5rem' }}>
                    Your custom Random Forest model has been trained only on your personal behaviors and is isolated from other users in the SQL database.
                  </p>
                </div>

                <div className="card">
                  <div className="card-header">
                    <h3 className="card-title">3-Second Checkout</h3>
                  </div>
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    Instead of filling out annoying sheets, the checkout prompt dynamically adjusts to your career path. Tapping the checkbox records your key delays and instantly aligns your personal prediction engine.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Note: ML Performance comparison page is deleted to represent a clean consumer application */}

          {/* TAB 4: LOG HISTORY */}
          {activeTab === 'history' && (
            <div className="card">
              <div className="card-header">
                <h3 className="card-title">
                  <History />
                  Departure Logs History
                </h3>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>Scroll through your 60-day historical logs</span>
              </div>
              
              <div className="table-wrapper" style={{ marginTop: '0.5rem' }}>
                <table className="logs-table">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Class/Event</th>
                      <th>Day</th>
                      <th>Planned Prep</th>
                      <th>Actual Prep</th>
                      <th>Actual Leave</th>
                      <th>Departure Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {logs.map((log) => (
                      <tr key={log.id}>
                        <td style={{ fontWeight: 500, color: 'var(--text-primary)' }}>{log.date}</td>
                        <td>{log.event_name}</td>
                        <td>{log.day_of_week}</td>
                        <td>{log.estimated_prep_duration_min} min</td>
                        <td>{log.actual_prep_duration_min} min</td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>{log.actual_departure.substring(0, 5)}</td>
                        <td>
                          <span className={`status-badge ${log.total_delay_min > 0.1 ? 'late' : 'on-time'}`}>
                            {log.total_delay_min > 0.1 ? `${Math.ceil(log.total_delay_min)}m late` : 'On Time'}
                          </span>
                        </td>
                      </tr>
                    ))}
                    {logs.length === 0 && (
                      <tr>
                        <td colSpan={7} style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                          No logs found.
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
