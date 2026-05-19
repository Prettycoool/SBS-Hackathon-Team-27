import { useEffect, useRef, useState } from 'react';
import PatientForm  from './components/PatientForm.jsx';
import ResultsPanel from './components/ResultsPanel.jsx';

const THEME_KEY = 'diana-theme';

function DianaLogo() {
  return (
    <svg
      className="header-logo-svg"
      width="50" height="50"
      viewBox="0 0 50 50"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      aria-hidden="true"
    >
      {/* Outer ring */}
      <circle cx="25" cy="25" r="23" fill="rgba(255,255,255,0.10)" stroke="rgba(255,255,255,0.30)" strokeWidth="1.5"/>
      {/* Inner subtle fill for depth */}
      <circle cx="25" cy="25" r="19" fill="rgba(255,255,255,0.05)"/>
      {/* Venus / female symbol — circle */}
      <circle cx="25" cy="18" r="9" stroke="white" strokeWidth="2.2" fill="none"/>
      {/* Stem */}
      <line x1="25" y1="27" x2="25" y2="39" stroke="white" strokeWidth="2.2" strokeLinecap="round"/>
      {/* Crossbar */}
      <line x1="19.5" y1="33" x2="30.5" y2="33" stroke="white" strokeWidth="2.2" strokeLinecap="round"/>
    </svg>
  );
}

const SECTIONS = [
  { id: 'section-demographics', label: 'Demographics' },
  { id: 'section-menstrual',    label: 'Menstrual'    },
  { id: 'section-hormonal',     label: 'Hormonal'     },
  { id: 'section-ultrasound',   label: 'Ultrasound'   },
  { id: 'section-symptoms',     label: 'Symptoms'     },
  { id: 'section-vitals',       label: 'Vitals'       },
];

function SectionProgress({ activeSection, onNavigate }) {
  return (
    <div className="section-progress">
      {SECTIONS.map((s, i) => (
        <span key={s.id} style={{ display: 'contents' }}>
          <button
            className={`progress-step ${activeSection === s.id ? 'active' : ''}`}
            onClick={() => onNavigate(s.id)}
          >
            {s.label}
          </button>
          {i < SECTIONS.length - 1 && <span className="progress-sep">›</span>}
        </span>
      ))}
    </div>
  );
}

const API_URL       = 'http://localhost:8000/diagnose';
const PARSE_NOTE_URL = 'http://localhost:8000/parse-note';

// Maps /parse-note response keys → form field names + optional coercion
const OCR_FIELD_MAP = {
  age:                 { key: 'age',               coerce: Number },
  bmi:                 { key: 'bmi',               coerce: Number },
  bp_systolic:         { key: 'bp_systolic',        coerce: v => Math.round(Number(v)) },
  bp_diastolic:        { key: 'bp_diastolic',       coerce: v => Math.round(Number(v)) },
  fsh:                 { key: 'fsh_miu_ml',         coerce: Number },
  lh:                  { key: 'lh_miu_ml',          coerce: Number },
  amh:                 { key: 'amh_ng_ml',          coerce: Number },
  tsh:                 { key: 'tsh_miu_l',          coerce: Number },
  prolactin:           { key: 'prl_ng_ml',          coerce: Number },
  progesterone:        { key: 'progesterone_ng_ml', coerce: Number },
  cycle_regular:       { key: 'cycle_regular',      coerce: v => Boolean(v) },
  cycle_length:        { key: 'cycle_length_days',  coerce: v => Math.max(1, Math.min(15, Math.round(Number(v)))) },
  follicle_no_l:       { key: 'follicle_no_l',      coerce: v => Math.round(Number(v)) },
  follicle_no_r:       { key: 'follicle_no_r',      coerce: v => Math.round(Number(v)) },
  avg_f_size_l:        { key: 'avg_f_size_l_mm',    coerce: Number },
  avg_f_size_r:        { key: 'avg_f_size_r_mm',    coerce: Number },
  weight_gain:         { key: 'weight_gain',        coerce: v => Boolean(v) },
  hair_growth:         { key: 'hair_growth',        coerce: v => Boolean(v) },
  skin_darkening:      { key: 'skin_darkening',     coerce: v => Boolean(v) },
  hair_loss:           { key: 'hair_loss',          coerce: v => Boolean(v) },
  pimples:             { key: 'pimples',            coerce: v => Boolean(v) },
  galactorrhea:        { key: 'galactorrhea',       coerce: v => Boolean(v) },
  chronic_pelvic_pain: { key: 'chronic_pelvic_pain', coerce: v => Math.max(0, Math.min(10, Math.round(Number(v)))) },
};

function DoctorsNotePanel({ onExtracted }) {
  const [note,       setNote]       = useState('');
  const [extracting, setExtracting] = useState(false);
  const [error,      setError]      = useState(null);

  const handleExtract = async () => {
    if (!note.trim()) return;
    setExtracting(true);
    setError(null);
    try {
      const res = await fetch(PARSE_NOTE_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ note }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail ?? `Server returned ${res.status}`);
      }
      const parsed = await res.json();
      const updates = {};
      for (const [ocrKey, { key, coerce }] of Object.entries(OCR_FIELD_MAP)) {
        const val = parsed[ocrKey];
        if (val !== null && val !== undefined) {
          updates[key] = coerce(val);
        }
      }
      onExtracted(updates);
    } catch (err) {
      setError(err.message);
    } finally {
      setExtracting(false);
    }
  };

  return (
    <div className="note-panel card">
      <div className="card-header"><h2>Paste Doctor's Note</h2></div>
      <div className="card-body">
        <p className="note-hint">
          Paste or type a clinical note. DIANA will extract all recognisable
          values and pre-fill the form — you can review and adjust before running
          the diagnostic.
        </p>
        <textarea
          className="note-textarea"
          placeholder="e.g. 28-year-old female. BMI 27.4. LH 12.3 mIU/mL, FSH 5.1 mIU/mL. Irregular cycle. Hirsutism noted. TSH 2.1 mIU/L. Follicles L: 14, R: 16..."
          value={note}
          onChange={e => setNote(e.target.value)}
          rows={10}
        />
        {error && <div className="note-error">{error}</div>}
        <button
          className="btn-extract"
          onClick={handleExtract}
          disabled={extracting || !note.trim()}
        >
          {extracting
            ? <><span className="spinner-inline" /> Extracting…</>
            : 'Extract & Fill Form'}
        </button>
      </div>
    </div>
  );
}

// Default form values match the training dataset population medians,
// so the slider positions represent a "typical" patient at startup.
const DEFAULTS = {
  age:              31,
  bmi:              24.2,
  cycle_regular:    true,
  cycle_length_days:5,
  fsh_miu_ml:       4.9,
  lh_miu_ml:        2.3,
  amh_ng_ml:        3.7,
  tsh_miu_l:        2.3,
  prl_ng_ml:          21.9,
  progesterone_ng_ml: 8.0,
  follicle_no_l:    5,
  follicle_no_r:    6,
  avg_f_size_l_mm:  15.0,
  avg_f_size_r_mm:  16.0,
  weight_gain:      false,
  hair_growth:      false,
  skin_darkening:   false,
  hair_loss:        false,
  pimples:             false,
  galactorrhea:        false,
  chronic_pelvic_pain: 0,
  bp_systolic:      110,
  bp_diastolic:     80,
};

export default function App() {
  const [form,           setForm]          = useState(DEFAULTS);
  const [results,        setResults]       = useState(null);
  const [loading,        setLoading]       = useState(false);
  const [error,          setError]         = useState(null);
  const [activeSection,  setActiveSection] = useState('section-demographics');
  const [isDark,         setIsDark]        = useState(
    () => localStorage.getItem(THEME_KEY) !== 'light'
  );
  const [iconFlipping,   setIconFlipping]  = useState(false);
  const [inputMode,      setInputMode]     = useState('manual'); // 'manual' | 'note'
  const [extractedCount, setExtractedCount] = useState(0);
  const resultsRef      = useRef(null);
  const isScrollingRef  = useRef(false);

  const navigateToSection = (id) => {
    const el  = document.getElementById(id);
    const bar = document.querySelector('.section-progress');
    if (!el) return;
    const offset = (bar?.offsetHeight ?? 44) + 16;
    const top    = el.getBoundingClientRect().top + window.scrollY - offset;
    isScrollingRef.current = true;
    setActiveSection(id);
    window.scrollTo({ top, behavior: 'smooth' });
    // Clear the gate once the browser signals the scroll is done.
    // scrollend is supported in Chrome 114+, FF 109+, Safari 16.2+;
    // the timeout is a fallback for older browsers.
    const done = () => { isScrollingRef.current = false; };
    window.addEventListener('scrollend', done, { once: true });
    setTimeout(done, 800);
  };

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
    localStorage.setItem(THEME_KEY, isDark ? 'dark' : 'light');
  }, [isDark]);

  const toggleTheme = () => {
    if (iconFlipping) return;
    setIconFlipping(true);
    setTimeout(() => setIsDark(d => !d), 210);
    setTimeout(() => setIconFlipping(false), 440);
  };

  useEffect(() => {
    const handleScroll = () => {
      if (isScrollingRef.current) return;
      const ids = SECTIONS.map(s => s.id);
      for (let i = ids.length - 1; i >= 0; i--) {
        const el = document.getElementById(ids[i]);
        if (el && el.getBoundingClientRect().top <= 120) {
          setActiveSection(ids[i]);
          return;
        }
      }
      setActiveSection(ids[0]);
    };
    window.addEventListener('scroll', handleScroll, { passive: true });
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const handleChange = (name, value) => {
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const handleExtracted = (updates) => {
    setForm(prev => ({ ...prev, ...updates }));
    setExtractedCount(Object.keys(updates).length);
    setInputMode('manual');
  };

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(API_URL, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(form),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail ?? `Server returned ${res.status}`);
      }
      const data = await res.json();
      setResults(data);
      // On narrow screens, scroll the results into view after they render
      setTimeout(() => resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">

      {/* Header */}
      <header className="app-header">
        <DianaLogo />
        <div>
          <h1>DIANA</h1>
          <p>Intelligent differential diagnosis for women's reproductive health</p>
        </div>
        <div className="header-right">
          <span className="header-version">v1.0 · SBS Hackathon 2026</span>
          <button
            className="theme-toggle"
            onClick={toggleTheme}
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            <span className={`theme-icon${iconFlipping ? ' flipping' : ''}`}>
              {isDark ? '🌙' : '☀️'}
            </span>
            {isDark ? 'Dark' : 'Light'}
          </button>
        </div>
      </header>

      {/* Disclaimer */}
      <div className="app-disclaimer">
        ⚠ For research and educational use only. Not a substitute for clinical diagnosis by a qualified physician.
      </div>

      {/* Section progress — hidden in note mode */}
      {inputMode === 'manual' && (
        <SectionProgress activeSection={activeSection} onNavigate={navigateToSection} />
      )}

      {/* Two-panel layout */}
      <main className="app-body">
        <div>
          {/* Input mode tabs */}
          <div className="input-mode-tabs">
            <button
              className={`input-tab ${inputMode === 'manual' ? 'active' : ''}`}
              onClick={() => setInputMode('manual')}
            >
              Manual Entry
            </button>
            <button
              className={`input-tab ${inputMode === 'note' ? 'active' : ''}`}
              onClick={() => { setInputMode('note'); setExtractedCount(0); }}
            >
              Paste Doctor's Note
            </button>
          </div>

          {/* Auto-fill success banner */}
          {extractedCount > 0 && inputMode === 'manual' && (
            <div className="note-success-banner">
              {extractedCount} field{extractedCount !== 1 ? 's' : ''} auto-filled
              from doctor's note — review and adjust as needed.
            </div>
          )}

          {inputMode === 'note'
            ? <DoctorsNotePanel onExtracted={handleExtracted} />
            : <PatientForm
                form={form}
                onChange={handleChange}
                onSubmit={handleSubmit}
                loading={loading}
              />
          }
        </div>

        <div className="results-panel" ref={resultsRef}>
          <ResultsPanel results={results} loading={loading} error={error} form={form} />
        </div>
      </main>

      <footer className="app-footer">
        DIANA · SBS Hackathon Team 27 · XGBoost + SHAP · Stage 1 ROC-AUC 0.954
      </footer>
    </div>
  );
}
