import { useRef, useState } from 'react';
import PatientForm  from './components/PatientForm.jsx';
import ResultsPanel from './components/ResultsPanel.jsx';

const API_URL = 'http://localhost:8000/diagnose';

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
  prl_ng_ml:        21.9,
  follicle_no_l:    5,
  follicle_no_r:    6,
  avg_f_size_l_mm:  15.0,
  avg_f_size_r_mm:  16.0,
  weight_gain:      false,
  hair_growth:      false,
  skin_darkening:   false,
  hair_loss:        false,
  pimples:          false,
  bp_systolic:      110,
  bp_diastolic:     80,
};

export default function App() {
  const [form,    setForm]    = useState(DEFAULTS);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState(null);
  const resultsRef = useRef(null);

  const handleChange = (name, value) => {
    setForm(prev => ({ ...prev, [name]: value }));
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
        <div className="app-header-icon">🧬</div>
        <div>
          <h1>DIANA</h1>
          <p>Intelligent differential diagnosis for women's reproductive health</p>
        </div>
      </header>

      {/* Disclaimer */}
      <div className="app-disclaimer">
        ⚠ For research and educational use only. Not a substitute for clinical diagnosis by a qualified physician.
      </div>

      {/* Two-panel layout */}
      <main className="app-body">
        <div>
          <PatientForm
            form={form}
            onChange={handleChange}
            onSubmit={handleSubmit}
            loading={loading}
          />
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
