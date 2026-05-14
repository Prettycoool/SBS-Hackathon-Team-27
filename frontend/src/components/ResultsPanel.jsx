// Results panel — shown after the /diagnose API call returns.
// Displays: gauge, primary diagnosis, Rotterdam checklist,
//           plain-English SHAP summary, SHAP bars, differential flags.

import GaugeChart from './GaugeChart.jsx';
import { generatePDF } from '../utils/generatePDF.js';

const MAX_SHAP = 2.0;

// ── SHAP bar ─────────────────────────────────────────────────────────────────

function ShapBar({ entry }) {
  const pct    = Math.min(Math.abs(entry.shap_value) / MAX_SHAP * 100, 100);
  const isUp   = entry.direction === 'increases_risk';
  const valStr = Number.isInteger(entry.feature_value)
    ? entry.feature_value
    : entry.feature_value.toFixed(2);

  return (
    <div className="shap-entry">
      <div className="shap-meta">
        <span className="shap-name">{entry.display_name}</span>
        <span className="shap-val">value: {valStr}</span>
      </div>
      <div className="shap-track">
        <div className={`shap-fill ${isUp ? 'up' : 'down'}`} style={{ width: `${pct}%` }} />
      </div>
      <div className={`shap-dir ${isUp ? 'up' : 'down'}`}>
        {isUp ? '▲ Increases PCOS risk' : '▼ Decreases PCOS risk'}
      </div>
    </div>
  );
}

// ── Plain-English SHAP summary ────────────────────────────────────────────────
// Maps the top SHAP feature to a human-readable sentence.

const SHAP_SENTENCES = {
  'Follicles — Right Ovary': {
    up:   v => `Elevated right-ovary follicle count (${v}) is the strongest indicator pointing toward PCOS.`,
    down: v => `Low right-ovary follicle count (${v}) is the primary reason PCOS appears unlikely.`,
  },
  'Follicles — Left Ovary': {
    up:   v => `Elevated left-ovary follicle count (${v}) is the strongest indicator pointing toward PCOS.`,
    down: v => `Low left-ovary follicle count (${v}) is the primary reason PCOS appears unlikely.`,
  },
  'Cycle Regularity': {
    up:   () => 'An irregular menstrual cycle is the strongest clinical factor pointing toward PCOS.',
    down: () => 'Regular menstrual cycles are the strongest single factor suggesting PCOS is unlikely.',
  },
  'Excessive Hair Growth': {
    up:   () => 'Reported excessive hair growth (hirsutism) is the leading clinical indicator in this prediction.',
    down: () => 'Absence of excessive hair growth is the main factor suggesting PCOS is unlikely.',
  },
  'Skin Darkening': {
    up:   () => 'Skin darkening (acanthosis nigricans) is the leading clinical sign driving this prediction.',
    down: () => 'No skin darkening is a key factor suggesting PCOS is unlikely.',
  },
  'Acne / Pimples': {
    up:   () => 'Hormonal acne pattern is the primary clinical driver of this prediction.',
    down: () => 'Absence of hormonal acne is a key factor suggesting PCOS is unlikely.',
  },
  'Weight Gain': {
    up:   () => 'Unexplained weight gain is the primary driver of this PCOS prediction.',
    down: () => 'No unexplained weight gain is the main factor reducing PCOS likelihood.',
  },
  'AMH': {
    up:   v => `Elevated AMH (${v} ng/mL) strongly suggests polycystic ovarian morphology.`,
    down: v => `Low AMH (${v} ng/mL) is the primary hormonal factor suggesting PCOS is unlikely.`,
  },
  'FSH / LH Ratio': {
    up:   v => `An abnormal FSH/LH ratio (${v}) is the primary hormonal driver of this prediction.`,
    down: v => `A healthy FSH/LH ratio (${v}) is the main hormonal factor suggesting PCOS is unlikely.`,
  },
  'LH': {
    up:   v => `Elevated LH (${v} mIU/mL) is the primary hormonal driver of this prediction.`,
    down: v => `Low LH (${v} mIU/mL) is the main hormonal factor suggesting PCOS is unlikely.`,
  },
  'Avg. Follicle Size (L)': {
    up:   v => `Small left-ovary follicle size (${v} mm) is consistent with polycystic morphology.`,
    down: v => `Normal left-ovary follicle size (${v} mm) is the main ultrasound factor against PCOS.`,
  },
  'Avg. Follicle Size (R)': {
    up:   v => `Small right-ovary follicle size (${v} mm) is consistent with polycystic morphology.`,
    down: v => `Normal right-ovary follicle size (${v} mm) is the main ultrasound factor against PCOS.`,
  },
};

function shapToSentence(entry) {
  if (!entry) return null;
  const { display_name, feature_value, direction } = entry;
  const isUp   = direction === 'increases_risk';
  const valStr = Number.isInteger(feature_value) ? feature_value : feature_value.toFixed(1);
  const lookup = SHAP_SENTENCES[display_name];
  if (lookup) return isUp ? lookup.up(valStr) : lookup.down(valStr);
  // Fallback for unmapped features
  return isUp
    ? `${display_name} (${valStr}) is the primary driver of this PCOS prediction.`
    : `${display_name} (${valStr}) is the strongest factor suggesting PCOS is unlikely.`;
}

// ── Rotterdam Criteria Tracker ────────────────────────────────────────────────

function RotterdamTracker({ form }) {
  if (!form) return null;

  const polycystic = (form.follicle_no_l ?? 0) > 12 || (form.follicle_no_r ?? 0) > 12;
  const hyperandro = !!(form.hair_growth || form.skin_darkening || form.pimples);
  const anovulation = form.cycle_regular === false;

  const criteria = [
    {
      icon:   '🔵',
      label:  'Polycystic Ovaries',
      detail: `Follicle count > 12 — L: ${form.follicle_no_l ?? '?'}, R: ${form.follicle_no_r ?? '?'}`,
      met:    polycystic,
    },
    {
      icon:   '🔴',
      label:  'Hyperandrogenism',
      detail: 'Hair growth, skin darkening, or acne reported',
      met:    hyperandro,
    },
    {
      icon:   '🟢',
      label:  'Anovulation',
      detail: 'Irregular menstrual cycle',
      met:    anovulation,
    },
  ];

  const metCount  = criteria.filter(c => c.met).length;
  const pcosLikely = metCount >= 2;

  return (
    <div className="card">
      <div className="card-header">
        <h2>Rotterdam Criteria Checklist</h2>
      </div>
      <div className="card-body" style={{ paddingBottom: 14 }}>
        {criteria.map(c => (
          <div key={c.label} className={`rotterdam-row ${c.met ? 'met' : 'unmet'}`}>
            <span className="rotterdam-icon">{c.icon}</span>
            <div className="rotterdam-text">
              <span className="rotterdam-label">{c.label}</span>
              <span className="rotterdam-detail">{c.detail}</span>
            </div>
            <span className="rotterdam-status">
              {c.met ? '✅ Met' : '⬜ Not Met'}
            </span>
          </div>
        ))}

        <div className={`rotterdam-note ${pcosLikely ? 'likely' : 'unlikely'}`}>
          <strong>{metCount}/3</strong> criteria present
          {pcosLikely
            ? ' — meets Rotterdam threshold for PCOS diagnosis'
            : ' — below Rotterdam threshold for PCOS diagnosis'}
        </div>

        <div className="rotterdam-consensus">
          PCOS diagnosed if 2 of 3 criteria met (Rotterdam consensus)
        </div>
      </div>
    </div>
  );
}

// ── Differential flag card ────────────────────────────────────────────────────

function FlagCard({ flagged, icon, label, biomarker, value, unit, threshold }) {
  return (
    <div className={`flag-card ${flagged ? 'flagged' : 'normal'}`}>
      <div className="flag-top">
        <span className="flag-cond-icon">{icon}</span>
        {label}
      </div>
      <div className={`flag-status-label ${flagged ? 'flagged' : 'normal'}`}>
        {flagged ? '⚠️ Flagged — Investigate' : 'Within Normal Range'}
      </div>
      <div className="flag-biomarker">
        {biomarker}: {value !== null ? `${value} ${unit}` : '—'}
      </div>
      <div className="flag-thresh">Threshold: &gt; {threshold} {unit}</div>
    </div>
  );
}

function EndoCard({ flagged, icon, probability }) {
  return (
    <div className={`flag-card ${flagged ? 'flagged' : 'normal'}`}>
      <div className="flag-top">
        <span className="flag-cond-icon">{icon}</span>
        Endometriosis Risk
      </div>
      <div className={`flag-status-label ${flagged ? 'flagged' : 'normal'}`}>
        {flagged ? '⚠️ Flagged — Investigate' : 'Within Normal Range'}
      </div>
      <div className="flag-biomarker">
        XGBoost probability: {probability !== null ? `${Math.round(probability * 100)}%` : '—'}
      </div>
      <div className="flag-thresh">Threshold: ≥ 50%</div>
    </div>
  );
}

// ── Main results panel ────────────────────────────────────────────────────────

export default function ResultsPanel({ results, loading, error, form }) {
  if (loading) {
    return (
      <div className="card">
        <div className="spinner-wrap"><div className="spinner" /></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="results-error">
        <strong>Error:</strong> {error}
        <br /><small>Make sure the FastAPI backend is running on port 8000.</small>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="card">
        <div className="results-empty">
          <div className="results-empty-icon">🔬</div>
          <p>Fill in patient data and click<br /><strong>Run Diagnostic Pipeline</strong></p>
        </div>
      </div>
    );
  }

  const { pcos_prediction, pcos_probability, differential_flags, shap_explanations } = results;
  const df      = differential_flags;
  const topShap = shap_explanations?.[0] ?? null;

  // ── [3] Inconclusive state: 40–60% probability ────────────────────────────
  const inconclusive = pcos_probability >= 0.40 && pcos_probability <= 0.60;
  const headerClass  = inconclusive ? 'inconclusive' : pcos_prediction ? 'positive' : 'negative';
  const badgeText    = inconclusive
    ? 'Inconclusive — Further Investigation Recommended'
    : pcos_prediction ? 'PCOS Detected' : 'PCOS Not Detected';
  const subText = inconclusive
    ? `Probability ${Math.round(pcos_probability * 100)}% — Stage 2 differential diagnosis has been automatically initiated`
    : `Stage 1 confidence: ${Math.round(
        pcos_prediction ? pcos_probability * 100 : (1 - pcos_probability) * 100
      )}%`;

  return (
    <>
      {/* ── [3] Primary diagnosis + gauge ──────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <h2>Diagnostic Result</h2>
        </div>
        <div className="card-body" style={{ paddingBottom: 12 }}>
          <div className={`diag-header ${headerClass}`}>
            <div className="diag-badge">
              <span className="diag-dot" />
              {badgeText}
            </div>
            <div className="diag-text">{subText}</div>
          </div>
        </div>

        {/* Gauge */}
        <div className="gauge-wrap">
          <GaugeChart value={pcos_probability} />
        </div>

        {/* [2] Plain-English summary of the top SHAP driver */}
        {topShap && (
          <div className="shap-summary">
            {shapToSentence(topShap)}
          </div>
        )}
        <div style={{ height: 14 }} />
      </div>

      {/* ── [1] Rotterdam criteria tracker ─────────────────────────────── */}
      <RotterdamTracker form={form} />

      {/* ── SHAP bar chart ──────────────────────────────────────────────── */}
      <div className="card">
        <div className="card-header">
          <h2>Key Contributing Factors</h2>
        </div>
        <div className="card-body">
          <div className="shap-list">
            {shap_explanations.map((e, i) => (
              <ShapBar key={i} entry={e} />
            ))}
          </div>
          <p style={{ fontSize: 11, color: 'var(--c-muted)', marginTop: 12 }}>
            SHAP values show each feature's contribution to the prediction.
            Bar width reflects magnitude; colour indicates direction.
          </p>
        </div>
      </div>

      {/* ── Differential flags (Not-PCOS only) ─────────────────────────── */}
      {!pcos_prediction && df && (
        <div className="card">
          <div className="card-header">
            <h2>Differential Diagnosis — Stage 2</h2>
          </div>
          <div className="card-body">
            <div className="flags-grid">
              <FlagCard icon="🦋" flagged={df.hypothyroidism}
                label="Hypothyroidism" biomarker="TSH"
                value={df.tsh_value} unit="mIU/L" threshold="4.0" />
              <FlagCard icon="💊" flagged={df.hyperprolactinemia}
                label="Hyperprolactinemia" biomarker="PRL"
                value={df.prl_value} unit="ng/mL" threshold="25" />
              <FlagCard icon="🫀" flagged={df.cushings_hypertension}
                label="Cushing's / HTN" biomarker="Systolic BP"
                value={df.sbp_value} unit="mmHg" threshold="140" />
              <EndoCard icon="🌸" flagged={df.endometriosis_risk}
                probability={df.endometriosis_probability} />
            </div>
            <p style={{ fontSize: 11, color: 'var(--c-muted)', marginTop: 12 }}>
              Flags indicate values exceeding published clinical thresholds.
              Results require clinical correlation and are not diagnostic alone.
            </p>
          </div>
        </div>
      )}

      {/* ── Download PDF report ─────────────────────────────────────────── */}
      <button
        className="btn-download"
        onClick={() => generatePDF({
          form,
          results,
          shapSentence: shapToSentence(topShap),
        })}
      >
        ⬇ Download PDF Report
      </button>
    </>
  );
}
