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

function FlagCard({ flagged, icon, label, biomarker, value, unit, threshold, note }) {
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
      {note && <div className="flag-note">{note}</div>}
    </div>
  );
}

function EndoCard({ flagged, icon, probability, pelvicPainNote }) {
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
      {pelvicPainNote && <div className="flag-note">{pelvicPainNote}</div>}
    </div>
  );
}

// ── Time Cost of Misdiagnosis card ───────────────────────────────────────────

const MISDIAGNOSIS_DATA = {
  A:        { years: 1.9, misdiagnosis: 'Idiopathic hirsutism' },
  B:        { years: 2.8, misdiagnosis: 'Idiopathic hirsutism (54% of cases)' },
  C:        { years: 2.4, misdiagnosis: 'Stress-related amenorrhea' },
  D:        { years: 3.8, misdiagnosis: 'Subclinical hypothyroidism' },
  Atypical: { years: 4.2, misdiagnosis: 'No diagnosis given' },
};

function TimeCostCard({ phenotype }) {
  const data = phenotype ? MISDIAGNOSIS_DATA[phenotype.code] : null;
  if (!data) return null;
  return (
    <div className="time-cost-card">
      <div className="time-cost-header">⏱ Without early detection tools:</div>
      <div className="time-cost-rows">
        <div className="time-cost-row">
          <span className="time-cost-label">Average time to diagnosis</span>
          <span className="time-cost-value">{data.years} years</span>
        </div>
        <div className="time-cost-row">
          <span className="time-cost-label">Doctors seen before diagnosis</span>
          <span className="time-cost-value">3–4 on average</span>
        </div>
        <div className="time-cost-row">
          <span className="time-cost-label">Most commonly misdiagnosed as</span>
          <span className="time-cost-value">{data.misdiagnosis}</span>
        </div>
      </div>
      <div className="time-cost-diana">
        ✓ With DIANA: Flagged today — estimated {data.years} years saved — fertility planning window preserved
      </div>
    </div>
  );
}

// ── Long-term Risk Profile card ───────────────────────────────────────────────

const RISK_PROFILE = {
  A:        { t2d: 4.0, ms: 3.2, cvd: 2.8, mh: 2.8, infertility: 72 },
  B:        { t2d: 3.2, ms: 2.8, cvd: 2.4, mh: 2.8, infertility: 68 },
  C:        { t2d: 2.8, ms: 2.4, cvd: 2.0, mh: 2.4, infertility: 55 },
  D:        { t2d: 2.4, ms: 2.0, cvd: 1.8, mh: 2.4, infertility: 48 },
  Atypical: { t2d: 2.5, ms: 2.2, cvd: 1.9, mh: 2.6, infertility: 52 },
};

const RISK_ROWS = [
  { key: 't2d', label: 'Type 2 Diabetes',      fmt: v => `${v}x relative risk` },
  { key: 'ms',  label: 'Metabolic Syndrome',    fmt: v => `${v}x relative risk` },
  { key: 'cvd', label: 'Cardiovascular Disease', fmt: v => `${v}x relative risk` },
  { key: 'mh',  label: 'Depression / Anxiety',  fmt: v => `${v}x relative risk` },
  { key: 'infertility', label: 'Infertility Risk', fmt: v => `${v}%` },
];

function riskColor(key, val) {
  if (key === 'infertility') return val >= 65 ? 'red' : val >= 55 ? 'amber' : 'yellow';
  return val >= 3 ? 'red' : val >= 2 ? 'amber' : 'yellow';
}

function RiskProfileCard({ phenotype }) {
  const data = phenotype ? RISK_PROFILE[phenotype.code] : null;
  if (!data) return null;
  return (
    <div className="risk-profile-card">
      <div className="risk-profile-header">Long-term Risk Profile — If Left Undiagnosed</div>
      <div className="risk-profile-rows">
        {RISK_ROWS.map(({ key, label, fmt }) => {
          const val = data[key];
          const tier = riskColor(key, val);
          return (
            <div key={key} className={`risk-row risk-row-${tier}`}>
              <span className="risk-label">{label}</span>
              <span className={`risk-value risk-value-${tier}`}>{fmt(val)}</span>
            </div>
          );
        })}
      </div>
      <div className="risk-goodnews">
        <div className="risk-goodnews-title">✓ Good news — Early intervention helps:</div>
        <p>Lifestyle modification reduces T2D risk by up to 58% (NEJM Diabetes Prevention Program). Metformin reduces metabolic risk. Mental health screening recommended.</p>
        <div className="risk-checklist-title">Recommended investigations:</div>
        <ul className="risk-checklist">
          <li>Fasting glucose + insulin</li>
          <li>Full lipid panel</li>
          <li>PHQ-9 depression screening</li>
          <li>Blood pressure monitoring</li>
        </ul>
      </div>
    </div>
  );
}

// ── Age-aware diagnostic flag card ───────────────────────────────────────────

function AgeFlagCard({ age_flag }) {
  if (age_flag === 'peak_gap') {
    return (
      <div className="age-flag-card amber">
        <div className="age-flag-title">⚠️ Peak Diagnostic Gap Zone</div>
        <div className="age-flag-text">
          Women aged 26–30 have the highest Rotterdam miss rate in our dataset (35.8%).
          DIANA applies heightened sensitivity for this age group — borderline presentations
          are flagged rather than dismissed.
        </div>
      </div>
    );
  }
  if (age_flag === 'adolescent') {
    return (
      <div className="age-flag-card blue">
        <div className="age-flag-title">⚠️ Adolescent Patient</div>
        <div className="age-flag-text">
          2023 international guidelines recommend against ultrasound and AMH in patients
          under 18 due to poor specificity. DIANA applies adolescent-adjusted criteria:
          both hyperandrogenism AND anovulation required for diagnosis. Ultrasound findings
          not weighted.
        </div>
      </div>
    );
  }
  return null;
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
      {/* ── Age-aware diagnostic flag card ─────────────────────────────── */}
      {results.age_flag && <AgeFlagCard age_flag={results.age_flag} />}

      {/* ── [3] Primary diagnosis + gauge ──────────────────────────────── */}
      <div className={`card${pcos_prediction && !inconclusive ? ' pcos-positive' : ''}`}>
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

          {/* Phenotype subtype — PCOS positive only */}
          {pcos_prediction && results.pcos_phenotype && (
            <div className="phenotype-row">
              <span className="phenotype-badge">{results.pcos_phenotype.label}</span>
              <span className="phenotype-desc">{results.pcos_phenotype.description}</span>
            </div>
          )}
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

      {/* ── Time Cost of Misdiagnosis — PCOS-positive only ─────────────── */}
      {pcos_prediction && results.pcos_phenotype && (
        <TimeCostCard phenotype={results.pcos_phenotype} />
      )}

      {/* ── Long-term Risk Profile — PCOS-positive only ────────────────── */}
      {pcos_prediction && results.pcos_phenotype && (
        <RiskProfileCard phenotype={results.pcos_phenotype} />
      )}

      {/* ── Rotterdam Gap Flag ─────────────────────────────────────────── */}
      {!pcos_prediction && results.rotterdam_gap_flag && (
        <div className="rotterdam-gap-card">
          <div className="rotterdam-gap-icon">⚠️</div>
          <div>
            <div className="rotterdam-gap-title">Atypical PCOS Signals Detected</div>
            <div className="rotterdam-gap-text">
              This patient does not meet Rotterdam criteria for PCOS but shows
              hormonal/metabolic signals consistent with atypical PCOS cases in
              our dataset. Do not exclude PCOS — recommend full clinical workup.
            </div>
          </div>
        </div>
      )}

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

      {/* ── Differential / Comorbidity flags (all patients) ───────────── */}
      {df && (
        <div className="card">
          <div className="card-header">
            <h2>{pcos_prediction ? 'Comorbidity Alerts' : 'Differential Diagnosis — Stage 2'}</h2>
          </div>
          <div className="card-body">
            <div className="flags-grid">
              <FlagCard icon="🦋" flagged={df.hypothyroidism}
                label="Hypothyroidism" biomarker="TSH"
                value={df.tsh_value} unit="mIU/L" threshold="4.0" />
              <FlagCard icon="💊" flagged={df.hyperprolactinemia}
                label="Hyperprolactinemia" biomarker="PRL"
                value={df.prl_value} unit="ng/mL" threshold="25"
                note={df.galactorrhea_strengthened ? 'Galactorrhea reported — supports hyperprolactinemia diagnosis.' : null} />
              <FlagCard icon="🫀" flagged={df.cushings_hypertension}
                label="Cushing's / HTN" biomarker="Systolic BP"
                value={df.sbp_value} unit="mmHg" threshold="140" />
              <EndoCard icon="🌸" flagged={df.endometriosis_risk}
                probability={df.endometriosis_probability}
                pelvicPainNote={df.pelvic_pain_note} />
            </div>
            <p style={{ fontSize: 11, color: 'var(--c-muted)', marginTop: 12 }}>
              {pcos_prediction
                ? 'PCOS patients may present with concurrent conditions. Flags indicate values exceeding clinical thresholds.'
                : 'Flags indicate values exceeding published clinical thresholds. Results require clinical correlation and are not diagnostic alone.'}
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
