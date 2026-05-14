/**
 * DIANA PDF Report Generator
 * Builds a structured A4 clinical report using jsPDF drawing primitives.
 * No HTML-to-canvas conversion — all layout is computed explicitly.
 */
import { jsPDF } from 'jspdf';

// ── Colours (RGB triplets) ─────────────────────────────────────────────────────
const C = {
  primary:    [13,  71, 112],
  teal:       [13, 128, 128],
  ok:         [29, 131,  72],
  danger:     [176, 58,  46],
  amber:      [200,120,   8],
  amberDark:  [ 93, 73,   0],
  muted:      [93, 109, 126],
  border:     [212,221, 232],
  bg:         [238,242, 247],
  bgLight:    [248,250, 252],
  white:      [255,255, 255],
  dark:       [ 26, 37,  47],
  purple:     [125, 60, 152],
};

// ── Page geometry ──────────────────────────────────────────────────────────────
const PW  = 210;         // page width  (mm)
const PH  = 297;         // page height (mm)
const M   = 16;          // margin (mm)
const CW  = PW - M * 2; // usable content width

// ── Form field metadata for Patient Input Summary table ───────────────────────
const FORM_META = [
  { key: 'age',               label: 'Age',                    unit: ' yrs'    },
  { key: 'bmi',               label: 'BMI',                    unit: ' kg/m²' },
  { key: 'cycle_regular',     label: 'Cycle',                  fmt: v => v ? 'Regular' : 'Irregular' },
  { key: 'cycle_length_days', label: 'Cycle Length',           unit: ' days'   },
  { key: 'fsh_miu_ml',        label: 'FSH',                    unit: ' mIU/mL' },
  { key: 'lh_miu_ml',         label: 'LH',                     unit: ' mIU/mL' },
  { key: 'amh_ng_ml',         label: 'AMH',                    unit: ' ng/mL'  },
  { key: 'tsh_miu_l',         label: 'TSH',                    unit: ' mIU/L'  },
  { key: 'prl_ng_ml',         label: 'Prolactin (PRL)',         unit: ' ng/mL'  },
  { key: 'follicle_no_l',     label: 'Follicles (L)',          unit: ''        },
  { key: 'follicle_no_r',     label: 'Follicles (R)',          unit: ''        },
  { key: 'avg_f_size_l_mm',   label: 'Avg Follicle Size (L)',  unit: ' mm'     },
  { key: 'avg_f_size_r_mm',   label: 'Avg Follicle Size (R)',  unit: ' mm'     },
  { key: 'bp_systolic',       label: 'Systolic BP',            unit: ' mmHg'   },
  { key: 'bp_diastolic',      label: 'Diastolic BP',           unit: ' mmHg'   },
  { key: 'weight_gain',       label: 'Weight Gain',            fmt: v => v ? 'Yes' : 'No' },
  { key: 'hair_growth',       label: 'Excessive Hair Growth',  fmt: v => v ? 'Yes' : 'No' },
  { key: 'skin_darkening',    label: 'Skin Darkening',         fmt: v => v ? 'Yes' : 'No' },
  { key: 'hair_loss',         label: 'Hair Loss',              fmt: v => v ? 'Yes' : 'No' },
  { key: 'pimples',           label: 'Acne / Pimples',         fmt: v => v ? 'Yes' : 'No' },
];

function fmtVal(key, rawVal, meta) {
  if (rawVal === null || rawVal === undefined) return '—';
  if (meta.fmt) return meta.fmt(rawVal);
  const num = typeof rawVal === 'number'
    ? (Number.isInteger(rawVal) ? String(rawVal) : rawVal.toFixed(1))
    : String(rawVal);
  return `${num}${meta.unit ?? ''}`;
}

// ── Main export ────────────────────────────────────────────────────────────────

/**
 * @param {object} form        - Patient form values
 * @param {object} results     - /diagnose API response
 * @param {string} shapSentence - Pre-computed plain-English top-SHAP sentence
 */
export function generatePDF({ form, results, shapSentence }) {
  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const now = new Date();
  const dateStr = now.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  const timeStr = now.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' });

  let y = M;

  // ── Drawing helpers ──────────────────────────────────────────────────────────

  // Check if remaining space is enough; add page if not
  const guard = (need = 15) => {
    if (y + need > PH - M) { doc.addPage(); y = M; }
  };

  // Dark section header bar with white title
  const sectionBar = (title) => {
    guard(12);
    doc.setFillColor(...C.primary);
    doc.rect(M, y, CW, 8, 'F');
    doc.setTextColor(...C.white);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9.5);
    doc.text(title.toUpperCase(), M + 3, y + 5.5);
    doc.setTextColor(...C.dark);
    y += 12;
  };

  // Two-column label: value row
  const kvRow = (label, val, xOff = 0) => {
    const x = M + xOff;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(...C.muted);
    doc.text(`${label}:`, x, y);
    doc.setTextColor(...C.dark);
    doc.text(String(val ?? '—'), x + 36, y);
  };

  // Wrapped paragraph of body text
  const para = (text, maxW = CW) => {
    guard(20);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(9);
    doc.setTextColor(...C.dark);
    const lines = doc.splitTextToSize(text, maxW);
    doc.text(lines, M, y);
    y += lines.length * 5 + 2;
  };

  // ── 1. HEADER ────────────────────────────────────────────────────────────────
  // Two-tone gradient strip
  doc.setFillColor(...C.primary);
  doc.rect(0, 0, PW, 26, 'F');
  doc.setFillColor(...C.teal);
  doc.rect(PW * 0.58, 0, PW * 0.42, 26, 'F');

  doc.setTextColor(...C.white);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(20);
  doc.text('DIANA', M, 16);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.text('Differential Intelligence for ANdrogen Assessment — Clinical Diagnostic Report', M, 22);
  doc.setFontSize(8.5);
  doc.text(`Generated: ${dateStr}  ${timeStr}`, PW - M, 22, { align: 'right' });

  doc.setTextColor(...C.dark);
  y = 34;

  // ── 2. DIAGNOSTIC RESULT ─────────────────────────────────────────────────────
  sectionBar('Diagnostic Result');

  const prob         = results.pcos_probability;
  const inconclusive = prob >= 0.4 && prob <= 0.6;
  const pcosDet      = results.pcos_prediction;

  const [resultBg, resultTc, resultLabel] = inconclusive
    ? [[255,243,205], C.amber,  'INCONCLUSIVE — Further Investigation Recommended']
    : pcosDet
      ? [[253,237,236], C.danger, 'PCOS DETECTED']
      : [[234,250,241], C.ok,     'PCOS NOT DETECTED'];

  guard(22);
  doc.setFillColor(...resultBg);
  doc.rect(M, y, CW, 14, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(11);
  doc.setTextColor(...resultTc);
  doc.text(resultLabel, M + 4, y + 6);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(9);
  const confLine = inconclusive
    ? `Probability: ${Math.round(prob * 100)}%  —  Stage 2 differential diagnosis has been automatically initiated`
    : `Probability: ${Math.round(prob * 100)}%  |  Confidence: ${Math.round(pcosDet ? prob * 100 : (1 - prob) * 100)}%`;
  doc.text(confLine, M + 4, y + 11.5);
  doc.setTextColor(...C.dark);
  y += 18;

  // Plain-English summary callout
  if (shapSentence) {
    guard(18);
    const lines = doc.splitTextToSize(shapSentence, CW - 10);
    const boxH  = lines.length * 5 + 7;
    doc.setFillColor(...C.bg);
    doc.rect(M, y, CW, boxH, 'F');
    doc.setFillColor(...C.primary);
    doc.rect(M, y, 2.5, boxH, 'F');
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(9);
    doc.setTextColor(...C.dark);
    doc.text(lines, M + 5, y + 5);
    y += boxH + 4;
  }

  // ── 3. ROTTERDAM CRITERIA ────────────────────────────────────────────────────
  sectionBar('Rotterdam Criteria Checklist');

  const polycystic  = (form.follicle_no_l ?? 0) > 12 || (form.follicle_no_r ?? 0) > 12;
  const hyperandro  = !!(form.hair_growth || form.skin_darkening || form.pimples);
  const anovulation = form.cycle_regular === false;
  const metCount    = [polycystic, hyperandro, anovulation].filter(Boolean).length;

  const rottRows = [
    {
      label:  'Polycystic Ovaries',
      detail: `Follicle count > 12  —  Left: ${form.follicle_no_l ?? '?'}, Right: ${form.follicle_no_r ?? '?'}`,
      met:    polycystic,
    },
    {
      label:  'Hyperandrogenism',
      detail: `Hair growth: ${form.hair_growth ? 'Yes' : 'No'}  |  Skin darkening: ${form.skin_darkening ? 'Yes' : 'No'}  |  Acne: ${form.pimples ? 'Yes' : 'No'}`,
      met:    hyperandro,
    },
    {
      label:  'Anovulation',
      detail: `Cycle regularity: ${form.cycle_regular ? 'Regular' : 'Irregular'}`,
      met:    anovulation,
    },
  ];

  rottRows.forEach(row => {
    guard(14);
    const [bg, tc] = row.met ? [[234,250,241], C.ok] : [[248,250,252], C.muted];
    doc.setFillColor(...bg);
    doc.rect(M, y, CW, 11, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(row.met ? C.ok[0] : C.muted[0], row.met ? C.ok[1] : C.muted[1], row.met ? C.ok[2] : C.muted[2]);
    doc.text(row.met ? '[MET]' : '[NOT MET]', M + CW - 2, y + 4.5, { align: 'right' });

    doc.setTextColor(...C.dark);
    doc.text(row.label, M + 4, y + 4.5);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    doc.setTextColor(...C.muted);
    doc.text(row.detail, M + 4, y + 9);

    doc.setTextColor(...C.dark);
    y += 13;
  });

  // Summary note
  guard(10);
  const [noteBg, noteTc] = metCount >= 2 ? [[253,237,236], C.danger] : [[234,250,241], C.ok];
  doc.setFillColor(...noteBg);
  doc.rect(M, y, CW, 8, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...noteTc);
  doc.text(
    `${metCount}/3 criteria present  —  PCOS ${metCount >= 2 ? 'MEETS' : 'does NOT meet'} Rotterdam consensus threshold`,
    M + CW / 2, y + 5.2, { align: 'center' },
  );
  doc.setTextColor(...C.dark);
  y += 12;

  doc.setFont('helvetica', 'italic');
  doc.setFontSize(8);
  doc.setTextColor(...C.muted);
  doc.text('PCOS diagnosed if 2 of 3 criteria met (Rotterdam 2003 consensus)', M, y);
  doc.setTextColor(...C.dark);
  y += 7;

  // ── 4. PATIENT INPUT SUMMARY ─────────────────────────────────────────────────
  sectionBar('Patient Input Summary');

  for (let i = 0; i < FORM_META.length; i += 2) {
    guard(7);
    const m1 = FORM_META[i];
    kvRow(m1.label, fmtVal(m1.key, form[m1.key], m1), 0);
    if (i + 1 < FORM_META.length) {
      const m2 = FORM_META[i + 1];
      kvRow(m2.label, fmtVal(m2.key, form[m2.key], m2), CW / 2);
    }
    y += 6;
  }

  // ── 5. KEY CONTRIBUTING FACTORS ──────────────────────────────────────────────
  y += 4;
  sectionBar('Key Contributing Factors (SHAP Analysis)');

  results.shap_explanations.forEach((e, idx) => {
    guard(14);
    const isUp    = e.direction === 'increases_risk';
    const barCol  = isUp ? C.danger : C.ok;
    const valStr  = Number.isInteger(e.feature_value) ? String(e.feature_value) : e.feature_value.toFixed(2);
    const shapStr = `${e.shap_value > 0 ? '+' : ''}${e.shap_value.toFixed(3)}`;

    doc.setFillColor(...C.bgLight);
    doc.rect(M, y, CW, 11, 'F');
    doc.setFillColor(...barCol);
    doc.rect(M, y, 2.5, 11, 'F');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...C.dark);
    doc.text(`${idx + 1}. ${e.display_name}`, M + 5, y + 4.5);

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(...C.muted);
    doc.text(
      `Value: ${valStr}  |  SHAP: ${shapStr}  |  ${isUp ? 'Increases PCOS risk' : 'Decreases PCOS risk'}`,
      M + 5, y + 9,
    );
    doc.setTextColor(...C.dark);
    y += 14;
  });

  // ── 6. STAGE 2 DIFFERENTIAL DIAGNOSIS ────────────────────────────────────────
  if (!pcosDet && results.differential_flags) {
    y += 4;
    sectionBar('Stage 2 Differential Diagnosis');

    const df = results.differential_flags;
    const flags = [
      { label: 'Hypothyroidism',     flagged: df.hypothyroidism,         bio: 'TSH',        val: df.tsh_value,                unit: 'mIU/L', thresh: '> 4.0 mIU/L'  },
      { label: 'Hyperprolactinemia', flagged: df.hyperprolactinemia,      bio: 'PRL',        val: df.prl_value,                unit: 'ng/mL', thresh: '> 25 ng/mL'    },
      { label: "Cushing's / HTN",    flagged: df.cushings_hypertension,   bio: 'Systolic BP',val: df.sbp_value,                unit: 'mmHg',  thresh: '> 140 mmHg'    },
      { label: 'Endometriosis Risk', flagged: df.endometriosis_risk,      bio: 'XGBoost p',  val: df.endometriosis_probability !== null ? `${Math.round(df.endometriosis_probability * 100)}%` : '—', unit: '', thresh: '>= 50%' },
    ];

    flags.forEach(f => {
      guard(12);
      const [bg, tc] = f.flagged ? [[245,238,248], C.purple] : [[234,250,241], C.ok];
      doc.setFillColor(...bg);
      doc.rect(M, y, CW, 10, 'F');

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9);
      doc.setTextColor(...tc);
      const statusText = f.flagged ? 'FLAGGED — INVESTIGATE' : 'WITHIN NORMAL RANGE';
      doc.text(statusText, M + CW - 2, y + 5.5, { align: 'right' });
      doc.setTextColor(...C.dark);
      doc.text(f.label, M + 4, y + 5.5);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8);
      doc.setTextColor(...C.muted);
      const bioLine = `${f.bio}: ${f.val}${f.unit ? ' ' + f.unit : ''}  |  Threshold: ${f.thresh}`;
      y += 12;
      guard(5);
      doc.text(bioLine, M + 4, y);
      doc.setTextColor(...C.dark);
      y += 6;
    });
  }

  // ── 7. DISCLAIMER ────────────────────────────────────────────────────────────
  y += 6;
  guard(24);
  doc.setFillColor(255,243,205);
  doc.setDrawColor(...C.amber);
  doc.roundedRect(M, y, CW, 22, 2, 2, 'FD');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(...C.amber);
  doc.text('DISCLAIMER', M + 4, y + 6);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(...C.amberDark);
  const disclaimer = 'For research and educational use only. Not a substitute for clinical diagnosis by a qualified physician. DIANA is an AI-assisted decision support tool and should not be used as the sole basis for any clinical decision.';
  const dLines = doc.splitTextToSize(disclaimer, CW - 8);
  doc.text(dLines, M + 4, y + 12);
  y += 26;

  // ── 8. PAGE FOOTER ───────────────────────────────────────────────────────────
  const totalPages = doc.getNumberOfPages();
  for (let p = 1; p <= totalPages; p++) {
    doc.setPage(p);
    doc.setFillColor(...C.border);
    doc.rect(0, PH - 11, PW, 11, 'F');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.5);
    doc.setTextColor(...C.muted);
    doc.text(`DIANA · SBS Hackathon Team 27 · Generated ${dateStr} ${timeStr}`, M, PH - 4.5);
    doc.text(`Page ${p} of ${totalPages}`, PW - M, PH - 4.5, { align: 'right' });
  }

  // ── SAVE ─────────────────────────────────────────────────────────────────────
  const stamp    = `${now.toISOString().slice(0, 10)}_${String(now.getHours()).padStart(2,'0')}${String(now.getMinutes()).padStart(2,'0')}`;
  doc.save(`DIANA_Report_${stamp}.pdf`);
}
