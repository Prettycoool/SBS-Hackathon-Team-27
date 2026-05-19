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
  { key: 'prl_ng_ml',          label: 'Prolactin (PRL)',         unit: ' ng/mL'  },
  { key: 'progesterone_ng_ml', label: 'Progesterone',            unit: ' ng/mL'  },
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
  { key: 'pimples',            label: 'Acne / Pimples',         fmt: v => v ? 'Yes' : 'No' },
  { key: 'galactorrhea',       label: 'Galactorrhea',            fmt: v => v ? 'Yes' : 'No' },
  { key: 'chronic_pelvic_pain', label: 'Chronic Pelvic Pain',   unit: ' / 10'   },
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

  // ── 2. AGE-AWARE DIAGNOSTIC FLAG ─────────────────────────────────────────────
  if (results.age_flag === 'peak_gap') {
    guard(22);
    doc.setFillColor(255, 251, 235);
    doc.setDrawColor(245, 158, 11);
    doc.roundedRect(M, y, CW, 4, 2, 2, 'F');  // top colour strip
    doc.setFillColor(255, 251, 235);
    doc.setDrawColor(245, 158, 11);
    doc.roundedRect(M, y, CW, 22, 2, 2, 'FD');
    doc.setFillColor(245, 158, 11);
    doc.rect(M, y, 3, 22, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(146, 64, 14);
    doc.text('⚠  Peak Diagnostic Gap Zone', M + 6, y + 6.5);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(92, 73, 0);
    const peakMsg = 'Women aged 26–30 have the highest Rotterdam miss rate in our dataset (35.8%). DIANA applies heightened sensitivity for this age group — borderline presentations are flagged rather than dismissed.';
    const peakLines = doc.splitTextToSize(peakMsg, CW - 10);
    doc.text(peakLines, M + 6, y + 13);
    doc.setTextColor(...C.dark);
    y += 26;
  } else if (results.age_flag === 'adolescent') {
    guard(22);
    doc.setFillColor(239, 246, 255);
    doc.setDrawColor(59, 130, 246);
    doc.roundedRect(M, y, CW, 22, 2, 2, 'FD');
    doc.setFillColor(59, 130, 246);
    doc.rect(M, y, 3, 22, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(30, 64, 175);
    doc.text('⚠  Adolescent Patient', M + 6, y + 6.5);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(30, 58, 138);
    const adolMsg = '2023 international guidelines recommend against ultrasound and AMH in patients under 18 due to poor specificity. DIANA applies adolescent-adjusted criteria: both hyperandrogenism AND anovulation required for diagnosis. Ultrasound findings not weighted.';
    const adolLines = doc.splitTextToSize(adolMsg, CW - 10);
    doc.text(adolLines, M + 6, y + 13);
    doc.setTextColor(...C.dark);
    y += 26;
  }

  // ── 3. DIAGNOSTIC RESULT ─────────────────────────────────────────────────────
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

  // Phenotype subtype row
  if (pcosDet && results.pcos_phenotype) {
    guard(12);
    doc.setFillColor(245, 238, 248);
    doc.rect(M, y, CW, 10, 'F');
    doc.setFillColor(...C.purple);
    doc.rect(M, y, 2.5, 10, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...C.purple);
    doc.text(results.pcos_phenotype.label, M + 5, y + 4.5);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(...C.muted);
    doc.text(results.pcos_phenotype.description, M + 5, y + 9);
    doc.setTextColor(...C.dark);
    y += 14;
  }

  // Time Cost of Misdiagnosis block (PCOS-positive with known phenotype)
  if (pcosDet && results.pcos_phenotype) {
    const MISDIAGNOSIS_PDF = {
      A:        { years: 1.9, misdiagnosis: 'Idiopathic hirsutism' },
      B:        { years: 2.8, misdiagnosis: 'Idiopathic hirsutism (54% of cases)' },
      C:        { years: 2.4, misdiagnosis: 'Stress-related amenorrhea' },
      D:        { years: 3.8, misdiagnosis: 'Subclinical hypothyroidism' },
      Atypical: { years: 4.2, misdiagnosis: 'No diagnosis given' },
    };
    const mData = MISDIAGNOSIS_PDF[results.pcos_phenotype.code];
    if (mData) {
      // Compute diana text lines first so card height is dynamic
      const dianaMsg = `With DIANA: Flagged today — estimated ${mData.years} years saved — fertility planning window preserved`;
      const dianaLines = doc.splitTextToSize(dianaMsg, CW - 16);
      const darkH  = 31;
      const stripH = Math.max(12, dianaLines.length * 5 + 8);
      const cardH  = darkH + stripH;

      guard(cardH);
      // Dark background
      doc.setFillColor(15, 32, 39);
      doc.rect(M, y, CW, darkH, 'F');
      // Green strip at bottom
      doc.setFillColor(6, 95, 70);
      doc.rect(M, y + darkH, CW, stripH, 'F');

      // Title (plain text — no emoji)
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9);
      doc.setTextColor(255, 255, 255);
      doc.text('Time Cost of Misdiagnosis — Without early detection tools:', M + 4, y + 7);

      // Data rows
      const rows = [
        ['Average time to diagnosis',      `${mData.years} years`],
        ['Doctors seen before diagnosis',   '3-4 on average'],
        ['Most commonly misdiagnosed as',   mData.misdiagnosis],
      ];
      let ry = y + 14;
      for (const [label, val] of rows) {
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(8.5);
        doc.setTextColor(180, 180, 190);
        doc.text(label + ':', M + 4, ry);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(251, 191, 36);
        doc.text(val, M + 4 + 70, ry);
        ry += 6.5;
      }

      // Diana green strip text — wrapped, padded, no problematic unicode
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8.5);
      doc.setTextColor(255, 255, 255);
      doc.text(dianaLines, M + 4, y + darkH + 6);

      doc.setTextColor(...C.dark);
      y += cardH + 4;
    }
  }

  // Long-term Risk Profile block (PCOS-positive with known phenotype)
  if (pcosDet && results.pcos_phenotype) {
    const RISK_PROFILE_PDF = {
      A:        { t2d: 4.0, ms: 3.2, cvd: 2.8, mh: 2.8, infertility: 72 },
      B:        { t2d: 3.2, ms: 2.8, cvd: 2.4, mh: 2.8, infertility: 68 },
      C:        { t2d: 2.8, ms: 2.4, cvd: 2.0, mh: 2.4, infertility: 55 },
      D:        { t2d: 2.4, ms: 2.0, cvd: 1.8, mh: 2.4, infertility: 48 },
      Atypical: { t2d: 2.5, ms: 2.2, cvd: 1.9, mh: 2.6, infertility: 52 },
    };
    const rp = RISK_PROFILE_PDF[results.pcos_phenotype.code];
    if (rp) {
      const riskRows = [
        { label: 'Type 2 Diabetes',       val: rp.t2d,         fmt: v => `${v}x relative risk`,  isX: true  },
        { label: 'Metabolic Syndrome',     val: rp.ms,          fmt: v => `${v}x relative risk`,  isX: true  },
        { label: 'Cardiovascular Disease', val: rp.cvd,         fmt: v => `${v}x relative risk`,  isX: true  },
        { label: 'Depression / Anxiety',   val: rp.mh,          fmt: v => `${v}x relative risk`,  isX: true  },
        { label: 'Infertility Risk',       val: rp.infertility, fmt: v => `${v}%`,                isX: false },
      ];

      // Header bar
      sectionBar('Long-term Risk Profile — If Left Undiagnosed');

      // Risk rows
      for (const row of riskRows) {
        guard(9);
        const tier = row.isX
          ? (row.val >= 3 ? 'red' : row.val >= 2 ? 'amber' : 'yellow')
          : (row.val >= 65 ? 'red' : row.val >= 55 ? 'amber' : 'yellow');
        const bgColor  = tier === 'red' ? [60, 20, 20] : tier === 'amber' ? [55, 35, 10] : [50, 45, 10];
        const valColor = tier === 'red' ? [248, 113, 113] : tier === 'amber' ? [251, 191, 36] : [253, 224, 71];

        doc.setFillColor(...bgColor);
        doc.rect(M, y, CW, 7, 'F');
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(8.5);
        doc.setTextColor(200, 200, 210);
        doc.text(row.label, M + 4, y + 4.8);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(...valColor);
        doc.text(row.fmt(row.val), M + CW - 4, y + 4.8, { align: 'right' });
        y += 8;
      }

      y += 3;

      // Good news / green section
      const goodNewsMsg = 'Lifestyle modification reduces T2D risk by up to 58% (NEJM Diabetes Prevention Program). Metformin reduces metabolic risk. Mental health screening recommended.';
      const gnLines = doc.splitTextToSize(goodNewsMsg, CW - 8);
      const checkItems = ['Fasting glucose + insulin', 'Full lipid panel', 'PHQ-9 depression screening', 'Blood pressure monitoring'];
      const gnBoxH = 8 + gnLines.length * 5 + 7 + checkItems.length * 5.5 + 6;

      guard(gnBoxH);
      doc.setFillColor(5, 46, 22);
      doc.rect(M, y, CW, gnBoxH, 'F');

      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9);
      doc.setTextColor(74, 222, 128);
      doc.text('✓  Good news — Early intervention helps:', M + 4, y + 6.5);

      doc.setFont('helvetica', 'normal');
      doc.setFontSize(8.5);
      doc.setTextColor(200, 220, 200);
      doc.text(gnLines, M + 4, y + 13);

      let cy = y + 13 + gnLines.length * 5 + 4;
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(8.5);
      doc.setTextColor(200, 240, 200);
      doc.text('Recommended investigations:', M + 4, cy);
      cy += 6;

      doc.setFont('helvetica', 'normal');
      doc.setTextColor(180, 220, 180);
      for (const item of checkItems) {
        doc.text(`•  ${item}`, M + 6, cy);
        cy += 5.5;
      }

      doc.setTextColor(...C.dark);
      y += gnBoxH + 4;
    }
  }

  // Rotterdam gap flag warning
  if (!pcosDet && results.rotterdam_gap_flag) {
    guard(22);
    doc.setFillColor(255, 243, 205);
    doc.setDrawColor(...C.amber);
    doc.roundedRect(M, y, CW, 20, 2, 2, 'FD');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    doc.setTextColor(...C.amber);
    doc.text('⚠  ATYPICAL PCOS SIGNALS DETECTED', M + 4, y + 6.5);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.setTextColor(...C.amberDark);
    const gapMsg = 'Patient does not meet Rotterdam criteria but shows hormonal/metabolic signals consistent with atypical PCOS cases in our dataset. Do not exclude PCOS — recommend full clinical workup.';
    const gapLines = doc.splitTextToSize(gapMsg, CW - 8);
    doc.text(gapLines, M + 4, y + 12.5);
    doc.setTextColor(...C.dark);
    y += 24;
  }

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

  // ── 4. ROTTERDAM CRITERIA ────────────────────────────────────────────────────
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

  // ── 5. PATIENT INPUT SUMMARY ─────────────────────────────────────────────────
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

  // ── 6. KEY CONTRIBUTING FACTORS ──────────────────────────────────────────────
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

  // ── 7. DIFFERENTIAL DIAGNOSIS / COMORBIDITY ALERTS (all patients) ───────────
  if (results.differential_flags) {
    y += 4;
    sectionBar(pcosDet ? 'Comorbidity Alerts' : 'Stage 2 Differential Diagnosis');

    const df = results.differential_flags;
    const flags = [
      { label: 'Hypothyroidism',     flagged: df.hypothyroidism,         bio: 'TSH',        val: df.tsh_value,                unit: 'mIU/L', thresh: '> 4.0 mIU/L',  note: '' },
      { label: 'Hyperprolactinemia', flagged: df.hyperprolactinemia,      bio: 'PRL',        val: df.prl_value,                unit: 'ng/mL', thresh: '> 25 ng/mL',    note: df.galactorrhea_strengthened ? 'Galactorrhea reported — supports hyperprolactinemia diagnosis.' : '' },
      { label: "Cushing's / HTN",    flagged: df.cushings_hypertension,   bio: 'Systolic BP',val: df.sbp_value,                unit: 'mmHg',  thresh: '> 140 mmHg',    note: '' },
      { label: 'Endometriosis Risk', flagged: df.endometriosis_risk,      bio: 'XGBoost p',  val: df.endometriosis_probability !== null ? `${Math.round(df.endometriosis_probability * 100)}%` : '—', unit: '', thresh: '>= 50%', note: df.pelvic_pain_note || '' },
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

      if (f.note) {
        guard(7);
        doc.setFillColor(255, 251, 235);
        doc.rect(M, y, CW, 6, 'F');
        doc.setFillColor(245, 158, 11);
        doc.rect(M, y, 2, 6, 'F');
        doc.setFont('helvetica', 'italic');
        doc.setFontSize(8);
        doc.setTextColor(146, 64, 14);
        const noteLines = doc.splitTextToSize(f.note, CW - 10);
        doc.text(noteLines, M + 5, y + 4.2);
        doc.setTextColor(...C.dark);
        y += 7;
      }
    });

    guard(8);
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(8);
    doc.setTextColor(...C.muted);
    doc.text(
      pcosDet
        ? 'PCOS patients may present with concurrent conditions. Flags indicate values exceeding clinical thresholds.'
        : 'Flags indicate values exceeding published clinical thresholds. Results require clinical correlation.',
      M, y,
    );
    doc.setTextColor(...C.dark);
    y += 6;
  }

  // ── 8. DISCLAIMER ────────────────────────────────────────────────────────────
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

  // ── 9. PAGE FOOTER ───────────────────────────────────────────────────────────
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
