// Patient data input form.
// Sliders for continuous values, pill toggles for binary symptoms.
// Organised into six clinical sections matching Rotterdam criteria groupings.

// ── Reusable primitives ──────────────────────────────────────────────────────

function Slider({ name, label, value, onChange, min, max, step, unit, refRange }) {
  const decimals = step < 1 ? 1 : 0;
  const display  = typeof value === 'number' ? value.toFixed(decimals) : value;
  const pct      = `${((value - min) / (max - min)) * 100}%`;

  return (
    <div className="field">
      <div className="field-header">
        <span className="field-label">{label}</span>
        <span className="field-value-badge">{display}{unit}</span>
      </div>
      {refRange && <div className="field-ref">Ref: {refRange}</div>}
      <input
        type="range"
        style={{ '--pct': pct }}
        min={min} max={max} step={step} value={value}
        onChange={e => onChange(name, parseFloat(e.target.value))}
      />
      <div className="field-minmax">
        <span>{min}{unit}</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  );
}

function Toggle({ name, label, sub, value, onChange, symptom }) {
  return (
    <div
      className={`toggle-field ${symptom ? 'symptom' : ''} ${value ? 'active' : ''}`}
      role="checkbox"
      aria-checked={value}
      onClick={() => onChange(name, !value)}
    >
      <div>
        <div className="toggle-name">{label}</div>
        {sub && <div className="toggle-sub">{sub}</div>}
      </div>
      <div className={`toggle-switch ${value ? 'on' : ''}`}>
        <div className="toggle-knob" />
      </div>
    </div>
  );
}

// ── Main form ────────────────────────────────────────────────────────────────

export default function PatientForm({ form, onChange, onSubmit, loading }) {
  return (
    <div className="form-panel">

      {/* ── Demographics ──────────────────────────────────────────────────── */}
      <div id="section-demographics" className="card border-blue">
        <div className="card-body">
          <div className="section-title">Demographics</div>
          <div className="form-grid">
            <Slider name="age"    label="Age"  value={form.age}   onChange={onChange}
                    min={15} max={50} step={1} unit=" yrs" />
            <Slider name="bmi"    label="BMI"  value={form.bmi}   onChange={onChange}
                    min={15} max={45} step={0.5} unit=" kg/m²" refRange="18.5 – 24.9" />
          </div>
        </div>
      </div>

      {/* ── Menstrual history ─────────────────────────────────────────────── */}
      <div id="section-menstrual" className="card border-purple">
        <div className="card-body">
          <div className="section-title">Menstrual History</div>
          <div className="form-grid">
            <Toggle name="cycle_regular" label="Cycle Regularity"
                    sub={form.cycle_regular ? 'Regular' : 'Irregular'}
                    value={form.cycle_regular} onChange={onChange} />
            <Slider name="cycle_length_days" label="Cycle Length"
                    value={form.cycle_length_days} onChange={onChange}
                    min={1} max={15} step={1} unit=" days" refRange="3 – 7 days" />
          </div>
        </div>
      </div>

      {/* ── Hormonal markers ──────────────────────────────────────────────── */}
      <div id="section-hormonal" className="card border-teal">
        <div className="card-body">
          <div className="section-title">Hormonal Markers</div>
          <div className="form-grid">
            <Slider name="fsh_miu_ml" label="FSH" value={form.fsh_miu_ml} onChange={onChange}
                    min={1} max={25} step={0.1} unit=" mIU/mL" refRange="3 – 10 mIU/mL" />
            <Slider name="lh_miu_ml"  label="LH"  value={form.lh_miu_ml}  onChange={onChange}
                    min={1} max={25} step={0.1} unit=" mIU/mL" refRange="2 – 15 mIU/mL" />
            <Slider name="amh_ng_ml"  label="AMH" value={form.amh_ng_ml}  onChange={onChange}
                    min={0} max={15} step={0.1} unit=" ng/mL"  refRange="1 – 3.5 ng/mL" />
            <Slider name="tsh_miu_l"  label="TSH" value={form.tsh_miu_l}  onChange={onChange}
                    min={0.1} max={10} step={0.1} unit=" mIU/L" refRange="0.4 – 4.0 mIU/L" />
            <Slider name="prl_ng_ml"  label="Prolactin (PRL)" value={form.prl_ng_ml} onChange={onChange}
                    min={2} max={100} step={0.5} unit=" ng/mL" refRange="< 25 ng/mL" />
          </div>
        </div>
      </div>

      {/* ── Ultrasound ────────────────────────────────────────────────────── */}
      <div id="section-ultrasound" className="card border-indigo">
        <div className="card-body">
          <div className="section-title">Ultrasound Findings</div>
          <div className="form-grid">
            <Slider name="follicle_no_l" label="Follicles — Left Ovary"
                    value={form.follicle_no_l} onChange={onChange}
                    min={0} max={25} step={1} unit="" refRange="< 12 normal" />
            <Slider name="follicle_no_r" label="Follicles — Right Ovary"
                    value={form.follicle_no_r} onChange={onChange}
                    min={0} max={25} step={1} unit="" refRange="< 12 normal" />
            <Slider name="avg_f_size_l_mm" label="Avg. Follicle Size (L)"
                    value={form.avg_f_size_l_mm} onChange={onChange}
                    min={5} max={35} step={0.5} unit=" mm" />
            <Slider name="avg_f_size_r_mm" label="Avg. Follicle Size (R)"
                    value={form.avg_f_size_r_mm} onChange={onChange}
                    min={5} max={35} step={0.5} unit=" mm" />
          </div>
        </div>
      </div>

      {/* ── Clinical symptoms ─────────────────────────────────────────────── */}
      <div id="section-symptoms" className="card border-rose">
        <div className="card-body">
          <div className="section-title">Clinical Symptoms</div>
          <div className="form-grid">
            <Toggle symptom name="weight_gain"   label="Weight Gain"
                    sub="Unexplained / recent"   value={form.weight_gain}   onChange={onChange} />
            <Toggle symptom name="hair_growth"   label="Excessive Hair Growth"
                    sub="Hirsutism"              value={form.hair_growth}   onChange={onChange} />
            <Toggle symptom name="skin_darkening" label="Skin Darkening"
                    sub="Acanthosis nigricans"   value={form.skin_darkening} onChange={onChange} />
            <Toggle symptom name="hair_loss"     label="Hair Loss"
                    sub="Scalp / androgenic"     value={form.hair_loss}     onChange={onChange} />
            <Toggle symptom name="pimples"       label="Acne / Pimples"
                    sub="Hormonal pattern"       value={form.pimples}       onChange={onChange} />
          </div>
        </div>
      </div>

      {/* ── Vitals ────────────────────────────────────────────────────────── */}
      <div id="section-vitals" className="card border-orange">
        <div className="card-body">
          <div className="section-title">Vitals</div>
          <div className="form-grid">
            <Slider name="bp_systolic"  label="Systolic BP"  value={form.bp_systolic}  onChange={onChange}
                    min={80} max={180} step={1} unit=" mmHg" refRange="< 120 mmHg" />
            <Slider name="bp_diastolic" label="Diastolic BP" value={form.bp_diastolic} onChange={onChange}
                    min={50} max={120} step={1} unit=" mmHg" refRange="< 80 mmHg" />
          </div>
        </div>
      </div>

      {/* ── Submit ────────────────────────────────────────────────────────── */}
      <button className="btn-submit" onClick={onSubmit} disabled={loading}>
        {loading ? 'Analysing…' : 'Run Diagnostic Pipeline'}
      </button>
    </div>
  );
}
