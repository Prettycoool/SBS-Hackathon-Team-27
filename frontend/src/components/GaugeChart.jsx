// Semicircular gauge — rebuilt with explicit SVG arc paths.
//
// Geometry:
//   Centre of diameter: CX=100, CY=82  (needle pivot, hub centre)
//   Radius: R=68,  stroke-width: SW=15
//   Arc spans from (32, 82) → top (100, 14) → (168, 82)
//   Value v∈[0,1] maps to angle π + v·π  (180° → 360° going CW through top)
//
// Layout (all y measured from top of viewBox "0 0 200 150"):
//   Arc band          y ≈ 14 – 82
//   Hub circle        y = 82
//   "Low" / "High"    y = 96   (just below arc endpoints)
//   Percentage        y = 113
//   Risk label        y = 128
//   "PCOS Prob…"      y = 143

const CX = 100, CY = 82, R = 68, SW = 15;
const NEEDLE_LEN = 50; // tip lands well inside the arc track

// Convert gauge value [0,1] → SVG (x,y) on the arc centreline
function pt(v) {
  const rad = Math.PI * (1 + v);   // π at v=0, 2π at v=1
  return {
    x: +(CX + R * Math.cos(rad)).toFixed(2),
    y: +(CY + R * Math.sin(rad)).toFixed(2),
  };
}

// SVG arc path string from gauge value v1 to v2
function arc(v1, v2) {
  const a = pt(v1), b = pt(v2);
  // The arc spans at most 180° total, each zone < 180°, so large-arc = 0
  // sweep = 1 (clockwise in SVG) traces the upper half correctly
  const large = (v2 - v1) >= 1.0 ? 1 : 0;
  return `M ${a.x} ${a.y} A ${R} ${R} 0 ${large} 1 ${b.x} ${b.y}`;
}

// Coloured background zones
const ZONES = [
  { v1: 0,    v2: 0.35, color: '#52BE80' }, // green  — low risk
  { v1: 0.35, v2: 0.65, color: '#F5B041' }, // amber  — moderate
  { v1: 0.65, v2: 1.0,  color: '#E74C3C' }, // red    — high risk
];

export default function GaugeChart({ value }) {
  const v = Math.max(0, Math.min(1, value ?? 0));

  const accentColor =
    v < 0.35 ? '#1E8449' :
    v < 0.65 ? '#D68910' :
               '#B03A2E';

  const riskLabel =
    v < 0.35 ? 'Low Risk' :
    v < 0.65 ? 'Moderate' :
               'High Risk';

  // Needle: draw pointing LEFT from hub, rotate CW by v·180°
  //   v=0 → 0° rotation  → points left  (arc start, 0%)
  //   v=0.5 → 90° CW     → points UP    (arc top,  50%)
  //   v=1 → 180° CW      → points right (arc end, 100%)
  const needleRot = +(v * 180).toFixed(2);

  // Arc endpoint coordinates for "Low" / "High" label anchoring
  const leftEnd  = pt(0);   // (32, 82)
  const rightEnd = pt(1);   // (168, 82)

  return (
    <svg
      viewBox="0 0 200 150"
      style={{ width: '100%', maxWidth: 260, display: 'block' }}
      aria-label={`PCOS probability ${Math.round(v * 100)}%`}
    >
      {/* ── 1. Background zone arcs ──────────────────────────────────────── */}
      {ZONES.map(({ v1, v2, color }) => (
        <path
          key={v1}
          d={arc(v1, v2)}
          fill="none"
          stroke={color}
          strokeWidth={SW}
          strokeLinecap="butt"
        />
      ))}

      {/* ── 2. Thin white dividers between zones ─────────────────────────── */}
      {[0.35, 0.65].map(tick => {
        const inner = { x: CX + (R - SW / 2) * Math.cos(Math.PI * (1 + tick)),
                        y: CY + (R - SW / 2) * Math.sin(Math.PI * (1 + tick)) };
        const outer = { x: CX + (R + SW / 2) * Math.cos(Math.PI * (1 + tick)),
                        y: CY + (R + SW / 2) * Math.sin(Math.PI * (1 + tick)) };
        return (
          <line key={tick}
            x1={inner.x.toFixed(1)} y1={inner.y.toFixed(1)}
            x2={outer.x.toFixed(1)} y2={outer.y.toFixed(1)}
            stroke="white" strokeWidth={2.5} strokeLinecap="round"
          />
        );
      })}

      {/* ── 3. Needle ────────────────────────────────────────────────────── */}
      <g transform={`rotate(${needleRot}, ${CX}, ${CY})`}>
        {/* Line from hub toward left; rotation sweeps it CW to the value angle */}
        <line
          x1={CX - NEEDLE_LEN} y1={CY}
          x2={CX - 8}          y2={CY}
          stroke="#1A252F" strokeWidth={3.5} strokeLinecap="round"
        />
      </g>

      {/* ── 4. Hub (drawn last so it sits on top of needle base) ─────────── */}
      <circle cx={CX} cy={CY} r={8}   fill="#1A252F" />
      <circle cx={CX} cy={CY} r={4.5} fill="white"   />

      {/* ── 5. "Low" / "High" labels anchored to arc endpoints ───────────── */}
      <text
        x={leftEnd.x} y={CY + 14}
        textAnchor="middle" fontSize={9}
        fill="#AABBCC" fontFamily="Inter, sans-serif"
      >
        Low
      </text>
      <text
        x={rightEnd.x} y={CY + 14}
        textAnchor="middle" fontSize={9}
        fill="#AABBCC" fontFamily="Inter, sans-serif"
      >
        High
      </text>

      {/* ── 6. Percentage ────────────────────────────────────────────────── */}
      <text
        x={CX} y={CY + 31}
        textAnchor="middle" fontSize={28} fontWeight="700"
        fill={accentColor} fontFamily="Inter, sans-serif"
      >
        {Math.round(v * 100)}%
      </text>

      {/* ── 7. Risk label ────────────────────────────────────────────────── */}
      <text
        x={CX} y={CY + 46}
        textAnchor="middle" fontSize={11}
        fill="#5D6D7E" fontFamily="Inter, sans-serif"
      >
        {riskLabel}
      </text>

      {/* ── 8. "PCOS Probability" footer ─────────────────────────────────── */}
      <text
        x={CX} y={143}
        textAnchor="middle" fontSize={9}
        fill="#8899AA" fontFamily="Inter, sans-serif" letterSpacing="0.4"
      >
        PCOS PROBABILITY
      </text>
    </svg>
  );
}
