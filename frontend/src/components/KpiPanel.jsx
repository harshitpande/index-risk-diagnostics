import "./KpiPanel.css";

// Today's value for the chart it sits beside. Display only — no thresholds or colour-coding,
// so it adds no interpretation beyond what snapshot.json already carries.
export default function KpiPanel({ label, value, caption }) {
  return (
    <section className="kpi-panel">
      <h2 className="kpi-panel-label">{label}</h2>
      <p className="kpi-panel-value">{value ?? "—"}</p>
      {caption && <p className="kpi-panel-caption">{caption}</p>}
    </section>
  );
}
