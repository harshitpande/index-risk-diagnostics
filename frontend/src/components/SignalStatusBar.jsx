import "./SignalStatusBar.css";

// Hardcoded for the layout-shell preview. Once wired up, `active` will come
// from the day's early-warning-signals output (early_warning_signals.pkl via
// the dashboard JSON export) instead of being hardcoded here.
const SIGNALS = [
  { key: "stress", label: "Stress", active: true },
  { key: "crisis", label: "Crisis", active: false },
  { key: "escalation", label: "Escalation", active: false },
];

export default function SignalStatusBar() {
  return (
    <div className="signal-bar" role="status" aria-label="Signal status">
      {SIGNALS.map(({ key, label, active }) => (
        <span
          key={key}
          className={`signal-chip signal-chip--${key} ${active ? "is-active" : "is-muted"}`}
        >
          {label}
        </span>
      ))}
    </div>
  );
}
