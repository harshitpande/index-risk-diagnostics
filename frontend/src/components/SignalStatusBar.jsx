import "./SignalStatusBar.css";

const SIGNALS = [
  { key: "stress", label: "Stress" },
  { key: "crisis", label: "Crisis" },
  { key: "escalation", label: "Escalation" },
];

// `signals` is snapshot.json's {stress, crisis, escalation} booleans. While it's undefined
// (snapshot still loading or failed), every chip renders muted so the bar's shape stays stable.
export default function SignalStatusBar({ signals }) {
  return (
    <div className="signal-bar" role="status" aria-label="Signal status">
      {SIGNALS.map(({ key, label }) => {
        const active = Boolean(signals?.[key]);
        return (
          <span
            key={key}
            className={`signal-chip signal-chip--${key} ${active ? "is-active" : "is-muted"}`}
          >
            {label}
          </span>
        );
      })}
    </div>
  );
}
