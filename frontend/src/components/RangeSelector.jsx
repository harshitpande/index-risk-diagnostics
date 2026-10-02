import { RANGES } from "../lib/timeAxis";
import "./RangeSelector.css";

// Shared range control for Charts 1-3 (Price, Volatility, Drawdown). Controlled by App, which
// owns the selected range and passes it to those three charts; Chart 4 is excluded.
export default function RangeSelector({ value, onChange }) {
  return (
    <div className="range-selector" role="group" aria-label="Time range">
      {RANGES.map((range) => (
        <button
          key={range}
          type="button"
          className={`range-item ${range === value ? "is-selected" : ""}`}
          aria-pressed={range === value}
          onClick={() => onChange(range)}
        >
          {range}
        </button>
      ))}
    </div>
  );
}
