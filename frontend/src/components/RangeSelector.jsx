import "./RangeSelector.css";

// Non-functional visual placeholder for now — will control Charts 1-3
// (Price, Volatility, Drawdown) together once wired up. Default window is 1Y.
const RANGES = ["1M", "6M", "1Y", "5Y", "All"];
const DEFAULT_RANGE = "1Y";

export default function RangeSelector() {
  return (
    <div className="range-selector" role="group" aria-label="Time range">
      {RANGES.map((range) => (
        <button
          key={range}
          type="button"
          className={`range-item ${range === DEFAULT_RANGE ? "is-selected" : ""}`}
          aria-current={range === DEFAULT_RANGE ? "true" : undefined}
        >
          {range}
        </button>
      ))}
    </div>
  );
}
