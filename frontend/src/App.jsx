import { useEffect, useState } from "react";
import Header from "./components/Header";
import VerdictBanner from "./components/VerdictBanner";
import RangeSelector from "./components/RangeSelector";
import KpiPanel from "./components/KpiPanel";
import ChartCard from "./components/ChartCard";
import PriceRegimeChart from "./components/PriceRegimeChart";
import VolatilityChart from "./components/VolatilityChart";
import DrawdownChart from "./components/DrawdownChart";
import MonteCarloChart from "./components/MonteCarloChart";
import { DEFAULT_RANGE, createZoomGroup } from "./lib/timeAxis";

const dateFormatter = new Intl.DateTimeFormat("en-IN", {
  year: "numeric",
  month: "short",
  day: "numeric",
});
const percentFormatter = new Intl.NumberFormat("en-IN", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

export default function App() {
  const [snapshot, setSnapshot] = useState(null);
  const [snapshotError, setSnapshotError] = useState(null);
  // Shared time window for Charts 1-3: the selected range, plus the group that keeps their
  // manual zoom in sync. Chart 4 (Monte Carlo) takes neither.
  // A fresh object per click, so re-clicking the active range still re-applies it (snapping back
  // after a manual zoom) — a bare string state would make that click a no-op.
  const [selection, setSelection] = useState({ range: DEFAULT_RANGE });
  const [zoomGroup] = useState(createZoomGroup);

  useEffect(() => {
    let cancelled = false;
    // Fetches frontend/public/snapshot.json, copied from data/dashboard/snapshot.json by
    // scripts/copy-data.js on every dev start/build (predev/prebuild; see frontend/.gitignore).
    fetch("/snapshot.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load snapshot.json (${res.status})`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setSnapshot(data);
      })
      .catch((err) => {
        if (!cancelled) setSnapshotError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const asOf = snapshot ? `as of ${dateFormatter.format(new Date(snapshot.date))}` : null;

  return (
    <div className="app-shell">
      <Header />

      <VerdictBanner snapshot={snapshot} error={snapshotError} />

      <div className="range-row">
        <RangeSelector value={selection.range} onChange={(range) => setSelection({ range })} />
      </div>

      <ChartCard title="Price History with Regime-Coloured Line">
        <PriceRegimeChart selection={selection} zoomGroup={zoomGroup} />
      </ChartCard>

      <div className="paired-row">
        <KpiPanel
          label="Realized volatility"
          value={snapshot ? percentFormatter.format(snapshot.realized_vol) : null}
          caption={asOf && `Annualised, ${asOf}`}
        />
        <ChartCard title="Realized vs GARCH Volatility">
          <VolatilityChart selection={selection} zoomGroup={zoomGroup} />
        </ChartCard>
      </div>

      <div className="paired-row">
        <KpiPanel
          label="Drawdown from peak"
          value={snapshot ? percentFormatter.format(snapshot.drawdown) : null}
          caption={asOf && `Below running peak, ${asOf}`}
        />
        <ChartCard title="Drawdown from Peak">
          <DrawdownChart selection={selection} zoomGroup={zoomGroup} />
        </ChartCard>
      </div>

      <ChartCard title="1-Month Monte Carlo Fan Chart">
        <MonteCarloChart />
      </ChartCard>
    </div>
  );
}
