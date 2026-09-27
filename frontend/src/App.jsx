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

  useEffect(() => {
    let cancelled = false;
    // Dev-only: fetches the manually copied frontend/public/snapshot.json, a duplicate of
    // data/dashboard/snapshot.json (see frontend/.gitignore). A build-time copy step will
    // replace this manual one later — for now, re-copy the file by hand after each pipeline run.
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
        <RangeSelector />
      </div>

      <ChartCard title="Price History with Regime-Coloured Line">
        <PriceRegimeChart />
      </ChartCard>

      <div className="paired-row">
        <KpiPanel
          label="Realized volatility"
          value={snapshot ? percentFormatter.format(snapshot.realized_vol) : null}
          caption={asOf && `Annualised, ${asOf}`}
        />
        <ChartCard title="Realized vs GARCH Volatility">
          <VolatilityChart />
        </ChartCard>
      </div>

      <div className="paired-row">
        <KpiPanel
          label="Drawdown from peak"
          value={snapshot ? percentFormatter.format(snapshot.drawdown) : null}
          caption={asOf && `Below running peak, ${asOf}`}
        />
        <ChartCard title="Drawdown from Peak">
          <DrawdownChart />
        </ChartCard>
      </div>

      <ChartCard title="1-Month Monte Carlo Fan Chart">
        <MonteCarloChart />
      </ChartCard>
    </div>
  );
}
