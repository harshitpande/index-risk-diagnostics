import Header from "./components/Header";
import SignalStatusBar from "./components/SignalStatusBar";
import RangeSelector from "./components/RangeSelector";
import ChartCard from "./components/ChartCard";
import PriceRegimeChart from "./components/PriceRegimeChart";
import VolatilityChart from "./components/VolatilityChart";
import DrawdownChart from "./components/DrawdownChart";
import MonteCarloChart from "./components/MonteCarloChart";

export default function App() {
  return (
    <div className="app-shell">
      <Header />

      <div className="controls-row">
        <SignalStatusBar />
        <RangeSelector />
      </div>

      <ChartCard title="Price History with Regime-Coloured Line">
        <PriceRegimeChart />
      </ChartCard>
      <ChartCard title="Realized vs GARCH Volatility">
        <VolatilityChart />
      </ChartCard>
      <ChartCard title="Drawdown from Peak">
        <DrawdownChart />
      </ChartCard>
      <ChartCard title="1-Month Monte Carlo Fan Chart">
        <MonteCarloChart />
      </ChartCard>
    </div>
  );
}
