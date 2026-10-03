import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import { ZOOM_MIN_SPAN_MS, useSyncedTimeWindow } from "../lib/timeAxis";
import "./PriceRegimeChart.css";

const REGIME_ORDER = ["Calm", "Pullback", "Stress", "Crisis"];

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function regimeColor(regime) {
  return cssVar(`--color-regime-${regime.toLowerCase()}`);
}

const dateFormatter = new Intl.DateTimeFormat("en-IN", {
  year: "numeric",
  month: "short",
  day: "numeric",
});
const numberFormatter = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 });

export default function PriceRegimeChart({ selection, zoomGroup }) {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const { onChartReady, onEvents } = useSyncedTimeWindow({ selection, rows, zoomGroup });

  useEffect(() => {
    let cancelled = false;
    // Fetches frontend/public/timeseries.json, copied from data/dashboard/timeseries.json by
    // scripts/copy-data.js on every dev start/build (predev/prebuild; see frontend/.gitignore).
    fetch("/timeseries.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load timeseries.json (${res.status})`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setRows(data);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const option = useMemo(() => {
    if (!rows) return null;

    const cardBg = cssVar("--color-card-bg");
    const cardBorder = cssVar("--color-card-border");
    const textMuted = cssVar("--color-card-text-muted");
    const textPrimary = cssVar("--color-card-text-primary");
    const fontFamily = cssVar("--font-family-base");

    return {
      grid: { left: 56, right: 16, top: 16, bottom: 32 },
      xAxis: {
        type: "time",
        axisLine: { lineStyle: { color: cardBorder } },
        axisLabel: { color: textMuted },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        axisLine: { show: false },
        axisLabel: { color: textMuted },
        splitLine: { lineStyle: { color: cardBorder } },
      },
      // Hidden — used only as the per-segment line-colouring mechanism (dimension 2 of each
      // series data tuple is the regime code). Not a visible/clickable legend; see the plain
      // .regime-legend markup below for the reader-facing colour key.
      visualMap: {
        show: false,
        type: "piecewise",
        dimension: 2,
        seriesIndex: 0,
        pieces: REGIME_ORDER.map((regime, i) => ({ value: i, color: regimeColor(regime) })),
      },
      // Inside-type zoom only (scroll/pinch + drag) — no visible slider UI. The window is shared
      // with Charts 2-3: set by the RangeSelector and kept in sync on manual zoom/pan via
      // useSyncedTimeWindow. minValueSpan floors zoom at one week so daily points never vanish.
      // TODO(mobile): verify inside-zoom doesn't fight page scroll on touch devices once the
      // mobile-responsive pass happens.
      dataZoom: [{ type: "inside", xAxisIndex: 0, minValueSpan: ZOOM_MIN_SPAN_MS }],
      tooltip: {
        trigger: "axis",
        backgroundColor: cardBg,
        borderColor: cardBorder,
        borderWidth: 1,
        padding: 10,
        textStyle: { color: textPrimary, fontFamily, fontSize: 13 },
        formatter: (params) => {
          const point = Array.isArray(params) ? params[0] : params;
          const [date, close, regimeCode] = point.data;
          const regime = REGIME_ORDER[regimeCode];
          const color = regimeColor(regime);
          return `
            <div style="border-left:3px solid ${color};padding-left:8px;">
              <div style="color:${color};font-weight:600;margin-bottom:2px;">${regime}</div>
              <div>${dateFormatter.format(new Date(date))}</div>
              <div>NIFTY 50: ${numberFormatter.format(close)}</div>
            </div>
          `;
        },
      },
      series: [
        {
          type: "line",
          data: rows.map((r) => [r.date, r.close, REGIME_ORDER.indexOf(r.regime)]),
          showSymbol: false,
          lineStyle: { width: 1.5 },
        },
      ],
    };
  }, [rows]);

  if (error) {
    return <span>Couldn&rsquo;t load price history: {error}</span>;
  }

  if (!option) {
    return <span>Loading price history&hellip;</span>;
  }

  return (
    <div className="price-regime-chart">
      <div className="price-regime-chart-canvas">
        <ReactECharts
          onChartReady={onChartReady}
          option={option}
          onEvents={onEvents}
          style={{ height: "100%", width: "100%" }}
          notMerge
        />
      </div>
      <div className="regime-legend">
        {REGIME_ORDER.map((regime) => (
          <span key={regime} className="regime-legend-item">
            <span
              className="regime-legend-swatch"
              style={{ background: `var(--color-regime-${regime.toLowerCase()})` }}
            />
            {regime}
          </span>
        ))}
      </div>
    </div>
  );
}
