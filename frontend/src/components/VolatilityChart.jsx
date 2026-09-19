import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import "./VolatilityChart.css";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

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

export default function VolatilityChart() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    // Dev-only: fetches the manually copied frontend/public/timeseries.json, a duplicate of
    // data/dashboard/timeseries.json (see frontend/.gitignore). A build-time copy step will
    // replace this manual one later — for now, re-copy the file by hand after each pipeline run.
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
    const realizedColor = cssVar("--color-series-realized-vol");
    const garchColor = cssVar("--color-series-garch-vol");

    return {
      grid: { left: 56, right: 16, top: 44, bottom: 32 },
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
        axisLabel: {
          color: textMuted,
          formatter: (val) => `${(val * 100).toFixed(0)}%`,
        },
        splitLine: { lineStyle: { color: cardBorder } },
      },
      // Native ECharts legend (not a static HTML block like Chart 1's regime legend) — this chart
      // has 2 real series, and REQUIREMENTS.md §6 requires legend-toggle for multi-series charts,
      // which a static HTML legend can't provide. Styled via textStyle/itemWidth/icon to visually
      // match Chart 1's small round-swatch, muted-text legend look as closely as ECharts allows.
      legend: {
        top: 8,
        left: "center",
        itemWidth: 10,
        itemHeight: 10,
        icon: "circle",
        itemGap: 16,
        textStyle: { color: textMuted, fontFamily, fontSize: 15 },
        inactiveColor: cardBorder,
      },
      // Inside-type zoom only (scroll/pinch + drag) — no visible slider UI. The shared
      // 1M/6M/1Y/5Y/All RangeSelector will drive Charts 1-3 together in a later pass; this is
      // a separate, per-chart interaction.
      // TODO(mobile): verify inside-zoom doesn't fight page scroll on touch devices once the
      // mobile-responsive pass happens.
      dataZoom: [{ type: "inside", xAxisIndex: 0 }],
      tooltip: {
        trigger: "axis",
        backgroundColor: cardBg,
        borderColor: cardBorder,
        borderWidth: 1,
        padding: 10,
        textStyle: { color: textPrimary, fontFamily, fontSize: 13 },
        formatter: (params) => {
          const date = params[0].data[0];
          const lines = params
            .map((p) => {
              const value = p.data[1];
              const display = value == null ? "—" : percentFormatter.format(value);
              return `<div><span style="color:${p.color};">●</span> ${p.seriesName}: ${display}</div>`;
            })
            .join("");
          return `
            <div>
              <div style="color:${textPrimary};font-weight:600;margin-bottom:2px;">${dateFormatter.format(new Date(date))}</div>
              ${lines}
            </div>
          `;
        },
      },
      series: [
        {
          name: "Realized Volatility",
          type: "line",
          data: rows.map((r) => [r.date, r.realized_vol]),
          showSymbol: false,
          lineStyle: { width: 1.5, color: realizedColor },
          itemStyle: { color: realizedColor },
          connectNulls: false,
        },
        {
          name: "GARCH Volatility",
          type: "line",
          data: rows.map((r) => [r.date, r.garch_vol]),
          showSymbol: false,
          lineStyle: { width: 1.5, color: garchColor },
          itemStyle: { color: garchColor },
          connectNulls: false,
        },
      ],
    };
  }, [rows]);

  if (error) {
    return <span>Couldn&rsquo;t load volatility history: {error}</span>;
  }

  if (!option) {
    return <span>Loading volatility history&hellip;</span>;
  }

  return (
    <div className="volatility-chart">
      <div className="volatility-chart-canvas">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge />
      </div>
    </div>
  );
}
