import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import "./DrawdownChart.css";

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

export default function DrawdownChart() {
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
    const drawdownColor = cssVar("--color-series-drawdown");
    const thresholdModerateColor = cssVar("--color-drawdown-threshold-moderate");
    const thresholdSevereColor = cssVar("--color-drawdown-threshold-severe");

    return {
      // Wider than the 16px used by Chart 1/2 — the "Severe (-30%)" markLine label is rendered
      // at the plot's right edge (position: "end") and needs room so it isn't clipped.
      grid: { left: 56, right: 92, top: 44, bottom: 32 },
      xAxis: {
        type: "time",
        axisLine: { lineStyle: { color: cardBorder } },
        axisLabel: { color: textMuted },
        splitLine: { show: false },
      },
      yAxis: {
        type: "value",
        scale: true,
        // Pinned (not `inverse: true`, which flips which end is drawn first and would put the
        // worst drawdown at the top). All drawdown values are <= 0, so max: 0 puts 0% flush at
        // the top edge while `scale` still lets the bottom pad naturally around the historical low.
        max: 0,
        axisLine: { show: false },
        axisLabel: {
          color: textMuted,
          formatter: (val) => `${(val * 100).toFixed(0)}%`,
        },
        splitLine: { lineStyle: { color: cardBorder } },
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
          const point = Array.isArray(params) ? params[0] : params;
          const [date, value] = point.data;
          const display = value == null ? "—" : percentFormatter.format(value);
          return `
            <div>
              <div style="color:${textPrimary};font-weight:600;margin-bottom:2px;">${dateFormatter.format(new Date(date))}</div>
              <div><span style="color:${drawdownColor};">●</span> Drawdown: ${display}</div>
            </div>
          `;
        },
      },
      series: [
        {
          name: "Drawdown",
          type: "line",
          data: rows.map((r) => [r.date, r.drawdown]),
          showSymbol: false,
          lineStyle: { width: 1.5, color: drawdownColor },
          itemStyle: { color: drawdownColor },
          // Subtle fill so it doesn't visually dominate the two threshold lines or the price/
          // regime chart above it — origin pinned to 0 explicitly rather than relying on
          // ECharts' 'auto' inference, so the fill-to-zero behavior can't silently change if
          // the axis bounds are ever adjusted.
          areaStyle: { color: drawdownColor, opacity: 0.16, origin: 0 },
          connectNulls: false,
          markLine: {
            silent: true,
            symbol: "none",
            animation: false,
            data: [
              {
                name: "Moderate",
                yAxis: -0.15,
                lineStyle: { color: thresholdModerateColor, type: "dashed", width: 1.5 },
                label: {
                  formatter: "Moderate (-15%)",
                  position: "end",
                  align: "right",
                  verticalAlign: "bottom",
                  color: thresholdModerateColor,
                  fontFamily,
                  fontSize: 12,
                  padding: [0, 4, 2, 4],
                  backgroundColor: cardBg,
                },
              },
              {
                name: "Severe",
                yAxis: -0.3,
                lineStyle: { color: thresholdSevereColor, type: "dashed", width: 1.5 },
                label: {
                  formatter: "Severe (-30%)",
                  position: "end",
                  align: "right",
                  verticalAlign: "bottom",
                  color: thresholdSevereColor,
                  fontFamily,
                  fontSize: 12,
                  padding: [0, 4, 2, 4],
                  backgroundColor: cardBg,
                },
              },
            ],
          },
        },
      ],
    };
  }, [rows]);

  if (error) {
    return <span>Couldn&rsquo;t load drawdown history: {error}</span>;
  }

  if (!option) {
    return <span>Loading drawdown history&hellip;</span>;
  }

  return (
    <div className="drawdown-chart">
      <div className="drawdown-chart-canvas">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge />
      </div>
    </div>
  );
}
