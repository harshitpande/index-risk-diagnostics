import { useEffect, useMemo, useState } from "react";
import ReactECharts from "echarts-for-react";
import "./MonteCarloChart.css";

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const dateFormatter = new Intl.DateTimeFormat("en-IN", {
  year: "numeric",
  month: "short",
  day: "numeric",
});
const numberFormatter = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 2 });
const axisNumberFormatter = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
const percentFormatter = new Intl.NumberFormat("en-IN", {
  style: "percent",
  minimumFractionDigits: 1,
  maximumFractionDigits: 1,
});

export default function MonteCarloChart() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    // Dev-only: fetches the manually copied frontend/public/montecarlo.json, a duplicate of
    // data/dashboard/montecarlo.json (see frontend/.gitignore). A build-time copy step will
    // replace this manual one later — for now, re-copy the file by hand after each pipeline run.
    fetch("/montecarlo.json")
      .then((res) => {
        if (!res.ok) throw new Error(`Failed to load montecarlo.json (${res.status})`);
        return res.json();
      })
      .then((json) => {
        if (!cancelled) setData(json);
      })
      .catch((err) => {
        if (!cancelled) setError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Lookup maps for the tooltip, keyed by parsed timestamp so the axis-trigger formatter can
  // resolve a hovered point back to its full row (percentiles for forecast, close for
  // historical) without threading that data through the plotted series values themselves.
  const { historicalByTime, forecastByTime } = useMemo(() => {
    if (!data) return { historicalByTime: null, forecastByTime: null };
    return {
      historicalByTime: new Map(data.historical.map((r) => [Date.parse(r.date), r])),
      forecastByTime: new Map(data.forecast.map((r) => [Date.parse(r.date), r])),
    };
  }, [data]);

  const option = useMemo(() => {
    if (!data) return null;

    const cardBg = cssVar("--color-card-bg");
    const cardBorder = cssVar("--color-card-border");
    const textMuted = cssVar("--color-card-text-muted");
    const textPrimary = cssVar("--color-card-text-primary");
    const fontFamily = cssVar("--font-family-base");
    const historicalColor = cssVar("--color-series-mc-historical");
    const bandColor = cssVar("--color-series-mc-band");
    const medianColor = cssVar("--color-series-mc-median");

    const endLabelBase = {
      show: true,
      fontFamily,
      fontSize: 12,
      padding: [2, 4],
      backgroundColor: cardBg,
      distance: 8,
    };

    return {
      grid: { left: 56, right: 96, top: 44, bottom: 32 },
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
          formatter: (val) => axisNumberFormatter.format(val),
        },
        splitLine: { lineStyle: { color: cardBorder } },
      },
      // Native ECharts legend, styled to match Chart 2/3. `_band_base` is deliberately excluded
      // from `data` so it can't be toggled independently — hiding only the invisible stack
      // baseline would shift the visible "90% Band" area down by its own value.
      legend: {
        data: ["Historical", "90% Band", "Median"],
        top: 8,
        left: "center",
        itemWidth: 10,
        itemHeight: 10,
        icon: "circle",
        itemGap: 16,
        textStyle: { color: textMuted, fontFamily, fontSize: 15 },
        inactiveColor: cardBorder,
      },
      // Inside-type zoom only (scroll/pinch + drag) — no visible slider UI. Chart 4 stays
      // excluded from the shared 1M/6M/1Y/5Y/All RangeSelector (REQUIREMENTS.md §6) since a
      // range control isn't meaningful for a short forward projection.
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
          const time = point.axisValue;
          const forecastRow = forecastByTime.get(time);
          const dateLabel = dateFormatter.format(new Date(time));

          if (forecastRow) {
            return `
              <div>
                <div style="color:${textPrimary};font-weight:600;margin-bottom:2px;">${dateLabel}</div>
                <div><span style="color:${bandColor};">●</span> 95th: ${numberFormatter.format(forecastRow.ptile_95)}</div>
                <div><span style="color:${bandColor};">●</span> 75th: ${numberFormatter.format(forecastRow.ptile_75)}</div>
                <div><span style="color:${medianColor};">●</span> 50th (median): ${numberFormatter.format(forecastRow.ptile_50)}</div>
                <div><span style="color:${bandColor};">●</span> 25th: ${numberFormatter.format(forecastRow.ptile_25)}</div>
                <div><span style="color:${bandColor};">●</span> 5th: ${numberFormatter.format(forecastRow.ptile_5)}</div>
              </div>
            `;
          }

          const historicalRow = historicalByTime.get(time);
          if (historicalRow) {
            return `
              <div>
                <div style="color:${textPrimary};font-weight:600;margin-bottom:2px;">${dateLabel}</div>
                <div><span style="color:${historicalColor};">●</span> NIFTY 50: ${numberFormatter.format(historicalRow.close)}</div>
              </div>
            `;
          }

          return dateLabel;
        },
      },
      series: [
        {
          name: "Historical",
          type: "line",
          data: data.historical.map((r) => [r.date, r.close]),
          showSymbol: false,
          lineStyle: { width: 1.5, color: historicalColor },
          itemStyle: { color: historicalColor },
          connectNulls: false,
        },
        {
          // Invisible stack baseline sitting at ptile_5 — establishes where the visible "90%
          // Band" area starts, per the standard ECharts stacked-area band technique.
          name: "_band_base",
          type: "line",
          stack: "mc-band",
          data: data.forecast.map((r) => [r.date, r.ptile_5]),
          showSymbol: false,
          symbol: "none",
          lineStyle: { opacity: 0 },
          tooltip: { show: false },
          endLabel: {
            ...endLabelBase,
            formatter: (p) => numberFormatter.format(p.value[1]),
            color: bandColor,
          },
        },
        {
          // Plotted value is the band WIDTH (ptile_95 - ptile_5), not ptile_95 itself — stacking
          // on top of "_band_base" makes the visible top of the area land at the true p95 height.
          // The raw p95 rides in data[2] so the end-label can show the true value, not the delta.
          name: "90% Band",
          type: "line",
          stack: "mc-band",
          data: data.forecast.map((r) => [r.date, r.ptile_95 - r.ptile_5, r.ptile_95]),
          showSymbol: false,
          symbol: "none",
          lineStyle: { opacity: 0 },
          areaStyle: { color: bandColor, opacity: 0.28 },
          tooltip: { show: false },
          endLabel: {
            ...endLabelBase,
            formatter: (p) => numberFormatter.format(p.value[2]),
            color: bandColor,
          },
        },
        {
          name: "Median",
          type: "line",
          data: data.forecast.map((r) => [r.date, r.ptile_50]),
          showSymbol: false,
          lineStyle: { width: 2, type: "dashed", color: medianColor },
          itemStyle: { color: medianColor },
          connectNulls: false,
          endLabel: {
            ...endLabelBase,
            formatter: (p) => numberFormatter.format(p.value[1]),
            color: medianColor,
          },
        },
      ],
    };
  }, [data, historicalByTime, forecastByTime]);

  if (error) {
    return <span>Couldn&rsquo;t load Monte Carlo forecast: {error}</span>;
  }

  if (!option) {
    return <span>Loading Monte Carlo forecast&hellip;</span>;
  }

  return (
    <div className="montecarlo-chart">
      <div className="montecarlo-context">
        Vol={percentFormatter.format(data.current_vol)} | Regime: {data.current_regime}
      </div>
      <div className="montecarlo-chart-canvas">
        <ReactECharts option={option} style={{ height: "100%", width: "100%" }} notMerge />
      </div>
      <p className="montecarlo-caption">
        Shaded region is a 90% probability band conditional on current risk state — a scenario
        distribution, not a price prediction.
      </p>
    </div>
  );
}
