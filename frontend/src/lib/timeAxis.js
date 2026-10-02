import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Shared time-axis interaction for the dashboard: the 1M/6M/1Y/5Y/All range selector that drives
// Charts 1-3 (Price, Volatility, Drawdown) together, the manual-zoom sync that keeps those three
// date-aligned, and the zoom floor used by every chart's dataZoom (including Chart 4).

export const RANGES = ["1M", "6M", "1Y", "5Y", "All"];
export const DEFAULT_RANGE = "1Y";

// Finest allowed zoom window. The data is daily (trading days only), so a sub-day window can land
// between two points and render a blank plot. 7 calendar days always spans ~5 trading sessions
// (3-4 in a holiday week), so the line stays visible and readable at maximum zoom.
export const ZOOM_MIN_SPAN_MS = 7 * 24 * 60 * 60 * 1000;

const RANGE_MONTHS = { "1M": 1, "6M": 6, "1Y": 12, "5Y": 60 };

// ECharts' time axis parses a bare "YYYY-MM-DD" as LOCAL midnight (unlike Date.parse, which uses
// UTC), so window bounds must be built the same way or the edge points fall outside the window.
function parseLocalDate(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

// Window for a range, anchored on the last data date (not today) so weekends/holidays never
// shorten it. "All" is every row in timeseries.json (the full history).
export function rangeWindow(range, rows) {
  const first = parseLocalDate(rows[0].date).getTime();
  const end = parseLocalDate(rows[rows.length - 1].date);
  const months = RANGE_MONTHS[range];
  if (months == null) return { startValue: first, endValue: end.getTime() };

  const start = new Date(end);
  start.setMonth(start.getMonth() - months);
  return { startValue: Math.max(start.getTime(), first), endValue: end.getTime() };
}

function dispatchWindow(chart, { startValue, endValue }) {
  chart.dispatchAction({ type: "dataZoom", dataZoomIndex: 0, startValue, endValue });
}

// Registry of the historical chart instances whose zoom is kept in sync. `relaying` guards
// against the feedback loop: dispatching dataZoom on a chart fires its own `datazoom` event,
// and dispatchAction is synchronous, so the flag is set for exactly the relayed dispatches.
export function createZoomGroup() {
  const members = new Set();
  let relaying = false;

  function guarded(fn) {
    relaying = true;
    try {
      fn();
    } finally {
      relaying = false;
    }
  }

  return {
    add: (chart) => members.add(chart),
    remove: (chart) => members.delete(chart),
    // Programmatic window change (range selector) — not relayed; every member applies its own.
    apply: (chart, window) => guarded(() => dispatchWindow(chart, window)),
    // Manual zoom/pan on `source` — copy its resolved value window (not start/end percentages)
    // to the other members so the charts stay aligned on dates.
    relayFrom: (source) => {
      if (relaying) return;
      const { startValue, endValue } = source.getOption().dataZoom[0];
      guarded(() =>
        members.forEach((chart) => {
          if (chart !== source) dispatchWindow(chart, { startValue, endValue });
        }),
      );
    },
  };
}

// Wires one historical chart into the shared time window: applies the selected range, and
// relays manual zoom/pan to the other members of the group. Only zoom is synced — tooltips and
// axis pointers stay per-chart. Returns props for
// <ReactECharts onChartReady={onChartReady} onEvents={onEvents}>.
//
// The instance comes from onChartReady, not a ref read in an effect: echarts-for-react's mount
// creates a temporary instance, waits for its first render, disposes it, then creates the real
// one asynchronously — an effect would grab (and zoom/register) the throwaway instance.
//
// `selection` is `{ range }`, a new object per click: keying the effect on its identity (not the
// range string) makes re-clicking the active range re-apply it after a manual zoom.
export function useSyncedTimeWindow({ selection, rows, zoomGroup }) {
  const [chart, setChart] = useState(null);
  // Latest selection/rows for onChartReady, which echarts-for-react captures once at mount.
  const latest = useRef({ selection, rows });
  useEffect(() => {
    latest.current = { selection, rows };
  });

  const onChartReady = useCallback(
    (instance) => {
      // Apply the window synchronously, before ECharts' next-frame render, so the chart never
      // flashes the full history before the selected range takes effect.
      const { selection: sel, rows: data } = latest.current;
      if (data) zoomGroup.apply(instance, rangeWindow(sel.range, data));
      setChart(instance);
    },
    [zoomGroup],
  );

  useEffect(() => {
    if (!chart) return undefined;
    zoomGroup.add(chart);
    return () => {
      zoomGroup.remove(chart);
    };
  }, [chart, zoomGroup]);

  // Range selector changes after mount.
  useEffect(() => {
    if (chart && rows) zoomGroup.apply(chart, rangeWindow(selection.range, rows));
  }, [chart, selection, rows, zoomGroup]);

  // echarts-for-react calls handlers as (params, instance).
  const onEvents = useMemo(
    () => ({ datazoom: (_params, instance) => zoomGroup.relayFrom(instance) }),
    [zoomGroup],
  );

  return { onChartReady, onEvents };
}
