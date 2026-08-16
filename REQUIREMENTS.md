# REQUIREMENTS.md — NIFTY 50 Risk Diagnostics Dashboard

> **Status:** v1 — locked, build-ready.
> **Scope:** Frontend dashboard only. Pipeline, JSON schema, and hosting are specified elsewhere (see ARCHITECTURE.md).
> **Purpose of this file:** The single source of truth for *what the dashboard shows and how it behaves*. A coding agent building the frontend should be able to work from this file without inferring intent. Where a choice is deliberate, it is stated; where something is explicitly out of scope, it is listed so it is not silently added back.

---

## 1. Product intent

The dashboard answers one question for a viewer: **what is the NIFTY 50's current market risk state, and how is it likely to evolve?**

It is a *diagnostic* tool, not a *predictive* one. Language, labelling, and framing throughout must communicate risk-state assessment, never price prediction. This constraint governs copy on every chart (axis labels, tooltips, titles, captions).

Audience: the builder's professional/portfolio audience (recruiters, interviewers, peers). It must read as considered and precise, not as a trading terminal.

---

## 2. Global layout

A **single-page dashboard**, top to bottom:

```
┌─────────────────────────────────────────────────────┐
│  SIGNAL STATUS BAR   [Stress] [Crisis] [Escalation]  │  ← always visible, slim
├─────────────────────────────────────────────────────┤
│  CHART 1: Price History with Regime-Coloured Line    │
├─────────────────────────────────────────────────────┤
│  CHART 2: Realized vs GARCH Volatility               │
├─────────────────────────────────────────────────────┤
│  CHART 3: Drawdown from Peak                         │
├─────────────────────────────────────────────────────┤
│  CHART 4: 1-Month Monte Carlo Fan Chart              │
└─────────────────────────────────────────────────────┘
```

There is **no separate "Early Warning System" section heading**. The three signals live in the status bar at the top; they *are* the early warning, surfaced by prominence rather than by a section. The four charts below are the diagnostic layer. Conceptually one dashboard, not two sections.

Charts stack vertically, full width, one per row. (Rationale: these are time series sharing an x-axis; stacking keeps dates visually aligned and leaves room for the fan chart's detail. Revisit only if a 2-column layout is explicitly requested.)

**Mobile responsive — required, first-class.** The dashboard must work well on mobile, not merely not-break. The single-column stack degrades naturally to narrow screens; charts resize to viewport width; the signal status bar wraps or stacks its three chips as needed; and because touch has no hover, **tooltips must be tap-to-show on touch devices** (tap a point to reveal its values). This is a primary requirement, weighted equally with the desktop experience — not an afterthought.

---

## 3. Theme and palette

**Page chrome:** white background (`#FFFFFF`), dark text. Clean, professional, non-terminal.

**Chart surfaces:** each chart sits in its own **dark card** (near-black navy panel, e.g. `#0E1117` or similar) with rounded corners and subtle shadow. This white-page / dark-panel split is deliberate: it keeps the page clean while preserving the dark backgrounds the existing chart palettes were designed for, and gives the signal chips a dark surface so their colours read clearly.

**Regime colour system** (used consistently everywhere a regime appears — the price line, tooltips, any regime reference):

| Regime    | Meaning                  | Colour (on dark panel) |
|-----------|--------------------------|------------------------|
| Calm      | Expansion / low risk     | Teal / green           |
| Pullback  | Normal-risk correction   | Amber / muted orange   |
| Stress    | Elevated volatility      | Red-orange             |
| Crisis    | Systemic stress          | Deep red / maroon      |

Exact hex values are the designer's choice at build time — tuned against the dark panel for contrast and verified on screen. They are a build decision, not an open question. The *mapping* above is fixed and must be identical across all components.

---

## 4. Signal status bar (top)

Three status indicators grouped in one container (single div, visually joined as a segmented control / chip group).

**Behaviour:** each is a **status light**, not an interactive control. It reflects today's pipeline output — lit/active when that signal is currently firing, muted/inactive otherwise. Clicking does nothing (no filtering, no navigation). *This is deliberate and must not be extended to interactive filtering without a spec change.*

**Signals and active-state colours:**

| Signal      | Active colour | Meaning                                    |
|-------------|---------------|--------------------------------------------|
| Stress      | Yellow        | Stress-regime early-warning signal active  |
| Crisis      | Orange        | Crisis-regime alert active                 |
| Escalation  | Red           | Regime-escalation signal active            |

Inactive state: muted/greyed version of the same chip, so the bar's shape is stable whether or not signals fire.

**Out of scope (explicitly removed):** the "signal strength" display. Do not render any strength meter, percentage, or intensity indicator. Active/inactive only.

---

## 5. Diagnostic charts

### Chart 1 — Price History with Regime-Coloured Line

- **Data:** daily NIFTY 50 close, full history in the JSON, with a regime label on each daily point.
- **Rendering:** a single price line whose **colour changes by regime segment** (using the regime colour system in §3). Background regime *bands* from the old version are **removed** — the colour lives in the line itself, not behind it.
- **Interaction — hover:** tooltip shows the date and the actual index value. The **tooltip/marker colour matches the regime** of the hovered point (hovering a Crisis-period point → deep-red tooltip accent, etc.).
- **Axes:** x = date; y = index level.

### Chart 2 — Realized vs GARCH Volatility

- **Data:** two daily series — realized (rolling, annualised) volatility and GJR-GARCH conditional volatility.
- **Rendering:** two lines, distinct colours, with a legend.
- **Interaction — hover:** tooltip shows the date and **both** volatility values at that point (realized and GARCH together), so the viewer can read the gap between them.
- **Axes:** x = date; y = annualised volatility (%).

### Chart 3 — Drawdown from Peak

- **Data:** daily drawdown series (percentage below running peak).
- **Rendering:** filled area/line showing drawdown depth over time. Reference lines at the −15% (moderate) and −30% (severe) thresholds, labelled.
- **Interaction — hover:** tooltip shows date and drawdown %.
- **Axes:** x = date; y = drawdown (%, 0 at top, negative below).

### Chart 4 — 1-Month Monte Carlo Fan Chart

- **Data:** recent historical index (for context) + forward simulation percentiles over a ~21-trading-day horizon: 5th, 25th, 50th (median), 75th, 95th. (Bands: 90% = 5th–95th, 50% = 25th–75th.)
- **Rendering:** historical line transitioning at "today" into a forward fan — a wider outer band (90%) and a narrower inner band (50%) around a dashed median line. Annotate the endpoint percentile values.
- **Framing (critical):** this is a **risk-conditioned scenario distribution, not a forecast**. The interval must be labelled as a **90% probability band conditional on current risk state**, never as a prediction or a confidence interval about a point forecast. Chart title/caption carries the current volatility and regime context (e.g. "Vol=12.3% | Regime: Pullback").
- **Interaction — hover:** tooltip shows the date and the percentile values at that horizon step.
- **Axes:** x = date (historical → forward); y = index level.

---

## 6. Interactivity baseline (all charts)

**Charting library: Apache ECharts** (via `echarts-for-react`). Locked. Every chart uses it, for consistent interaction behaviour, tooltips, and theming across the dashboard.

Every chart must support, at minimum:
- Hover tooltips (per-chart contents specified above); tap-to-show on touch (see §2).
- Zoom / pan on the time axis.
- Legend toggle where a chart has multiple series.

**Range selector (selectable window, Google-Finance style).** A shared row of buttons — **1M / 6M / 1Y / 5Y / All (since 2020)** — reframes the x-axis on click. It controls the three historical time-series charts together (Chart 1 Price, Chart 2 Volatility, Chart 3 Drawdown) so they stay date-aligned as one view. Default window: **1Y**. The Monte Carlo fan chart (Chart 4) is **excluded** from the selector — it keeps a fixed recent-history-plus-forward window, since a range control is not meaningful for a short forward projection. *(This shared-vs-per-chart choice is a deliberate recommendation; flag if per-chart selectors are preferred instead.)*

Cross-filtering between charts (click one chart → others filter) is **out of scope for v1**. If added later it requires its own spec.

---

## 7. Explicitly out of scope for v1

Listed so they are not silently re-added:
- **Composite Stress Probability chart** — removed for now. May return later if the simpler dashboard proves insufficient; would need its own spec (definition of "composite", data source, framing).
- **Softmax probability bands (4-class stack)** — not shown live.
- **Predicted-vs-Actual regime panel** — model-evaluation content; belongs in METHODOLOGY.md, not the dashboard.
- **ARIMA forecast charts** — diagnostic/validation artefacts, not viewer-facing.
- **Signal strength meter** — removed (see §4).
- **Cross-filtering / drill-through** — see §6.

---

## 8. Decisions locked (previously open)

Every item that was open in the draft is now resolved:
- **Charting library:** Apache ECharts (§6).
- **Regime / panel hex values:** designer's choice at build, verified on screen (§3).
- **Historical window:** selectable range — 1M / 6M / 1Y / 5Y / All since 2020, shared across the three historical charts (§6).
- **Mobile responsive:** required and first-class, with tap-to-show tooltips on touch (§2).

No open items remain. The spec is build-ready.
