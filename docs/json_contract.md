# JSON Data Contract — NIFTY 50 Risk Diagnostics Dashboard

> **Status:** v1 — locked.
> **Purpose:** The single agreement between the Python pipeline (which *writes* these files) and the React/ECharts frontend (which *reads* them). Both sides develop independently against this contract. If the pipeline and frontend ever disagree about a field name, type, or shape, this document is the arbiter — change it here first, then change both sides to match.
> **Companion docs:** `REQUIREMENTS.md` (what the dashboard displays), `docs/json_schema_notes.md` (what the old dashboard produced, historical reference).

---

## Design principles this contract follows

These are *why* the files are shaped as they are. Keep them in mind when extending the contract later.

1. **Separation of concerns.** The JSON carries *data and decisions*, never *presentation*. The pipeline decides what is true (regime, signals, percentiles); the frontend decides how it looks (colours, layout). No hex colours, no styling, in the data.
2. **Single source of truth for logic.** All business logic — signal computation, regime assignment, percentile calculation — happens once, in the pipeline. The frontend never re-derives a decision from raw inputs. It reads answers.
3. **Consumer-shaped, not producer-shaped.** These files are shaped around what the four charts and the signal bar need to read — not around the pipeline's internal `.pkl` structures.
4. **Cohesion / low coupling.** Data read together lives together (charts 1–3 share `timeseries.json`); data with a different shape or lifecycle lives apart (`snapshot`, `montecarlo`).

---

## Global conventions (apply to all three files)

- **Dates:** always ISO 8601 strings, `"YYYY-MM-DD"` (e.g. `"2026-08-15"`). No other date format anywhere. ISO dates sort lexicographically and parse natively in ECharts.
- **Field naming:** `snake_case` throughout, matching the Python pipeline. Never mix in camelCase.
- **Numbers:** plain JSON numbers. Volatility and drawdown are expressed as **decimals, not percentages** — e.g. `0.123` means 12.3%. The frontend formats the `%` for display. (One rule, applied everywhere: the data is unformatted; the frontend formats.)
- **Regime labels:** the canonical string set is `"Calm"`, `"Pullback"`, `"Stress"`, `"Crisis"`. Exactly these spellings, everywhere a regime appears. The frontend maps label → colour via the REQUIREMENTS.md §3 palette.
- **Missing / not-yet-available values:** `null`, never `0`, never an empty string. (A missing GARCH value on day 1 is `null`, not `0` — `0` would be a lie the charts would plot.)

---

## File 1 — `snapshot.json`

**Job:** today's state. Paints the top of the dashboard (signal status bar + regime headline) instantly, before the larger history file loads. Tiny, changes daily.

**Shape:** a single object (not an array).

```json
{
  "date": "2026-08-15",
  "current_regime": "Pullback",
  "days_in_regime": 12,
  "signals": {
    "stress": true,
    "crisis": false,
    "escalation": false
  }
}
```

**Fields:**
| Field | Type | Meaning |
|-------|------|---------|
| `date` | string (ISO date) | The date this snapshot describes (latest pipeline run). |
| `current_regime` | string | One of the four canonical regime labels. Today's regime. |
| `days_in_regime` | integer | Number of consecutive days the market has been in `current_regime`. |
| `signals.stress` | boolean | Whether the Stress signal is currently active. **Computed by the pipeline** (`early_warning/signals.py`), not derived on the frontend. |
| `signals.crisis` | boolean | Whether the Crisis alert is currently active. Pipeline-computed. |
| `signals.escalation` | boolean | Whether the Escalation signal is currently active. Pipeline-computed. |

**Notes:**
- The three signal booleans map directly to the three chips in REQUIREMENTS.md §4 (Stress=yellow, Crisis=orange, Escalation=red). The frontend lights a chip when its boolean is `true`. No threshold logic on the frontend.
- `signal_strength` is intentionally excluded (REQUIREMENTS.md §4 — active/inactive only).
- Room to grow: current index level, current vols, etc. can be added here later if the top strip needs them. Additive changes (new fields) don't break the frontend; renames/removals do — so add, don't rename.

---

## File 2 — `timeseries.json`

**Job:** the full daily history. Feeds Charts 1 (price + regime line), 2 (realized vs GARCH vol), and 3 (drawdown) — all three read from this one file. Carries the **entire history** (from `START_DATE` 2007-09-17 to latest), deliberately, so the dashboard can show how current regime rules would have classified the full past.

**Shape:** an array of daily-row objects, ordered oldest → newest.

```json
[
  {
    "date": "2007-09-17",
    "close": 4494.65,
    "regime": "Calm",
    "drawdown": 0.0,
    "realized_vol": null,
    "garch_vol": null
  },
  {
    "date": "2026-08-15",
    "close": 24310.15,
    "regime": "Pullback",
    "drawdown": -0.038,
    "realized_vol": 0.121,
    "garch_vol": 0.134
  }
]
```

**Fields (per row):**
| Field | Type | Meaning |
|-------|------|---------|
| `date` | string (ISO date) | Trading day. |
| `close` | number | NIFTY 50 closing index level. |
| `regime` | string | Rule-based regime label for that day (one of the four canonical labels). Drives the regime-coloured line and its tooltip in Chart 1. |
| `drawdown` | number | Drawdown from running peak, as a decimal (e.g. `-0.038` = −3.8%). Chart 3. |
| `realized_vol` | number \| null | Rolling realized volatility, annualised, decimal. `null` for early rows before the rolling window fills. Chart 2. |
| `garch_vol` | number \| null | GJR-GARCH conditional volatility, annualised, decimal. `null` where unavailable. Chart 2. |

**Notes:**
- **No `regime_color` field.** Colour is presentation — the frontend maps `regime` label → hex via the palette. Changing the palette must never require re-running the pipeline.
- The range selector (1M/6M/1Y/5Y/All) filters this array client-side by `date`. No separate files per range — the frontend slices the one array. This is why full history lives in one file.
- Early rows will have `null` vols (rolling windows not yet filled). The frontend must handle `null` gracefully (gap in the line, not a plotted zero).
- Ordering is guaranteed oldest→newest so the frontend can trust array order without re-sorting.

---

## File 3 — `montecarlo.json`

**Job:** the 1-month forward fan chart (Chart 4). Structurally different from the daily series — it has a *historical context leg* (recent actual prices) transitioning into a *forward projection* (percentile bands). Hence its own file.

**Shape:** an object with metadata + two arrays.

```json
{
  "generated_on": "2026-08-15",
  "current_vol": 0.123,
  "current_regime": "Pullback",
  "horizon_days": 21,
  "historical": [
    { "date": "2026-07-15", "close": 24010.30 },
    { "date": "2026-08-15", "close": 24310.15 }
  ],
  "forecast": [
    {
      "date": "2026-08-18",
      "ptile_5": 24120.0,
      "ptile_25": 24230.0,
      "ptile_50": 24350.0,
      "ptile_75": 24470.0,
      "ptile_95": 24590.0
    }
  ]
}
```

**Top-level fields:**
| Field | Type | Meaning |
|-------|------|---------|
| `generated_on` | string (ISO date) | The "today" point where history meets forecast. |
| `current_vol` | number | Current volatility (decimal), for the chart title. |
| `current_regime` | string | Current regime label, for the chart title. |
| `horizon_days` | integer | Forward horizon (≈21 trading days = 1 month). |
| `historical` | array | Recent actual prices leading up to `generated_on` (context leg of the fan). |
| `forecast` | array | Forward projection, one row per horizon day, with percentile bands. |

**`historical[]` row:** `{ date, close }` — recent actuals only.

**`forecast[]` row:**
| Field | Type | Meaning |
|-------|------|---------|
| `date` | string (ISO date) | Forward trading day. |
| `ptile_5` | number | 5th percentile index level. |
| `ptile_25` | number | 25th percentile. |
| `ptile_50` | number | 50th percentile (median). |
| `ptile_75` | number | 75th percentile. |
| `ptile_95` | number | 95th percentile. |

**Notes:**
- `historical` rows have `close` and no percentiles; `forecast` rows have percentiles and no `close`. Clean separation — the frontend draws a solid line for `historical`, then the fan (90% band = p5–p95, 50% band = p25–p75, median = p50) for `forecast`.
- The 90%/50% band framing must be labelled in the UI as a *risk-conditioned probability band, not a forecast* (REQUIREMENTS.md Chart 4). The contract carries the numbers; the framing lives in the frontend copy.
- How much `historical` context to include (e.g. last ~1 month) is a frontend-display choice; the pipeline can ship a reasonable window (say 21 trading days back) and the frontend uses what it needs.

---

## Locked decisions

- **File location:** the three files are written to `data/dashboard/` (`data/dashboard/snapshot.json`, `data/dashboard/timeseries.json`, `data/dashboard/montecarlo.json`). Kept separate from the pipeline's internal `.pkl` outputs.
- **Snapshot contents:** `date`, `current_regime`, `days_in_regime`, `signals{stress,crisis,escalation}`. Enrichment (index level, vols) deferred — additive only if the top strip needs it.
- **Monte Carlo historical window:** ~21 trading days of actual price context before the fan begins.
- **Units:** volatility and drawdown ship as **decimals**; the frontend formats to `%` for display (formatting only — no computation on the frontend).

## Critical first step of the pipeline refactor (step 3): verify current pkl units

The current pipeline's `.pkl` outputs are binary and cannot be eyeballed. Before writing any JSON-export code, the refactor must **load the existing outputs and print sample values + ranges** (a throwaway inspection script) to determine whether each quantity is currently stored as a decimal or a percentage. Rationale: the original notebook scaled GARCH input by ×100 (percentage) while realized vol was computed as a decimal, so the pipeline may hold a **mix**. Plotting a percentage series against a decimal series would put one line ~100× the other.

The JSON-export layer must convert every shipped value to the **decimal** convention above, regardless of how it is stored internally. Verify first, then convert — do not assume the internal units.
