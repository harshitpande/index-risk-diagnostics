# JSON schema notes — migration reference

Captured from `dashboard_app.py` and `visualization/dashboards.py` before their
deletion (old Streamlit/matplotlib dashboard architecture), as reference material
for designing the new pipeline's JSON output contract
(GitHub Actions → JSON → React/ECharts). Not itself a schema definition — just the
findings to design one from.

## `get_current_state()` snapshot schema

`dashboard_app.py`'s `get_current_state(ews, feat) -> dict` was the most complete
version of the "today's snapshot" object:

```python
{
  'date': str,               'close': float,
  'regime': int,              'regime_label': str,
  'actual_regime': int,       'drawdown': float,
  'garch_vol': float,         'realized_vol': float,
  'p_calm': float, 'p_pullback': float, 'p_stress': float, 'p_crisis': float,
  'stress_combined': float,
  'active_signals': list[str],   # e.g. ['STRESS_SIGNAL', 'CRISIS_ALERT']
  'signal_strength': int,        # 0-3
}
```

## Consolidate the two state-builders

Both `dashboard_app.py`'s `get_current_state()` and `visualization/dashboards.py`'s
`build_current_state()` independently reconstructed the same snapshot from `ews`/
`feat`, with `build_current_state()` missing `actual_regime`, `realized_vol`, and
`signal_strength` relative to `get_current_state()`. The new pipeline should build
this snapshot in exactly one place (one function, one JSON file) rather than
re-deriving it in multiple consumers.

## `signal_strength` is excluded from the new dashboard

`signal_strength` appeared in `get_current_state()`'s output, but REQUIREMENTS.md §4
explicitly puts it out of scope for the new dashboard ("Active/inactive only" for
signals) — it likely does not need to be included in the new JSON export at all.

## `active_signals` naming convention (stable — carry forward)

Both old files independently used the same naming for active signal entries:
`STRESS_SIGNAL`, `CRISIS_ALERT`, `ESCALATION_SIGNAL` (upper-cased signal-column
names). Since this convention was consistent across two independently-written
consumers, it's a safe naming convention to reuse in the new JSON contract.

## Per-chart data needs (beyond the snapshot)

Chart-level data lived inline inside the old `render_*`/`plot_*` functions, not in
any loader — the new JSON export needs to cover these explicitly:

- **Monte Carlo fan chart**: `p05`, `p25`, `p50`, `p75`, `p95` per forecast day
  (percentiles computed from `mc['paths']`, a 21-day × 10,000-sim path matrix) —
  the new JSON only needs the per-day percentile series, not the raw path matrix.
- **Early warning / stress chart**: the full `stress_combined_prob` time series,
  plus the signal columns (`stress_signal`, `crisis_alert`, `escalation_signal`)
  needed to mark active-signal points on the series.
- **Regime overlay** (price chart background): the `regime` column aligned to
  price/date, used to shade Calm/Pullback/Stress/Crisis periods.
- **Model evaluation** (if surfaced): `ev['tier1']['per_class']` (per-regime
  precision/recall/F1) and `ev['tier2']['episode_detail']` (historical stress
  episode capture table) from the already-JSON `evaluation_results.json`.
