# visualization/dashboards.py
# Step 14 -- Visualization Layer
# NIFTY 50 Equity Risk Diagnostics System
#
# Produces four publication-grade outputs:
#   14a. regime_probability_ts.png     -- softmax probability bands
#   14b. stress_signal_dashboard.png   -- composite stress signal
#   14c. monte_carlo_fanchart.png      -- regime-conditional scenario fan
#   14d. full_system_dashboard.png     -- combined 4-panel dashboard
#
# Design principles (Bogle):
#   - Uncertainty widens with horizon
#   - Probabilistic outputs, not point forecasts
#   - Current state always visible at terminal data point
#   - CURRENT_STATE is read dynamically from pkl files -- never hardcoded

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker
from matplotlib.gridspec import GridSpec

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REGIME_LABELS,
    REGIME_COLORS,
    PROB_COLORS,
    STRESS_EPISODES,
    REGIME_PROBS_PKL,
    EWS_PKL,
    FEATURES_PKL,
    MONTE_CARLO_PKL,
    OUTPUTS_DIR,
)

os.makedirs(OUTPUTS_DIR, exist_ok=True)

OUT_14A = os.path.join(OUTPUTS_DIR, 'regime_probability_ts.png')
OUT_14B = os.path.join(OUTPUTS_DIR, 'stress_signal_dashboard.png')
OUT_14C = os.path.join(OUTPUTS_DIR, 'monte_carlo_fanchart.png')
OUT_14D = os.path.join(OUTPUTS_DIR, 'full_system_dashboard.png')

# Theme constants -- visualization only, not business logic
DARK_BG  = '#0d0d1a'
PANEL_BG = '#11112a'
GRID_CLR = '#222244'
TEXT_CLR = '#dde0f0'
ACCENT   = '#7b89d4'


# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════

def load_all():
    print("\n[LOAD] Loading visualization inputs...")

    with open(REGIME_PROBS_PKL, 'rb') as f:
        rp = pickle.load(f)
    with open(EWS_PKL, 'rb') as f:
        ews = pickle.load(f)
    with open(FEATURES_PKL, 'rb') as f:
        feat = pickle.load(f)

    rp.index   = pd.to_datetime(rp.index)
    ews.index  = pd.to_datetime(ews.index)
    feat.index = pd.to_datetime(feat.index)

    mc = None
    if os.path.exists(MONTE_CARLO_PKL):
        with open(MONTE_CARLO_PKL, 'rb') as f:
            mc = pickle.load(f)
        print(f"  monte_carlo_output.pkl loaded")
    else:
        print(f"  monte_carlo_output.pkl not found -- 14c will generate paths")

    print(f"  regime_probs : {rp.shape}")
    print(f"  ews          : {ews.shape}")
    print(f"  features     : {feat.shape}")

    return rp, ews, feat, mc


def build_current_state(ews: pd.DataFrame, feat: pd.DataFrame) -> dict:
    """
    Build current state dict dynamically from latest pipeline outputs.
    Never hardcoded -- reads from early_warning_signals.pkl and features.pkl.
    """
    latest_ews  = ews.iloc[-1]
    latest_date = ews.index[-1]

    # Get Close and drawdown from features, aligned to latest EWS date
    feat_latest = feat.reindex([latest_date], method='ffill').iloc[0]

    predicted_regime = int(latest_ews.get('predicted_regime', -1))

    return {
        'date':           str(latest_date.date()),
        'close':          float(feat_latest.get('Close', 0)),
        'regime':         predicted_regime,
        'regime_label':   REGIME_LABELS.get(predicted_regime, 'Unknown'),
        'drawdown':       float(feat_latest.get('drawdown', 0)),
        'garch_vol':      float(feat_latest.get('GARCH_Vol', 0)) * 100,
        'p_calm':         float(latest_ews.get('P_regime_0', 0)),
        'p_pullback':     float(latest_ews.get('P_regime_1', 0)),
        'p_stress':       float(latest_ews.get('P_regime_2', 0)),
        'p_crisis':       float(latest_ews.get('P_regime_3', 0)),
        'stress_combined': float(latest_ews.get('stress_combined_prob', 0)),
        'active_signals': [
            sig for sig, col in [
                ('STRESS_SIGNAL',     'stress_signal'),
                ('CRISIS_ALERT',      'crisis_alert'),
                ('ESCALATION_SIGNAL', 'escalation_signal'),
            ] if col in latest_ews and latest_ews[col]
        ]
    }


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _style_ax(ax, title=None, xlabel=None, ylabel=None, fontsize=9):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.6)
    if title:
        ax.set_title(title, color=TEXT_CLR, fontsize=fontsize, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=fontsize - 1)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=fontsize - 1)


def _mark_current(ax, current_date, color='white', linestyle='--', lw=1.2):
    ax.axvline(pd.Timestamp(current_date), color=color,
               linestyle=linestyle, linewidth=lw, zorder=10,
               label=f'Current ({current_date})')


def _format_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')


# ════════════════════════════════════════════════════════════════
# 14a -- REGIME PROBABILITY TIME SERIES
# ════════════════════════════════════════════════════════════════

def plot_regime_probability_ts(rp, current_state):
    print("\n[14a] Regime probability time series...")

    prob_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3']
    missing = [c for c in prob_cols if c not in rp.columns]
    if missing:
        print(f"  [WARNING] Missing columns: {missing}. Skipping 14a.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(16, 9),
                             gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.35})
    fig.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    _style_ax(ax,
              title='GRU Regime Classifier -- Softmax Probability Bands (Test Period: 2024-present)',
              ylabel='P(Regime)')

    dates = rp.index
    ax.stackplot(dates,
                 rp['P_regime_0'].values, rp['P_regime_1'].values,
                 rp['P_regime_2'].values, rp['P_regime_3'].values,
                 colors=[PROB_COLORS[i] for i in range(4)],
                 alpha=0.82,
                 labels=[REGIME_LABELS[i] for i in range(4)])

    if 'actual_regime' in rp.columns:
        actual_norm = rp['actual_regime'] / 3.0
        ax.plot(dates, actual_norm, color='white', linewidth=0.8,
                alpha=0.5, linestyle=':', label='Actual regime (normalised)')

    _mark_current(ax, current_state['date'])
    ax.set_ylim(0, 1)
    ax.annotate(
        f"P(Stress)={current_state['p_stress']:.3f}\n"
        f"P(Crisis)={current_state['p_crisis']:.3f}\n"
        f"Signal: {', '.join(current_state['active_signals']) or 'NONE'}",
        xy=(pd.Timestamp(current_state['date']), current_state['p_stress']),
        xytext=(-120, 20), textcoords='offset points',
        color='#ff6b6b', fontsize=7.5, fontfamily='monospace',
        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=0.8),
        bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor='#ff6b6b', alpha=0.9)
    )
    ax.legend(loc='upper left', fontsize=8, facecolor=PANEL_BG,
              labelcolor=TEXT_CLR, framealpha=0.9)
    _format_date_axis(ax)

    ax2 = axes[1]
    _style_ax(ax2, title='Predicted vs Actual Regime',
              ylabel='Regime', xlabel='Date')
    if 'predicted_regime' in rp.columns and 'actual_regime' in rp.columns:
        ax2.step(dates, rp['actual_regime'].values,
                 color=ACCENT, linewidth=1.0, where='post', label='Actual', alpha=0.9)
        ax2.step(dates, rp['predicted_regime'].values,
                 color='#ff9f43', linewidth=0.8, where='post',
                 linestyle='--', label='Predicted', alpha=0.9)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels([REGIME_LABELS[i] for i in range(4)],
                        color=TEXT_CLR, fontsize=7.5)
    ax2.set_ylim(-0.5, 3.8)
    _mark_current(ax2, current_state['date'])
    ax2.legend(loc='upper left', fontsize=8, facecolor=PANEL_BG,
               labelcolor=TEXT_CLR, framealpha=0.9)
    _format_date_axis(ax2)

    fig.suptitle('NIFTY 50 -- Regime Probability Time Series',
                 color=TEXT_CLR, fontsize=13, fontweight='bold', y=0.99)
    plt.savefig(OUT_14A, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] {OUT_14A}")


# ════════════════════════════════════════════════════════════════
# 14b -- STRESS SIGNAL DASHBOARD
# ════════════════════════════════════════════════════════════════

def plot_stress_signal_dashboard(ews, feat, current_state):
    print("\n[14b] Stress signal dashboard...")

    ews_df = ews.copy()
    dates  = ews_df.index

    fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                             gridspec_kw={'height_ratios': [2.5, 1.2, 1], 'hspace': 0.42})
    fig.patch.set_facecolor(DARK_BG)

    combined = ews_df.get(
        'stress_combined_prob',
        ews_df.get('P_regime_2', 0) + ews_df.get('P_regime_3', 0)
    )

    ax1 = axes[0]
    _style_ax(ax1, title='Composite Stress Probability: P(Stress) + P(Crisis)',
              ylabel='Combined Probability')
    ax1.fill_between(dates, combined, where=combined < 0.40,
                     color='#27ae60', alpha=0.4, label='Low (<0.40)')
    ax1.fill_between(dates, combined,
                     where=(combined >= 0.40) & (combined < 0.60),
                     color='#f39c12', alpha=0.5, label='Moderate (0.40-0.60)')
    ax1.fill_between(dates, combined, where=combined >= 0.60,
                     color='#e74c3c', alpha=0.6, label='High (>=0.60)')
    ax1.plot(dates, combined, color='#ecf0f1', linewidth=0.7, alpha=0.8)
    ax1.axhline(0.40, color='#f39c12', linestyle='--', linewidth=1.0,
                alpha=0.8, label='Sustained threshold (0.40)')
    ax1.axhline(0.60, color='#e74c3c', linestyle='--', linewidth=1.0,
                alpha=0.8, label='Override threshold (0.60)')

    for sig_col, (label_char, color, marker, ms) in {
        'stress_signal':     ('S', '#f39c12', '^', 80),
        'crisis_alert':      ('C', '#e74c3c', 'D', 90),
        'escalation_signal': ('E', '#9b59b6', 's', 70),
    }.items():
        if sig_col in ews_df.columns:
            sig_mask = ews_df[sig_col].astype(bool)
            if sig_mask.any():
                ax1.scatter(dates[sig_mask], combined[sig_mask],
                            color=color, marker=marker, s=ms, zorder=8,
                            label=f'{label_char} Signal',
                            edgecolors='white', linewidths=0.4)

    _mark_current(ax1, current_state['date'])
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left', fontsize=7.5, facecolor=PANEL_BG,
               labelcolor=TEXT_CLR, framealpha=0.9, ncol=2)
    _format_date_axis(ax1)

    last_date = dates[-1]
    active_str = ', '.join(current_state['active_signals']) or 'NONE'
    ax1.annotate(
        f"Combined={current_state['stress_combined']:.4f}\n{active_str}",
        xy=(last_date, current_state['stress_combined']),
        xytext=(-150, -35), textcoords='offset points',
        color='#ff6b6b', fontsize=7.5, fontfamily='monospace',
        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=0.8),
        bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor='#ff6b6b', alpha=0.95)
    )

    ax2 = axes[1]
    _style_ax(ax2, title='P(Crisis) -- Crisis Probability', ylabel='P(Crisis)')
    p_crisis = ews_df.get('P_regime_3', pd.Series(0, index=dates))
    ax2.fill_between(dates, p_crisis, color='#922b21', alpha=0.55)
    ax2.plot(dates, p_crisis, color='#ff6b6b', linewidth=0.7)
    ax2.axhline(0.25, color='#e74c3c', linestyle=':', linewidth=1.0,
                alpha=0.7, label='Crisis alert threshold (0.25)')
    ax2.axhline(0.50, color='#c0392b', linestyle='--', linewidth=1.0,
                alpha=0.7, label='Crisis override (0.50)')
    if 'crisis_alert' in ews_df.columns:
        ca_mask = ews_df['crisis_alert'].astype(bool)
        if ca_mask.any():
            ax2.scatter(dates[ca_mask], p_crisis[ca_mask],
                        color='#e74c3c', marker='D', s=60, zorder=8,
                        label='Crisis Alert', edgecolors='white', linewidths=0.4)
    _mark_current(ax2, current_state['date'])
    ax2.legend(loc='upper right', fontsize=7.5, facecolor=PANEL_BG,
               labelcolor=TEXT_CLR, framealpha=0.9)
    _format_date_axis(ax2)

    ax3 = axes[2]
    _style_ax(ax3, title='Signal Strength (0-3 active signals)',
              ylabel='Active Signals', xlabel='Date')
    if 'signal_strength' in ews_df.columns:
        sig_strength = ews_df['signal_strength'].fillna(0)
        color_map = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c', 3: '#922b21'}
        colors_bar = sig_strength.map(color_map).fillna('#27ae60')
        ax3.bar(dates, sig_strength, color=colors_bar, alpha=0.75, width=1.5)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.set_ylim(-0.1, 3.5)
    _mark_current(ax3, current_state['date'])
    _format_date_axis(ax3)

    signal_label = ', '.join(current_state['active_signals']) or 'NONE'
    fig.suptitle(
        f"NIFTY 50 -- Early Warning Signal Dashboard  |  "
        f"{signal_label}  |  {current_state['date']}",
        color='#ff6b6b' if current_state['active_signals'] else TEXT_CLR,
        fontsize=12, fontweight='bold', y=0.99
    )
    plt.savefig(OUT_14B, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] {OUT_14B}")


# ════════════════════════════════════════════════════════════════
# 14c -- MONTE CARLO FAN CHART
# ════════════════════════════════════════════════════════════════

def plot_monte_carlo_fanchart(mc, feat, current_state):
    print("\n[14c] Monte Carlo fan chart...")

    S0          = current_state['close']
    sigma_daily = current_state['garch_vol'] / 100 / np.sqrt(252)
    regime_drift = {
        0: 0.000736,
        1: 0.000366,
        2: -0.000128,
        3: -0.000437
    }
    current_regime_id = current_state.get('predicted_regime', 1)
    mu_daily = regime_drift.get(int(current_regime_id), 0.000366)
    horizon     = 63
    n_sims      = 20000

    paths = None
    if mc is not None and isinstance(mc, dict):
        try:
            stored = mc.get('paths', None)
            if stored is not None and stored.shape[0] >= horizon:
                paths = stored[:horizon, :]
        except Exception:
            paths = None

    if paths is None:
        print(f"  Generating {n_sims} paths (horizon={horizon}d) from current state...")
        np.random.seed(42)
        paths = np.zeros((horizon, n_sims))
        paths[0] = S0
        for t in range(1, horizon):
            shock = np.random.normal(0, 1, n_sims)
            paths[t] = paths[t-1] * np.exp(
                mu_daily - 0.5 * sigma_daily**2 + sigma_daily * shock
            )

    p05 = np.percentile(paths, 5,  axis=1)
    p25 = np.percentile(paths, 25, axis=1)
    p50 = np.percentile(paths, 50, axis=1)
    p75 = np.percentile(paths, 75, axis=1)
    p95 = np.percentile(paths, 95, axis=1)

    feat_c    = feat.copy().sort_index()
    hist_tail = feat_c['Close'].dropna().tail(126)
    last_date = feat_c.index[-1]
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    _style_ax(ax,
              title=f'1-Quarter Monte Carlo Fan Chart\n'
                    f"Regime: {current_state['regime_label']} | "
                    f"Vol={current_state['garch_vol']:.1f}% ann.",
              ylabel='NIFTY Index Level', xlabel='Date')
    ax.plot(hist_tail.index, hist_tail.values,
            color=ACCENT, linewidth=1.5, label='NIFTY (Historical)', zorder=5)
    ax.fill_between(forecast_dates, p05, p95, color='#e74c3c', alpha=0.15,
                    label='90% interval (5th-95th)')
    ax.fill_between(forecast_dates, p25, p75, color='#e74c3c', alpha=0.30,
                    label='50% interval (25th-75th)')
    ax.plot(forecast_dates, p50, color='#ff6b6b', linewidth=1.5,
            linestyle='--', label='Median path', zorder=6)
    for pct, val, clr in [(5, p05[-1], '#aaaacc'),
                           (25, p25[-1], '#ccaaaa'),
                           (50, p50[-1], '#ff6b6b'),
                           (75, p75[-1], '#ccaaaa'),
                           (95, p95[-1], '#aaaacc')]:
        ax.text(forecast_dates[-1] + pd.Timedelta(days=2),
                val, f'{pct}th\n{val:,.0f}',
                color=clr, fontsize=6.5, va='center')
    ax.axvline(last_date, color='white', linestyle=':', linewidth=1.0,
               alpha=0.6, label=f'Forecast start ({current_state["date"]})')
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )
    ax.legend(loc='upper left', fontsize=7.5, facecolor=PANEL_BG,
              labelcolor=TEXT_CLR, framealpha=0.9)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    ax2 = axes[1]
    _style_ax(ax2, title='Terminal Distribution (1-Quarter Horizon)',
              xlabel='NIFTY Level', ylabel='Density')
    end_prices = paths[-1]
    ax2.hist(end_prices, bins=80, density=True, color='#e74c3c', alpha=0.6)
    for pct, val, clr, ls in [
        (5,  p05[-1], '#aaaacc', ':'),
        (50, p50[-1], '#ff6b6b', '--'),
        (95, p95[-1], '#aaaacc', ':'),
    ]:
        ax2.axvline(val, color=clr, linestyle=ls, linewidth=1.2,
                    label=f'{pct}th: {val:,.0f}')
    ax2.axvline(S0, color='white', linestyle='-', linewidth=1.0,
                alpha=0.6, label=f'Current: {S0:,.0f}')
    ax2.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )
    ax2.legend(loc='upper left', fontsize=7.5, facecolor=PANEL_BG,
               labelcolor=TEXT_CLR, framealpha=0.9)

    fig.text(0.5, -0.02,
             "Probabilistic risk assessment, not a price forecast. "
             "Uncertainty widens with horizon. "
             "This chart diagnoses risk states -- it does not predict markets.",
             ha='center', va='top', color='#888899', fontsize=7.5, style='italic')

    fig.suptitle('NIFTY 50 -- Regime-Conditional Monte Carlo Fan Chart',
                 color=TEXT_CLR, fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(OUT_14C, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] {OUT_14C}")


# ════════════════════════════════════════════════════════════════
# 14d -- FULL SYSTEM DASHBOARD
# ════════════════════════════════════════════════════════════════

def plot_full_system_dashboard(rp, ews, feat, current_state):
    print("\n[14d] Full system dashboard...")

    fig = plt.figure(figsize=(20, 11))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32,
                  left=0.06, right=0.97, top=0.91, bottom=0.07)

    feat_c = feat.copy().sort_index()
    rp_c   = rp.copy()
    ews_c  = ews.copy()

    # Panel A -- NIFTY Close + Regime Background
    ax_a = fig.add_subplot(gs[0, 0])
    _style_ax(ax_a, title='A -- NIFTY 50: Price & Drawdown (2020-2026)',
              ylabel='Index Level')
    feat_2020 = feat_c[feat_c.index >= '2020-01-01']
    if 'regime' in feat_2020.columns:
        reg = feat_2020['regime'].fillna(0).astype(int)
        for r_id, r_color in REGIME_COLORS.items():
            mask = reg == r_id
            ax_a.fill_between(feat_2020.index, 0, 1,
                              where=mask, transform=ax_a.get_xaxis_transform(),
                              color=r_color, alpha=0.12, zorder=0)
    ax_a.plot(feat_2020.index, feat_2020['Close'],
              color='#ecf0f1', linewidth=1.2, zorder=5, label='NIFTY Close')
    if 'drawdown' in feat_2020.columns:
        ax_a2 = ax_a.twinx()
        ax_a2.set_facecolor(PANEL_BG)
        ax_a2.fill_between(feat_2020.index, feat_2020['drawdown'],
                           color='#e74c3c', alpha=0.35, label='Drawdown')
        ax_a2.plot(feat_2020.index, feat_2020['drawdown'],
                   color='#e74c3c', linewidth=0.6, alpha=0.6)
        ax_a2.set_ylabel('Drawdown', color='#e74c3c', fontsize=8)
        ax_a2.tick_params(colors='#e74c3c', labelsize=7)
        ax_a2.set_ylim(-0.5, 0.05)
        ax_a2.spines['right'].set_edgecolor('#e74c3c')
        ax_a2.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:.0%}')
        )
    ax_a.axvline(pd.Timestamp(current_state['date']),
                 color='#ff6b6b', linestyle='--', linewidth=1.2, zorder=10)
    ax_a.annotate(
        f"Regime: {current_state['regime_label']}\n"
        f"Close: {current_state['close']:,.0f}\n"
        f"DD: {current_state['drawdown']:.1%}",
        xy=(pd.Timestamp(current_state['date']), current_state['close']),
        xytext=(-80, 15), textcoords='offset points',
        color='#ff6b6b', fontsize=6.5, fontfamily='monospace',
        arrowprops=dict(arrowstyle='->', color='#ff6b6b', lw=0.7),
        bbox=dict(boxstyle='round', facecolor=DARK_BG, edgecolor='#ff6b6b', alpha=0.9)
    )
    ax_a.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
    )
    ax_a.legend(loc='upper left', fontsize=7, facecolor=PANEL_BG,
                labelcolor=TEXT_CLR, framealpha=0.85)
    _format_date_axis(ax_a)

    # Panel B -- Stacked Regime Probabilities
    ax_b = fig.add_subplot(gs[0, 1])
    _style_ax(ax_b, title='B -- GRU Regime Probabilities (Test Period)',
              ylabel='P(Regime)')
    prob_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3']
    if all(c in rp_c.columns for c in prob_cols):
        ax_b.stackplot(rp_c.index,
                       rp_c['P_regime_0'], rp_c['P_regime_1'],
                       rp_c['P_regime_2'], rp_c['P_regime_3'],
                       colors=[PROB_COLORS[i] for i in range(4)],
                       alpha=0.82, labels=[REGIME_LABELS[i] for i in range(4)])
        ax_b.set_ylim(0, 1)
    _mark_current(ax_b, current_state['date'])
    ax_b.legend(loc='lower left', fontsize=7, facecolor=PANEL_BG,
                labelcolor=TEXT_CLR, framealpha=0.85)
    _format_date_axis(ax_b)

    # Panel C -- Composite Stress Signal
    ax_c = fig.add_subplot(gs[1, 0])
    _style_ax(ax_c, title='C -- Composite Stress Probability + Signals',
              ylabel='P(Stress) + P(Crisis)', xlabel='Date')
    combined = ews_c.get(
        'stress_combined_prob',
        ews_c.get('P_regime_2', 0) + ews_c.get('P_regime_3', 0)
    )
    ax_c.fill_between(ews_c.index, combined, where=combined < 0.40,
                      color='#27ae60', alpha=0.4)
    ax_c.fill_between(ews_c.index, combined,
                      where=(combined >= 0.40) & (combined < 0.60),
                      color='#f39c12', alpha=0.5)
    ax_c.fill_between(ews_c.index, combined, where=combined >= 0.60,
                      color='#e74c3c', alpha=0.6)
    ax_c.plot(ews_c.index, combined, color='#ecf0f1', linewidth=0.6, alpha=0.7)
    ax_c.axhline(0.40, color='#f39c12', linestyle='--', linewidth=0.9, alpha=0.8)
    ax_c.axhline(0.60, color='#e74c3c', linestyle='--', linewidth=0.9, alpha=0.8)
    ax_c.set_ylim(0, 1.05)
    for sig_col, color, marker in [
        ('stress_signal',    '#f39c12', '^'),
        ('crisis_alert',     '#e74c3c', 'D'),
        ('escalation_signal','#9b59b6', 's'),
    ]:
        if sig_col in ews_c.columns:
            m = ews_c[sig_col].astype(bool)
            if m.any():
                ax_c.scatter(ews_c.index[m], combined[m],
                             color=color, marker=marker, s=50, zorder=8,
                             edgecolors='white', linewidths=0.4)
    _mark_current(ax_c, current_state['date'])
    _format_date_axis(ax_c)

    # Panel D -- Current State Summary Card (dynamic)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_facecolor('#0a0a1a')
    ax_d.axis('off')

    active_str = ', '.join(current_state['active_signals']) or 'NONE'
    signal_color = '#ff6b6b' if current_state['active_signals'] else '#2ecc71'

    card_lines = [
        ('NIFTY 50 -- Current Risk State',          TEXT_CLR,    11,  'bold',   'normal'),
        ('',                                         TEXT_CLR,     8,  'normal', 'normal'),
        (f"Date         : {current_state['date']}",  TEXT_CLR,   8.5, 'normal', 'normal'),
        (f"Close        : {current_state['close']:,.0f}", TEXT_CLR, 8.5, 'normal', 'normal'),
        (f"Regime       : {current_state['regime']} -- {current_state['regime_label']}",
         '#e74c3c', 9, 'bold', 'normal'),
        (f"Drawdown     : {current_state['drawdown']:.1%}", '#e74c3c', 8.5, 'normal', 'normal'),
        (f"GARCH Vol    : {current_state['garch_vol']:.1f}% (annualised)", TEXT_CLR, 8.5, 'normal', 'normal'),
        ('',                                         TEXT_CLR,     8, 'normal', 'normal'),
        ('GRU Probabilities:',                       TEXT_CLR,   8.5, 'bold',   'normal'),
        (f"  P(Calm)    : {current_state['p_calm']:.4f}",     '#2ecc71', 8.5, 'normal', 'normal'),
        (f"  P(Pullback): {current_state['p_pullback']:.4f}", '#f39c12', 8.5, 'normal', 'normal'),
        (f"  P(Stress)  : {current_state['p_stress']:.4f}",  '#e74c3c', 9,   'bold',   'normal'),
        (f"  P(Crisis)  : {current_state['p_crisis']:.4f}",  '#c0392b', 8.5, 'normal', 'normal'),
        ('',                                         TEXT_CLR,     8, 'normal', 'normal'),
        (f"Combined     : {current_state['stress_combined']:.4f}", '#e74c3c', 9, 'bold', 'normal'),
        (f"Active Signal: {active_str}",             signal_color, 9, 'bold',   'normal'),
        ('',                                         TEXT_CLR,     8, 'normal', 'normal'),
        ('-' * 38,                                   '#444466',  7.5, 'normal', 'normal'),
        ('Model: GRU Regime Classifier (frozen)',    '#888899',  7.5, 'normal', 'italic'),
        ('Evaluation: Out-of-sample (2024-present)', '#888899',  7.5, 'normal', 'italic'),
        ('Bodie / Karasan / Bogle framework',        '#888899',  7.5, 'normal', 'italic'),
    ]

    y_pos = 0.97
    for text, color, fsize, weight, style in card_lines:
        ax_d.text(0.05, y_pos, text, transform=ax_d.transAxes,
                  color=color, fontsize=fsize, fontweight=weight,
                  fontstyle=style, fontfamily='monospace', va='top')
        y_pos -= 0.048 if fsize >= 9 else 0.042

    for spine in ax_d.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#e74c3c' if current_state['active_signals'] else '#2ecc71')
        spine.set_linewidth(1.5)

    regime_patches = [
        mpatches.Patch(color=REGIME_COLORS[i], label=REGIME_LABELS[i], alpha=0.8)
        for i in range(4)
    ]
    fig.legend(handles=regime_patches, loc='lower center', ncol=4, fontsize=8,
               facecolor=PANEL_BG, labelcolor=TEXT_CLR,
               framealpha=0.85, bbox_to_anchor=(0.5, 0.01))

    title_color = '#ff6b6b' if current_state['active_signals'] else TEXT_CLR
    fig.suptitle(
        f"NIFTY 50 -- Equity Risk Diagnostics System  |  "
        f"{active_str}  |  {current_state['date']}",
        color=title_color, fontsize=14, fontweight='bold', y=0.96
    )

    plt.savefig(OUT_14D, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] {OUT_14D}")


# ════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run_dashboards():
    print("=" * 65)
    print("STEP 14 -- VISUALIZATION LAYER")
    print("NIFTY 50 Equity Risk Diagnostics System")
    print("=" * 65)

    rp, ews, feat, mc = load_all()

    # Build current state dynamically -- never hardcoded
    current_state = build_current_state(ews, feat)
    print(f"\n[STATE] Date={current_state['date']} | "
          f"Regime={current_state['regime_label']} | "
          f"Combined={current_state['stress_combined']:.4f} | "
          f"Signals={current_state['active_signals']}")

    plot_regime_probability_ts(rp, current_state)
    plot_stress_signal_dashboard(ews, feat, current_state)
    plot_monte_carlo_fanchart(mc, feat, current_state)
    plot_full_system_dashboard(rp, ews, feat, current_state)

    print("\n" + "=" * 65)
    print("STEP 14 COMPLETE")
    print(f"  regime_probability_ts.png    -> {OUT_14A}")
    print(f"  stress_signal_dashboard.png  -> {OUT_14B}")
    print(f"  monte_carlo_fanchart.png     -> {OUT_14C}")
    print(f"  full_system_dashboard.png    -> {OUT_14D}")
    print("=" * 65)


if __name__ == '__main__':
    run_dashboards()
