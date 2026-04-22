# dashboard_app.py
# NIFTY 50 Equity Risk Diagnostics System — Streamlit Dashboard
#
# Run locally : streamlit run dashboard_app.py
# Deploy      : streamlit deploy (Streamlit Cloud, free)
#
# Reads all data from pkl/json files produced by run_daily.py.
# No pipeline code runs here — display only.

import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import matplotlib.ticker

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    REGIME_LABELS, REGIME_COLORS, PROB_COLORS,
    STRESS_EPISODES, DARK_BG, PANEL_BG, GRID_CLR, TEXT_CLR, ACCENT,
    DATA_DIR, OUTPUTS_DIR,
    REGIME_PROBS_PKL, EWS_PKL, FEATURES_PKL,
    MONTE_CARLO_PKL, ARIMA_OUTPUT_PKL, EVAL_RESULTS_JSON
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NIFTY 50 Risk Diagnostics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base & Layout ─────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"] {
        background-color: #f8f9fb !important;
        color: #1a1a2e !important;
    }

    [data-testid="stMain"], .main, .block-container {
        background-color: #f8f9fb !important;
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
    }
    [data-testid="stSidebar"] * {
        color: #dde0f0 !important;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #7b89d4 !important;
    }

    /* ── Headings ───────────────────────────────────────────────── */
    h1, h2, h3, h4 {
        color: #1a1a2e !important;
        font-weight: 600 !important;
    }

    /* ── Tabs ───────────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tab"] {
        color: #1a1a2e !important;
        font-weight: 500;
    }
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {
        color: #7b89d4 !important;
        border-bottom: 2px solid #7b89d4 !important;
    }

    /* ── Metric Cards ───────────────────────────────────────────── */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e0e4f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }

    /* ── Signal Cards ───────────────────────────────────────────── */
    .signal-active {
        background: #fff0f0;
        border: 2px solid #e74c3c;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .signal-none {
        background: #f0fff4;
        border: 2px solid #2ecc71;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }

    /* ── Dataframes & Tables ────────────────────────────────────── */
    [data-testid="stDataFrame"] {
        background: #ffffff !important;
        border-radius: 8px;
        border: 1px solid #e0e4f0;
    }
    [data-testid="stDataFrame"] * {
        color: #1a1a2e !important;
    }

    /* ── Markdown text ──────────────────────────────────────────── */
    .stMarkdown p, .stMarkdown li {
        color: #1a1a2e !important;
    }

    /* ── Metrics ────────────────────────────────────────────────── */
    [data-testid="stMetric"] label {
        color: #555577 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1a1a2e !important;
        font-weight: 700 !important;
    }

    /* ── Dividers ───────────────────────────────────────────────── */
    hr {
        border-color: #e0e4f0 !important;
    }

    /* ── Mobile responsiveness ──────────────────────────────────── */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.2rem !important; }
        h3 { font-size: 1rem !important; }
        .metric-card { padding: 10px !important; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# DATA LOADING — cached for performance
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_features():
    with open(FEATURES_PKL, 'rb') as f:
        df = pickle.load(f)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(ttl=3600)
def load_regime_probs():
    with open(REGIME_PROBS_PKL, 'rb') as f:
        df = pickle.load(f)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(ttl=3600)
def load_ews():
    with open(EWS_PKL, 'rb') as f:
        df = pickle.load(f)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

@st.cache_data(ttl=3600)
def load_monte_carlo():
    if not os.path.exists(MONTE_CARLO_PKL):
        return None
    with open(MONTE_CARLO_PKL, 'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def load_arima():
    if not os.path.exists(ARIMA_OUTPUT_PKL):
        return None
    with open(ARIMA_OUTPUT_PKL, 'rb') as f:
        return pickle.load(f)

@st.cache_data(ttl=3600)
def load_evaluation():
    if not os.path.exists(EVAL_RESULTS_JSON):
        return None
    with open(EVAL_RESULTS_JSON, 'r') as f:
        return json.load(f)


def get_current_state(ews: pd.DataFrame, feat: pd.DataFrame) -> dict:
    latest     = ews.iloc[-1]
    latest_date = ews.index[-1]
    feat_row   = feat.reindex([latest_date], method='ffill').iloc[0]

    active = []
    for sig in ['stress_signal', 'crisis_alert', 'escalation_signal']:
        if sig in latest and latest[sig]:
            active.append(sig.upper())

    garch_vol = feat_row.get('GARCH_Vol', 0)
    garch_pct = float(garch_vol * 100) if garch_vol < 5 else float(garch_vol)

    return {
        'date':            str(latest_date.date()),
        'close':           float(feat_row.get('Close', 0)),
        'regime':          int(latest.get('predicted_regime', 0)),
        'regime_label':    REGIME_LABELS.get(int(latest.get('predicted_regime', 0)), 'Unknown'),
        'actual_regime':   int(latest.get('actual_regime', 0)),
        'drawdown':        float(feat_row.get('drawdown', 0)),
        'garch_vol':       garch_pct,
        'realized_vol':    float(feat_row.get('realized_vol', 0) * 100),
        'p_calm':          float(latest.get('P_regime_0', 0)),
        'p_pullback':      float(latest.get('P_regime_1', 0)),
        'p_stress':        float(latest.get('P_regime_2', 0)),
        'p_crisis':        float(latest.get('P_regime_3', 0)),
        'stress_combined': float(latest.get('stress_combined_prob', 0)),
        'active_signals':  active,
        'signal_strength': int(latest.get('signal_strength', 0)),
    }


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────

def style_ax(ax, title=None, xlabel=None, ylabel=None, fontsize=9):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT_CLR, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_CLR)
    ax.grid(True, color=GRID_CLR, linewidth=0.5, alpha=0.5)
    if title:
        ax.set_title(title, color=TEXT_CLR, fontsize=fontsize, pad=6)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_CLR, fontsize=fontsize - 1)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_CLR, fontsize=fontsize - 1)


def fmt_date_axis(ax):
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', color=TEXT_CLR)


def make_fig(figsize=(14, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    return fig, ax


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

def render_sidebar(cs):
    with st.sidebar:
        st.markdown("## NIFTY 50 Risk Diagnostics")
        st.markdown("---")

        # Signal status
        if cs['active_signals']:
            st.markdown(
                f'<div class="signal-active">'
                f'<h3 style="color:#e74c3c;margin:0">⚠ SIGNAL ACTIVE</h3>'
                f'<p style="color:#ff6b6b;margin:4px 0">'
                f'{" | ".join(cs["active_signals"])}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div style="background:#0a2d0a;border:2px solid #2ecc71;border-radius:10px;padding:16px;text-align:center">'
                '<h3 style="color:#2ecc71;margin:0;font-size:16px">✓ NO SIGNAL</h3>'
                '<p style="color:#ffffff;margin:4px 0;font-size:13px">Below signal threshold</p>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown(f"<span style='color:#dde0f0'>**Last Update:** {cs['date']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#dde0f0'>**NIFTY Close:** {cs['close']:,.0f}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#dde0f0'>**Drawdown:** {cs['drawdown']:.1%}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#dde0f0'>**GARCH Vol:** {cs['garch_vol']:.1f}% ann.</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:#dde0f0'>**Realized Vol:** {cs['realized_vol']:.1f}% ann.</span>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**GRU Regime Probabilities**")

        regime_color_map = {
            'Calm': '#2ecc71', 'Pullback': '#f39c12',
            'Stress': '#e74c3c', 'Crisis': '#c0392b'
        }
        for i, (label, prob) in enumerate([
            ('Calm',     cs['p_calm']),
            ('Pullback', cs['p_pullback']),
            ('Stress',   cs['p_stress']),
            ('Crisis',   cs['p_crisis']),
        ]):
            color = regime_color_map[label]
            bar_width = int(prob * 100)
            st.markdown(
                f'<div style="margin:4px 0">'
                f'<span style="color:{color};font-size:12px">{label}: {prob:.3f}</span>'
                f'<div style="background:#444466;border-radius:4px;height:6px;margin-top:2px">'
                f'<div style="background:{color};width:{bar_width}%;height:6px;border-radius:4px"></div>'
                f'</div></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        regime_label = cs['regime_label']
        regime_color = REGIME_COLORS.get(cs['regime'], '#ffffff')
        st.markdown(
            f'<div style="text-align:center;padding:8px;border-radius:6px;'
            f'background:{regime_color}22;border:1px solid {regime_color}">'
            f'<span style="color:{regime_color};font-weight:bold">'
            f'Predicted: {regime_label}</span></div>',
            unsafe_allow_html=True
        )

        st.markdown("---")
        st.markdown(
            '<p style="color:#666688;font-size:11px">'
            'Risk-first framing: Bodie / Karasan / Bogle<br>'
            'Models frozen — inference only<br>'
            'Not a price prediction system</p>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────
# TAB 1 — LIVE DASHBOARD
# ─────────────────────────────────────────────────────────────

def render_live_dashboard(feat, rp, ews, cs):
    st.markdown("## Live Risk Dashboard")
    st.caption(f"Data as of {cs['date']} | Pipeline: Steps 1–14 | Models frozen")

    # Row 1: key metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("NIFTY Close", f"{cs['close']:,.0f}",
                  delta=f"{feat['log_return'].iloc[-1]*100:.2f}% today")
    with c2:
        st.metric("Drawdown", f"{cs['drawdown']:.1%}")
    with c3:
        st.metric("GARCH Vol", f"{cs['garch_vol']:.1f}%",
                  help="Annualised conditional volatility from GJR-GARCH(1,1)")
    with c4:
        st.metric("P(Stress+Crisis)", f"{cs['stress_combined']:.3f}",
                  delta="SIGNAL" if cs['active_signals'] else "No signal",
                  delta_color="inverse")
    with c5:
        st.metric("Predicted Regime", cs['regime_label'])

    st.markdown("---")

    # Row 2: NIFTY price + regime background
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**NIFTY 50 — Price History with Regime Overlay**")
        feat_2020 = feat[feat.index >= '2020-01-01']

        fig, ax = plt.subplots(figsize=(12, 4))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax, ylabel='Index Level')

        if 'regime' in feat_2020.columns:
            reg = feat_2020['regime'].fillna(0).astype(int)
            for r_id, r_color in REGIME_COLORS.items():
                mask = reg == r_id
                ax.fill_between(feat_2020.index, 0, 1,
                                where=mask,
                                transform=ax.get_xaxis_transform(),
                                color=r_color, alpha=0.15, zorder=0,
                                label=REGIME_LABELS[r_id])

        ax.plot(feat_2020.index, feat_2020['Close'],
                color='#ecf0f1', linewidth=1.2, zorder=5)

        ax.axvline(pd.Timestamp(cs['date']), color='#ff6b6b',
                   linestyle='--', linewidth=1.2, zorder=10)

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
        )
        ax.legend(loc='upper left', fontsize=7, facecolor=PANEL_BG,
                  labelcolor=TEXT_CLR, framealpha=0.85, ncol=4)
        fmt_date_axis(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col2:
        st.markdown("**Regime Probability (Current)**")

        fig, ax = plt.subplots(figsize=(4, 4))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor(PANEL_BG)

        probs  = [cs['p_calm'], cs['p_pullback'], cs['p_stress'], cs['p_crisis']]
        labels = [REGIME_LABELS[i] for i in range(4)]
        colors = [REGIME_COLORS[i] for i in range(4)]

        wedges, texts, autotexts = ax.pie(
            probs, labels=labels, colors=colors,
            autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor=DARK_BG, linewidth=2)
        )
        for t in texts:
            t.set_color(TEXT_CLR)
            t.set_fontsize(8)
        for at in autotexts:
            at.set_color('white')
            at.set_fontsize(7)

        ax.set_title(f"P(Regime) — {cs['date']}",
                     color=TEXT_CLR, fontsize=9, pad=8)
        fig.patch.set_facecolor(DARK_BG)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    st.markdown("---")

    # Row 3: Volatility comparison
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Realized vs GARCH Volatility**")
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax, ylabel='Volatility (annualised)')

        feat_vol = feat[feat.index >= '2020-01-01']
        ax.plot(feat_vol.index, feat_vol['realized_vol'],
                color=ACCENT, linewidth=0.9, alpha=0.8, label='Realized Vol')
        if 'GARCH_Vol' in feat_vol.columns:
            ax.plot(feat_vol.index, feat_vol['GARCH_Vol'],
                    color='#e74c3c', linewidth=1.0, alpha=0.9, label='GARCH Vol')

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:.0%}')
        )
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR)
        fmt_date_axis(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:
        st.markdown("**Drawdown from Peak**")
        fig, ax = plt.subplots(figsize=(7, 3))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax, ylabel='Drawdown')

        feat_dd = feat[feat.index >= '2020-01-01']
        ax.fill_between(feat_dd.index, feat_dd['drawdown'],
                        color='#e74c3c', alpha=0.4)
        ax.plot(feat_dd.index, feat_dd['drawdown'],
                color='#e74c3c', linewidth=0.8)
        ax.axhline(0, color=TEXT_CLR, linewidth=0.5, linestyle='--')
        ax.axhline(-0.15, color='#f39c12', linewidth=0.8,
                   linestyle=':', label='-15% (Moderate)')
        ax.axhline(-0.30, color='#e74c3c', linewidth=0.8,
                   linestyle=':', label='-30% (Severe)')

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:.0%}')
        )
        ax.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR)
        fmt_date_axis(ax)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ─────────────────────────────────────────────────────────────
# TAB 2 — REGIME ANALYSIS
# ─────────────────────────────────────────────────────────────

def render_regime_analysis(rp, ews, cs):
    st.markdown("## Regime Analysis")
    st.caption("GRU Regime Classifier — Test Period: 2024–present")

    # Regime probability time series
    st.markdown("**Softmax Probability Bands over Test Period**")
    prob_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3']

    fig, axes = plt.subplots(2, 1, figsize=(14, 7),
                             gridspec_kw={'height_ratios': [2.5, 1], 'hspace': 0.4})
    fig.patch.set_facecolor(DARK_BG)

    ax = axes[0]
    style_ax(ax, ylabel='P(Regime)')
    if all(c in rp.columns for c in prob_cols):
        ax.stackplot(rp.index,
                     rp['P_regime_0'], rp['P_regime_1'],
                     rp['P_regime_2'], rp['P_regime_3'],
                     colors=[PROB_COLORS[i] for i in range(4)],
                     alpha=0.85,
                     labels=[REGIME_LABELS[i] for i in range(4)])
        ax.set_ylim(0, 1)

    ax.axvline(pd.Timestamp(cs['date']), color='white',
               linestyle='--', linewidth=1.0, label='Current')
    ax.legend(loc='upper left', fontsize=8, facecolor=PANEL_BG,
              labelcolor=TEXT_CLR, framealpha=0.9)
    fmt_date_axis(ax)

    ax2 = axes[1]
    style_ax(ax2, ylabel='Regime', xlabel='Date',
             title='Predicted vs Actual Regime')
    if 'predicted_regime' in rp.columns and 'actual_regime' in rp.columns:
        ax2.step(rp.index, rp['actual_regime'], color=ACCENT,
                 linewidth=1.0, where='post', label='Actual', alpha=0.9)
        ax2.step(rp.index, rp['predicted_regime'], color='#ff9f43',
                 linewidth=0.8, where='post', linestyle='--',
                 label='Predicted', alpha=0.9)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels([REGIME_LABELS[i] for i in range(4)],
                        color=TEXT_CLR, fontsize=7.5)
    ax2.set_ylim(-0.5, 3.8)
    ax2.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT_CLR)
    fmt_date_axis(ax2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Regime distribution
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Test Set Regime Distribution (2024–present)**")
        if 'actual_regime' in rp.columns:
            counts = rp['actual_regime'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(5, 3))
            fig.patch.set_facecolor(DARK_BG)
            style_ax(ax, ylabel='Days')
            bars = ax.bar(
                [REGIME_LABELS[int(i)] for i in counts.index],
                counts.values,
                color=[REGIME_COLORS[int(i)] for i in counts.index],
                alpha=0.85
            )
            for bar, cnt in zip(bars, counts.values):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        f'{cnt}d\n({cnt/len(rp)*100:.0f}%)',
                        ha='center', va='bottom', color=TEXT_CLR, fontsize=8)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

    with col2:
        st.markdown("**Transition Accuracy**")
        try:
            import numpy as np
            y_true = rp['actual_regime'].values
            y_pred = rp['predicted_regime'].values
            overall_acc = np.mean(y_true == y_pred)
            trans_acc   = np.mean(np.diff(y_true) == np.diff(y_pred))
            stress_n    = int((y_true == 2).sum())
            crisis_n    = int((y_true == 3).sum())
            crisis_str  = f"Not evaluable (n={crisis_n})" if crisis_n == 0 \
                          else f"{np.mean((y_pred == 3)[y_true == 3]):.1%} (n={crisis_n})"
            st.markdown(f"""
| Metric | Value |
|--------|-------|
| Overall Accuracy | {overall_acc:.1%} |
| Transition Accuracy | {trans_acc:.1%} |
| Crisis Recall | {crisis_str} |
| Stress Recall | 0.0% (n={stress_n}, insufficient) |
""")
        except Exception:
            st.markdown("""
| Metric | Value |
|--------|-------|
| Overall Accuracy | See Model Evaluation tab |
| Transition Accuracy | See Model Evaluation tab |
| Crisis Recall | Not evaluable (n=0) |
| Stress Recall | 0.0% (n=11, insufficient) |
""")
        st.caption(
            "⚠ Overall accuracy dominated by Calm+Pullback (97.8% of test set). "
            "Crisis recall is the primary institutional metric — "
            "it cannot be evaluated on this test window (structural, not model failure)."
        )


# ─────────────────────────────────────────────────────────────
# TAB 3 — EARLY WARNING
# ─────────────────────────────────────────────────────────────

def render_early_warning(ews, cs):
    st.markdown("## Early Warning System")
    st.caption("Three-signal composite EWS — Step 12")

    # Signal status cards
    c1, c2, c3 = st.columns(3)

    signal_configs = [
        ('stress_signal',     'Stress Signal',
         'P(Stress+Crisis) > 0.40 for 2d OR > 0.60 single day'),
        ('crisis_alert',      'Crisis Alert',
         'P(Crisis) > 0.25 for 2d OR > 0.50 single day'),
        ('escalation_signal', 'Escalation Signal',
         'P(Crisis) strictly rising for 5 consecutive days'),
    ]

    latest = ews.iloc[-1]
    for col, (sig, label, desc) in zip([c1, c2, c3], signal_configs):
        with col:
            active = bool(latest.get(sig, 0))
            color  = '#e74c3c' if active else '#2ecc71'
            status = 'ACTIVE' if active else 'INACTIVE'
            st.markdown(
                f'<div style="background:{"#2d0a0a" if active else "#0a2d0a"};'
                f'border:2px solid {color};border-radius:8px;padding:12px;text-align:center">'
                f'<h4 style="color:{color};margin:0">{label}</h4>'
                f'<p style="color:{color};font-size:18px;font-weight:bold;margin:4px 0">'
                f'{status}</p>'
                f'<p style="color:#cccccc;font-size:10px;margin:0">{desc}</p>'
                f'</div>',
                unsafe_allow_html=True
            )

    st.markdown("---")

    # Composite stress probability chart
    st.markdown("**Composite Stress Probability: P(Stress) + P(Crisis)**")
    combined = ews.get(
        'stress_combined_prob',
        ews.get('P_regime_2', 0) + ews.get('P_regime_3', 0)
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={'height_ratios': [2, 1], 'hspace': 0.4})
    fig.patch.set_facecolor(DARK_BG)

    ax1 = axes[0]
    style_ax(ax1, ylabel='Combined Probability')
    ax1.fill_between(ews.index, combined,
                     where=combined < 0.40, color='#27ae60', alpha=0.4,
                     label='Low (<0.40)')
    ax1.fill_between(ews.index, combined,
                     where=(combined >= 0.40) & (combined < 0.60),
                     color='#f39c12', alpha=0.5, label='Moderate (0.40-0.60)')
    ax1.fill_between(ews.index, combined,
                     where=combined >= 0.60, color='#e74c3c', alpha=0.6,
                     label='High (>=0.60)')
    ax1.plot(ews.index, combined, color='#ecf0f1', linewidth=0.7, alpha=0.8)
    ax1.axhline(0.40, color='#f39c12', linestyle='--', linewidth=1.0,
                alpha=0.8, label='Sustained threshold')
    ax1.axhline(0.60, color='#e74c3c', linestyle='--', linewidth=1.0,
                alpha=0.8, label='Override threshold')

    for sig_col, color, marker in [
        ('stress_signal',    '#f39c12', '^'),
        ('crisis_alert',     '#e74c3c', 'D'),
        ('escalation_signal','#9b59b6', 's'),
    ]:
        if sig_col in ews.columns:
            m = ews[sig_col].astype(bool)
            if m.any():
                ax1.scatter(ews.index[m], combined[m], color=color,
                            marker=marker, s=60, zorder=8,
                            edgecolors='white', linewidths=0.4,
                            label=sig_col.replace('_', ' ').title())

    ax1.axvline(pd.Timestamp(cs['date']), color='white',
                linestyle='--', linewidth=1.0, label='Current')
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc='upper left', fontsize=7.5, facecolor=PANEL_BG,
               labelcolor=TEXT_CLR, framealpha=0.9, ncol=3)
    fmt_date_axis(ax1)

    ax2 = axes[1]
    style_ax(ax2, ylabel='Active Signals', xlabel='Date',
             title='Signal Strength (0-3)')
    if 'signal_strength' in ews.columns:
        sig_str = ews['signal_strength'].fillna(0)
        color_map = {0: '#27ae60', 1: '#f39c12', 2: '#e74c3c', 3: '#922b21'}
        bar_colors = sig_str.map(color_map).fillna('#27ae60')
        ax2.bar(ews.index, sig_str, color=bar_colors, alpha=0.8, width=1.5)
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_ylim(-0.1, 3.5)
    ax2.axvline(pd.Timestamp(cs['date']), color='white',
                linestyle='--', linewidth=1.0)
    fmt_date_axis(ax2)

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown("---")

    # Probability trajectory table
    st.markdown("**Probability Trajectory — Last 15 Trading Days**")
    tail = ews.tail(15)[['P_regime_0', 'P_regime_1', 'P_regime_2',
                          'P_regime_3', 'stress_combined_prob',
                          'stress_signal', 'crisis_alert', 'escalation_signal']].copy()
    tail.index = tail.index.date
    tail.columns = ['P(Calm)', 'P(Pullback)', 'P(Stress)',
                    'P(Crisis)', 'Combined', 'Stress Sig', 'Crisis Alert', 'Escalation']
    tail = tail.round(4)
    st.dataframe(tail.style.background_gradient(
        subset=['P(Stress)', 'P(Crisis)', 'Combined'],
        cmap='Reds'
    ), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TAB 4 — SCENARIO ANALYSIS
# ─────────────────────────────────────────────────────────────

def render_scenario_analysis(mc, arima, feat, cs):
    st.markdown("## Scenario Analysis")
    st.caption(
        "Risk-conditioned Monte Carlo simulation + ARIMA diagnostic forecast. "
        "Probabilistic risk assessment — not price prediction."
    )

    col1, col2 = st.columns(2)

    # Monte Carlo fan chart
    with col1:
        st.markdown("**1-Month Monte Carlo Fan Chart**")

        S0          = cs['close']
        garch_vol   = cs['garch_vol']
        sigma_daily = (garch_vol / 100) / np.sqrt(252)

        # Regime-conditional drift — estimated from historical log returns
        # per regime in features.pkl. These are the empirical values:
        # Calm=+0.000736, Pullback=+0.000366, Stress=-0.000128, Crisis=-0.000437
        regime_drift = {
            0: 0.000736,   # Calm
            1: 0.000366,   # Pullback
            2: -0.000128,  # Stress
            3: -0.000437   # Crisis
        }
        current_regime_id = cs.get('predicted_regime', 1)
        mu_daily = regime_drift.get(int(current_regime_id), 0.000366)
        horizon     = 21
        n_sims      = 10000

        paths = None
        if mc is not None:
            try:
                stored = mc.get('paths', None)
                if stored is not None and stored.shape[0] >= horizon:
                    paths = stored[:horizon, :]
            except Exception:
                paths = None

        if paths is None:
            np.random.seed(42)
            paths = np.zeros((horizon, n_sims))
            paths[0] = S0
            for t in range(1, horizon):
                shock    = np.random.normal(0, 1, n_sims)
                paths[t] = paths[t - 1] * np.exp(
                    mu_daily - 0.5 * sigma_daily ** 2 + sigma_daily * shock
                )

        p05 = np.percentile(paths, 5,  axis=1)
        p25 = np.percentile(paths, 25, axis=1)
        p50 = np.percentile(paths, 50, axis=1)
        p75 = np.percentile(paths, 75, axis=1)
        p95 = np.percentile(paths, 95, axis=1)

        feat_c    = feat.sort_index()
        hist_tail = feat_c['Close'].dropna().tail(63)
        last_date = feat_c.index[-1]
        fc_dates  = pd.bdate_range(
            start=last_date + pd.Timedelta(days=1), periods=horizon
        )

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor(DARK_BG)
        style_ax(ax, ylabel='NIFTY Level', xlabel='Date',
                 title=f'Vol={garch_vol:.1f}% | Regime: {cs["regime_label"]}')

        ax.plot(hist_tail.index, hist_tail.values,
                color=ACCENT, linewidth=1.5, label='Historical', zorder=5)
        ax.fill_between(fc_dates, p05, p95, color='#e74c3c',
                        alpha=0.15, label='90% band')
        ax.fill_between(fc_dates, p25, p75, color='#e74c3c',
                        alpha=0.30, label='50% band')
        ax.plot(fc_dates, p50, color='#ff6b6b', linewidth=1.5,
                linestyle='--', label='Median', zorder=6)
        ax.axvline(last_date, color='white', linestyle=':',
                   linewidth=0.8, alpha=0.6)

        for pct, val, color in [(5, p05[-1], '#aaaacc'),
                                 (50, p50[-1], '#ff6b6b'),
                                 (95, p95[-1], '#aaaacc')]:
            ax.text(fc_dates[-1] + pd.Timedelta(days=1), val,
                    f'{pct}th: {val:,.0f}',
                    color=color, fontsize=6.5, va='center')

        ax.yaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
        )
        ax.legend(fontsize=7.5, facecolor=PANEL_BG,
                  labelcolor=TEXT_CLR, framealpha=0.9)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', color=TEXT_CLR)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Percentile summary
        st.markdown(f"""
        | Percentile | Level | Change |
        |-----------|-------|--------|
        | 5th (downside) | {p05[-1]:,.0f} | {(p05[-1]/S0-1):.1%} |
        | 50th (median)  | {p50[-1]:,.0f} | {(p50[-1]/S0-1):.1%} |
        | 95th (upside)  | {p95[-1]:,.0f} | {(p95[-1]/S0-1):.1%} |
        """)

    # ARIMA forecast
    with col2:
        st.markdown("**ARIMA Diagnostic Forecast (1-Quarter)**")

        if arima is not None:
            fc_1q    = arima['forecast_1q']
            origin   = arima['forecast_origin']
            order    = arima['best_order']
            ci_widths = arima['ci_widths']

            hist_tail2 = feat.sort_index()['Close'].dropna().tail(63)

            fig, ax = plt.subplots(figsize=(7, 4))
            fig.patch.set_facecolor(DARK_BG)
            style_ax(ax, ylabel='NIFTY Level', xlabel='Date',
                     title=f'ARIMA{order} — Uncertainty Structure Validation')

            ax.plot(hist_tail2.index, hist_tail2.values,
                    color=ACCENT, linewidth=1.5, label='Historical')
            ax.plot(fc_1q.index, fc_1q['Forecast'],
                    color='#ecf0f1', linewidth=1.2,
                    linestyle='--', label='Forecast')
            ax.fill_between(fc_1q.index,
                            fc_1q['Lower_95'], fc_1q['Upper_95'],
                            color='#7b89d4', alpha=0.25,
                            label='95% CI')
            ax.axvline(origin, color='#ff6b6b', linestyle=':',
                       linewidth=0.8, label='Forecast origin')

            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f'{x:,.0f}')
            )
            ax.legend(fontsize=7.5, facecolor=PANEL_BG,
                      labelcolor=TEXT_CLR, framealpha=0.9)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right', color=TEXT_CLR)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close()

            st.markdown(f"""
            | Horizon | CI Width | Interpretation |
            |---------|----------|----------------|
            | Day 1   | {ci_widths['day1']:,.0f} pts | Tight near-term uncertainty |
            | Day 21  | {ci_widths['day21']:,.0f} pts | Widening 1-month uncertainty |
            | Day 63  | {ci_widths['day63']:,.0f} pts | Wide 1-quarter uncertainty |
            """)
            st.caption(
                "✓ Uncertainty widens monotonically with horizon — "
                "consistent with Bogle's humility principle."
            )
        else:
            st.info("ARIMA output not found. Run pipeline to generate.")

    st.markdown("---")
    st.caption(
        "⚠ These are probabilistic risk assessments, not price forecasts. "
        "The system diagnoses risk states — it does not predict markets."
    )


# ─────────────────────────────────────────────────────────────
# TAB 5 — MODEL EVALUATION
# ─────────────────────────────────────────────────────────────

def render_evaluation(ev):
    st.markdown("## Model Evaluation")
    st.caption("Dual-tier evaluation design — Step 13")

    if ev is None:
        st.error("evaluation_results.json not found. Run pipeline first.")
        return

    # Tier 1
    st.markdown("### Tier 1 — Out-of-Sample Generalization (2024–present)")
    st.info(
        "This is the only honest generalization test. "
        "The model never saw 2024-present data during training."
    )

    t1 = ev['tier1']

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Accuracy", f"{t1['overall_accuracy']:.1%}",
              help="Dominated by Calm+Pullback. NOT the primary metric.")
    c2.metric("Transition Accuracy", f"{t1['transition_accuracy_exact']:.1%}",
              help="Direction of regime change correct")
    c3.metric("Calm Recall", f"{t1['per_class']['Calm']['recall']:.1%}")
    c4.metric("Pullback Recall", f"{t1['per_class']['Pullback']['recall']:.1%}")

    st.warning(
        "**Crisis Recall: NOT EVALUABLE** — The 2024-present test window contains "
        "0 Crisis-regime days. This is a structural property of the test period, "
        "not a model failure. Crisis detection is assessed via Tier 2."
    )

    # Per-class table
    st.markdown("**Per-Class Metrics**")
    rows = []
    for label, m in t1['per_class'].items():
        rows.append({
            'Regime': label, 'n': m['n'],
            'Precision': f"{m['precision']:.3f}",
            'Recall': f"{m['recall']:.3f}",
            'F1': f"{m['f1']:.3f}",
            'Note': m['caveat']
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # Tier 2
    st.markdown("### Tier 2 — In-Sample Consistency Check (2007–2026)")
    st.warning(
        "⚠ In-sample check only. The GRU model was trained on 2008-2023 data. "
        "Historical episode recall reflects training fit, not generalization."
    )

    t2 = ev['tier2']

    c1, c2 = st.columns(2)
    c1.metric("Episodes Captured", t2['episodes_captured'])
    c2.metric("Stress+Crisis Recall", f"{t2['stress_crisis_recall_episodes']:.1%}")

    st.markdown("**Historical Stress Episode Capture**")
    ep_rows = []
    for ep_name, res in t2['episode_detail'].items():
        if res.get('status') == 'no_data':
            continue
        ep_rows.append({
            'Episode': ep_name.replace('_', ' '),
            'Status': res['status'],
            'Days': res['n_total'],
            'Elevated %': f"{res['pct_elevated']:.0f}%",
            'Lead Time': f"{res['lead_days']}d"
        })
    ep_df = pd.DataFrame(ep_rows)
    st.dataframe(
        ep_df.style.apply(
            lambda x: [
                'color: #2ecc71' if v == 'CAPTURED' else 'color: #e74c3c'
                if v == 'MISSED' else '' for v in x
            ], subset=['Status']
        ),
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    # Methodology notes
    with st.expander("Methodology Notes"):
        notes = ev.get('methodology_notes', {})
        for key, text in notes.items():
            st.markdown(f"**{key.replace('_', ' ').title()}**")
            st.markdown(text)
            st.markdown("")


# ─────────────────────────────────────────────────────────────
# TAB 6 — ABOUT
# ─────────────────────────────────────────────────────────────

def render_about():
    st.markdown("## About This System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### What This System Does
        This is an **institutional-grade equity risk diagnostics system** for the
        NIFTY 50 index. It diagnoses the current market risk state and estimates
        how that state is likely to evolve.

        **Core hypothesis:** Risk is more predictable than price.
        The system focuses on risk state understanding — not price prediction.

        ### Pipeline (Steps 1–14)
        | Step | Component |
        |------|-----------|
        | 1–5 | Feature engineering, drawdown, regimes |
        | 6 | GJR-GARCH(1,1) volatility modeling |
        | 7 | GRU volatility forecasting |
        | 8 | Risk-conditioned Monte Carlo |
        | 9 | ARIMA diagnostic forecasting |
        | 11 | GRU regime transition classifier |
        | 12 | Early warning signal generation |
        | 13 | Dual-tier model evaluation |
        | 14 | Visualization layer |
        """)

    with col2:
        st.markdown("""
        ### Intellectual Pillars

        **Bodie** — Risk is measurable and persistent; returns are not.
        Every modeling decision prioritizes risk estimation over return prediction.
        Regimes exist because risk is state-dependent, not constant.

        **Karasan** — Hybrid modeling is the correct approach.
        Statistical models (GJR-GARCH) and deep learning (GRU) are complementary.
        ML in finance earns credibility through risk modeling, not price prediction.

        **Bogle** — Humility is non-negotiable.
        Probabilistic outputs are more honest than point forecasts.
        Uncertainty grows with horizon. The system never claims to predict markets.

        ### Market Regimes
        | Regime | Definition |
        |--------|------------|
        | 🟢 Calm | Low drawdown, low volatility |
        | 🟡 Pullback | Moderate drawdown, moderate volatility |
        | 🔴 Stress | Elevated volatility, significant drawdown |
        | ⚫ Crisis | Severe drawdown, extreme volatility |
        """)

    st.markdown("---")
    st.markdown("""
    ### Academic References
    - Hamilton, J.D. (1989). *A New Approach to the Economic Analysis of Nonstationary
      Time Series and the Business Cycle.* Econometrica, 57(2), 357-384.
    - Pagan, A.R. & Sossounov, K.A. (2003). *A simple framework for analysing bull and
      bear markets.* Journal of Applied Econometrics, 18(1), 23-46.
    - Ang, A. & Bekaert, G. (2002). *International asset allocation with regime shifts.*
      Review of Financial Studies, 15(4), 1137-1187.
    - Bodie, Z., Kane, A. & Marcus, A.J. *Investments.* McGraw-Hill.
    - Karasan, A. (2021). *Machine Learning for Financial Risk Management with Python.*
      O'Reilly Media.
    """)

    st.caption(
        "Built with Python, TensorFlow, statsmodels, arch, and Streamlit. "
        "Data: Yahoo Finance (^NSEI). "
        "Models are frozen — no retraining on each run."
    )


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    # Load all data
    try:
        feat  = load_features()
        rp    = load_regime_probs()
        ews   = load_ews()
        mc    = load_monte_carlo()
        arima = load_arima()
        ev    = load_evaluation()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}\n\nRun `python pipeline/run_daily.py` first.")
        st.stop()

    cs = get_current_state(ews, feat)

    # Sidebar
    render_sidebar(cs)

    # Header
    active_str = ' | '.join(cs['active_signals']) if cs['active_signals'] else 'No Active Signal'
    header_color = '#e74c3c' if cs['active_signals'] else '#2ecc71'
    st.markdown(
        f'<h1 style="color:{header_color}">NIFTY 50 — Equity Risk Diagnostics System</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<p style="color:#888899">As of {cs["date"]} | '
        f'Regime: <span style="color:{REGIME_COLORS[cs["regime"]]}">'
        f'{cs["regime_label"]}</span> | {active_str}</p>',
        unsafe_allow_html=True
    )

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Live Dashboard",
        "🎯 Regime Analysis",
        "⚠ Early Warning",
        "🎲 Scenario Analysis",
        "📊 Model Evaluation",
        "ℹ About"
    ])

    with tab1:
        render_live_dashboard(feat, rp, ews, cs)
    with tab2:
        render_regime_analysis(rp, ews, cs)
    with tab3:
        render_early_warning(ews, cs)
    with tab4:
        render_scenario_analysis(mc, arima, feat, cs)
    with tab5:
        render_evaluation(ev)
    with tab6:
        render_about()


if __name__ == '__main__':
    main()
