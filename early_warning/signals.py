# early_warning/signals.py
# Step 12: Early Warning System — Stress Signal Generation

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive — no blocking plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REGIME_LABELS,
    STRESS_EPISODES,
    STRESS_PROB_THRESHOLD,
    STRESS_SUSTAIN_DAYS,
    STRESS_OVERRIDE_THRESHOLD,
    CRISIS_PROB_THRESHOLD,
    CRISIS_SUSTAIN_DAYS,
    CRISIS_OVERRIDE_THRESHOLD,
    ESCALATION_WINDOW,
    REGIME_N_CLASSES,
)

# ─────────────────────────────────────────────────────────────
# SIGNAL LABELS
# ─────────────────────────────────────────────────────────────
SIGNAL_LABELS = {
    'STRESS_SIGNAL':     'Stress Signal',
    'CRISIS_ALERT':      'Crisis Alert',
    'ESCALATION_SIGNAL': 'Escalation Signal'
}


# ─────────────────────────────────────────────────────────────
# SECTION 1: SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_early_warning_signals(prob_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all three early warning signals.

    Stress Signal fires if EITHER:
      (a) P(Stress)+P(Crisis) > 0.40 for 2+ consecutive days  [sustained]
      (b) P(Stress)+P(Crisis) > 0.60 on any single day        [override]

    Crisis Alert fires if EITHER:
      (a) P(Crisis) > 0.25 for 2+ consecutive days            [sustained]
      (b) P(Crisis) > 0.50 on any single day                  [override]

    Escalation Signal fires if P(Crisis) strictly rising
    for 5 consecutive days (no override -- trend signal only).

    Thresholds imported from config.py -- single source of truth.
    """
    df = prob_df.copy()

    required = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Signal 1: Stress Signal (sustained OR single-day override)
    df['stress_combined_prob'] = df['P_regime_2'] + df['P_regime_3']

    stress_sustained = (
        (df['stress_combined_prob'] > STRESS_PROB_THRESHOLD)
        .astype(int)
        .rolling(window=STRESS_SUSTAIN_DAYS)
        .min()
        .fillna(0)
        .astype(int)
    )
    stress_override = (df['stress_combined_prob'] > STRESS_OVERRIDE_THRESHOLD).astype(int)
    df['stress_signal'] = ((stress_sustained == 1) | (stress_override == 1)).astype(int)

    # Signal 2: Crisis Alert (sustained OR single-day override)
    crisis_sustained = (
        (df['P_regime_3'] > CRISIS_PROB_THRESHOLD)
        .astype(int)
        .rolling(window=CRISIS_SUSTAIN_DAYS)
        .min()
        .fillna(0)
        .astype(int)
    )
    crisis_override = (df['P_regime_3'] > CRISIS_OVERRIDE_THRESHOLD).astype(int)
    df['crisis_alert'] = ((crisis_sustained == 1) | (crisis_override == 1)).astype(int)

    # Signal 3: Escalation Signal (trend only -- no override)
    crisis_diff = df['P_regime_3'].diff()
    df['escalation_signal'] = (
        (crisis_diff > 0)
        .astype(int)
        .rolling(window=ESCALATION_WINDOW)
        .min()
        .fillna(0)
        .astype(int)
    )

    # Composite
    df['any_signal'] = (
        (df['stress_signal'] | df['crisis_alert'] | df['escalation_signal'])
        .astype(int)
    )
    df['signal_strength'] = (
        df['stress_signal'] + df['crisis_alert'] + df['escalation_signal']
    )

    return df


# ─────────────────────────────────────────────────────────────
# SECTION 2: SIGNAL SUMMARY
# ─────────────────────────────────────────────────────────────

def summarise_signals(signals_df: pd.DataFrame,
                      actual_stress_periods: dict = None) -> dict:
    results = {}

    print("\n" + "=" * 60)
    print("STEP 12: EARLY WARNING SIGNAL SUMMARY")
    print(f"Period: {signals_df.index.min().date()} to {signals_df.index.max().date()}")
    print(f"n = {len(signals_df)}")
    print("=" * 60)

    for signal_type in ['stress_signal', 'crisis_alert', 'escalation_signal']:
        active     = signals_df[signal_type]
        n_active   = active.sum()
        pct_active = n_active / len(active) * 100
        episodes   = _find_signal_episodes(active)

        results[signal_type] = {
            'n_active_days': int(n_active),
            'pct_active':    round(pct_active, 2),
            'n_episodes':    len(episodes),
            'episodes':      episodes
        }

        label_key = signal_type.upper()
        print(f"\n  {SIGNAL_LABELS.get(label_key, signal_type)}")
        print(f"    Active days   : {n_active} ({pct_active:.1f}% of period)")
        print(f"    Episodes      : {len(episodes)}")
        if len(episodes) > 0:
            avg_dur = np.mean([e['duration'] for e in episodes])
            print(f"    Avg duration  : {avg_dur:.1f} days")
            for ep in episodes:
                print(f"      {ep['start'].date()} to {ep['end'].date()} "
                      f"({ep['duration']} days)")

    if actual_stress_periods is not None and len(actual_stress_periods) > 0:
        print("\n  LEAD TIME ANALYSIS (days before stress onset)")
        print(f"  {'Episode':30s} {'Detected':10s} {'Lead (days)':12s}")
        print(f"  {'-'*55}")

        lead_times = []
        detected   = 0

        for name, (start, end) in actual_stress_periods.items():
            start_dt     = pd.Timestamp(start)
            window_start = max(
                start_dt - pd.Timedelta(days=60),
                signals_df.index.min()
            )
            window = signals_df.loc[window_start:start_dt, 'any_signal']

            if len(window) == 0 or window.sum() == 0:
                print(f"  {name:30s} {'NO':10s} {'n/a':12s}")
                continue

            first_signal = window[window == 1].index.min()
            lead_days    = (start_dt - first_signal).days
            lead_times.append(lead_days)
            detected    += 1
            print(f"  {name:30s} {'YES':10s} {lead_days:12d}")

        detection_rate = detected / len(actual_stress_periods)
        avg_lead       = np.mean(lead_times) if lead_times else 0
        print(f"\n  Detection rate  : {detected}/{len(actual_stress_periods)} "
              f"({detection_rate*100:.0f}%)")
        print(f"  Average lead    : {avg_lead:.1f} days")
        results['detection_rate'] = detection_rate
        results['avg_lead_days']  = avg_lead
    else:
        print("\n  LEAD TIME ANALYSIS: Deferred to Step 13")
        print("  No completed stress episodes within the test window (2024-present).")
        print("  Full-dataset lead-time evaluation runs in Step 13, Tier 2.")

    return results


# ─────────────────────────────────────────────────────────────
# SECTION 3: CURRENT SIGNAL STATE
# ─────────────────────────────────────────────────────────────

def get_current_signal_state(signals_df: pd.DataFrame) -> dict:
    """Return current signal state from the latest row."""
    latest = signals_df.iloc[-1]
    state = {
        'date':             str(signals_df.index[-1].date()),
        'predicted_regime': int(latest.get('predicted_regime', -1)),
        'stress_combined':  round(float(latest['stress_combined_prob']), 4),
        'p_calm':           round(float(latest['P_regime_0']), 4),
        'p_pullback':       round(float(latest['P_regime_1']), 4),
        'p_stress':         round(float(latest['P_regime_2']), 4),
        'p_crisis':         round(float(latest['P_regime_3']), 4),
        'active_signals':   []
    }
    if latest['stress_signal']:     state['active_signals'].append('STRESS_SIGNAL')
    if latest['crisis_alert']:      state['active_signals'].append('CRISIS_ALERT')
    if latest['escalation_signal']: state['active_signals'].append('ESCALATION_SIGNAL')
    return state


# ─────────────────────────────────────────────────────────────
# SECTION 4: PROBABILITY TRAJECTORY TABLE
# ─────────────────────────────────────────────────────────────

def print_probability_trajectory(signals_df: pd.DataFrame, n_days: int = 15):
    print(f"\n{'='*60}")
    print(f"PROBABILITY TRAJECTORY -- Last {n_days} trading days")
    print("=" * 60)
    print(f"  {'Date':12s} {'P(Calm)':>8s} {'P(Pull)':>8s} "
          f"{'P(Stress)':>10s} {'P(Crisis)':>10s} "
          f"{'Combined':>10s} {'Signals':>10s}")
    print(f"  {'-'*74}")

    for date, row in signals_df.tail(n_days).iterrows():
        sigs = []
        if row['stress_signal']:     sigs.append('S')
        if row['crisis_alert']:      sigs.append('C')
        if row['escalation_signal']: sigs.append('E')
        sig_str = '+'.join(sigs) if sigs else '--'

        override  = row['stress_combined_prob'] > STRESS_OVERRIDE_THRESHOLD
        sustained = row['stress_combined_prob'] > STRESS_PROB_THRESHOLD
        flag = ' <<' if override else (' <' if sustained else '')

        print(f"  {str(date.date()):12s} "
              f"{row['P_regime_0']:8.3f} "
              f"{row['P_regime_1']:8.3f} "
              f"{row['P_regime_2']:10.3f} "
              f"{row['P_regime_3']:10.3f} "
              f"{row['stress_combined_prob']:10.3f}{flag} "
              f"{sig_str:>8s}")

    print(f"\n  <  = above sustained threshold ({STRESS_PROB_THRESHOLD})")
    print(f"  << = above override threshold  ({STRESS_OVERRIDE_THRESHOLD}) -- fires immediately")
    print(f"  S=Stress Signal  C=Crisis Alert  E=Escalation")


# ─────────────────────────────────────────────────────────────
# SECTION 5: HELPERS
# ─────────────────────────────────────────────────────────────

def _find_signal_episodes(signal_series: pd.Series) -> list:
    """Find contiguous active signal episodes."""
    episodes = []
    in_episode = False
    start = None

    for date, val in signal_series.items():
        if val == 1 and not in_episode:
            in_episode = True
            start = date
        elif val == 0 and in_episode:
            in_episode = False
            episodes.append({
                'start':    start,
                'end':      date,
                'duration': (date - start).days
            })

    if in_episode:
        end = signal_series.index[-1]
        episodes.append({
            'start':    start,
            'end':      end,
            'duration': (end - start).days
        })

    return episodes


def _add_stress_overlays(ax, stress_periods: dict, dates: pd.DatetimeIndex):
    """Shade stress episode windows on a matplotlib axis."""
    for name, (start, end) in stress_periods.items():
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        if e >= dates.min() and s <= dates.max():
            ax.axvspan(max(s, dates.min()), min(e, dates.max()),
                       alpha=0.12, color='#c0392b', zorder=0)


# ─────────────────────────────────────────────────────────────
# SECTION 6: VISUALIZATION
# ─────────────────────────────────────────────────────────────

def plot_early_warning_dashboard(signals_df, features_df=None,
                                 actual_stress_periods=None,
                                 save_path='outputs/early_warning_dashboard.png'):

    regime_colors_map = ['#2ecc71', '#f39c12', '#e74c3c', '#1a1a2e']
    fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
    fig.suptitle('NIFTY 50 Early Warning System -- Step 12 Dashboard',
                 fontsize=14, fontweight='bold')

    dates = signals_df.index

    # Panel 1: NIFTY close
    ax1 = axes[0]
    if features_df is not None and 'Close' in features_df.columns:
        close = features_df['Close'].reindex(dates)
        if not close.isna().all():
            ax1.plot(dates, close, color='#2c3e50', lw=1.2, label='NIFTY 50')
            if 'actual_regime' in signals_df.columns:
                for r, color in enumerate(regime_colors_map):
                    mask = (signals_df['actual_regime'] == r).values
                    ax1.fill_between(dates, close.min(), close.max(),
                                     where=mask, alpha=0.15, color=color,
                                     label=REGIME_LABELS[r])
    ax1.set_title('NIFTY 50 (Regime-Shaded)', fontsize=10)
    ax1.set_ylabel('Index Level')
    ax1.legend(fontsize=8, ncol=5, loc='lower left')
    ax1.grid(alpha=0.2)
    if actual_stress_periods:
        _add_stress_overlays(ax1, actual_stress_periods, dates)

    # Panel 2: Probability stacks
    ax2 = axes[1]
    prob_cols = ['P_regime_0', 'P_regime_1', 'P_regime_2', 'P_regime_3']
    probs     = signals_df[prob_cols].values.T
    ax2.stackplot(dates, probs,
                  labels=['Calm', 'Pullback', 'Stress', 'Crisis'],
                  colors=regime_colors_map, alpha=0.75)
    ax2.set_title('Regime Transition Probabilities P(Regime_k)', fontsize=10)
    ax2.set_ylabel('Probability')
    ax2.legend(loc='upper left', fontsize=8, ncol=4)
    ax2.set_ylim(0, 1)
    ax2.grid(alpha=0.2)

    # Panel 3: Signal strength
    ax3 = axes[2]
    sc = {0: '#bdc3c7', 1: '#f39c12', 2: '#e74c3c', 3: '#8e44ad'}
    bar_colors = [sc[min(s, 3)] for s in signals_df['signal_strength']]
    ax3.bar(dates, signals_df['signal_strength'],
            color=bar_colors, width=1.0, alpha=0.85)
    ax3.set_title('Signal Strength (0=None to 3=All Active)', fontsize=10)
    ax3.set_ylabel('Active Signals')
    ax3.set_ylim(0, 3.5)
    ax3.set_yticks([0, 1, 2, 3])
    ax3.legend(handles=[
        mpatches.Patch(color='#bdc3c7', label='No Signal'),
        mpatches.Patch(color='#f39c12', label='1 Signal'),
        mpatches.Patch(color='#e74c3c', label='2 Signals'),
        mpatches.Patch(color='#8e44ad', label='3 Signals'),
    ], fontsize=8, ncol=4)
    ax3.grid(alpha=0.2)
    if actual_stress_periods:
        _add_stress_overlays(ax3, actual_stress_periods, dates)

    # Panel 4: Probability with thresholds
    ax4 = axes[3]
    ax4.plot(dates, signals_df['P_regime_3'],
             color='#c0392b', lw=1.2, label='P(Crisis)')
    ax4.plot(dates, signals_df['stress_combined_prob'],
             color='#e67e22', lw=1.0, alpha=0.8,
             linestyle='--', label='P(Stress)+P(Crisis)')
    ax4.axhline(CRISIS_PROB_THRESHOLD,
                color='#c0392b', linestyle=':', lw=1.5,
                label=f'Crisis threshold ({CRISIS_PROB_THRESHOLD})')
    ax4.axhline(STRESS_PROB_THRESHOLD,
                color='#e67e22', linestyle=':', lw=1.5,
                label=f'Stress threshold ({STRESS_PROB_THRESHOLD})')
    ax4.axhline(STRESS_OVERRIDE_THRESHOLD,
                color='#e74c3c', linestyle='--', lw=1.2,
                label=f'Override threshold ({STRESS_OVERRIDE_THRESHOLD})')
    ax4.set_title('Crisis & Stress Probability with Alert Thresholds', fontsize=10)
    ax4.set_ylabel('Probability')
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize=8, ncol=2)
    ax4.grid(alpha=0.2)
    if actual_stress_periods:
        _add_stress_overlays(ax4, actual_stress_periods, dates)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Dashboard saved: {save_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':

    prob_df = pd.read_pickle('data/regime_probs.pkl')
    prob_df.index = pd.to_datetime(prob_df.index)

    print("Loaded regime_probs.pkl")
    print(f"Period: {prob_df.index.min().date()} to {prob_df.index.max().date()}")
    print(f"Rows: {len(prob_df)}")

    print("\nComputing early warning signals...")
    signals_df = compute_early_warning_signals(prob_df)
    signals_df.to_pickle('data/early_warning_signals.pkl')
    print("Signals saved: data/early_warning_signals.pkl")

    # Use STRESS_EPISODES from config -- no local redefinition
    test_start  = prob_df.index.min()
    test_stress = {
        k: v for k, v in STRESS_EPISODES.items()
        if pd.Timestamp(v[1]) >= test_start
    }

    if len(test_stress) == 0:
        results = summarise_signals(signals_df, actual_stress_periods=None)
    else:
        results = summarise_signals(signals_df, test_stress)

    print_probability_trajectory(signals_df, n_days=15)

    print(f"\n{'='*60}")
    print("CURRENT SIGNAL STATE")
    print("=" * 60)
    current = get_current_signal_state(signals_df)
    for k, v in current.items():
        print(f"  {k:30s}: {v}")

    features = pd.read_pickle('data/features.pkl')
    features.index = pd.to_datetime(features.index)
    plot_early_warning_dashboard(signals_df, features, STRESS_EPISODES)
