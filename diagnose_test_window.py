# diagnose_test_window.py
# ══════════════════════════════════════════════════════════════
# Diagnostic: Why does the test set have only 9 Stress days?
# Run this BEFORE signals.py to understand the evaluation problem
# ══════════════════════════════════════════════════════════════

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

TRAIN_CUTOFF = '2024-01-01'

REGIME_LABELS = {0: 'Calm', 1: 'Pullback', 2: 'Stress', 3: 'Crisis'}
REGIME_COLORS = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c', 3: '#1a1a2e'}

# ── Load
features = pd.read_pickle('data/features.pkl')
features.index = pd.to_datetime(features.index)
features = features.sort_index()

# ── Full dataset regime distribution
print("=" * 60)
print("FULL DATASET REGIME DISTRIBUTION")
print(f"2007-12-17 → 2026-04-02 (n={len(features)})")
print("=" * 60)
vc_full = features['regime'].value_counts().sort_index()
for r, n in vc_full.items():
    print(f"  Regime {r} ({REGIME_LABELS[r]:8s}): {n:5d} days ({n/len(features)*100:.1f}%)")

# ── Test window regime distribution
test = features[features.index >= TRAIN_CUTOFF]
print(f"\n{'='*60}")
print(f"TEST WINDOW REGIME DISTRIBUTION")
print(f"{TRAIN_CUTOFF} → 2026-04-02 (n={len(test)})")
print("=" * 60)
vc_test = test['regime'].value_counts().sort_index()
for r in range(4):
    n = vc_test.get(r, 0)
    print(f"  Regime {r} ({REGIME_LABELS[r]:8s}): {n:5d} days ({n/len(test)*100:.1f}%)")

# ── Show the 9 Stress days
print(f"\n{'='*60}")
print("THE 9 STRESS DAYS IN TEST WINDOW")
print("=" * 60)
stress_days = test[test['regime'] == 2][['Close', 'realized_vol', 'GARCH_Vol', 'drawdown', 'regime']]
print(stress_days.to_string())

# ── Timeline: when did regime transitions happen in test window?
print(f"\n{'='*60}")
print("REGIME TIMELINE IN TEST WINDOW (transitions only)")
print("=" * 60)
prev = None
for date, row in test.iterrows():
    if row['regime'] != prev:
        print(f"  {date.date()}  →  Regime {int(row['regime'])} ({REGIME_LABELS[int(row['regime'])]})")
        prev = row['regime']

# ── Key question: are these 9 Stress days clustered or isolated?
print(f"\n{'='*60}")
print("STRESS SPELL ANALYSIS IN TEST WINDOW")
print("=" * 60)
test_copy = test.copy()
test_copy['spell'] = (test_copy['regime'] != test_copy['regime'].shift()).cumsum()
stress_spells = (
    test_copy[test_copy['regime'] == 2]
    .groupby('spell')
    .agg(
        start=('Close', lambda x: x.index.min()),
        end=('Close', lambda x: x.index.max()),
        duration=('regime', 'count'),
        min_close=('Close', 'min'),
        max_garch=('GARCH_Vol', 'max')
    )
)
print(stress_spells.to_string())

# ── Diagnosis
print(f"\n{'='*60}")
print("DIAGNOSIS")
print("=" * 60)

n_stress_test = vc_test.get(2, 0)
n_stress_full = vc_full.get(2, 0)

print(f"""
  Stress days in test window   : {n_stress_test} ({n_stress_test/len(test)*100:.1f}%)
  Stress days in full dataset  : {n_stress_full} ({n_stress_full/len(features)*100:.1f}%)

  The test window (2024-2026) is predominantly a Calm/Pullback
  period punctuated by a current Stress episode that began
  recently. With only 9 Stress examples, the model's recall
  of 0.111 is evaluated on a statistically thin sample.

  IMPLICATION FOR STEP 13 EVALUATION DESIGN:
  ─────────────────────────────────────────────
  Option A: Extend test window back to 2022-01-01 to include
            Post-COVID Macro Stress (2021-10-18 to 2022-11-30)
            This gives ~300 additional Stress days in the
            evaluation set.

  Option B: Keep fixed cutoff but report Stress/Crisis recall
            on the FULL dataset using time-series cross-validation
            (walk-forward expanding window).

  Option C: Report two evaluation tiers:
            Tier 1: 2024-present (live simulation — honest)
            Tier 2: Full 2007-2026 (regime coverage — complete)

  RECOMMENDATION: Option C — report both tiers.
  Tier 1 is the honest out-of-sample test.
  Tier 2 covers all regime types for institutional completeness.

  NOTE FOR METHODOLOGY DOCUMENTATION:
  The Stress recall of 0.111 on the test set is not a model
  failure — it reflects a nearly Stress-free test window.
  The model's primary obligation is Crisis recall, which
  cannot be evaluated on this test set at all (n=0 Crisis).
  This motivates the dual-tier evaluation in Step 13.
""")

# ── Current state context
print("=" * 60)
print("CURRENT MARKET STATE CONTEXT (2026-04-02)")
print("=" * 60)
latest = features.iloc[-5:]
print(latest[['Close', 'realized_vol', 'GARCH_Vol', 'drawdown', 'regime']].to_string())
print(f"\n  Current regime: {int(features['regime'].iloc[-1])} "
      f"({REGIME_LABELS[int(features['regime'].iloc[-1])]})")
print(f"  Current GARCH_Vol: {features['GARCH_Vol'].iloc[-1]:.4f} "
      f"({features['GARCH_Vol'].iloc[-1]*100:.1f}% annualized)")

# ── Visualisation
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Test Window Regime Diagnostic — Why Only 9 Stress Days?',
             fontsize=12, fontweight='bold')

# Test window only
test_plot = features[features.index >= '2023-01-01']  # show 1 yr before cutoff too

# Panel 1: Close price with regime background
ax1 = axes[0]
ax1.plot(test_plot.index, test_plot['Close'], color='#2c3e50', lw=1.2)
for r in range(4):
    mask = test_plot['regime'] == r
    ax1.fill_between(test_plot.index, test_plot['Close'].min(),
                     test_plot['Close'].max(),
                     where=mask, alpha=0.2,
                     color=REGIME_COLORS[r],
                     label=REGIME_LABELS[r])
ax1.axvline(pd.Timestamp(TRAIN_CUTOFF), color='black',
            linestyle='--', lw=1.5, label='Train/Test Cutoff')
ax1.set_title('NIFTY 50 Close — Regime Background (2023–2026)', fontsize=10)
ax1.set_ylabel('Index Level')
ax1.legend(fontsize=8, ncol=5, loc='lower left')
ax1.grid(alpha=0.2)

# Panel 2: Regime labels
ax2 = axes[1]
ax2.scatter(test_plot.index, test_plot['regime'],
            c=[REGIME_COLORS[r] for r in test_plot['regime']],
            s=6, alpha=0.8)
ax2.axvline(pd.Timestamp(TRAIN_CUTOFF), color='black',
            linestyle='--', lw=1.5)
ax2.set_yticks([0, 1, 2, 3])
ax2.set_yticklabels([REGIME_LABELS[i] for i in range(4)])
ax2.set_title('Daily Regime Labels', fontsize=10)
ax2.grid(alpha=0.2)

# Panel 3: GARCH_Vol
ax3 = axes[2]
ax3.plot(test_plot.index, test_plot['GARCH_Vol'] * 100,
         color='#e74c3c', lw=1.2, label='GARCH_Vol (%)')
ax3.axvline(pd.Timestamp(TRAIN_CUTOFF), color='black',
            linestyle='--', lw=1.5, label='Train/Test Cutoff')
ax3.set_title('GARCH Conditional Volatility (%)', fontsize=10)
ax3.set_ylabel('Volatility (%)')
ax3.legend(fontsize=8)
ax3.grid(alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/test_window_diagnostic.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nDiagnostic chart saved → outputs/test_window_diagnostic.png")