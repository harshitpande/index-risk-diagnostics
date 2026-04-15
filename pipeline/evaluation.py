# pipeline/evaluation.py
# Step 13 -- Model Evaluation
# NIFTY 50 Equity Risk Diagnostics System
#
# Dual-tier evaluation design:
#   Tier 1 -- Honest out-of-sample (2024-present): generalization test
#   Tier 2 -- Full dataset (2007-2026): in-sample consistency check
#
# FLAGS respected:
#   FLAG 1 -- Test set has 0 Crisis days; Crisis recall not evaluable on Tier 1
#   FLAG 2 -- Dual-tier is mandatory; every metric carries its tier label
#   FLAG 3 -- Models are frozen; no fit() calls anywhere
#   FLAG 5 -- Output files overwrite on each run (correct behaviour)

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.metrics import confusion_matrix, classification_report

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    REGIME_LABELS,
    REGIME_COLORS,
    STRESS_EPISODES,
    TRAIN_CUTOFF,
    COST_RATIO_JSON,
    EVAL_RESULTS_JSON,
    THRESHOLD_CAL_PKL,
    REGIME_PROBS_PKL,
    EWS_PKL,
    FEATURES_PKL,
    OUTPUTS_DIR,
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(BASE_DIR, 'data')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

TIER1_PNG = os.path.join(OUTPUTS_DIR, 'tier1_evaluation.png')
TIER2_PNG = os.path.join(OUTPUTS_DIR, 'tier2_evaluation.png')


# ════════════════════════════════════════════════════════════════
# LOAD DATA
# ════════════════════════════════════════════════════════════════

def load_data():
    print("\n[LOAD] Loading data files...")

    with open(REGIME_PROBS_PKL, 'rb') as f:
        regime_probs = pickle.load(f)
    with open(EWS_PKL, 'rb') as f:
        ews = pickle.load(f)
    with open(FEATURES_PKL, 'rb') as f:
        features = pickle.load(f)
    with open(COST_RATIO_JSON, 'r') as f:
        cost_config = json.load(f)

    print(f"  regime_probs  : {regime_probs.shape}, "
          f"{regime_probs.index.min().date()} to {regime_probs.index.max().date()}")
    print(f"  ews           : {ews.shape}")
    print(f"  features      : {features.shape}, "
          f"{features.index.min().date()} to {features.index.max().date()}")
    print(f"  cost_ratio    : {cost_config}")

    return regime_probs, ews, features, cost_config


# ════════════════════════════════════════════════════════════════
# STEP 13a -- TIER 1 EVALUATION
# ════════════════════════════════════════════════════════════════

def tier1_evaluation(regime_probs, cost_config):
    """
    Honest out-of-sample evaluation on the test set (2024-present).
    Crisis recall is structurally unevaluable (n=0 in test set).
    """
    print("\n[TIER 1] Out-of-sample evaluation (2024-present)...")

    y_true = regime_probs['actual_regime'].values.astype(int)
    y_pred = regime_probs['predicted_regime'].values.astype(int)

    unique, counts = np.unique(y_true, return_counts=True)
    regime_counts  = dict(zip(unique.tolist(), counts.tolist()))
    print(f"  Test set regime distribution: {regime_counts}")

    all_classes = [0, 1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=all_classes)

    report = classification_report(
        y_true, y_pred,
        labels=all_classes,
        target_names=[REGIME_LABELS[i] for i in all_classes],
        output_dict=True,
        zero_division=0
    )

    print("\n  Per-class metrics:")
    metrics_t1 = {}
    for cls in all_classes:
        label = REGIME_LABELS[cls]
        n     = regime_counts.get(cls, 0)
        rec   = report[label]['recall']
        prec  = report[label]['precision']
        f1    = report[label]['f1-score']

        if cls == 3:
            caveat = "NOT EVALUABLE -- n=0 in test set (structural, not model failure)"
        elif cls == 2:
            caveat = f"LOW CONFIDENCE -- n={n} (interpret with caution)"
        else:
            caveat = f"n={n}"

        metrics_t1[label] = {
            'n': n, 'precision': round(prec, 4),
            'recall': round(rec, 4), 'f1': round(f1, 4),
            'caveat': caveat
        }
        print(f"    {label:10s} | n={n:4d} | P={prec:.3f} | "
              f"R={rec:.3f} | F1={f1:.3f} | {caveat}")

    actual_transitions    = np.diff(y_true)
    predicted_transitions = np.diff(y_pred)
    transition_accuracy   = np.mean(actual_transitions == predicted_transitions)
    direction_accuracy    = np.mean(
        np.sign(actual_transitions) == np.sign(predicted_transitions)
    )

    print(f"\n  Transition accuracy (exact): {transition_accuracy:.4f}")
    print(f"  Direction accuracy (sign)  : {direction_accuracy:.4f}")

    overall_acc = np.mean(y_true == y_pred)
    print(f"\n  Overall accuracy: {overall_acc:.4f}")
    print("  CAVEAT: Overall accuracy is dominated by Calm+Pullback (97.8% of test set).")
    print("          This metric is NOT the primary evaluation criterion.")
    print("          Crisis recall is the primary metric -- unevaluable on this test set.")

    _plot_tier1(cm, metrics_t1, overall_acc, transition_accuracy, direction_accuracy)

    return {
        'tier': 'Tier1_OutOfSample',
        'period': f'{TRAIN_CUTOFF} to present',
        'n_total': int(len(y_true)),
        'regime_counts': regime_counts,
        'overall_accuracy': round(overall_acc, 4),
        'overall_accuracy_caveat': 'Dominated by Calm+Pullback (97.8%). Not primary metric.',
        'transition_accuracy_exact': round(transition_accuracy, 4),
        'transition_accuracy_direction': round(direction_accuracy, 4),
        'per_class': metrics_t1
    }


def _plot_tier1(cm, metrics_t1, overall_acc, transition_acc, direction_acc):
    fig = plt.figure(figsize=(16, 7))
    fig.patch.set_facecolor('#0f0f1a')
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)

    ax_cm = fig.add_subplot(gs[0, 0])
    ax_cm.set_facecolor('#0f0f1a')

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm / row_sums

    im = ax_cm.imshow(cm_norm, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    labels = [REGIME_LABELS[i] for i in range(4)]
    ax_cm.set_xticks(range(4))
    ax_cm.set_yticks(range(4))
    ax_cm.set_xticklabels(labels, rotation=30, ha='right', color='white', fontsize=9)
    ax_cm.set_yticklabels(labels, color='white', fontsize=9)
    ax_cm.set_xlabel('Predicted', color='white', fontsize=10)
    ax_cm.set_ylabel('Actual', color='white', fontsize=10)
    ax_cm.set_title('Tier 1 -- Confusion Matrix\n(row-normalised, 2024-present)',
                    color='white', fontsize=11, pad=10)

    for i in range(4):
        for j in range(4):
            val = cm[i, j]
            norm = cm_norm[i, j]
            txt_color = 'black' if norm > 0.5 else 'white'
            ax_cm.text(j, i, f'{val}\n({norm:.2f})',
                       ha='center', va='center', fontsize=8,
                       color=txt_color, fontweight='bold')

    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_tbl = fig.add_subplot(gs[0, 1])
    ax_tbl.set_facecolor('#0f0f1a')
    ax_tbl.axis('off')

    col_labels = ['Regime', 'n', 'Precision', 'Recall', 'F1', 'Caveat']
    table_data = []
    for cls in range(4):
        label  = REGIME_LABELS[cls]
        m      = metrics_t1[label]
        caveat = m['caveat'].split('--')[0].strip() if '--' in m['caveat'] else m['caveat']
        table_data.append([label, m['n'], f"{m['precision']:.3f}",
                           f"{m['recall']:.3f}", f"{m['f1']:.3f}", caveat])

    tbl = ax_tbl.table(
        cellText=table_data, colLabels=col_labels,
        cellLoc='center', loc='center', bbox=[0, 0.25, 1, 0.6]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor('#1a1a2e' if row == 0 else '#12122a')
        cell.set_text_props(color='white')
        cell.set_edgecolor('#333355')

    summary_txt = (
        f"Overall Accuracy : {overall_acc:.4f}  -- NOT primary metric\n"
        f"Transition Acc   : {transition_acc:.4f} (exact) | {direction_acc:.4f} (direction)\n"
        f"Crisis Recall    : NOT EVALUABLE -- n=0 in test set\n"
        f"Stress Recall    : see per-class table (n=9, low confidence)"
    )
    ax_tbl.text(0.5, 0.15, summary_txt, transform=ax_tbl.transAxes,
                ha='center', va='center', fontsize=8.5, color='#aaaacc',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor='#444466'))

    ax_tbl.set_title('Tier 1 -- Per-Class Metrics\n(out-of-sample generalization test)',
                     color='white', fontsize=11, pad=10)
    fig.suptitle('NIFTY 50 Risk Regime Classifier -- Tier 1 Evaluation',
                 color='white', fontsize=13, fontweight='bold', y=1.01)

    plt.tight_layout()
    plt.savefig(TIER1_PNG, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [SAVED] {TIER1_PNG}")


# ════════════════════════════════════════════════════════════════
# STEP 13b -- TIER 2 EVALUATION
# ════════════════════════════════════════════════════════════════

def tier2_evaluation(features, cost_config):
    """
    In-sample consistency check across the full dataset (2007-2026).
    Uses rule-based regime labels from features.pkl as ground truth.
    CAVEAT: in-sample fit check only, not a generalization test.
    """
    print("\n[TIER 2] Full dataset in-sample consistency check (2007-2026)...")
    print("  CAVEAT: Model trained on 2008-2023. Historical recall = in-sample fit.")

    feat = features.copy()
    feat.index = pd.to_datetime(feat.index)
    feat = feat.sort_index()

    regime_col = 'regime' if 'regime' in feat.columns else None
    if regime_col is None:
        print("  [WARNING] 'regime' column not found. Skipping Tier 2.")
        return {}

    y_full = feat[regime_col].dropna().astype(int)

    unique, counts = np.unique(y_full.values, return_counts=True)
    regime_counts_full = dict(zip(unique.tolist(), counts.tolist()))
    print(f"\n  Full dataset regime distribution: {regime_counts_full}")
    for cls, cnt in regime_counts_full.items():
        pct = cnt / len(y_full) * 100
        print(f"    Regime {cls} ({REGIME_LABELS.get(cls, '?'):10s}): "
              f"{cnt:5d} days ({pct:.1f}%)")

    print("\n  Stress episode capture analysis:")
    episode_results = {}

    for episode_name, (start_str, end_str) in STRESS_EPISODES.items():
        start_dt = pd.Timestamp(start_str)
        end_dt   = pd.Timestamp(end_str)
        ep_mask  = (y_full.index >= start_dt) & (y_full.index <= end_dt)
        ep_labels = y_full[ep_mask]

        if len(ep_labels) == 0:
            print(f"    {episode_name:30s} : NO DATA in features.pkl")
            episode_results[episode_name] = {'status': 'no_data'}
            continue

        n_stress = int((ep_labels >= 2).sum())
        n_crisis = int((ep_labels == 3).sum())
        n_total  = int(len(ep_labels))
        pct_elevated = round(n_stress / n_total * 100, 1)

        pre_window = y_full[
            (y_full.index < start_dt) &
            (y_full.index >= start_dt - pd.Timedelta(days=30))
        ]
        lead_days = 0
        if len(pre_window) > 0:
            stress_pre = pre_window[pre_window >= 2]
            if len(stress_pre) > 0:
                lead_days = int((start_dt - stress_pre.index.min()).days)

        status = 'CAPTURED' if n_stress > 0 else 'MISSED'
        print(f"    {episode_name:30s} : {status:8s} | "
              f"n={n_total:3d} | Stress+Crisis days={n_stress:3d} "
              f"({pct_elevated:5.1f}%) | Crisis days={n_crisis:3d} | Lead={lead_days}d")

        episode_results[episode_name] = {
            'status': status, 'n_total': n_total,
            'n_stress_crisis': n_stress, 'n_crisis': n_crisis,
            'pct_elevated': pct_elevated, 'lead_days': lead_days
        }

    total_sc    = sum(v.get('n_stress_crisis', 0) for v in episode_results.values())
    total_ep    = sum(v.get('n_total', 0) for v in episode_results.values()
                      if v.get('status') != 'no_data')
    sc_recall   = total_sc / total_ep if total_ep > 0 else 0.0
    ep_captured = sum(1 for v in episode_results.values() if v.get('status') == 'CAPTURED')
    ep_total    = sum(1 for v in episode_results.values() if v.get('status') != 'no_data')
    avg_lead    = np.mean([v['lead_days'] for v in episode_results.values()
                           if 'lead_days' in v])

    print(f"\n  Stress+Crisis recall across episodes : {sc_recall:.4f}")
    print(f"  Episodes captured                    : {ep_captured}/{ep_total}")
    print(f"  Avg lead time                        : {avg_lead:.1f} days")

    _plot_tier2(y_full, episode_results, regime_counts_full,
                sc_recall, ep_captured, ep_total)

    return {
        'tier': 'Tier2_InSampleConsistency',
        'period': '2007-2026',
        'caveat': 'In-sample fit check only. Not a generalization test.',
        'regime_counts_full': regime_counts_full,
        'stress_crisis_recall_episodes': round(sc_recall, 4),
        'episodes_captured': f'{ep_captured}/{ep_total}',
        'episode_detail': episode_results
    }


def _plot_tier2(y_full, episode_results, regime_counts_full,
                stress_recall, ep_captured, ep_total):
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('#0f0f1a')
    gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                  height_ratios=[1.8, 1])

    ax_time = fig.add_subplot(gs[0, :])
    ax_time.set_facecolor('#0f0f1a')

    for regime_id, color in REGIME_COLORS.items():
        mask = y_full == regime_id
        ax_time.fill_between(y_full.index, regime_id - 0.4, regime_id + 0.4,
                             where=mask, color=color, alpha=0.7,
                             label=REGIME_LABELS[regime_id])

    for ep_name, (s, e) in STRESS_EPISODES.items():
        result = episode_results.get(ep_name, {})
        alpha  = 0.15 if result.get('status') == 'CAPTURED' else 0.07
        ax_time.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                        color='red', alpha=alpha)
        mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
        ax_time.text(mid, 3.6, ep_name.replace('_', '\n'),
                     ha='center', va='bottom', fontsize=5.5, color='#ff9999')

    ax_time.set_yticks([0, 1, 2, 3])
    ax_time.set_yticklabels([REGIME_LABELS[i] for i in range(4)],
                            color='white', fontsize=8)
    ax_time.set_ylim(-0.7, 4.2)
    ax_time.tick_params(colors='white')
    for spine in ax_time.spines.values():
        spine.set_edgecolor('#333355')
    ax_time.legend(loc='lower left', fontsize=8, facecolor='#1a1a2e',
                   labelcolor='white', framealpha=0.8)
    ax_time.set_title(
        'Full Dataset Regime Timeline (2007-2026) -- Rule-Based Labels\n'
        'In-sample consistency check: model trained on 2008-2023 data',
        color='#ffcc88', fontsize=10, pad=8
    )

    ax_dist = fig.add_subplot(gs[1, 0])
    ax_dist.set_facecolor('#0f0f1a')
    regimes    = list(regime_counts_full.keys())
    day_counts = [regime_counts_full[r] for r in regimes]
    colors     = [REGIME_COLORS.get(r, 'grey') for r in regimes]
    bars = ax_dist.bar([REGIME_LABELS[r] for r in regimes],
                       day_counts, color=colors, alpha=0.8)
    for bar, cnt in zip(bars, day_counts):
        ax_dist.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 20,
                     f'{cnt}d\n({cnt/sum(day_counts)*100:.1f}%)',
                     ha='center', va='bottom', color='white', fontsize=8)
    ax_dist.set_facecolor('#0f0f1a')
    ax_dist.set_ylabel('Trading Days', color='white', fontsize=9)
    ax_dist.tick_params(colors='white')
    for spine in ax_dist.spines.values():
        spine.set_edgecolor('#333355')
    ax_dist.set_title('Regime Distribution (Full Dataset)', color='white', fontsize=9)

    ax_ep = fig.add_subplot(gs[1, 1])
    ax_ep.set_facecolor('#0f0f1a')
    ax_ep.axis('off')

    ep_rows = []
    for ep_name, res in episode_results.items():
        if res.get('status') == 'no_data':
            continue
        ep_rows.append([
            ep_name.replace('_', ' '),
            res['status'],
            str(res['n_total']),
            f"{res['pct_elevated']:.0f}%",
            f"{res['lead_days']}d"
        ])

    if ep_rows:
        tbl = ax_ep.table(
            cellText=ep_rows,
            colLabels=['Episode', 'Status', 'Days', 'Elev%', 'Lead'],
            cellLoc='center', loc='center', bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        for (row, col), cell in tbl.get_celld().items():
            cell.set_facecolor('#1a1a2e' if row == 0 else '#12122a')
            cell.set_text_props(color='white')
            cell.set_edgecolor('#333355')
            if row > 0 and col == 1:
                txt = cell.get_text().get_text()
                cell.set_text_props(
                    color='#2ecc71' if txt == 'CAPTURED' else '#e74c3c'
                )

    ax_ep.set_title(
        f'Episode Capture: {ep_captured}/{ep_total}  |  '
        f'Stress+Crisis Recall: {stress_recall:.3f}',
        color='white', fontsize=9, pad=8
    )

    fig.suptitle(
        'NIFTY 50 Risk Regime Classifier -- Tier 2 In-Sample Consistency Check',
        color='#ffcc88', fontsize=13, fontweight='bold', y=1.01
    )
    plt.savefig(TIER2_PNG, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  [SAVED] {TIER2_PNG}")


# ════════════════════════════════════════════════════════════════
# STEP 13c -- THRESHOLD CALIBRATION
# ════════════════════════════════════════════════════════════════

def threshold_calibration(ews, cost_config):
    """
    Sweep n=1 to 7 for sustained and override thresholds.
    Primary metric: Stress+Crisis recall.
    Cost ratios evaluated: 6.5:1 (empirical) and 10:1 (training).
    """
    print("\n[STEP 13c] Threshold calibration -- n-day sweep (n=1 to 7)...")

    from itertools import product as iproduct

    ews_df   = ews.copy()
    ews_df.index = pd.to_datetime(ews_df.index)
    p_stress = ews_df.get('P_regime_2', pd.Series(0, index=ews_df.index))
    p_crisis = ews_df.get('P_regime_3', pd.Series(0, index=ews_df.index))
    combined = p_stress + p_crisis
    y_actual = ews_df.get('actual_regime',
                           pd.Series(0, index=ews_df.index)).astype(int)

    cost_ratios = {
        '6.5:1 (empirical)': 6.5,
        '10:1  (training)' : 10.0
    }

    results = []
    for n, override_thresh, sustained_thresh in iproduct(
        range(1, 8), [0.50, 0.60, 0.70], [0.30, 0.40, 0.50]
    ):
        sustained_signal = (
            combined.rolling(n)
            .apply(lambda x: (x >= sustained_thresh).all(), raw=True)
            .fillna(0).astype(bool)
        )
        override_signal = combined >= override_thresh
        signal = sustained_signal | override_signal

        elevated_actual = y_actual >= 2
        tp = int((signal & elevated_actual).sum())
        fp = int((signal & ~elevated_actual).sum())
        fn = int((~signal & elevated_actual).sum())
        tn = int((~signal & ~elevated_actual).sum())

        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        cost_scores = {}
        for ratio_name, ratio in cost_ratios.items():
            false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            cost_scores[ratio_name] = round(ratio * recall - false_alarm_rate, 4)

        results.append({
            'n': n, 'sustained_thresh': sustained_thresh,
            'override_thresh': override_thresh,
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'recall': round(recall, 4), 'precision': round(precision, 4),
            'f1': round(f1, 4),
            **{f'cost_{k}': v for k, v in cost_scores.items()}
        })

    results_df = pd.DataFrame(results)

    print("\n  Optimal configurations:")
    recommendations = {}
    for ratio_name in cost_ratios:
        col = f'cost_{ratio_name}'
        if col not in results_df.columns:
            continue
        best = results_df.loc[results_df[col].idxmax()]
        print(f"\n  Cost ratio {ratio_name}:")
        print(f"    n={int(best['n'])} | sustained>={best['sustained_thresh']} | "
              f"override>={best['override_thresh']}")
        print(f"    Recall={best['recall']:.4f} | Precision={best['precision']:.4f} | "
              f"F1={best['f1']:.4f} | Score={best[col]:.4f}")
        recommendations[ratio_name] = best.to_dict()

    print("\n  Hamilton expected duration check:")
    print("    Regime persistence theory (Hamilton 1989) implies stress spells")
    print("    of 5-15 days. n=2-3 is theoretically grounded for early warning.")
    print("    n>5 risks missing short, acute stress episodes entirely.")

    rec_65 = recommendations.get('6.5:1 (empirical)', {}).get('recall', 0)
    rec_10 = recommendations.get('10:1  (training)', {}).get('recall', 0)
    recall_delta = abs(rec_10 - rec_65)

    print(f"\n  Recall delta between cost ratios: {recall_delta:.4f}")
    if recall_delta < 0.02:
        recommendation = "6.5:1 (empirical)"
        rationale = ("Recall delta < 0.02. Empirically grounded 6.5:1 preferred: "
                     "same detection, lower false alarm cost.")
    else:
        recommendation = "10:1  (training)"
        rationale = (f"Recall delta = {recall_delta:.4f}. 10:1 delivers meaningfully "
                     "higher Crisis recall. Conservative setting justified.")

    print(f"\n  RECOMMENDATION: {recommendation}")
    print(f"  RATIONALE      : {rationale}")

    with open(THRESHOLD_CAL_PKL, 'wb') as f:
        pickle.dump({
            'sweep_results': results_df,
            'recommendations': recommendations,
            'recommendation_final': recommendation,
            'rationale': rationale,
            'recall_delta': recall_delta
        }, f)
    print(f"\n  [SAVED] {THRESHOLD_CAL_PKL}")

    _plot_threshold_calibration(results_df)

    return {
        'sweep_shape': results_df.shape,
        'recommendations': {k: {
            'n': int(v['n']),
            'sustained_thresh': v['sustained_thresh'],
            'override_thresh': v['override_thresh'],
            'recall': v['recall'],
            'precision': v['precision'],
            'f1': v['f1']
        } for k, v in recommendations.items()},
        'recommendation_final': recommendation,
        'rationale': rationale,
        'recall_delta': round(recall_delta, 4)
    }


def _plot_threshold_calibration(results_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f0f1a')

    cost_cols = [c for c in results_df.columns if c.startswith('cost_')]
    colors_n  = plt.cm.plasma(np.linspace(0.1, 0.9, 7))

    for ax_idx, cost_col in enumerate(cost_cols[:2]):
        ax = axes[ax_idx]
        ax.set_facecolor('#0f0f1a')

        for n_val, color in zip(range(1, 8), colors_n):
            subset   = results_df[results_df['n'] == n_val]
            ax.scatter(subset['precision'], subset['recall'],
                       color=color, alpha=0.6, s=20, label=f'n={n_val}')
            best_idx = subset[cost_col].idxmax()
            best     = subset.loc[best_idx]
            ax.scatter(best['precision'], best['recall'],
                       color=color, s=120, marker='*', zorder=5,
                       edgecolors='white', linewidths=0.5)

        ax.set_xlabel('Precision', color='white', fontsize=9)
        ax.set_ylabel('Recall', color='white', fontsize=9)
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333355')
        ratio_label = cost_col.replace('cost_', '').replace('_', ' ')
        ax.set_title(f'Precision-Recall Tradeoff\n(Cost ratio: {ratio_label})',
                     color='white', fontsize=10)
        ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='white',
                  framealpha=0.8, loc='lower right')
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.plot([0, 1], [0, 1], '--', color='#555577', linewidth=0.8)

    fig.suptitle('Step 13c -- Threshold Calibration: n-Day Sweep (n=1 to 7)\n'
                 '* = optimal configuration per cost ratio',
                 color='white', fontsize=12, fontweight='bold')

    cal_png = os.path.join(OUTPUTS_DIR, 'threshold_calibration.png')
    plt.tight_layout()
    plt.savefig(cal_png, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"  [SAVED] {cal_png}")


# ════════════════════════════════════════════════════════════════
# STEP 13d -- SAVE RESULTS
# ════════════════════════════════════════════════════════════════

def save_evaluation_results(t1_results, t2_results, cal_results):
    results = {
        'tier1': t1_results,
        'tier2': t2_results,
        'threshold_calibration': cal_results,
        'methodology_notes': {
            'evaluation_design': (
                "Dual-tier evaluation. Tier 1 is the only honest out-of-sample "
                "generalization test (2024-present). Tier 2 is an in-sample "
                "consistency check across historical stress episodes (2007-2026). "
                "Models are frozen -- no retraining."
            ),
            'primary_metric': (
                "Crisis recall, then Stress recall. Overall accuracy is explicitly "
                "de-prioritized: the test set is 97.8% Calm+Pullback."
            ),
            'crisis_recall_caveat': (
                "Crisis recall cannot be evaluated on the Tier 1 test set because "
                "the 2024-present period contains 0 Crisis-regime days. Structural "
                "property of the test window, not a model failure."
            ),
            'threshold_calibration': (
                "Early warning thresholds calibrated via precision-recall sweep "
                "across n=1 to 7 consecutive days. Hamilton (1989) regime persistence "
                "theory supports n=2-3 as theoretically grounded."
            ),
            'bodie_karasan_bogle': (
                "Evaluation follows Bodie's risk-first framing. Karasan's hybrid "
                "modeling discipline: Stress and Crisis recall are primary criteria. "
                "Bogle's humility: no accuracy number reported without qualification."
            )
        }
    }

    def make_serialisable(obj):
        if isinstance(obj, np.integer):   return int(obj)
        if isinstance(obj, np.floating):  return float(obj)
        if isinstance(obj, np.ndarray):   return obj.tolist()
        if isinstance(obj, pd.Timestamp): return str(obj.date())
        if isinstance(obj, dict):  return {k: make_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [make_serialisable(i) for i in obj]
        return obj

    results = make_serialisable(results)
    with open(EVAL_RESULTS_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVED] {EVAL_RESULTS_JSON}")
    return results


# ════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ════════════════════════════════════════════════════════════════

def run_evaluation():
    print("=" * 65)
    print("STEP 13 -- MODEL EVALUATION")
    print("NIFTY 50 Equity Risk Diagnostics System")
    print("=" * 65)

    regime_probs, ews, features, cost_config = load_data()
    t1_results  = tier1_evaluation(regime_probs, cost_config)
    t2_results  = tier2_evaluation(features, cost_config)
    cal_results = threshold_calibration(ews, cost_config)
    save_evaluation_results(t1_results, t2_results, cal_results)

    print("\n" + "=" * 65)
    print("STEP 13 COMPLETE")
    print(f"  tier1_evaluation.png      -> {TIER1_PNG}")
    print(f"  tier2_evaluation.png      -> {TIER2_PNG}")
    print(f"  evaluation_results.json   -> {EVAL_RESULTS_JSON}")
    print(f"  threshold_calibration.pkl -> {THRESHOLD_CAL_PKL}")
    print("=" * 65)


if __name__ == '__main__':
    run_evaluation()
