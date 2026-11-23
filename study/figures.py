#!/usr/bin/env python3
"""Figures for RESULTS.md from results/study_final/tables (python study/figures.py)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'results/study_final'
TAB = RES / 'tables'
FIG = RES / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
MARGIN = 0.03


def forest():
    c = pd.read_csv(TAB / 'contrasts.csv')
    c = c[c.family.isin(['primary', 'secondary_B'])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(c) / 2 + 1.6), sharey=True)
    for ax, mod in zip(axes, ['miRNA', 'mRNA']):
        s = c[c.modality == mod].reset_index(drop=True)
        y = np.arange(len(s))[::-1]
        ax.axvspan(-MARGIN, MARGIN, color='0.9', zorder=0)
        ax.axvline(0, color='0.4', lw=0.8)
        ax.errorbar(s.delta_macro_f1, y, xerr=[s.delta_macro_f1 - s.corrected_ci95_lo, s.corrected_ci95_hi - s.delta_macro_f1],
                    fmt='o', color='k', capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{h.split("_")[0]}: {ct.replace("single_mi_", "single_<test mod>_").replace("single_mr_", "single_<test mod>_")}'
                            for h, ct in zip(s.hypothesis, s.contrast)], fontsize=8)
        ax.set_title(f'{mod} test examples')
        ax.set_xlabel('Δ macro-F1 (corrected 95% CI); grey = ±0.03 margin')
    fig.tight_layout()
    fig.savefig(FIG / 'fig1_hypotheses_forest.png', dpi=160)


def arms():
    s = pd.read_csv(TAB / 'arm_summary.csv')
    order = s.groupby('arm').macro_f1.mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8, 0.33 * len(order) + 1.5))
    for j, (mod, col) in enumerate((('miRNA', 'C0'), ('mRNA', 'C1'))):
        d = s[s.modality == mod].set_index('arm').reindex(order)
        y = np.arange(len(order)) + (j - 0.5) * 0.3
        ax.errorbar(d.macro_f1, y, xerr=[d.macro_f1 - d.cluster_boot_ci95_lo, d.cluster_boot_ci95_hi - d.macro_f1], fmt='o', color=col, label=mod, capsize=2)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlabel('macro-F1, 5x5 grouped CV (patient-cluster bootstrap 95% CI)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / 'fig2_arms_and_baselines.png', dpi=160)


def per_class():
    c = pd.read_csv(TAB / 'contrasts.csv')
    c = c[c.family.isin(['primary', 'secondary_B'])]
    cols = [x for x in c.columns if x.startswith('delta_f1_')]
    M = c[cols].values
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(c) + 1.5))
    v = np.nanmax(np.abs(M))
    im = ax.imshow(M, cmap='RdBu', vmin=-v, vmax=v, aspect='auto')
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([x.replace('delta_f1_', '') for x in cols])
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels([f'{h.split("_")[0]} {m}' for h, m in zip(c.hypothesis, c.modality)], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:+.3f}', ha='center', va='center', fontsize=7)
    fig.colorbar(im, label='Δ per-class F1')
    fig.tight_layout()
    fig.savefig(FIG / 'fig3_per_class_gain.png', dpi=160)


def ladder():
    f = TAB / 'shortcut_ladder.csv'
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for ax, mod in zip(axes, ['miRNA', 'mRNA']):
        d = s[s.modality == mod].sort_values('auroc_met_vs_primary')
        ax.barh(d.feature_set, d.auroc_met_vs_primary, xerr=d.auroc_sd_repeats.fillna(0), color=['C3' if x.startswith('model:') else 'C7' for x in d.feature_set])
        ax.axvline(0.5, color='k', lw=0.8)
        ax.set_xlim(0.4, 1.0)
        ax.set_title(f'{mod}: LN metastasis vs primary')
        ax.set_xlabel('AUROC (mean over repeats)')
    fig.tight_layout()
    fig.savefig(FIG / 'fig4_shortcut_ladder.png', dpi=160)


def learning():
    f = TAB / 'learning_curve.csv'
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True)
    for ax, mod in zip(axes, ['miRNA', 'mRNA']):
        for m, d in s[s.modality == mod].groupby('model'):
            d = d.sort_values('fraction')
            ax.errorbar(d.fraction * 100, d.macro_f1, yerr=d.macro_f1_sd_runs.fillna(0), marker='o', label=m, capsize=2)
        ax.set_xlabel('% of inner-fit training groups (repeat 0)')
        ax.set_title(mod)
    axes[0].set_ylabel('macro-F1')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig5_learning_curve.png', dpi=160)


def positive():
    f = TAB / 'positive_control.csv'
    if not f.exists():
        return
    s = pd.read_csv(f)
    d = s[s.row.str.contains(' - ')]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    lab = [f'{a} | {b} | {c.replace("joint_", "")}' for a, b, c in zip(d.scheme, d.modality, d.row)]
    y = np.arange(len(d))
    ax.errorbar(d.macro_f1, y, xerr=[d.macro_f1 - d.ci95_lo, d.ci95_hi - d.macro_f1], fmt='o', color='k', capsize=3)
    ax.axvline(0, color='0.4', lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(lab, fontsize=7)
    ax.set_xlabel('Δ macro-F1 (corrected 95% CI), planted signal along miRDB edges')
    fig.tight_layout()
    fig.savefig(FIG / 'fig6_positive_control.png', dpi=160)


def reliability():
    f = TAB / 'reliability.csv'
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, mod in zip(axes, ['miRNA', 'mRNA']):
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
        for a, d in s[s.modality == mod].groupby('arm'):
            ax.plot(d.confidence, d.accuracy, marker='o', label=a)
        ax.set_title(f'{mod}: reliability (top-label, pooled CV)')
        ax.set_xlabel('confidence')
    axes[0].set_ylabel('accuracy')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig7_reliability.png', dpi=160)


def zeroshot():
    f = TAB / 'zero_shot_transfer.csv'
    if not f.exists():
        return
    s = pd.read_csv(f)
    d = s[~s.edges.str.contains(' - ')]
    fig, ax = plt.subplots(figsize=(6, 3))
    lab = [f'{a}→{b}: {e}' for a, b, e in zip(d.trained_on, d.tested_on, d.edges)]
    ax.barh(lab, d.macro_f1, color='C2')
    ax.axvline(1 / 3, color='k', lw=0.8, ls=':')
    ax.set_xlabel('macro-F1 on the unseen modality (zero-shot)')
    fig.tight_layout()
    fig.savefig(FIG / 'fig8_zero_shot.png', dpi=160)


def edgeval():
    f = RES / 'edge_validation/split_half_replication.csv'
    if not f.exists():
        return
    s = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.hist(s.replication_random, bins=20, alpha=0.7, label='random pairs')
    ax.hist(s.replication_topK, bins=20, alpha=0.7, label='top-K TCGA pairs (half A)')
    ax.set_xlabel('replication rate in half B (r<0, one-sided p<0.05)')
    ax.set_ylabel('splits (of 100)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'fig9_edge_split_half.png', dpi=160)


if __name__ == '__main__':
    for f in (forest, arms, per_class, ladder, learning, positive, reliability, zeroshot, edgeval):
        try:
            f()
            print('ok', f.__name__)
        except Exception as e:
            print('skip', f.__name__, type(e).__name__, e)
