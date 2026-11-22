#!/usr/bin/env python3
"""TCGA-BRCA TNBC, patient-matched miRNA + mRNA: networks from TRUE pairs, multi-network GAT vs
baselines. Usage: 01_tcga_matched_gat.py <cohort_dir> <out_dir> <task> [n_seeds] [n_rewire]

task = tumor_vs_normal : TNBC primary tumor (01) vs solid tissue normal (11), patient-grouped.
task = node            : TNBC primary tumors, pathologic N0 vs N1-3 (metastatic-spread proxy).

Protocol (no tuning): fixed patient-grouped stratified 75/25 train/test split; a further
patient-grouped 20% of train is the early-stopping validation set. Feature selection (Welch t on
the train split only), z-scoring and network construction use training samples only. Features
are restricted to molecules on the GEO GSE45498 NanoString panels so the same nodes exist in the
replication cohort. GAT hyper-parameters are the repo's fixed GAT_FIX_PARAMS.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_common import (REPO, build_network, cluster_bootstrap_ci, edges_to_tensor, geo_panels,
                         map_geo_mirna_to_tcga, metrics, mimat_to_name, rewire, train_gat,
                         welch_select)

torch.set_num_threads(int(__import__('os').environ.get('TORCH_THREADS', '1')))
COH, OUT, TASK = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
N_SEEDS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
N_REWIRE = int(sys.argv[5]) if len(sys.argv) > 5 else 50
K_MIR, K_MR = 60, 60
OUT.mkdir(parents=True, exist_ok=True)

meta = pd.read_csv(COH / 'tcga_tnbc_meta.csv', index_col=0)
mir = pd.read_csv(COH / 'tcga_tnbc_mirna.csv.gz', index_col=0)
mr = pd.read_csv(COH / 'tcga_tnbc_mrna.csv.gz', index_col=0)
names = mimat_to_name(sys.argv[6] if len(sys.argv) > 6 else str(Path.home() / '.capy/work/mirbase/mature.fa'))
mir = mir.rename(columns=names)
mir = mir.loc[:, ~mir.columns.duplicated()]

geo_mir, geo_mr = geo_panels()
mir_map = map_geo_mirna_to_tcga(geo_mir, mir)
mir_cols = sorted(set(mir_map.values()))
mr_cols = sorted(set(geo_mr) & set(mr.columns))
mir, mr = mir[mir_cols], mr[mr_cols]
pd.Series(mir_map, name='tcga_mirna').to_csv(OUT / 'geo_to_tcga_mirna_map.csv')

if TASK == 'tumor_vs_normal':
    keep = meta.tissue.isin(['tumor', 'normal'])
    y_all = (meta.tissue == 'tumor').astype(int)
    class_names = ['normal', 'tumor']
elif TASK == 'node':
    keep = (meta.tissue == 'tumor') & meta.node.isin(['N0', 'Npos'])
    y_all = (meta.node == 'Npos').astype(int)
    class_names = ['N0', 'Npos']
else:
    sys.exit(f'unknown task {TASK!r}; use tumor_vs_normal or node')
ids = meta.index[keep]
y = y_all[ids].values
groups = meta.loc[ids, 'patient'].values

sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
tr_i, te_i = next(sgk.split(ids, y, groups))
sgk2 = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fit_j, va_j = next(sgk2.split(ids[tr_i], y[tr_i], groups[tr_i]))
fit_i, va_i = tr_i[fit_j], tr_i[va_j]
assert not set(groups[tr_i]) & set(groups[te_i])

Xmir, Xmr = mir.loc[ids], mr.loc[ids]
sel_mir, de_mir = welch_select(Xmir.iloc[tr_i], y[tr_i], K_MIR)
sel_mr, de_mr = welch_select(Xmr.iloc[tr_i], y[tr_i], K_MR)
pd.concat([de_mir.assign(modality='miRNA'), de_mr.assign(modality='mRNA')]).to_csv(OUT / 'train_split_de.csv')
nodes = sel_mir + sel_mr
node_types = torch.tensor([0] * len(sel_mir) + [1] * len(sel_mr))
X = pd.concat([Xmir[sel_mir], Xmr[sel_mr]], axis=1)
mu, sd = X.iloc[tr_i].mean(), X.iloc[tr_i].std().replace(0, 1)
Z = ((X - mu) / sd).values.astype(np.float32)

net_edges, net_stats = {}, {}
for c, cname in enumerate(class_names):
    rows = tr_i[y[tr_i] == c]
    e, s = build_network(X.iloc[rows][sel_mir], X.iloc[rows][sel_mr])
    e.to_csv(OUT / f'network_{cname}_edges.csv', index=False)
    net_edges[cname] = e
    net_stats[cname] = s | {'n_train_samples': int(len(rows))} | e.edge_type.value_counts().to_dict()

def nets_from(edge_dict):
    return {k: edges_to_tensor(v, nodes) for k, v in edge_dict.items()}

empty = {k: (torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))) for k in class_names}
real = nets_from(net_edges)
Xfit, yfit, Xva, yva, Xte, yte = Z[fit_i], y[fit_i], Z[va_i], y[va_i], Z[te_i], y[te_i]
gte = groups[te_i]

rows, probs = [], {}
t0 = time.time()
for variant, nets, use_expr in (('GAT (real network)', real, True), ('GAT (edges removed)', empty, True),
                                ('GAT graph-only (real network)', real, False)):
    ps = []
    for seed in range(N_SEEDS):
        p = train_gat(Xfit, yfit, Xva, yva, Xte, nets, node_types, 2, seed, use_expr)
        ps.append(p)
        rows.append({'model': variant, 'seed': seed} | metrics(yte, p))
    probs[variant] = np.mean(ps, 0)
    print(f'{variant}: {time.time() - t0:.0f}s', flush=True)

for seed in range(N_SEEDS):
    xgb = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=seed, eval_metric='logloss')
    xgb.fit(Z[tr_i], y[tr_i])
    p = xgb.predict_proba(Xte)
    rows.append({'model': 'XGBoost (same features)', 'seed': seed} | metrics(yte, p))
    probs.setdefault('_xgb', []).append(p)
probs['XGBoost (same features)'] = np.mean(probs.pop('_xgb'), 0)
lr = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(Z[tr_i], y[tr_i])
probs['Logistic regression (L2)'] = lr.predict_proba(Xte)
rows.append({'model': 'Logistic regression (L2)', 'seed': 0} | metrics(yte, probs['Logistic regression (L2)']))
for strat in ('most_frequent', 'stratified'):
    for seed in range(N_SEEDS):
        d = DummyClassifier(strategy=strat, random_state=seed).fit(Z[tr_i], y[tr_i])
        rows.append({'model': f'Naive ({strat})', 'seed': seed} | metrics(yte, d.predict_proba(Xte)))

rew_rng = np.random.default_rng(7)
rew_scores = []
for r_i in range(N_REWIRE):
    rnets = nets_from({k: rewire(v, rew_rng) for k, v in net_edges.items()})
    p = train_gat(Xfit, yfit, Xva, yva, Xte, rnets, node_types, 2, r_i % N_SEEDS)
    rew_scores.append(metrics(yte, p) | {'rewire': r_i})
rew = pd.DataFrame(rew_scores)
rew.to_csv(OUT / 'rewired_null.csv', index=False)
print(f'rewiring done: {time.time() - t0:.0f}s', flush=True)

per_seed = pd.DataFrame(rows)
per_seed.to_csv(OUT / 'per_seed_metrics.csv', index=False)
summ = per_seed.groupby('model').agg(macro_f1_mean=('macro_f1', 'mean'), macro_f1_sd=('macro_f1', 'std'),
                                     auroc_mean=('auroc', 'mean'), auroc_sd=('auroc', 'std'),
                                     acc_mean=('accuracy', 'mean'), n_runs=('seed', 'size'))
for m, p in probs.items():
    summ.loc[m, 'ensemble_macro_f1'] = metrics(yte, p)['macro_f1']
    summ.loc[m, 'ensemble_auroc'] = metrics(yte, p)['auroc']
    lo, hi = cluster_bootstrap_ci(yte, p, gte)
    summ.loc[m, 'f1_ci_lo'], summ.loc[m, 'f1_ci_hi'] = lo, hi
    lo, hi = cluster_bootstrap_ci(yte, p, gte, key='auroc')
    summ.loc[m, 'auroc_ci_lo'], summ.loc[m, 'auroc_ci_hi'] = lo, hi
real_seed_mean = per_seed[per_seed.model == 'GAT (real network)'].macro_f1.mean()
real_auc_mean = per_seed[per_seed.model == 'GAT (real network)'].auroc.mean()
# The null is single-seed per rewiring; compare to the per-seed real distribution's mean.
perm = {'rewired_f1_mean': rew.macro_f1.mean(), 'rewired_f1_sd': rew.macro_f1.std(),
        'p_f1_real_vs_rewired': float((1 + (rew.macro_f1 >= real_seed_mean).sum()) / (1 + len(rew))),
        'rewired_auroc_mean': rew.auroc.mean(),
        'p_auroc_real_vs_rewired': float((1 + (rew.auroc >= real_auc_mean).sum()) / (1 + len(rew))),
        'n_rewire': len(rew)}
summ.to_csv(OUT / 'summary.csv')
info = {'task': TASK, 'n_samples': int(len(ids)), 'n_patients': int(len(set(groups))),
        'class_counts': {class_names[c]: int((y == c).sum()) for c in range(2)},
        'n_train': int(len(tr_i)), 'n_fit': int(len(fit_i)), 'n_val': int(len(va_i)), 'n_test': int(len(te_i)),
        'test_class_counts': {class_names[c]: int((yte == c).sum()) for c in range(2)},
        'geo_panel_mirna_mapped': len(mir_cols), 'geo_panel_mrna_in_tcga': len(mr_cols),
        'nodes': {'miRNA': len(sel_mir), 'mRNA': len(sel_mr)},
        'de_train_fdr05': {'miRNA': int((de_mir.q < 0.05).sum()), 'mRNA': int((de_mr.q < 0.05).sum())},
        'networks': net_stats, 'permutation': perm, 'seconds': time.time() - t0}
json.dump(info, open(OUT / 'run_info.json', 'w'), indent=2, default=float)
pd.DataFrame({'sample': ids[te_i], 'patient': gte, 'y': yte} | {m: p[:, 1] for m, p in probs.items()}).to_csv(OUT / 'test_predictions.csv', index=False)
print(json.dumps(info, indent=1, default=float))
print(summ.round(3).to_string())
