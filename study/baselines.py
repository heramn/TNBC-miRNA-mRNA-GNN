#!/usr/bin/env python3
"""Non-GNN baselines on the same outer folds, features and inner split as the GAT arms.

Every model gets the same budget: 16 configurations (the first is a conventional default, the rest
drawn with a fixed RNG), each fit on the inner-fit split and scored by class-weighted log-loss on
the inner-validation split; the best configuration is then run with the remaining seeds. No model
is refit on the full outer-training fold (the GAT cannot be, because it early-stops on that split).

  python study/baselines.py main      # XGBoost (joint NaN, routed), elastic net (routed, joint), target-set EN, naive
  python study/baselines.py shortcut  # met-vs-primary shortcut ladder
  python study/baselines.py export    # fold matrices for tabpfn_baseline.py (separate venv)
"""
import itertools
import os
import sys
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import prep as P
from run_gat import RUNS

warnings.filterwarnings('ignore')


def xgb_configs():
    grid = list(itertools.product([2, 3, 4, 6], [100, 200, 400], [0.03, 0.1, 0.3], [0.7, 0.9], [0.5, 0.8], [1, 3]))
    first = (3, 200, 0.1, 0.9, 0.8, 1)
    grid = [g for g in grid if g != first]
    pick = np.random.default_rng(23456).choice(len(grid), C.N_CONFIGS - 1, replace=False)
    keys = ['max_depth', 'n_estimators', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight']
    return [dict(zip(keys, first))] + [dict(zip(keys, grid[i])) for i in pick]


def enet_configs():
    rng = np.random.default_rng(34567)
    out = [dict(C=0.1, l1_ratio=0.5)]
    for _ in range(C.N_CONFIGS - 1):
        out.append(dict(C=float(10 ** rng.uniform(-3, 1)), l1_ratio=float(rng.choice([0.0, 0.1, 0.5, 0.9, 1.0]))))
    return out


def class_weights(y):
    cnt = np.bincount(y, minlength=3).astype(float)
    return len(y) / (3 * np.maximum(cnt, 1))


def wlogloss(y, p, w):
    p = np.clip(p, 1e-7, 1)
    return float(np.sum(w[y] * -np.log(p[np.arange(len(y)), y])) / np.sum(w[y]))


def make_model(kind, cfg, seed):
    if kind == 'xgb':
        return XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=seed, n_jobs=1,
                             verbosity=0, **cfg)
    return LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000, class_weight='balanced',
                              C=cfg['C'], l1_ratio=cfg['l1_ratio'], random_state=seed)


def fit_proba(kind, cfg, seed, Xf, yf, Xo):
    m = make_model(kind, cfg, seed)
    if kind == 'xgb':
        m.fit(Xf, yf, sample_weight=class_weights(yf)[yf])
    else:
        m.fit(Xf, yf)
    p = np.zeros((len(Xo), 3))
    p[:, m.classes_] = m.predict_proba(Xo)
    return p


def features(fd, ids, node_set, mode):
    """mode: 'joint_nan' (both modalities' columns, unobserved = NaN), 'joint_zero' (0 + modality
    indicator), 'mi' / 'mr' (one modality), 'mi_ts' / 'mr_ts' (one modality + miRDB target-set means)."""
    t = fd['samples']
    mi, mr = fd['nodes'][node_set]
    if mode in ('mi', 'mr', 'mi_ts', 'mr_ts'):
        m = 0 if mode.startswith('mi') else 1
        X = fd['Z'][m].loc[ids, mi if m == 0 else mr]
        if mode.endswith('_ts'):
            md = P.mirdb()
            md = md[md.probe.isin(mi) & md.gene.isin(mr)]
            if m == 1:
                agg = {f'ts_{p}': X[g.gene.tolist()].mean(1) for p, g in md.groupby('probe')}
            else:
                agg = {f'rs_{g}': X[s.probe.tolist()].mean(1) for g, s in md.groupby('gene')}
            X = pd.concat([X, pd.DataFrame(agg, index=X.index)], axis=1)
        return X.values
    X = np.full((len(ids), len(mi) + len(mr)), np.nan, np.float32)
    mods = t.loc[ids, 'modality'].values
    for m, cols, sl in ((0, mi, slice(0, len(mi))), (1, mr, slice(len(mi), None))):
        sel = np.where(mods == m)[0]
        X[sel, sl] = fd['Z'][m].loc[[ids[i] for i in sel], cols].values
    if mode == 'joint_zero':
        X = np.column_stack([np.nan_to_num(X), mods])
    return X


MODELS = {
    # name: (estimator, feature mode, node set, trained/test modalities, seeds)
    'xgb_joint': ('xgb', 'joint_nan', 'subset', (0, 1), C.SEEDS_SECONDARY),
    'xgb_joint_full': ('xgb', 'joint_nan', 'full', (0, 1), C.SEEDS_SECONDARY),
    'xgb_routed_mi': ('xgb', 'mi', 'subset', (0,), C.SEEDS_SECONDARY),
    'xgb_routed_mr': ('xgb', 'mr', 'subset', (1,), C.SEEDS_SECONDARY),
    'enet_routed_mi': ('enet', 'mi', 'subset', (0,), 1),
    'enet_routed_mr': ('enet', 'mr', 'subset', (1,), 1),
    'enet_joint': ('enet', 'joint_zero', 'subset', (0, 1), 1),
    'enet_targetset_mi': ('enet', 'mi_ts', 'subset', (0,), 1),
    'enet_targetset_mr': ('enet', 'mr_ts', 'subset', (1,), 1),
}


def run_model_fold(args):
    name, r, k, fit_ids = args[0], args[1], args[2], (args[3] if len(args) > 3 else None)
    est, mode, ns, mods, n_seeds = MODELS[name]
    tag = '' if fit_ids is None else args[4]
    out = RUNS / ('main' if not tag else tag) / name / f'r{r}_f{k}.csv'
    if out.exists():
        return out
    fd = P.prep('main', r, k)
    t = fd['samples']
    sel = lambda ids: [i for i in ids if t.at[i, 'modality'] in mods]
    fit, val, test = sel(fit_ids or fd['fit']), sel(fd['val']), sel(fd['test'])
    Xf, Xv, Xt = (features(fd, ids, ns, mode) for ids in (fit, val, test))
    yf, yv = t.loc[fit, 'y'].values, t.loc[val, 'y'].values
    w = class_weights(yf)
    cfgs = xgb_configs() if est == 'xgb' else enet_configs()
    rows = []
    vls = []
    for c, cfg in enumerate(cfgs):
        p = fit_proba(est, cfg, 0, Xf, yf, np.vstack([Xv, Xt]))
        vls.append(wlogloss(yv, p[:len(val)], w))
        rows.append((c, 0, vls[-1], p[len(val):]))
    best = int(np.argmin(vls))
    for s in range(1, n_seeds):
        rows.append((best, s, vls[best], fit_proba(est, cfgs[best], s, Xf, yf, Xt)))
    dfs = []
    for c, s, vl, p in rows:
        dfs.append(pd.DataFrame({'id': test, 'modality': t.loc[test, 'modality'].values, 'y': t.loc[test, 'y'].values,
                                 'group': t.loc[test, 'group'].values, 'p0': p[:, 0], 'p1': p[:, 1], 'p2': p[:, 2],
                                 'scheme': tag or 'main', 'arm': name, 'repeat': r, 'fold': k, 'config': c, 'seed': s,
                                 'val_loss': vl, 'selected': c == best}))
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(dfs).to_csv(out, index=False, float_format='%.6g')
    return out


def naive(r, k):
    fd = P.prep('main', r, k)
    t = fd['samples']
    rows = []
    for m in (0, 1):
        fit = [i for i in fd['fit'] if t.at[i, 'modality'] == m]
        test = [i for i in fd['test'] if t.at[i, 'modality'] == m]
        prior = np.bincount(t.loc[fit, 'y'], minlength=3) / len(fit)
        maj = np.eye(3)[prior.argmax()] * 0.98 + 0.01
        for name, p in (('naive_prior', prior), ('naive_majority', maj)):
            rows.append(pd.DataFrame({'id': test, 'modality': m, 'y': t.loc[test, 'y'].values, 'group': t.loc[test, 'group'].values,
                                      'p0': p[0], 'p1': p[1], 'p2': p[2], 'scheme': 'main', 'arm': name, 'repeat': r,
                                      'fold': k, 'config': 0, 'seed': 0, 'val_loss': np.nan, 'selected': True}))
    return pd.concat(rows)


# ---- shortcut ladder: met vs primary, binary elastic net per modality ----
def shortcut_sets(fd):
    t = fd['samples']
    mi_ctrl = list(fd['ctrl'][0].columns)
    mr_ctrl = list(fd['ctrl'][1].columns)
    lymph = [g for g in C.LYMPHOID_GENES if g in fd['Z'][1].columns]
    imm = [g for g in C.IMMUNE_MIRNAS if g in fd['Z'][0].columns]
    mi, mr = fd['nodes']['subset']
    return {0: {'control_probes': ('ctrl', mi_ctrl), 'immune_mirnas': ('Z', imm), 'all_subset_features': ('Z', list(mi))},
            1: {'control_probes': ('ctrl', mr_ctrl), 'lymphoid_myeloid_genes': ('Z', lymph),
                'lymphoid_score_mean': ('score', lymph), 'all_subset_features': ('Z', list(mr))}}


def shortcut_fold(args):
    r, k = args
    fd = P.prep('main', r, k)
    t = fd['samples']
    rows = []
    for m, sets in shortcut_sets(fd).items():
        pick = lambda ids: [i for i in ids if t.at[i, 'modality'] == m and t.at[i, 'y'] in (1, 2)]
        fit, val, test = pick(fd['fit']), pick(fd['val']), pick(fd['test'])
        for name, (src, cols) in sets.items():
            if src == 'score':
                get = lambda ids: fd['Z'][m].loc[ids, cols].mean(1).values[:, None]
            else:
                get = lambda ids, s=src: fd[s][m].loc[ids, cols].values
            Xf, Xv, Xt = get(fit), get(val), get(test)
            yf, yv = (t.loc[fit, 'y'].values == 2).astype(int), (t.loc[val, 'y'].values == 2).astype(int)
            best, bl = None, np.inf
            for cfg in enet_configs():
                mdl = LogisticRegression(penalty='elasticnet', solver='saga', max_iter=3000, class_weight='balanced',
                                         C=cfg['C'], l1_ratio=cfg['l1_ratio']).fit(Xf, yf)
                pv = np.clip(mdl.predict_proba(Xv)[:, 1], 1e-7, 1 - 1e-7)
                wv = np.where(yv == 1, 0.5 / max(yv.mean(), 1e-9), 0.5 / max(1 - yv.mean(), 1e-9))
                ll = float(np.sum(wv * -(yv * np.log(pv) + (1 - yv) * np.log(1 - pv))) / wv.sum())
                if ll < bl:
                    bl, best = ll, mdl
            p = best.predict_proba(Xt)[:, 1]
            rows.append(pd.DataFrame({'id': test, 'modality': m, 'y_met': (t.loc[test, 'y'].values == 2).astype(int),
                                      'group': t.loc[test, 'group'].values, 'p_met': p, 'feature_set': name,
                                      'n_features': len(cols), 'repeat': r, 'fold': k}))
    return pd.concat(rows)


def export_folds():
    """Fold matrices for the TabPFN script (runs in its own venv)."""
    d = RUNS.parent / 'tabpfn_inputs'
    d.mkdir(parents=True, exist_ok=True)
    for r in range(C.N_REPEATS):
        for k in range(C.N_FOLDS):
            fd = P.prep('main', r, k)
            t = fd['samples']
            for m, mode in ((0, 'mi'), (1, 'mr')):
                sel = lambda ids: [i for i in ids if t.at[i, 'modality'] == m]
                fit, val, test = sel(fd['fit']), sel(fd['val']), sel(fd['test'])
                np.savez(d / f'r{r}_f{k}_m{m}.npz', Xf=features(fd, fit, 'subset', mode), yf=t.loc[fit, 'y'].values,
                         Xv=features(fd, val, 'subset', mode), yv=t.loc[val, 'y'].values,
                         Xt=features(fd, test, 'subset', mode), yt=t.loc[test, 'y'].values,
                         ids=np.array(test), groups=t.loc[test, 'group'].values)


def main():
    cmd = sys.argv[1]
    folds = [(r, k) for r in range(C.N_REPEATS) for k in range(C.N_FOLDS)]
    workers = int(os.environ.get('WORKERS', os.cpu_count()))
    if cmd == 'main':
        jobs = [(name, r, k) for name in MODELS for r, k in folds]
        with Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(run_model_fold, jobs)):
                if (i + 1) % 25 == 0:
                    print(f'{i + 1}/{len(jobs)}', flush=True)
        nv = pd.concat([naive(r, k) for r, k in folds])
        for name, g in nv.groupby('arm'):
            (RUNS / 'main' / name).mkdir(parents=True, exist_ok=True)
            g.to_csv(RUNS / 'main' / name / 'all.csv', index=False)
    elif cmd == 'lc':
        import run_gat as R
        jobs = []
        for f in C.LC_FRACTIONS:
            for k in range(C.N_FOLDS):
                fit_ids = R.learning_subset(P.prep('main', 0, k), f)
                jobs += [(name, 0, k, fit_ids, f'lc_{f}') for name in ('xgb_joint', 'xgb_routed_mi', 'xgb_routed_mr', 'enet_routed_mi', 'enet_routed_mr')]
        with Pool(workers) as pool:
            list(pool.imap_unordered(run_model_fold, jobs))
    elif cmd == 'shortcut':
        with Pool(workers) as pool:
            out = pd.concat(pool.map(shortcut_fold, folds))
        out.to_csv(RUNS.parent / 'shortcut_predictions.csv', index=False, float_format='%.6g')
    elif cmd == 'export':
        export_folds()


if __name__ == '__main__':
    main()
