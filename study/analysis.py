#!/usr/bin/env python3
"""Analysis of the final study (pre-specified in STUDY_PLAN.md; written before the runs finished).

  python study/analysis.py collect   # consolidate run files -> results/study_final/predictions_final.csv.gz
  python study/analysis.py analyze   # tables -> results/study_final/tables/
"""
import glob
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C

REPO = Path(__file__).resolve().parents[1]
RES = REPO / 'results/study_final'
RUNS = Path(os.environ.get('STUDY_RUNS', RES / 'runs'))
TAB = RES / 'tables'
TAB.mkdir(parents=True, exist_ok=True)
PCOLS = ['p0', 'p1', 'p2']
MODN = {0: 'miRNA', 1: 'mRNA'}
CLS = C.CLASSES
RNG = np.random.default_rng(20261122)
ALL_ARMS = C.ARMS | C.SENS_ARMS


# ---------------------------------------------------------------- collect
def _read(f):
    return pd.read_csv(f)


def collect():
    sels = {}
    for f in glob.glob(str(RUNS / 'selection' / '*.json')):
        sch, arm, rk = Path(f).stem.split('__')
        r, k = (int(x[1:]) for x in rk.split('_'))
        sels[(sch, arm, r, k)] = json.load(open(f))
    pd.DataFrame([{'scheme': s, 'arm': a, 'repeat': r, 'fold': k, 'config': v['config'], **v['params'],
                   'val_loss_selected': min(v['val_losses'])} for (s, a, r, k), v in sels.items()]).to_csv(RES / 'selected_configs.csv', index=False)
    files = [f for f in glob.glob(str(RUNS / '*' / '*' / 'r*_f*_c*_s*.csv'))]
    with ThreadPoolExecutor(16) as ex:
        df = pd.concat(ex.map(_read, files), ignore_index=True)
    tune = df[df.seed == 0][['scheme', 'arm', 'repeat', 'fold', 'config', 'val_loss', 'epochs', 'n_cross', 'n_nodes_mi', 'n_nodes_mr', 'secs']].drop_duplicates()
    tune.to_csv(RES / 'gat_runs_summary.csv.gz', index=False)
    keep = []
    for (sch, arm, r, k), g in df.groupby(['scheme', 'arm', 'repeat', 'fold']):
        a = ALL_ARMS.get(arm, {})
        if sch == 'main' and a.get('tune'):
            c = sels[(sch, arm, r, k)]['config']
        elif sch == 'main' and 'config_from' in a:
            c = sels[(sch, a['config_from'], r, k)]['config']
        elif sch.startswith('lc_'):
            c = sels[('main', arm, r, k)]['config']
        else:
            c = 0
        keep.append(g[g.config == c])
    gat = pd.concat(keep)
    base = []
    for f in glob.glob(str(RUNS / 'main' / '*' / 'r*_f*.csv')) + glob.glob(str(RUNS / 'lc_*' / '*' / 'r*_f*.csv')):
        if Path(f).stem.count('_') != 1:
            continue
        b = pd.read_csv(f)
        base.append(b[b.selected])
    for f in glob.glob(str(RUNS / 'main' / 'naive_*' / 'all.csv')):
        base.append(pd.read_csv(f))
    tp = RUNS / 'main' / 'tabpfn_all.csv'
    if tp.exists():
        b = pd.read_csv(tp)
        base.append(b[b.selected])
    cols = ['scheme', 'arm', 'repeat', 'fold', 'config', 'seed', 'id', 'modality', 'y', 'group'] + PCOLS
    out = pd.concat([gat[cols + ['n_cross']]] + [b.assign(n_cross=np.nan)[cols + ['n_cross']] for b in base], ignore_index=True)
    out.to_csv(RES / 'predictions_final.csv.gz', index=False, float_format='%.5g')
    print(out.groupby(['scheme', 'arm']).size().to_string())


# ---------------------------------------------------------------- metrics
def f1_from_conf(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    denom = cm.sum(-2) + cm.sum(-1)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1e-12), 0.0)


def conf(y, p, w=None):
    return np.bincount(y * 3 + p, weights=w, minlength=9).reshape(3, 3)


def macro_f1(y, p):
    return float(f1_from_conf(conf(y, p)).mean())


def logloss_vec(y, P):
    return -np.log(np.clip(P[np.arange(len(y)), y], 1e-7, 1))


def ece(y, P, bins=10):
    conf_ = P.max(1)
    pred = P.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        s = (conf_ > lo) & (conf_ <= lo + 1 / bins)
        if s.any():
            e += s.mean() * abs((pred[s] == y[s]).mean() - conf_[s].mean())
    return e


class Arm:
    """Predictions of one arm on one modality's test examples: arrays indexed [run, sample]."""

    def __init__(self, df, mod, scheme='main'):
        d = df[(df.modality == mod)]
        self.ids = np.array(sorted(d.id.unique()))
        idx = {i: j for j, i in enumerate(self.ids)}
        first = d.drop_duplicates('id').set_index('id')
        self.y = first.loc[self.ids, 'y'].values.astype(int)
        self.group = first.loc[self.ids, 'group'].values
        runs = sorted(d.groupby(['repeat', 'seed']).groups)
        self.runs = runs
        self.P = np.full((len(runs), len(self.ids), 3), np.nan)
        self.fold = np.full((len(runs), len(self.ids)), -1)
        for j, ((r, s), g) in enumerate(d.groupby(['repeat', 'seed'])):
            ii = g.id.map(idx).values
            self.P[j, ii] = g[PCOLS].values
            self.fold[j, ii] = g.fold.values
        self.complete = ~np.isnan(self.P).any((1, 2))

    def per_run_f1(self):
        return np.array([macro_f1(self.y, self.P[j].argmax(1)) for j in range(len(self.runs)) if self.complete[j]])

    def sample_ll(self):
        return np.nanmean(-np.log(np.clip(np.take_along_axis(self.P, self.y[None, :, None], 2)[..., 0], 1e-7, 1)), 0)

    def fold_f1(self):
        """{(repeat, fold): mean over seeds of macro-F1 on that fold's test samples}."""
        out = {}
        for j, (r, s) in enumerate(self.runs):
            for k in np.unique(self.fold[j]):
                if k < 0:
                    continue
                m = self.fold[j] == k
                out.setdefault((r, int(k)), []).append(macro_f1(self.y[m], self.P[j, m].argmax(1)))
        return {key: float(np.mean(v)) for key, v in out.items()}

    def summary(self):
        Pm = self.P[self.complete]
        f1 = self.per_run_f1()
        yy = np.broadcast_to(self.y, Pm.shape[:2])
        ll = np.mean([logloss_vec(self.y, Pm[j]).mean() for j in range(len(Pm))])
        brier = np.mean([((Pm[j] - np.eye(3)[self.y]) ** 2).sum(1).mean() for j in range(len(Pm))])
        acc = np.mean(Pm.argmax(2) == yy)
        cls_f1 = np.mean([f1_from_conf(conf(self.y, Pm[j].argmax(1))) for j in range(len(Pm))], 0)
        ff = self.fold_f1()
        by = {}
        for j, (r, s) in enumerate(self.runs):
            for k in np.unique(self.fold[j]):
                m = self.fold[j] == k
                by.setdefault((r, k), []).append(macro_f1(self.y[m], self.P[j, m].argmax(1)))
        seed_var = np.mean([np.var(v, ddof=1) for v in by.values() if len(v) > 1]) if any(len(v) > 1 for v in by.values()) else np.nan
        return {'n_samples': len(self.ids), 'n_groups': len(np.unique(self.group)), 'n_runs': int(self.complete.sum()),
                'macro_f1': f1.mean(), 'macro_f1_sd_runs': f1.std(ddof=1) if len(f1) > 1 else np.nan,
                'accuracy': acc, 'log_loss': ll, 'brier': brier, 'ece': np.mean([ece(self.y, Pm[j]) for j in range(len(Pm))]),
                **{f'f1_{c}': v for c, v in zip(CLS, cls_f1)},
                'fold_sd_of_seedmean_f1': float(np.std(list(ff.values()), ddof=1)), 'seed_sd_within_fold_f1': float(np.sqrt(seed_var))}

    def boot_f1(self, n=C.N_BOOT):
        """Patient-cluster bootstrap 95% CI of the run-mean pooled macro-F1."""
        Pm = self.P[self.complete].argmax(2)
        ug, gi = np.unique(self.group, return_inverse=True)
        vals = []
        for _ in range(n):
            w = np.bincount(RNG.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
            cms = np.stack([conf(self.y, Pm[j], w) for j in range(len(Pm))])
            vals.append(f1_from_conf(cms).mean(-1).mean())
        return np.percentile(vals, [2.5, 97.5])


def align(a, b):
    assert (a.ids == b.ids).all() and (a.y == b.y).all()


def signflip(d, groups, n=C.N_SIGNFLIP):
    gm = pd.Series(d).groupby(groups).mean().values
    obs = gm.mean()
    flips = RNG.choice([-1.0, 1.0], size=(n, len(gm)))
    null = (flips * gm).mean(1)
    return float(obs), float((1 + np.sum(np.abs(null) >= abs(obs))) / (n + 1))


def contrast(A, B, name, mod, margin=C.TOST_MARGIN):
    align(A, B)
    fa, fb = A.fold_f1(), B.fold_f1()
    keys = sorted(set(fa) & set(fb))
    d = np.array([fa[k] - fb[k] for k in keys])
    J = len(d)
    n_te = len(A.ids) / C.N_FOLDS
    ratio = n_te / (len(A.ids) - n_te)
    se = np.sqrt((1 / J + ratio) * d.var(ddof=1)) if J > 1 else np.nan
    delta = A.per_run_f1().mean() - B.per_run_f1().mean()
    tc = stats.t.ppf(0.975, J - 1)
    t90 = stats.t.ppf(0.95, J - 1)
    p_lo = stats.t.sf((delta + margin) / se, J - 1)
    p_hi = stats.t.sf((margin - delta) / se, J - 1)
    nc = margin / se
    power = stats.nct.sf(tc, J - 1, nc) + stats.nct.cdf(-tc, J - 1, nc)
    p_t = 2 * stats.t.sf(abs(delta) / se, J - 1)
    lla, llb = A.sample_ll(), B.sample_ll()
    dll, p_sf = signflip(lla - llb, A.group)
    ug, gi = np.unique(A.group, return_inverse=True)
    boots = []
    Pa, Pb = A.P[A.complete].argmax(2), B.P[B.complete].argmax(2)
    for _ in range(C.N_BOOT):
        w = np.bincount(RNG.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
        fa_ = f1_from_conf(np.stack([conf(A.y, Pa[j], w) for j in range(len(Pa))])).mean(-1).mean()
        fb_ = f1_from_conf(np.stack([conf(B.y, Pb[j], w) for j in range(len(Pb))])).mean(-1).mean()
        boots.append(fa_ - fb_)
    row = {'contrast': name, 'modality': MODN[mod], 'n_samples': len(A.ids), 'n_groups': len(ug),
           'f1_a': A.per_run_f1().mean(), 'f1_b': B.per_run_f1().mean(), 'delta_macro_f1': delta,
           'corrected_ci95_lo': delta - tc * se, 'corrected_ci95_hi': delta + tc * se,
           'corrected_ci90_lo': delta - t90 * se, 'corrected_ci90_hi': delta + t90 * se,
           'corrected_t_p': p_t, 'tost_p': max(p_lo, p_hi), 'equivalent_within_margin': bool(max(p_lo, p_hi) < 0.05),
           'power_at_margin': power, 'n_fold_pairs': J,
           'cluster_boot_ci95_lo': np.percentile(boots, 2.5), 'cluster_boot_ci95_hi': np.percentile(boots, 97.5),
           'delta_logloss': dll, 'signflip_p_logloss': p_sf}
    for c in range(3):
        m = A.y == c
        row[f'delta_logloss_{CLS[c]}'] = float((lla - llb)[m].mean())
        row[f'delta_f1_{CLS[c]}'] = float(np.mean([f1_from_conf(conf(A.y, Pa[j]))[c] for j in range(len(Pa))]) -
                                          np.mean([f1_from_conf(conf(B.y, Pb[j]))[c] for j in range(len(Pb))]))
    return row


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0
    for rank, i in enumerate(o):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def load():
    return pd.read_csv(RES / 'predictions_final.csv.gz')


def arm(df, scheme, name, mod, repeats=None, seeds=None):
    d = df[(df.scheme == scheme) & (df.arm == name)]
    if repeats is not None:
        d = d[d.repeat.isin(repeats)]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    return Arm(d, mod)


def analyze():
    df = load()
    main = df[df.scheme == 'main']
    routed = {'xgb_routed': ('xgb_routed_mi', 'xgb_routed_mr'), 'enet_routed': ('enet_routed_mi', 'enet_routed_mr'),
              'tabpfn_routed': ('tabpfn_routed_mi', 'tabpfn_routed_mr'), 'enet_targetset': ('enet_targetset_mi', 'enet_targetset_mr'),
              'single_nocross (routed GAT)': ('single_mi_nocross', 'single_mr_nocross'),
              'single_mirdb (routed GAT)': ('single_mi_mirdb', 'single_mr_mirdb'),
              'single_rewired (routed GAT)': ('single_mi_rewired', 'single_mr_rewired')}
    present = set(main.arm)
    rows, arms = [], {}
    joint_names = [a for a in ['joint_mirdb', 'joint_tcga', 'joint_nocross', 'joint_rewired', 'joint_mirdb_full', 'gcn_mirdb', 'mlp_only',
                               'graphonly_mirdb', 'xgb_joint', 'xgb_joint_full', 'enet_joint', 'naive_prior', 'naive_majority'] if a in present]
    for mod in (0, 1):
        for name in joint_names:
            arms[(name, mod)] = arm(main, 'main', name, mod)
        for name, pair in routed.items():
            if pair[mod] in present:
                arms[(name, mod)] = arm(main, 'main', pair[mod], mod)
    for (name, mod), A in arms.items():
        s = A.summary()
        lo, hi = A.boot_f1(500)
        rows.append({'arm': name, 'modality': MODN[mod], **s, 'cluster_boot_ci95_lo': lo, 'cluster_boot_ci95_hi': hi})
    summ = pd.DataFrame(rows)
    summ.to_csv(TAB / 'arm_summary.csv', index=False)
    print(summ[['arm', 'modality', 'n_runs', 'macro_f1', 'cluster_boot_ci95_lo', 'cluster_boot_ci95_hi', 'log_loss', 'ece']].round(3).to_string())

    # hypotheses
    hrows = []
    for h, (a, b) in C.HYPOTHESES.items():
        for mod in (0, 1):
            bb = b[MODN[mod]] if isinstance(b, dict) else b
            key_b = 'single_nocross (routed GAT)' if isinstance(b, dict) else bb
            if (a, mod) in arms and (key_b, mod) in arms:
                hrows.append({'family': 'primary', 'hypothesis': h} | contrast(arms[(a, mod)], arms[(key_b, mod)], f'{a} - {bb}', mod))
    if hrows:
        adj = holm([r['signflip_p_logloss'] for r in hrows])
        for r, p in zip(hrows, adj):
            r['holm_p'] = p
    for h, (a, b) in C.SECONDARY.items():
        for mod in (0, 1):
            if (a, mod) in arms and (b, mod) in arms:
                hrows.append({'family': 'secondary_B', 'hypothesis': h} | contrast(arms[(a, mod)], arms[(b, mod)], f'{a} - {b}', mod))
    extra = [('joint_mirdb_full', 'joint_mirdb'), ('joint_mirdb', 'xgb_joint'), ('joint_mirdb', 'xgb_routed'), ('joint_mirdb', 'enet_routed'),
             ('joint_mirdb', 'tabpfn_routed'), ('joint_mirdb', 'gcn_mirdb'), ('joint_mirdb', 'mlp_only'), ('joint_nocross', 'mlp_only'),
             ('joint_mirdb', 'graphonly_mirdb'), ('single_mirdb (routed GAT)', 'single_nocross (routed GAT)'),
             ('single_mirdb (routed GAT)', 'single_rewired (routed GAT)'), ('joint_mirdb', 'single_mirdb (routed GAT)'),
             ('enet_targetset', 'enet_routed'), ('joint_mirdb', 'enet_targetset')]
    for a, b in extra:
        for mod in (0, 1):
            if (a, mod) in arms and (b, mod) in arms:
                hrows.append({'family': 'descriptive', 'hypothesis': ''} | contrast(arms[(a, mod)], arms[(b, mod)], f'{a} - {b}', mod))
    hyp = pd.DataFrame(hrows)
    hyp.to_csv(TAB / 'contrasts.csv', index=False)
    print(hyp[['family', 'contrast', 'modality', 'delta_macro_f1', 'corrected_ci95_lo', 'corrected_ci95_hi', 'tost_p', 'power_at_margin',
               'delta_logloss', 'signflip_p_logloss'] + (['holm_p'] if 'holm_p' in hyp else [])].round(4).to_string())

    # rewired null, descriptive (per-run OOF macro-F1)
    nrow = []
    for mod in (0, 1):
        if ('joint_mirdb', mod) in arms and ('joint_rewired', mod) in arms:
            real, null = arms[('joint_mirdb', mod)].per_run_f1(), arms[('joint_rewired', mod)].per_run_f1()
            nrow.append({'modality': MODN[mod], 'real_mean': real.mean(), 'null_mean': null.mean(), 'null_sd': null.std(ddof=1),
                         'n_real_runs': len(real), 'n_null_runs': len(null),
                         'empirical_p_null_ge_real_mean': (1 + np.sum(null >= real.mean())) / (1 + len(null)),
                         'mannwhitney_p_real_gt_null': stats.mannwhitneyu(real, null, alternative='greater').pvalue})
    pd.DataFrame(nrow).to_csv(TAB / 'rewired_null.csv', index=False)

    # zero-shot cross-modality transfer: single-modality-trained arms scored on the other modality
    z = []
    for tr_mod, prefix in ((0, 'single_mi_'), (1, 'single_mr_')):
        te_mod = 1 - tr_mod
        zz = {}
        for e in ('nocross', 'mirdb', 'rewired'):
            if prefix + e in present:
                zz[e] = arm(main, 'main', prefix + e, te_mod)
                s = zz[e].summary()
                z.append({'trained_on': MODN[tr_mod], 'tested_on': MODN[te_mod], 'edges': e, 'macro_f1': s['macro_f1'],
                          'log_loss': s['log_loss'], 'accuracy': s['accuracy'], **{f'f1_{c}': s[f'f1_{c}'] for c in CLS}})
        for a, b in (('mirdb', 'nocross'), ('mirdb', 'rewired')):
            if a in zz and b in zz:
                c = contrast(zz[a], zz[b], f'zero-shot {MODN[tr_mod]}->{MODN[te_mod]}: {a} - {b}', te_mod)
                z.append({'trained_on': MODN[tr_mod], 'tested_on': MODN[te_mod], 'edges': f'{a} - {b}', 'macro_f1': c['delta_macro_f1'],
                          'ci95_lo': c['corrected_ci95_lo'], 'ci95_hi': c['corrected_ci95_hi'], 'log_loss': c['delta_logloss'],
                          'signflip_p': c['signflip_p_logloss']})
    pd.DataFrame(z).to_csv(TAB / 'zero_shot_transfer.csv', index=False)

    # shortcut ladder
    sc = RES / 'shortcut_predictions.csv'
    if sc.exists():
        s = pd.read_csv(sc)
        lad = []
        for (mod, fs), g in s.groupby(['modality', 'feature_set']):
            au = [roc_auc_score(x.y_met, x.p_met) for _, x in g.groupby('repeat')]
            lad.append({'modality': MODN[mod], 'feature_set': fs, 'n_features': int(g.n_features.iloc[0]), 'auroc_met_vs_primary': np.mean(au),
                        'auroc_sd_repeats': np.std(au, ddof=1)})
        for mod in (0, 1):
            for name in ('joint_mirdb', 'joint_nocross', 'xgb_joint'):
                if (name, mod) in arms:
                    A = arms[(name, mod)]
                    m = A.y > 0
                    sc_ = A.P[A.complete][:, m, 2] / (A.P[A.complete][:, m, 1] + A.P[A.complete][:, m, 2])
                    au = [roc_auc_score(A.y[m] == 2, sc_[j]) for j in range(len(sc_))]
                    lad.append({'modality': MODN[mod], 'feature_set': f'model:{name}', 'n_features': np.nan, 'auroc_met_vs_primary': np.mean(au),
                                'auroc_sd_repeats': np.std(au, ddof=1)})
        lad = pd.DataFrame(lad)
        for mod in ('miRNA', 'mRNA'):
            sub = lad[lad.modality == mod]
            best = sub[~sub.feature_set.str.startswith('model:') & (sub.feature_set != 'all_subset_features')].auroc_met_vs_primary.max()
            lad.loc[lad.modality == mod, 'margin_over_best_shortcut'] = lad.loc[lad.modality == mod, 'auroc_met_vs_primary'] - best
        # bootstrap CI of the GAT margin over the best shortcut (per-sample averaged scores, group bootstrap)
        mrows = []
        for mod in (0, 1):
            sub = lad[(lad.modality == MODN[mod]) & ~lad.feature_set.str.startswith('model:') & (lad.feature_set != 'all_subset_features')]
            bfs = sub.sort_values('auroc_met_vs_primary').feature_set.iloc[-1]
            g = s[(s.modality == mod) & (s.feature_set == bfs)].groupby('id').agg(y=('y_met', 'first'), p=('p_met', 'mean'), grp=('group', 'first'))
            A = arms[('joint_mirdb', mod)]
            m = A.y > 0
            gs = pd.Series(np.nanmean(A.P[A.complete][:, m, 2] / (A.P[A.complete][:, m, 1] + A.P[A.complete][:, m, 2]), 0), index=A.ids[m])
            g = g.join(gs.rename('gat'), how='inner')
            ug = g.grp.unique()
            bs = []
            for _ in range(C.N_BOOT):
                pick = RNG.choice(ug, len(ug))
                gg = pd.concat([g[g.grp == u] for u in pick])
                if gg.y.nunique() == 2:
                    bs.append(roc_auc_score(gg.y, gg.gat) - roc_auc_score(gg.y, gg.p))
            mrows.append({'modality': MODN[mod], 'best_shortcut': bfs, 'auroc_gat_avg_scores': roc_auc_score(g.y, g.gat),
                          'auroc_shortcut_avg_scores': roc_auc_score(g.y, g.p),
                          'margin': roc_auc_score(g.y, g.gat) - roc_auc_score(g.y, g.p),
                          'ci95_lo': np.percentile(bs, 2.5), 'ci95_hi': np.percentile(bs, 97.5)})
        lad.to_csv(TAB / 'shortcut_ladder.csv', index=False)
        pd.DataFrame(mrows).to_csv(TAB / 'shortcut_margin.csv', index=False)

    # positive control
    pcr = []
    for sch in sorted(s_ for s_ in df.scheme.unique() if s_.startswith('pc_')):
        pc = df[df.scheme == sch]
        for mod in (0, 1):
            A = {a: Arm(pc[pc.arm == a], mod) for a in C.PC_ARMS if a in set(pc.arm)}
            for a, X in A.items():
                pcr.append({'scheme': sch, 'modality': MODN[mod], 'row': a, 'macro_f1': X.summary()['macro_f1'], 'log_loss': X.summary()['log_loss']})
            for a, b in (('joint_mirdb', 'joint_rewired'), ('joint_mirdb', 'joint_nocross')):
                if a in A and b in A:
                    c = contrast(A[a], A[b], f'{a} - {b}', mod)
                    pcr.append({'scheme': sch, 'modality': MODN[mod], 'row': f'{a} - {b}', 'macro_f1': c['delta_macro_f1'],
                                'ci95_lo': c['corrected_ci95_lo'], 'ci95_hi': c['corrected_ci95_hi'],
                                'log_loss': c['delta_logloss'], 'signflip_p': c['signflip_p_logloss']})
    pd.DataFrame(pcr).to_csv(TAB / 'positive_control.csv', index=False)

    # learning curve (repeat 0)
    lc = []
    for name, pair in [('joint_mirdb', None), ('joint_nocross', None), ('xgb_joint', None), ('xgb_routed', ('xgb_routed_mi', 'xgb_routed_mr')),
                       ('enet_routed', ('enet_routed_mi', 'enet_routed_mr'))]:
        for mod in (0, 1):
            an = pair[mod] if pair else name
            for frac, sch in ((0.25, 'lc_0.25'), (0.5, 'lc_0.5'), (1.0, 'main')):
                d = df[(df.scheme == sch) & (df.arm == an) & (df.repeat == 0) & (df.seed < C.SEEDS_SECONDARY)]
                if len(d) and (d.modality == mod).any():
                    X = Arm(d, mod)
                    if X.complete.any():
                        lc.append({'model': name, 'modality': MODN[mod], 'fraction': frac, 'macro_f1': X.summary()['macro_f1'],
                                   'macro_f1_sd_runs': X.summary()['macro_f1_sd_runs'], 'log_loss': X.summary()['log_loss']})
    pd.DataFrame(lc).to_csv(TAB / 'learning_curve.csv', index=False)

    # sensitivity analyses (repeat 0, 5 seeds) and continuity
    sens = []
    r0 = df[(df.repeat == 0) & (df.seed < C.SEEDS_SECONDARY)]
    pairs = [('main', 'sens_tcga_normal', 'main', 'joint_tcga'), ('main', 'sens_tcga_estimate', 'main', 'joint_tcga'),
             ('main', 'sens_tcga_normal', 'main', 'joint_mirdb'), ('main', 'sens_aug_pairs', 'main', 'joint_mirdb'),
             ('main', 'joint_mirdb', 'main', 'joint_nocross'),
             ('conservative', 'joint_mirdb', 'conservative', 'joint_nocross'), ('subset230', 'joint_mirdb', 'subset230', 'joint_nocross')]
    for sa, a, sb, b in pairs:
        for mod in (0, 1):
            da, db = r0[(r0.scheme == sa) & (r0.arm == a)], r0[(r0.scheme == sb) & (r0.arm == b)]
            if len(da) and len(db):
                A, B = Arm(da, mod), Arm(db, mod)
                c = contrast(A, B, f'{sa}:{a} - {sb}:{b}', mod)
                sens.append({k: c[k] for k in ('contrast', 'modality', 'f1_a', 'f1_b', 'delta_macro_f1', 'corrected_ci95_lo', 'corrected_ci95_hi',
                                               'delta_logloss', 'signflip_p_logloss')})
    pd.DataFrame(sens).to_csv(TAB / 'sensitivity_repeat0.csv', index=False)
    cont = []
    asp = df[df.scheme == 'asplit']
    for mod in (0, 1):
        A = {a: Arm(asp[asp.arm == a], mod) for a in sorted(set(asp.arm))}
        for a, X in A.items():
            s = X.summary()
            lo, hi = X.boot_f1(500)
            cont.append({'row': a, 'modality': MODN[mod], 'macro_f1': s['macro_f1'], 'sd_seeds': s['macro_f1_sd_runs'], 'boot_lo': lo, 'boot_hi': hi})
        for a, b in (('joint_mirdb', 'joint_nocross'), ('joint_mirdb', 'joint_rewired'), ('joint_tcga', 'joint_mirdb')):
            if a in A and b in A:
                Pa, Pb = A[a].P.argmax(2), A[b].P.argmax(2)
                ug, gi = np.unique(A[a].group, return_inverse=True)
                bs = []
                for _ in range(C.N_BOOT):
                    w = np.bincount(RNG.integers(0, len(ug), len(ug)), minlength=len(ug))[gi].astype(float)
                    bs.append(f1_from_conf(np.stack([conf(A[a].y, x, w) for x in Pa])).mean(-1).mean() -
                              f1_from_conf(np.stack([conf(A[b].y, x, w) for x in Pb])).mean(-1).mean())
                cont.append({'row': f'{a} - {b}', 'modality': MODN[mod], 'macro_f1': A[a].per_run_f1().mean() - A[b].per_run_f1().mean(),
                             'boot_lo': np.percentile(bs, 2.5), 'boot_hi': np.percentile(bs, 97.5)})
    pd.DataFrame(cont).to_csv(TAB / 'continuity_190_40.csv', index=False)

    # reliability data for the figure
    rel = []
    for (name, mod), A in arms.items():
        if name in ('joint_mirdb', 'joint_nocross', 'xgb_joint', 'enet_routed', 'tabpfn_routed'):
            Pm = A.P[A.complete]
            conf_ = Pm.max(2).ravel()
            corr = (Pm.argmax(2) == A.y[None]).ravel()
            for lo in np.linspace(0, 1, 11)[:-1]:
                s_ = (conf_ > lo) & (conf_ <= lo + 0.1)
                if s_.sum():
                    rel.append({'arm': name, 'modality': MODN[mod], 'bin_lo': lo, 'n': int(s_.sum()), 'confidence': conf_[s_].mean(), 'accuracy': corr[s_].mean()})
    pd.DataFrame(rel).to_csv(TAB / 'reliability.csv', index=False)


if __name__ == '__main__':
    {'collect': collect, 'analyze': analyze}[sys.argv[1]]()
