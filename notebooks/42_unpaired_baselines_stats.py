"""Baselines and statistics for the pairing-free multi-omic GAT (design A).

Reads results/fix_multiomic_A/predictions_*.csv written by 41 and writes:
  xgb_predictions.csv        XGBoost on the same inputs, 20 seeds
  summary.csv                macro-F1 / accuracy per model x condition x modality
  bootstrap_cis.csv          patient-cluster bootstrap 95% CI of the 20-seed mean
  late_fusion.csv            specimen-level late fusion under random within-tissue pairings
  permutation_test.csv       real miRDB cross edges vs degree-preserving rewirings
  mcnemar.csv                GAT vs XGBoost on seed-majority predictions
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from xgboost import XGBClassifier

OUT = 'results/fix_multiomic_A/'
SEEDS = 20
N_BOOT = 2000
N_PAIRINGS = 50
spec = importlib.util.spec_from_file_location('g', 'notebooks/41_unpaired_multiomic_gat.py')
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
MOD_NAME = {0: 'miRNA', 1: 'mRNA', 2: 'both (paired)'}


def mf1(y, p):
    return f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0)


def xgb_runs():
    d = g.load()
    n_mi, n_mr = d['train_mirna'].shape[1], d['train_mrna'].shape[1]
    rows = []
    for cond in ('joint', 'single_mirna', 'single_mrna', 'paired_random', 'paired_order'):
        best = None
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            if cond.startswith('paired'):
                if cond == 'paired_random':
                    trp, tep = g.within_tissue_pairing(d['train_y'], rng), g.within_tissue_pairing(d['test_y'], rng)
                else:
                    trp, tep = [(s, s) for s in d['train_y'].index], [(s, s) for s in d['test_y'].index]
                tr = g.make_examples(d, 'train', None, n_mi, n_mr, trp)
                te = g.make_examples(d, 'test', None, n_mi, n_mr, tep)
            else:
                mods = {'single_mirna': [0], 'single_mrna': [1]}.get(cond, [0, 1])
                tr = g.make_examples(d, 'train', mods, n_mi, n_mr)
                te = g.make_examples(d, 'test', mods, n_mi, n_mr)
            Xtr = np.where(tr[1].numpy() > 0, tr[0].numpy(), np.nan)
            Xte = np.where(te[1].numpy() > 0, te[0].numpy(), np.nan)
            if cond.startswith('single'):
                keep = ~np.isnan(Xtr).all(0)
                Xtr, Xte = Xtr[:, keep], Xte[:, keep]
            ytr, gtr = tr[2], tr[4]
            if best is None:
                grid = GridSearchCV(
                    XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', subsample=0.8,
                                  colsample_bytree=0.8, random_state=0, n_jobs=1, verbosity=0),
                    {'max_depth': [3, 5, 7], 'n_estimators': [100, 300, 500], 'learning_rate': [0.01, 0.05, 0.1]},
                    scoring='f1_macro', cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42), n_jobs=2)
                grid.fit(Xtr, ytr, groups=gtr)
                best = grid.best_params_
                print(cond, best, round(grid.best_score_, 4), flush=True)
            clf = XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', subsample=0.8,
                                colsample_bytree=0.8, random_state=seed, n_jobs=2, verbosity=0, **best)
            clf.fit(Xtr, ytr)
            prob = clf.predict_proba(Xte)
            rows.append(pd.DataFrame({'id': te[5], 'modality': te[3], 'group': te[4], 'y': te[2],
                                      'p_normal': prob[:, 0], 'p_primary': prob[:, 1], 'p_metastatic': prob[:, 2],
                                      'pred': prob.argmax(1), 'seed': seed, 'condition': cond, 'perm': -1}))
    return pd.concat(rows)


def seed_scores(df):
    return df.groupby('seed').apply(lambda s: pd.Series({'f1': mf1(s.y, s.pred), 'acc': accuracy_score(s.y, s.pred)}),
                                    include_groups=False)


def cluster_bootstrap(df, rng):
    """Patient-cluster bootstrap of the across-seed mean macro-F1 (resample test patient groups)."""
    groups = df.loc[df.seed == df.seed.min(), 'group'].unique()
    by_seed = {s: x.reset_index(drop=True) for s, x in df.groupby('seed')}
    idx_by_group = {s: x.groupby('group').indices for s, x in by_seed.items()}
    vals = []
    for _ in range(N_BOOT):
        pick = rng.choice(groups, size=len(groups), replace=True)
        f = []
        for s, x in by_seed.items():
            ii = np.concatenate([idx_by_group[s][gr] for gr in pick])
            f.append(mf1(x.y.values[ii], x.pred.values[ii]))
        vals.append(np.mean(f))
    return np.percentile(vals, [2.5, 97.5])


def late_fusion(df, rng):
    """Fuse a test miRNA sample's and a test mRNA sample's class probabilities (sum of log-probs)
    under random within-tissue pairings, plus the order-based pairing of the 01 build."""
    out = []
    P = ['p_normal', 'p_primary', 'p_metastatic']
    for seed, s in df.groupby('seed'):
        mi = s[s.modality == 0].assign(sid=lambda x: x.id.str.split(':').str[1]).set_index('sid')
        mr = s[s.modality == 1].assign(sid=lambda x: x.id.str.split(':').str[1]).set_index('sid')
        y = mi.y
        lp = lambda a, b: np.log(np.clip(mi.loc[a, P].values, 1e-9, 1)) + np.log(np.clip(mr.loc[b, P].values, 1e-9, 1))
        fo = lp(list(y.index), list(y.index)).argmax(1)
        out.append({'seed': seed, 'pairing': 'order (01 build)', 'f1': mf1(y.values, fo)})
        for k in range(N_PAIRINGS):
            a, b, yy = [], [], []
            for c in range(3):
                ids = list(y.index[y == c])
                a += ids; b += list(rng.permutation(ids)); yy += [c] * len(ids)
            out.append({'seed': seed, 'pairing': 'random within tissue', 'f1': mf1(np.array(yy), lp(a, b).argmax(1))})
    return pd.DataFrame(out)


def mcnemar(a, b, y):
    ca, cb = a == y, b == y
    n01, n10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    return n01, n10, p


def main():
    rng = np.random.default_rng(0)
    xgb_path = f'{OUT}xgb_predictions.csv'
    if not os.path.exists(xgb_path):
        xgb_runs().to_csv(xgb_path, index=False)
    gat = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(f'{OUT}predictions_*.csv'))])
    xgb = pd.read_csv(xgb_path)
    gat['model'], xgb['model'] = 'GAT', 'XGBoost'
    allp = pd.concat([gat, xgb])
    real = allp[allp.condition != 'joint_rewired']

    summ, cis = [], []
    for (model, cond, mod), df in real.groupby(['model', 'condition', 'modality']):
        sc = seed_scores(df)
        lo, hi = cluster_bootstrap(df, rng)
        summ.append({'model': model, 'condition': cond, 'modality': MOD_NAME[mod], 'n_seeds': len(sc),
                     'n_test': int((df.seed == df.seed.min()).sum()),
                     'macro_f1_mean': sc.f1.mean(), 'macro_f1_sd': sc.f1.std(ddof=1),
                     'macro_f1_min': sc.f1.min(), 'macro_f1_max': sc.f1.max(),
                     'acc_mean': sc.acc.mean(), 'ci95_lo': lo, 'ci95_hi': hi})
    summary = pd.DataFrame(summ)
    summary.to_csv(f'{OUT}summary.csv', index=False)
    print(summary.round(3).to_string())

    fus = []
    for (model, cond), df in real[real.modality < 2].groupby(['model', 'condition']):
        if cond != 'joint' and cond not in ('single_mirna', 'single_mrna'):
            continue
        fus.append(late_fusion(df, rng).assign(model=model, condition=cond))
    # single-modality models: fuse the miRNA-only model with the mRNA-only model
    for model in ('GAT', 'XGBoost'):
        sm = real[(real.model == model) & real.condition.isin(['single_mirna', 'single_mrna'])]
        fus.append(late_fusion(sm, rng).assign(model=model, condition='single_mirna + single_mrna'))
    fus = pd.concat(fus)
    fus = fus[~fus.condition.isin(['single_mirna', 'single_mrna'])]
    fsum = fus.groupby(['model', 'condition', 'pairing']).f1.agg(
        mean='mean', sd='std', p2_5=lambda x: np.percentile(x, 2.5), p97_5=lambda x: np.percentile(x, 97.5), n='size').reset_index()
    fsum.to_csv(f'{OUT}late_fusion.csv', index=False)
    print(fsum.round(3).to_string())

    perm = gat[gat.condition == 'joint_rewired']
    prow = []
    if len(perm):
        real_joint = gat[gat.condition == 'joint']
        for mod in (0, 1, 'all'):
            sel = (lambda x: x) if mod == 'all' else (lambda x, m=mod: x[x.modality == m])
            real_mean = seed_scores(sel(real_joint)).f1.mean()
            null = seed_scores(sel(perm)).f1.values
            nocross = seed_scores(sel(gat[gat.condition == 'joint_nocross'])).f1.mean()
            prow.append({'modality': {0: 'miRNA', 1: 'mRNA', 'all': 'miRNA+mRNA test samples'}[mod],
                         'real_mean': real_mean, 'no_cross_edges_mean': nocross,
                         'null_mean': null.mean(), 'null_sd': null.std(ddof=1), 'n_perm': len(null),
                         'p_empirical': (1 + (null >= real_mean).sum()) / (1 + len(null))})
        pd.DataFrame(prow).to_csv(f'{OUT}permutation_test.csv', index=False)
        print(pd.DataFrame(prow).round(3).to_string())

    mc = []
    for cond_g, cond_x in (('joint', 'joint'), ('single_mirna', 'single_mirna'), ('single_mrna', 'single_mrna'),
                           ('paired_order', 'paired_order')):
        for mod in sorted(real[real.condition == cond_g].modality.unique()):
            def maj(df):
                return df.groupby('id').agg(y=('y', 'first'), pred=('pred', lambda p: np.bincount(p, minlength=3).argmax()))
            a = maj(real[(real.model == 'GAT') & (real.condition == cond_g) & (real.modality == mod)])
            b = maj(real[(real.model == 'XGBoost') & (real.condition == cond_x) & (real.modality == mod)]).loc[a.index]
            n01, n10, p = mcnemar(a.pred.values, b.pred.values, a.y.values)
            mc.append({'condition': cond_g, 'modality': MOD_NAME[mod], 'gat_only_correct': n01,
                       'xgb_only_correct': n10, 'mcnemar_exact_p': p,
                       'gat_majority_f1': mf1(a.y, a.pred), 'xgb_majority_f1': mf1(b.y, b.pred)})
    pd.DataFrame(mc).to_csv(f'{OUT}mcnemar.csv', index=False)
    print(pd.DataFrame(mc).round(3).to_string())


if __name__ == '__main__':
    main()
