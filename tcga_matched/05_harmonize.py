#!/usr/bin/env python3
"""Harmonizing GEO GSE45498 (NanoString, primary cohort) with TCGA-BRCA TNBC (RNA-seq).
Usage: 05_harmonize.py <tcga_cohort_dir> <out_dir> diagnostics
       05_harmonize.py <tcga_cohort_dir> <out_dir> eval <method> [n_gat_seeds] [n_xgb_seeds]
       05_harmonize.py <tcga_cohort_dir> <out_dir> summarize

Features: the GEO DE nodes of the paper's GAT that map one-to-one to TCGA (GEO miRNA names ->
miRBase arm, mRNA symbols). Scales: GEO = log2(normalized count + 1), z-scored on the GEO training
split (data/processed, notebooks/01); TCGA = log2(RSEM + 1) and log2(RPM + 1).
Classes: normal, primary (TCGA TNBC primary tumors), metastatic (GEO only; TCGA has one TNBC met,
dropped). Correction parameters are fit on TRAINING patients only (GEO 190-sample split, TCGA
patient-grouped 75%) and applied unchanged to held-out patients; no test label is ever used.

Methods
  none        : no cross-platform correction.
  combat_ref  : empirical-Bayes ComBat (location + scale), GEO as reference batch (GEO values are
                left unchanged), tissue class as protected covariate, fit on training samples of
                the classes present in both batches (normal, primary). Applied label-free: the
                covariate term for a TCGA sample is the TCGA-training class mix, so the per-feature
                map is affine and identical for training and held-out TCGA samples.
  zscore      : per-cohort feature z-score (each cohort's own training mean / SD).
  quantile    : per-feature quantile mapping of TCGA onto the GEO training distribution (empirical
                CDF fit on TCGA training samples).
  rank        : within-sample rank scoring per modality (rank / n_features), no fitting.
After every method a single pooled-training feature z-score (batch-blind) is applied for models.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcga_common import REPO, edges_to_tensor, map_geo_mirna_to_tcga, mimat_to_name, train_gat

torch.set_num_threads(1)
COH, OUT, PART = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
P = REPO / 'data/processed'
CLASSES = ['normal', 'primary', 'metastatic']
SHARED = ['normal', 'primary']
METHODS = ['none', 'combat_ref', 'zscore', 'quantile', 'rank']

# ---------------- data ----------------
meta = pd.read_csv(COH / 'tcga_tnbc_meta.csv', index_col=0)
meta = meta[meta.tissue.isin(['tumor', 'normal'])]
tmir = pd.read_csv(COH / 'tcga_tnbc_mirna.csv.gz', index_col=0).rename(
    columns=mimat_to_name(str(Path.home() / '.capy/work/mirbase/mature.fa')))
tmir = tmir.loc[:, ~tmir.columns.duplicated()]
tmr = pd.read_csv(COH / 'tcga_tnbc_mrna.csv.gz', index_col=0)
gmir = pd.concat([pd.read_csv(P / 'train_mirna.csv', index_col=0), pd.read_csv(P / 'test_mirna.csv', index_col=0)])
gmr = pd.concat([pd.read_csv(P / 'train_mrna.csv', index_col=0), pd.read_csv(P / 'test_mrna.csv', index_col=0)])
glab = pd.concat([pd.read_csv(P / 'train_labels.csv', index_col=0), pd.read_csv(P / 'test_labels.csv', index_col=0)]).tissue_group
gpid = pd.read_csv(P / 'patient_ids.csv', index_col=0).patient_id
split = json.load(open(P / 'split_indices.json'))
mmap = map_geo_mirna_to_tcga(list(gmir.columns), tmir)
mmap = {g: t for g, t in mmap.items() if list(mmap.values()).count(t) == 1}
mi = [g for g in gmir.columns if g in mmap]
mr = [g for g in gmr.columns if g in tmr.columns]
nodes = mi + mr
is_mi = np.array([True] * len(mi) + [False] * len(mr))

G = pd.concat([gmir[mi], gmr[mr]], axis=1)
T = pd.concat([tmir[[mmap[g] for g in mi]].set_axis(mi, axis=1), tmr[mr]], axis=1).loc[meta.index]
X = pd.concat([G, T])
batch = np.array(['GEO'] * len(G) + ['TCGA'] * len(T))
cls = np.concatenate([glab.values, np.where(meta.tissue == 'tumor', 'primary', 'normal')])
patient = np.concatenate([gpid[G.index].values, meta.patient.values])

tcga_ids = list(meta.index)
ty = (meta.tissue == 'tumor').astype(int).values
sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
ttr, tte = next(sgk.split(tcga_ids, ty, meta.patient.values))
train_set = set(split['train_sample_ids']) | {tcga_ids[i] for i in ttr}
is_train = np.array([s in train_set for s in X.index])
is_test = ~is_train

# ---------------- harmonization ----------------
def combat_ref(Xv):
    fit = is_train & np.isin(cls, SHARED)
    ref = batch == 'GEO'
    tc = batch == 'TCGA'
    prim = (cls == 'primary').astype(float)
    D = np.column_stack([np.ones(len(Xv)), tc.astype(float), prim])
    B, *_ = np.linalg.lstsq(D[fit], Xv[fit], rcond=None)
    alpha, beta = B[0], B[2]
    resid_ref = Xv[fit & ref] - D[fit & ref] @ B
    sigma = np.sqrt((resid_ref ** 2).mean(0)) + 1e-8
    fT = fit & tc
    z = (Xv[fT] - alpha - np.outer(prim[fT], beta)) / sigma
    n = z.shape[0]
    g_hat, d_hat = z.mean(0), z.var(0, ddof=1)
    g_bar, t2 = g_hat.mean(), g_hat.var(ddof=1)
    m, s2 = d_hat.mean(), d_hat.var(ddof=1)
    a, b = (2 * s2 + m ** 2) / s2, (m * s2 + m ** 3) / s2
    g_s, d_s = g_hat.copy(), d_hat.copy()
    for _ in range(1000):
        g_new = (n * t2 * g_hat + d_s * g_bar) / (n * t2 + d_s)
        d_new = (b + 0.5 * ((z - g_new) ** 2).sum(0)) / (n / 2 + a - 1)
        done = max(np.abs(g_new - g_s).max(), np.abs(d_new - d_s).max()) < 1e-6
        g_s, d_s = g_new, d_new
        if done:
            break
    mix = alpha + prim[fT].mean() * beta
    out = Xv.copy()
    out[tc] = (Xv[tc] - mix - sigma * g_s) / np.sqrt(d_s) + mix
    return out

def harmonize(method):
    Xv = X.values.astype(float).copy()
    if method == 'combat_ref':
        Xv = combat_ref(Xv)
    elif method == 'zscore':
        for bb in ('GEO', 'TCGA'):
            fit = is_train & (batch == bb)
            Xv[batch == bb] = (Xv[batch == bb] - Xv[fit].mean(0)) / (Xv[fit].std(0) + 1e-8)
    elif method == 'quantile':
        gref = np.sort(Xv[is_train & (batch == 'GEO')], axis=0)
        tfit = np.sort(Xv[is_train & (batch == 'TCGA')], axis=0)
        tc = batch == 'TCGA'
        for j in range(Xv.shape[1]):
            u = (np.searchsorted(tfit[:, j], Xv[tc, j], side='right')) / len(tfit)
            u = np.clip(u, 0.5 / len(tfit), 1 - 0.5 / len(tfit))
            Xv[tc, j] = np.quantile(gref[:, j], u)
    elif method == 'rank':
        for mask in (is_mi, ~is_mi):
            Xv[:, mask] = stats.rankdata(Xv[:, mask], axis=1) / mask.sum()
    mu, sd = Xv[is_train].mean(0), Xv[is_train].std(0) + 1e-8
    return (Xv - mu) / sd

# ---------------- diagnostics ----------------
def kbet_rate(E, b, k=25):
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E)
    idx = nn.kneighbors(E, return_distance=False)[:, 1:]
    p_glob = (b == 'TCGA').mean()
    obs = (b[idx] == 'TCGA').sum(1)
    expct = np.array([k * (1 - p_glob), k * p_glob])
    chi = ((k - obs - expct[0]) ** 2) / expct[0] + ((obs - expct[1]) ** 2) / expct[1]
    return float((stats.chi2.sf(chi, 1) < 0.05).mean())

def batch_auc(Xh, mask):
    yb = (batch[mask] == 'TCGA').astype(int)
    if yb.min() == yb.max():
        return np.nan
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    pr = np.zeros(mask.sum())
    for a, b2 in cv.split(Xh[mask], yb, patient[mask]):
        pr[b2] = LogisticRegression(C=0.1, max_iter=5000).fit(Xh[mask][a], yb[a]).predict_proba(Xh[mask][b2])[:, 1]
    return roc_auc_score(yb, pr)

def tstats(Xh, bb):
    m = (batch == bb) & np.isin(cls, SHARED)
    return stats.ttest_ind(Xh[m & (cls == 'primary')], Xh[m & (cls == 'normal')], equal_var=False)[0]

if PART == 'diagnostics':
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        import umap
    except ImportError:
        umap = None
    ct = pd.crosstab(pd.Series(cls, name='class'), pd.Series(batch, name='batch'))
    ct_split = pd.crosstab([pd.Series(cls, name='class'), pd.Series(np.where(is_train, 'train', 'test'), name='split')],
                           pd.Series(batch, name='batch'))
    ct.to_csv(OUT / 'class_by_batch.csv'); ct_split.to_csv(OUT / 'class_by_batch_by_split.csv')
    sh = np.isin(cls, SHARED)
    t_raw_T, t_geo = tstats(X.values, 'TCGA'), tstats(X.values, 'GEO')
    rows = []
    ncols = 4 if umap else 2
    fig, axes = plt.subplots(len(METHODS), ncols, figsize=(4.2 * ncols, 3.6 * len(METHODS)))
    for r_i, mth in enumerate(METHODS):
        Xh = harmonize(mth)
        pca = PCA(n_components=10, random_state=0).fit(Xh[is_train & sh])
        E = pca.transform(Xh[sh])
        bsh, csh = batch[sh], cls[sh]
        rec = {'method': mth,
               'silhouette_batch': silhouette_score(E, bsh), 'silhouette_class': silhouette_score(E, csh),
               'kbet_reject_normal': kbet_rate(E[csh == 'normal'], bsh[csh == 'normal']),
               'kbet_reject_primary': kbet_rate(E[csh == 'primary'], bsh[csh == 'primary']),
               'batch_auroc_within_normal': batch_auc(Xh, sh & (cls == 'normal')),
               'batch_auroc_within_primary': batch_auc(Xh, sh & (cls == 'primary')),
               'pc1_var': pca.explained_variance_ratio_[0]}
        tT = tstats(Xh, 'TCGA')
        rec['de_sign_preserved_vs_raw_tcga'] = float((np.sign(tT) == np.sign(t_raw_T)).mean())
        rec['de_sign_concordance_tcga_vs_geo'] = float((np.sign(tT) == np.sign(t_geo)).mean())
        rec['de_spearman_tcga_vs_geo'] = stats.spearmanr(tT, t_geo)[0]
        m_tr = is_train & sh
        pooled_t = stats.ttest_ind(Xh[m_tr & (cls == 'primary')], Xh[m_tr & (cls == 'normal')], equal_var=False)[0]
        rec['pooled_train_de_sign_vs_geo'] = float((np.sign(pooled_t) == np.sign(t_geo)).mean())
        rows.append(rec)
        embs = [('PCA', E)]
        if umap:
            embs.append(('UMAP', umap.UMAP(random_state=0, n_neighbors=20).fit_transform(Xh[sh])))
        for e_i, (ename, EE) in enumerate(embs):
            for c_i, (lab, vals, cmap) in enumerate((('batch', bsh, {'GEO': '#1f77b4', 'TCGA': '#d62728'}),
                                                     ('class', csh, {'normal': '#2ca02c', 'primary': '#9467bd'}))):
                ax = axes[r_i, e_i * 2 + c_i]
                for v, col in cmap.items():
                    mm = vals == v
                    ax.scatter(EE[mm, 0], EE[mm, 1], s=6, c=col, label=v, alpha=0.7)
                ax.set_title(f'{mth} | {ename} by {lab}', fontsize=9)
                ax.set_xticks([]); ax.set_yticks([])
                if r_i == 0:
                    ax.legend(fontsize=7, markerscale=2)
    plt.tight_layout()
    plt.savefig(OUT / 'harmonization_embeddings.png', dpi=110)
    diag = pd.DataFrame(rows).set_index('method')
    diag.to_csv(OUT / 'harmonization_diagnostics.csv')
    print(ct.to_string()); print(ct_split.to_string()); print(diag.round(3).to_string())
    print('umap', bool(umap), 'nodes', len(mi), len(mr))
    sys.exit(0)

# ---------------- evaluation ----------------
def geo_nets():
    out = {}
    for c in CLASSES:
        e = pd.read_csv(REPO / f'results/networks/{c}_edges_top150.csv').rename(columns={'pearson_r': 'r'})
        out[c] = edges_to_tensor(e[e.source.isin(nodes) & e.target.isin(nodes)], nodes)
    return out

def cohort_metrics(prob, m):
    yy = np.array([CLASSES.index(c) for c in cls[m]])
    s = 1 - prob[:, 0]
    tum = (yy != 0).astype(int)
    out = {'auroc_normal_vs_tumor': roc_auc_score(tum, s),
           'f1_normal_vs_tumor': f1_score(tum, (prob.argmax(1) != 0).astype(int), average='macro')}
    if (batch[m] == 'GEO').all():
        out['macro_f1_3class'] = f1_score(yy, prob.argmax(1), average='macro', labels=[0, 1, 2])
    return out

if PART == 'eval':
    method = sys.argv[4]
    n_gat = int(sys.argv[5]) if len(sys.argv) > 5 else 10
    n_xgb = int(sys.argv[6]) if len(sys.argv) > 6 else 20
    Xh = harmonize(method).astype(np.float32)
    yall = np.array([CLASSES.index(c) for c in cls])
    node_types = torch.tensor([0] * len(mi) + [1] * len(mr))
    nets = geo_nets()
    exps = {'pooled': (is_train, is_test),
            'GEO->TCGA': (is_train & (batch == 'GEO'), is_test & (batch == 'TCGA')),
            'TCGA->GEO': (is_train & (batch == 'TCGA'), is_test & (batch == 'GEO'))}
    rows = []
    for exp, (trm, tem) in exps.items():
        tr_idx = np.where(trm)[0]
        sg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        fj, vj = next(sg.split(tr_idx, yall[tr_idx], patient[tr_idx]))
        fi, vi = tr_idx[fj], tr_idx[vj]
        te_idx = np.where(tem)[0]
        def record(model, seed, prob):
            rec = {'method': method, 'experiment': exp, 'model': model, 'seed': seed}
            for bb in ('GEO', 'TCGA'):
                mm = batch[te_idx] == bb
                if mm.any():
                    for k, v in cohort_metrics(prob[mm], te_idx[mm]).items():
                        rec[f'{bb}_{k}'] = v
                    for c_i, cname in enumerate(SHARED):
                        pass
            if exp == 'pooled':
                rec['pooled_macro_f1_3class'] = f1_score(yall[te_idx], prob.argmax(1), average='macro', labels=[0, 1, 2])
                for cname in SHARED:
                    mm = cls[te_idx] == cname
                    yb = (batch[te_idx][mm] == 'TCGA').astype(int)
                    rec[f'batch_auroc_from_pred_within_{cname}'] = roc_auc_score(yb, prob[mm][:, 0])
            rows.append(rec)
        for seed in range(n_gat):
            p = train_gat(Xh[fi], yall[fi], Xh[vi], yall[vi], Xh[te_idx], nets, node_types, 3, seed)
            record('GAT (GEO networks)', seed, p)
        for seed in range(n_xgb):
            ytr_e = yall[tr_idx]
            present = np.unique(ytr_e)
            enc = {c: i for i, c in enumerate(present)}
            m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                              random_state=seed, eval_metric='mlogloss').fit(Xh[tr_idx], np.array([enc[c] for c in ytr_e]))
            pp = np.zeros((len(te_idx), 3)); pp[:, present] = m.predict_proba(Xh[te_idx])
            record('XGBoost', seed, pp)
        lr = LogisticRegression(C=0.1, max_iter=5000, class_weight='balanced').fit(Xh[tr_idx], yall[tr_idx])
        pp = np.zeros((len(te_idx), 3)); pp[:, lr.classes_] = lr.predict_proba(Xh[te_idx])
        record('Logistic regression (L2)', 0, pp)
        print(exp, 'done', flush=True)
    pd.DataFrame(rows).to_csv(OUT / f'eval_{method}.csv', index=False)
    print(pd.DataFrame(rows).groupby(['experiment', 'model']).mean(numeric_only=True).round(3).to_string())
    sys.exit(0)

if PART == 'summarize':
    ev = pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('eval_*.csv'))])
    agg = ev.groupby(['experiment', 'method', 'model']).agg(['mean', 'std']).drop(columns='seed', level=0)
    agg.columns = [f'{a}_{b}' for a, b in agg.columns]
    agg = agg.dropna(axis=1, how='all')
    agg.to_csv(OUT / 'harmonization_eval_summary.csv')
    print(agg.round(3).to_string())
