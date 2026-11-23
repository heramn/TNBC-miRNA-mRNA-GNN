#!/usr/bin/env python3
"""Validation of the frozen TCGA edge list (TCGA only; no GEO data). Run after the freeze.

1. Split-half replication: 100 splits of the 94 purity-called TNBC tumours into 47/47, stratified by
   nodal status. In half A, the top-K most negative purity-adjusted partial Spearman pairs on the
   reference node subset (74 miRNA x 157 mRNA, K = 394, as in the GEO arm) are selected; the
   replication rate is the share with partial r < 0 and one-sided p < 0.05 in half B. Null: K random
   pairs from the same subset, scored in the same half B.
2. Sign agreement in non-TNBC BRCA primary tumours with ABSOLUTE purity: share of frozen pairs with
   negative purity-adjusted partial Spearman, against 1,000 random pair sets of the same size.
3. miRTarBase 9.0 enrichment (strong-evidence MTIs when the evidence field is available), Fisher.
4. miRDB v6.0 (score > 70) pairs vs all other pairs in TNBC tumours (purity-adjusted), and
   overlap of the frozen TCGA pairs with miRDB.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib
import mappings as mp

te = importlib.import_module('01_tcga_edges')
CACHE = te.CACHE
OUT = mp.REPO / 'results/study_final/edge_validation'
OUT.mkdir(parents=True, exist_ok=True)
REF = json.load(open(mp.FROZEN / 'tcga_edges_info.json'))['reference_190_build']
K = REF['mirdb_pairs_mappable_subset_K']


def main():
    A = pd.read_csv(CACHE / 'tcga_mirna_on_geo_panel.csv.gz', index_col=0)
    B = pd.read_csv(CACHE / 'tcga_mrna_on_geo_panel.csv.gz', index_col=0)
    meta = pd.read_csv(CACHE / 'tcga_meta.csv', index_col=0)
    ranked = pd.read_csv(mp.FROZEN / 'tcga_tumor_edges_ranked.csv')
    P = mp.REPO / 'data/processed'
    de_mi = list(pd.read_csv(P / 'train_mirna.csv', index_col=0, nrows=0).columns)
    de_mr = list(pd.read_csv(P / 'train_mrna.csv', index_col=0, nrows=0).columns)
    mi = [g for g in de_mi if g in A.columns]
    mr = [g for g in de_mr if g in B.columns]
    tum = meta.index[(meta.group == 'tnbc_tumor') & meta.absolute.notna()]
    rng = np.random.default_rng(2026)
    res = {'reference_subset': {'mirna': len(mi), 'mrna': len(mr), 'K': K}}

    # 1. split-half replication
    strata = meta.loc[tum, 'node'].values
    rows = []
    for s in range(100):
        half = np.zeros(len(tum), bool)
        for st in np.unique(strata):
            idx = np.where(strata == st)[0]
            half[rng.permutation(idx)[:len(idx) // 2]] = True
        a, b = tum[half], tum[~half]
        ra, _ = te.partial_spearman(A.loc[a, mi], B.loc[a, mr], meta.loc[a, 'absolute'].values)
        rb, pb = te.partial_spearman(A.loc[b, mi], B.loc[b, mr], meta.loc[b, 'absolute'].values)
        top = np.argsort(ra.ravel())[:K]
        rnd = rng.choice(ra.size, K, replace=False)
        rep = lambda ix: float(np.mean((rb.ravel()[ix] < 0) & (pb.ravel()[ix] / 2 < 0.05)))
        rows.append({'split': s, 'n_a': len(a), 'n_b': len(b), 'replication_topK': rep(top), 'replication_random': rep(rnd),
                     'sign_agree_topK': float(np.mean(rb.ravel()[top] < 0)), 'sign_agree_random': float(np.mean(rb.ravel()[rnd] < 0))})
    sh = pd.DataFrame(rows)
    sh.to_csv(OUT / 'split_half_replication.csv', index=False)
    res['split_half'] = {k: [float(sh[k].mean()), float(sh[k].quantile(0.025)), float(sh[k].quantile(0.975))]
                         for k in ['replication_topK', 'replication_random', 'sign_agree_topK', 'sign_agree_random']}
    res['split_half']['frac_splits_topK_gt_random'] = float((sh.replication_topK > sh.replication_random).mean())

    # 2. sign agreement in non-TNBC BRCA
    non = meta.index[(meta.group == 'nontnbc_tumor') & meta.absolute.notna()]
    rn, pn = te.partial_spearman(A.loc[non, mi], B.loc[non, mr], meta.loc[non, 'absolute'].values)
    rn = pd.DataFrame(rn, index=mi, columns=mr)
    pn = pd.DataFrame(pn, index=mi, columns=mr)
    sub = ranked[ranked.probe.isin(mi) & ranked.gene.isin(mr)].head(K)
    vals = np.array([rn.at[p, g] for p, g in zip(sub.probe, sub.gene)])
    pv = np.array([pn.at[p, g] for p, g in zip(sub.probe, sub.gene)])
    null = [float(np.mean(rng.choice(rn.values.ravel(), len(vals), replace=False) < 0)) for _ in range(1000)]
    res['nontnbc_sign_agreement'] = {'n_tumours': int(len(non)), 'pairs': int(len(vals)),
                                     'frac_negative': float(np.mean(vals < 0)),
                                     'frac_negative_p05': float(np.mean((vals < 0) & (pv < 0.05))),
                                     'random_frac_negative_mean_95': [float(np.mean(null)), float(np.quantile(null, 0.025)), float(np.quantile(null, 0.975))],
                                     'empirical_p': float((1 + np.sum(np.array(null) >= np.mean(vals < 0))) / 1001)}

    # 3/4. prior databases on the whole universe (all frozen pairs) and on the reference subset
    allp = pd.read_csv(mp.FROZEN / 'tcga_tumor_pairs_all_absolute.csv.gz')
    allp['tcga_edge'] = (allp.q < 0.05) & (allp.r < 0)
    md = mp.mirdb_pairs(70)
    allp['mirdb'] = pd.MultiIndex.from_frame(allp[['probe', 'gene']]).isin(pd.MultiIndex.from_frame(md[['probe', 'gene']]))
    mtb = OUT.parent / 'mirtarbase_pairs.csv'
    dbs = ['mirdb'] + (['mirtarbase_strong', 'mirtarbase_any'] if mtb.exists() else [])
    if mtb.exists():
        t = pd.read_csv(mtb)
        for col in ('mirtarbase_strong', 'mirtarbase_any'):
            s = t[t[col]]
            allp[col] = pd.MultiIndex.from_frame(allp[['probe', 'gene']]).isin(pd.MultiIndex.from_frame(s[['probe', 'gene']]))
    enr = []
    for scope, df in (('universe', allp), ('reference_subset', allp[allp.probe.isin(mi) & allp.gene.isin(mr)])):
        for db in dbs:
            for edge_def, sel in (('frozen_q05_negative', df.tcga_edge),
                                  ('topK_reference', df.set_index(['probe', 'gene']).index.isin(list(zip(sub.probe, sub.gene))))):
                if scope == 'universe' and edge_def == 'topK_reference':
                    continue
                tab = [[int((sel & df[db]).sum()), int((sel & ~df[db]).sum())], [int((~sel & df[db]).sum()), int((~sel & ~df[db]).sum())]]
                orr, p = stats.fisher_exact(tab, alternative='greater')
                enr.append({'scope': scope, 'prior': db, 'edges': edge_def, 'edges_in_prior': tab[0][0], 'n_edges': int(sel.sum()),
                            'prior_pairs': int(df[db].sum()), 'pairs': int(len(df)), 'odds_ratio': orr, 'fisher_p_greater': p})
        for db in dbs:
            x, y = df.loc[df[db], 'r'], df.loc[~df[db], 'r']
            enr.append({'scope': scope, 'prior': db, 'edges': 'partial_r_prior_vs_rest', 'edges_in_prior': int(len(x)),
                        'mean_r_prior': float(x.mean()), 'mean_r_rest': float(y.mean()),
                        'frac_sig_negative_prior': float(((df.q < 0.05) & (df.r < 0))[df[db]].mean()),
                        'frac_sig_negative_rest': float(((df.q < 0.05) & (df.r < 0))[~df[db]].mean()),
                        'mannwhitney_p_prior_more_negative': float(stats.mannwhitneyu(x, y, alternative='less').pvalue)})
    pd.DataFrame(enr).to_csv(OUT / 'prior_enrichment.csv', index=False)
    res['prior_enrichment'] = enr
    sens = {}
    for name, f in (('estimate', 'tcga_tumor_edges_ranked_estimate_sensitivity.csv'), ('normal', 'tcga_normal_edges_ranked_sensitivity.csv')):
        o = pd.read_csv(mp.FROZEN / f)
        a = set(zip(ranked.probe, ranked.gene))
        bset = set(zip(o.probe, o.gene))
        sens[name] = {'pairs': len(bset), 'overlap_with_primary': len(a & bset), 'jaccard': len(a & bset) / len(a | bset)}
    res['sensitivity_overlap'] = sens
    json.dump(res, open(OUT / 'edge_validation_summary.json', 'w'), indent=1, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != 'prior_enrichment'}, indent=1, default=float))
    print(pd.DataFrame(enr).round(4).to_string())


if __name__ == '__main__':
    main()
