#!/usr/bin/env python3
"""Discover TCGA-BRCA TNBC miRNA-mRNA anti-correlated pairs and FREEZE them (study-final, design B).

TCGA is used ONLY to discover edges. No GEO sample or label is read here except the GEO probe
and gene NAMES (panel universe) and, for the descriptive reference count at the end, the DE node
list of the existing 190-sample build (names only).

Primary edge set (pre-specified):
  cohort      TCGA-BRCA primary tumours (sample type 01) of IHC-TNBC patients with miRNA-seq and
              RNA-seq on the same sample barcode (UCSC Xena TCGA.BRCA.sampleMap), restricted to
              those with an ABSOLUTE purity call (PanCanAtlas mastercalls; complete-case).
  statistic   partial Spearman correlation adjusted for ABSOLUTE purity (rank-transform, residualise
              each rank vector on purity rank, Pearson of residuals; t-test with n-3 df).
  universe    every mappable GEO miRNA probe x every mappable GEO panel mRNA.
  selection   negative correlations with BH q < 0.05 over the whole universe, ranked by |r|.
  per fold    K_f = number of miRDB (score > 70) pairs on the fold's TCGA-mappable DE node subset;
              the TCGA arm takes the top-K_f ranked pairs inside the same subset.
  edges       undirected, binary; called "anti-correlated pairs consistent with repression".
Sensitivities (not primary): ESTIMATE-derived purity on all tumours; TCGA normals (plain Spearman).

Usage: 01_tcga_edges.py   (reads ~/.capy/work/ext, writes study/frozen/ and ~/.capy/work/tcga_cache/)
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mappings as mp

EXT, FROZEN = mp.EXT, mp.FROZEN
CACHE = Path.home() / '.capy/work/tcga_cache'
FROZEN.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
Q = 0.05


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def tnbc_patients(clin):
    n = lambda s: s.astype(str).str.strip().str.lower()
    er, pr = n(clin['breast_carcinoma_estrogen_receptor_status']), n(clin['breast_carcinoma_progesterone_receptor_status'])
    hi, hs = n(clin['lab_proc_her2_neu_immunohistochemistry_receptor_status']), n(clin['lab_procedure_her2_neu_in_situ_hybrid_outcome_type'])
    h2n = (hi == 'negative') | ((hi == 'equivocal') & (hs == 'negative')) | (hi.isin(['nan', 'indeterminate', '']) & (hs == 'negative'))
    h2p = (hi == 'positive') | (hs == 'positive')
    return set(clin.loc[(er == 'negative') & (pr == 'negative') & h2n & ~h2p, '_PATIENT'])


def probe_matrix(mir, probe_map):
    """TCGA log2(RPM+1) per GEO probe; merged probes = log2(sum of member RPM + 1)."""
    cols = {}
    for probe, g in probe_map.groupby('probe', sort=False):
        mem = [m for m in g.mimat if m in mir.columns]
        if not mem:
            continue
        cols[probe] = np.log2((2 ** mir[mem] - 1).clip(lower=0).sum(1) + 1) if len(mem) > 1 else mir[mem[0]]
    return pd.DataFrame(cols)


def partial_spearman(A, B, z=None):
    """r[i, j] = Spearman(A_i, B_j | z) with p-values (t, n - 2 - k df)."""
    ra = np.apply_along_axis(stats.rankdata, 0, A.values)
    rb = np.apply_along_axis(stats.rankdata, 0, B.values)
    n = ra.shape[0]
    if z is not None:
        Z = np.column_stack([np.ones(n), stats.rankdata(z)])
        P = Z @ np.linalg.pinv(Z)
        ra, rb, k = ra - P @ ra, rb - P @ rb, 1
    else:
        k = 0
    ra = (ra - ra.mean(0)) / (ra.std(0) + 1e-12)
    rb = (rb - rb.mean(0)) / (rb.std(0) + 1e-12)
    r = np.clip(ra.T @ rb / n, -0.999999, 0.999999)
    df = n - 2 - k
    p = 2 * stats.t.sf(np.abs(r * np.sqrt(df / (1 - r ** 2))), df)
    return r, p


def pair_table(A, B, z=None):
    r, p = partial_spearman(A, B, z)
    ii, jj = np.meshgrid(np.arange(A.shape[1]), np.arange(B.shape[1]), indexing='ij')
    t = pd.DataFrame({'probe': A.columns[ii.ravel()], 'gene': B.columns[jj.ravel()], 'r': r.ravel(), 'p': p.ravel()})
    t['q'] = multipletests(t.p, method='fdr_bh')[1]
    return t


def ranked_negative(t):
    e = t[(t.q < Q) & (t.r < 0)].copy()
    e['abs_r'] = e.r.abs()
    e = e.sort_values(['abs_r', 'probe', 'gene'], ascending=[False, True, True]).reset_index(drop=True)
    e['rank'] = np.arange(1, len(e) + 1)
    return e[['rank', 'probe', 'gene', 'r', 'p', 'q']]


def main():
    xena = EXT / 'xena'
    clin = pd.read_csv(xena / 'BRCA_clinicalMatrix', sep='\t', index_col=0, low_memory=False)
    mrna = pd.read_csv(xena / 'HiSeqV2.gz', sep='\t', index_col=0)
    mir = pd.read_csv(xena / 'miRNA_HiSeq_gene.gz', sep='\t', index_col=0)

    # frozen mapping tables
    gpl = mp.gpl16231(EXT / 'gpl/GPL16231.txt')
    gpl.to_csv(FROZEN / 'GPL16231_annotation.tsv', sep='\t', index=False)
    pmap = mp.build_mirna_probe_table(FROZEN / 'GPL16231_annotation.tsv', EXT / 'mirbase/mature.fa')
    pmap.to_csv(FROZEN / 'mirna_probe_map.csv', index=False)
    smap = mp.build_mrna_symbol_table(list(mrna.index))
    smap.to_csv(FROZEN / 'mrna_symbol_map.csv', index=False)

    tn = tnbc_patients(clin)
    both = sorted(set(mrna.columns) & set(mir.columns))
    tum = [s for s in both if s[13:15] == '01' and s[:12] in tn]
    nontn = [s for s in both if s[13:15] == '01' and s[:12] not in tn]
    nor = [s for s in both if s[13:15] == '11']
    absl = pd.read_csv(EXT / 'purity/TCGA_mastercalls.abs_tables_JSedit.fixed.txt', sep='\t').set_index('array')['purity']
    est = pd.read_csv(EXT / 'estimate/breast_RNAseqV2.txt', sep='\t').set_index('ID')['ESTIMATE_score']
    est_purity = np.cos(0.6049872018 + 0.0001467884 * est)
    pur = pd.DataFrame({'absolute': [absl.get(s[:15], np.nan) for s in both],
                        'estimate': [est_purity.get(s[:15], np.nan) for s in both]}, index=both)

    # TCGA features on the GEO panels
    mirT = mir[both].T
    keep = mirT.notna().mean() >= 0.9
    mirT = mirT.loc[:, keep].fillna(0.0)
    pm = pmap[pmap.mimat != '']
    A_all = probe_matrix(mirT, pm)
    A_all = A_all.loc[:, A_all.loc[tum].std() > 0]
    gm = smap[smap.tcga_symbol != '']
    B_all = mrna.loc[gm.tcga_symbol, both].T
    B_all.columns = list(gm.gene)
    B_all = B_all.loc[:, B_all.loc[tum].std() > 0]
    for name, df in (('mirna', A_all), ('mrna', B_all)):
        df.to_csv(CACHE / f'tcga_{name}_on_geo_panel.csv.gz')
    pd.DataFrame({'modality': ['miRNA'] * A_all.shape[1] + ['mRNA'] * B_all.shape[1],
                  'node': list(A_all.columns) + list(B_all.columns)}).to_csv(FROZEN / 'tcga_mappable_nodes.csv', index=False)
    meta = pd.DataFrame({'patient': [s[:12] for s in both],
                         'group': ['tnbc_tumor' if s in tum else 'nontnbc_tumor' if s in nontn else 'normal' if s in nor else 'other'
                                   for s in both]}, index=both).join(pur)
    node = clin.drop_duplicates('_PATIENT').set_index('_PATIENT')['pathologic_N'].astype(str)
    meta['node'] = [('N0' if node.get(p, 'nan').startswith('N0') else 'Npos' if node.get(p, 'nan')[:2] in ('N1', 'N2', 'N3') else 'NA')
                    for p in meta.patient]
    meta.to_csv(CACHE / 'tcga_meta.csv')

    # primary: TNBC tumours with ABSOLUTE purity, partial Spearman | purity
    t_abs = [s for s in tum if pd.notna(pur.at[s, 'absolute'])]
    prim = pair_table(A_all.loc[t_abs], B_all.loc[t_abs], pur.loc[t_abs, 'absolute'].values)
    prim.to_csv(FROZEN / 'tcga_tumor_pairs_all_absolute.csv.gz', index=False, float_format='%.6g')
    ranked = ranked_negative(prim)
    ranked.to_csv(FROZEN / 'tcga_tumor_edges_ranked.csv', index=False, float_format='%.6g')

    # sensitivities
    t_est = [s for s in tum if pd.notna(pur.at[s, 'estimate'])]
    sens_est = ranked_negative(pair_table(A_all.loc[t_est], B_all.loc[t_est], pur.loc[t_est, 'estimate'].values))
    sens_est.to_csv(FROZEN / 'tcga_tumor_edges_ranked_estimate_sensitivity.csv', index=False, float_format='%.6g')
    nA = A_all.loc[nor]
    nA = nA.loc[:, nA.std() > 0]
    sens_nor = ranked_negative(pair_table(nA, B_all.loc[nor]))
    sens_nor.to_csv(FROZEN / 'tcga_normal_edges_ranked_sensitivity.csv', index=False, float_format='%.6g')

    # descriptive reference count on the existing 190-sample build DE nodes (names only)
    ref = {}
    P = mp.REPO / 'data/processed'
    if (P / 'train_mirna.csv').exists():
        de_mi = list(pd.read_csv(P / 'train_mirna.csv', index_col=0, nrows=0).columns)
        de_mr = list(pd.read_csv(P / 'train_mrna.csv', index_col=0, nrows=0).columns)
        mi_map = [g for g in de_mi if g in A_all.columns]
        mr_map = [g for g in de_mr if g in B_all.columns]
        md = mp.mirdb_pairs()
        k_full = int((md.probe.isin(de_mi) & md.gene.isin(de_mr)).sum())
        k_sub = int((md.probe.isin(mi_map) & md.gene.isin(mr_map)).sum())
        sub = ranked[ranked.probe.isin(mi_map) & ranked.gene.isin(mr_map)]
        ref = {'de_mirna': len(de_mi), 'de_mrna': len(de_mr), 'mappable_mirna': len(mi_map), 'mappable_mrna': len(mr_map),
               'unmappable_de_mirna': [g for g in de_mi if g not in mi_map],
               'unmappable_de_mrna': [g for g in de_mr if g not in mr_map],
               'mirdb_pairs_full_250': k_full, 'mirdb_pairs_mappable_subset_K': k_sub,
               'tcga_sig_negative_pairs_in_subset': int(len(sub)),
               'tcga_topK_min_abs_r': float(sub.r.abs().iloc[min(k_sub, len(sub)) - 1]) if len(sub) else None,
               'tcga_topK_overlap_with_mirdb': int(len(set(zip(sub.probe[:k_sub], sub.gene[:k_sub])) &
                                                     set(zip(md.probe, md.gene))))}

    info = {
        'cohort': {'tnbc_patients_ihc': len(tn), 'tnbc_tumours_both_modalities': len(tum),
                   'tnbc_tumours_with_absolute_purity': len(t_abs), 'tnbc_tumours_with_estimate_purity': len(t_est),
                   'nontnbc_primary_tumours_both_modalities': len(nontn), 'normals_both_modalities': len(nor),
                   'absolute_purity_median_iqr': [float(np.nanmedian(pur.loc[t_abs, 'absolute'])),
                                                  float(np.nanpercentile(pur.loc[t_abs, 'absolute'], 25)),
                                                  float(np.nanpercentile(pur.loc[t_abs, 'absolute'], 75))],
                   'spearman_absolute_vs_estimate_purity': float(stats.spearmanr(pur.loc[t_abs, 'absolute'], pur.loc[t_abs, 'estimate'], nan_policy='omit')[0])},
        'mapping': {'geo_mirna_probes': int(pmap.probe.nunique()),
                    'probes_with_mimat': int(pmap[pmap.mimat != ''].probe.nunique()),
                    'probes_with_mirbase_v22_name': int(pmap[pmap.mirbase_v22 != ''].probe.nunique()),
                    'merged_probes': int(pmap[pmap.merged_probe].probe.nunique()),
                    'merged_probe_handling': 'TCGA: sum of member arms on the linear RPM scale; miRDB: union of member targets (max score)',
                    'probes_on_tcga_after_filter': int(A_all.shape[1]),
                    'panel_mrna': int(len(smap)), 'mrna_on_tcga': int(B_all.shape[1]),
                    'mrna_recovered_by_hgnc_alias': smap[smap.how == 'hgnc_alias'][['gene', 'tcga_symbol']].values.tolist(),
                    'mrna_unmapped': smap[smap.how == 'unmapped'].gene.tolist()},
        'primary_edges': {'pairs_tested': int(len(prim)), 'negative_q_lt_0.05': int(len(ranked)),
                          'positive_q_lt_0.05': int(((prim.q < Q) & (prim.r > 0)).sum())},
        'sensitivity_edges': {'estimate_purity_negative_q_lt_0.05': int(len(sens_est)),
                              'normal_negative_q_lt_0.05': int(len(sens_nor)),
                              'overlap_top500_absolute_vs_estimate': int(len(set(zip(ranked.probe[:500], ranked.gene[:500])) &
                                                                          set(zip(sens_est.probe[:500], sens_est.gene[:500]))))},
        'reference_190_build': ref,
        'input_sha256': {p.name: sha(p) for p in [EXT / 'xena/HiSeqV2.gz', EXT / 'xena/miRNA_HiSeq_gene.gz', EXT / 'xena/BRCA_clinicalMatrix',
                                                   EXT / 'purity/TCGA_mastercalls.abs_tables_JSedit.fixed.txt', EXT / 'estimate/breast_RNAseqV2.txt',
                                                   EXT / 'mirbase/mature.fa', EXT / 'gpl/GPL16231.txt', EXT / 'hgnc_complete_set.txt',
                                                   mp.RAW / 'miRDB_v6.0_prediction_result.txt.gz']},
        'output_sha256': {p.name: sha(p) for p in sorted(FROZEN.glob('*')) if p.is_file() and p.suffix != '.json'},
    }
    json.dump(info, open(FROZEN / 'tcga_edges_info.json', 'w'), indent=2, default=str)
    print(json.dumps({k: v for k, v in info.items() if k not in ('input_sha256', 'output_sha256')}, indent=1, default=str))


if __name__ == '__main__':
    main()
