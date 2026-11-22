"""GSE45498 per-modality data for the pairing-free study (study-final).

Every confident profile of each modality is one example with its own GEO label:
  miRNA: 282 GPL16231 profiles (raw columns verified 282/282 against their GSM titles)
  mRNA : 249 GPL16299 profiles whose raw column is confidently matched to its GSM
         (metadata/mrna_raw_to_gsm_mapping.csv; 5 profiles without a confident match are excluded)
Groups: mRNA = patient id (union of GEO-title id and raw-header id, as in notebooks/01);
miRNA = one group per profile (GEO gives no miRNA patient id; correlation shows no near-duplicate
structure, max nearest-neighbour Spearman 0.946 with no gap). The 'conservative' scheme links each
miRNA profile to the mRNA patient of its order-alignment partner (repro/inferred_*_alignment.csv).
"""
import gzip
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / 'data/raw'
STATUS = {'normal': 0, 'primary triple negative cancer': 1, 'Metastatic': 2}


def series_matrix(path):
    out = {}
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if not line.startswith('!Sample_'):
                continue
            key, *vals = line.rstrip('\n').split('\t')
            vals = [v.strip().strip('"') for v in vals]
            if key == '!Sample_characteristics_ch1':
                k = vals[0].split(':')[0]
                out[k] = [v.split(':', 1)[1].strip() if ':' in v else v for v in vals]
            else:
                out[key[8:]] = vals
    return pd.DataFrame(out)


def _uf():
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)
    return find, union


def _title_pid(title):
    t = str(title).replace('_mRNA', '').strip()
    m = re.match(r'^(\d+)[NT]', t)
    if m:
        return 'P' + m.group(1)
    m = re.match(r'^TAS10-15-(\d+)', t)
    return 'MET' + m.group(1) if m else 'P_' + t


def _header_pid(raw_col):
    m = re.match(r'^(?:TAS09 12 |Normal_)(\d+)', raw_col)
    if m:
        return 'P' + m.group(1)
    m = re.match(r'^TAS10-15-(\d+)', raw_col)
    return 'MET' + m.group(1) if m else 'P_' + raw_col


@lru_cache(maxsize=1)
def load():
    """Returns dict with counts/controls per modality and a sample table
    (index = example id 'mi:<title>' / 'mr:<title>'; columns gsm, title, modality, y, group, group_conservative, raw_col)."""
    # miRNA
    mi_raw = pd.read_csv(RAW / 'GSE45498_raw_data.txt.gz', sep='\t')
    mi_meta = series_matrix(RAW / 'GSE45498-GPL16231_series_matrix.txt.gz')
    endo = mi_raw[mi_raw['Code Class'] == 'Endogenous1'].set_index('Name').iloc[:, 2:]
    ctrl = mi_raw[mi_raw['Code Class'].isin(['Positive', 'Negative', 'Housekeeping'])].set_index('Name').iloc[:, 2:]
    assert list(endo.columns) and set(endo.columns) == set(mi_meta.title)
    mi_meta = mi_meta.set_index('title').loc[endo.columns]
    mi_ids = ['mi:' + t for t in endo.columns]
    mi_counts = endo.T.astype(float).set_axis(mi_ids)
    mi_ctrl = ctrl.T.astype(float).set_axis(mi_ids).add_prefix('miCTRL_')
    mi_tab = pd.DataFrame({'gsm': mi_meta.geo_accession.values, 'title': list(endo.columns), 'modality': 0,
                           'y': mi_meta.status.map(STATUS).values, 'raw_col': list(endo.columns)}, index=mi_ids)

    # mRNA
    mr_raw = pd.read_csv(RAW / 'GSE45498_mRNA_non-normalized_data.txt.gz', sep='\t')
    mr_raw.columns = [c.strip() for c in mr_raw.columns]
    mr_meta = series_matrix(RAW / 'GSE45498-GPL16299_series_matrix.txt.gz').set_index('geo_accession')
    mp = pd.read_csv(REPO / 'metadata/mrna_raw_to_gsm_mapping.csv')
    mp = mp[mp.confident]
    gsms = list(mp.best_gsm)
    raw_cols = list(mp.raw_col.str.strip())
    titles = list(mr_meta.loc[gsms, 'title'])
    mr_ids = ['mr:' + t for t in titles]
    mendo = mr_raw[mr_raw['Code Class'] == 'Endogenous'].set_index('Name')
    mctrl = mr_raw[mr_raw['Code Class'].isin(['Positive', 'Negative'])].set_index('Name')
    mr_counts = mendo[raw_cols].T.astype(float).set_axis(mr_ids)
    mr_ctrl = mctrl[raw_cols].T.astype(float).set_axis(mr_ids).add_prefix('mrCTRL_')
    find, union = _uf()
    for g, rc in zip(mp.best_gsm, mp.raw_col.str.strip()):
        union(_title_pid(mr_meta.at[g, 'title']), _header_pid(rc))
    mr_tab = pd.DataFrame({'gsm': gsms, 'title': titles, 'modality': 1,
                           'y': mr_meta.loc[gsms, 'status'].map(STATUS).values, 'raw_col': raw_cols}, index=mr_ids)
    mr_tab['group'] = ['G_' + find(_title_pid(t)) for t in titles]
    mi_tab['group'] = ['M_' + t for t in mi_tab.title]

    # conservative scheme: link miRNA profiles to the patient of their order-alignment mRNA partner
    al = pd.read_csv(REPO / 'repro/inferred_mirna_mrna_pairs_order_alignment.csv')
    gsm2grp = dict(zip(mr_tab.gsm, mr_tab.group))
    mi_gsm2id = dict(zip(mi_tab.gsm, mi_tab.index))
    cons = dict(zip(mi_tab.index, mi_tab.group)) | dict(zip(mr_tab.index, mr_tab.group))
    find2, union2 = _uf()
    for a, b in zip(al.mirna_gsm, al.mrna_gsm):
        if a in mi_gsm2id and b in gsm2grp:
            union2(cons[mi_gsm2id[a]], gsm2grp[b])
    tab = pd.concat([mi_tab, mr_tab])
    tab['group_conservative'] = [find2(cons[i]) for i in tab.index]
    assert tab.y.notna().all()
    tab['y'] = tab.y.astype(int)
    # 230-sample subset of the corrected 01 build (order-paired samples), for the sensitivity scheme
    sub = REPO / 'study/frozen/subset230_ids.csv'
    tab['in_subset230'] = tab.index.isin(pd.read_csv(sub).id) if sub.exists() else False
    return {'counts': {0: mi_counts, 1: mr_counts}, 'controls': {0: mi_ctrl, 1: mr_ctrl}, 'samples': tab}
