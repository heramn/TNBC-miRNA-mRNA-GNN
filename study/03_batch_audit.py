#!/usr/bin/env python3
"""Batch audit from the deposited GEO files only (no author contact). Writes
results/study_final/batch_audit/*.csv. The deposited data are lane-normalised counts without the
NanoString RCC files, so cartridge, lane, scan date and binding density cannot be audited; the RCC
files would allow a real batch analysis.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import geo_data

OUT = geo_data.REPO / 'results/study_final/batch_audit'
OUT.mkdir(parents=True, exist_ok=True)
CLS = ['normal', 'primary', 'ln_metastasis']


def main():
    d = geo_data.load()
    t = d['samples']
    rows = []
    for plat, m in (('GPL16231', 0), ('GPL16299', 1)):
        sm = geo_data.series_matrix(geo_data.RAW / f'GSE45498-{plat}_series_matrix.txt.gz').set_index('geo_accession')
        tm = t[t.modality == m]
        sm = sm.loc[tm.gsm]
        for col in ('submission_date', 'last_update_date', 'source_name_ch1', 'tissue', 'extract_protocol_ch1', 'label_protocol_ch1',
                    'data_processing', 'hyb_protocol', 'scan_protocol', 'contact_name'):
            if col in sm:
                for (val, y), n in pd.Series(list(zip(sm[col].str[:120], tm.y.map(dict(enumerate(CLS)))))).value_counts().items():
                    rows.append({'platform': plat, 'field': col, 'value': val, 'class': y, 'n': int(n)})
    fields = pd.DataFrame(rows)
    fields.to_csv(OUT / 'series_matrix_fields_by_class.csv', index=False)
    # raw-header project prefixes (mRNA) and naming (miRNA)
    pref = lambda s: re.sub(r'[\d_ ]+[A-Za-z]?$', '', s).strip() or s
    hdr = t.assign(prefix=t.raw_col.map(lambda s: re.match(r'^(TAS\d\d[- ]\d\d|Normal|Tumor|Mets)', s).group(1) if re.match(r'^(TAS\d\d[- ]\d\d|Normal|Tumor|Mets)', s) else pref(s)),
                   cls=t.y.map(dict(enumerate(CLS))))
    hdr.groupby(['modality', 'prefix', 'cls']).size().rename('n').reset_index().to_csv(OUT / 'raw_header_prefix_by_class.csv', index=False)
    # control probes and value granularity
    feats = []
    for m in (0, 1):
        ctrl = d['controls'][m]
        cnt = d['counts'][m]
        ids = cnt.index
        f = pd.DataFrame(index=ids)
        pos = ctrl[[c for c in ctrl.columns if 'POS_' in c]]
        neg = ctrl[[c for c in ctrl.columns if 'NEG_' in c]]
        f['pos_spike_geomean'] = np.exp(np.log(pos.clip(lower=1)).mean(1))
        f['neg_background_mean'] = neg.mean(1)
        hk = ctrl[[c for c in ctrl.columns if not ('POS_' in c or 'NEG_' in c)]]
        if hk.shape[1]:
            f['housekeeping_geomean'] = np.exp(np.log(hk.clip(lower=1)).mean(1))
        f['endogenous_total_counts'] = cnt.sum(1)
        f['frac_endogenous_below_neg_max'] = (cnt.values < neg.max(1).values[:, None]).mean(1)
        f['frac_non_integer_values'] = (np.abs(cnt.values - np.round(cnt.values)) > 1e-9).mean(1)
        f['modality'] = m
        f['y'] = t.loc[ids, 'y']
        feats.append(f)
    feats = pd.concat(feats)
    feats.to_csv(OUT / 'per_sample_qc.csv')
    rows = []
    for m, g in feats.groupby('modality'):
        for col in [c for c in g.columns if c not in ('modality', 'y')]:
            v = g[col].dropna()
            if v.nunique() < 2:
                continue
            kw = stats.kruskal(*[v[g.y == c] for c in range(3) if (g.y == c).any()]).pvalue
            mp_ = g[g.y > 0]
            au = roc_auc_score(mp_.y == 2, mp_[col])
            rows.append({'modality': ['miRNA', 'mRNA'][m], 'qc_feature': col, **{f'median_{CLS[c]}': g.loc[g.y == c, col].median() for c in range(3)},
                         'kruskal_p': kw, 'auroc_met_vs_primary': au, 'auroc_distance_from_0.5': abs(au - 0.5)})
    pd.DataFrame(rows).to_csv(OUT / 'qc_by_class.csv', index=False)
    print(pd.DataFrame(rows).round(4).to_string())
    print(hdr.groupby(['modality', 'prefix', 'cls']).size().to_string())
    print(fields[fields.field.isin(['submission_date', 'data_processing'])].to_string()[:3000])


if __name__ == '__main__':
    main()
