#!/usr/bin/env python3
"""Map every column of the raw mRNA supplementary file to its GEO sample (GSM).

Why: GSE45498_mRNA_non-normalized_data.txt.gz is ordered by patient number with
normal/tumor columns interleaved (and partly mislabelled headers), while the
GPL16299 series matrix -- the source of the tissue labels and patient titles --
is ordered metastases -> normals -> tumors. Pairing the two by column position
(the original 01_preprocessing_DE.py behaviour) attached the wrong mRNA profile
to 186 of 235 samples.

How: the series-matrix data table holds the quantile-normalised values of each
GSM. Quantile normalisation is rank-preserving within a sample, so the raw column
of a GSM has Spearman rho ~= 1 with that GSM's normalised profile. For every raw
column we compute Spearman rho against every GSM (236 endogenous genes; pairwise
complete for the one raw column with missing values) and accept a match only if
it is the mutual best match, rho >= 0.90, and it beats the runner-up by >= 0.01
in both directions. Unmatched GSMs are dropped downstream.

Output: metadata/mrna_raw_to_gsm_mapping.csv (one row per raw column).
Run from repo root:  python scripts/derive_mrna_gsm_mapping.py
"""
import collections
import gzip
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

RAW = Path('data/raw')
OUT = Path('metadata/mrna_raw_to_gsm_mapping.csv')
MIN_RHO, MIN_MARGIN = 0.90, 0.01


def read_series_matrix(path):
    meta, table, in_table = collections.defaultdict(list), [], False
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith('!series_matrix_table_begin'):
                in_table = True
            elif line.startswith('!series_matrix_table_end'):
                in_table = False
            elif in_table:
                table.append(line)
            elif line.startswith('!Sample_'):
                key, *vals = line.rstrip('\n').split('\t')
                meta[key].append([v.strip('"') for v in vals])
    return meta, pd.read_csv(io.StringIO(''.join(table)), sep='\t', index_col=0)


def characteristic(meta, key):
    for row in meta['!Sample_characteristics_ch1']:
        if row[0].startswith(key + ':'):
            return [v.split(': ', 1)[1] for v in row]
    raise KeyError(key)


def main():
    meta, norm = read_series_matrix(RAW / 'GSE45498-GPL16299_series_matrix.txt.gz')
    gsms = meta['!Sample_geo_accession'][0]
    title = dict(zip(gsms, meta['!Sample_title'][0]))
    status = dict(zip(gsms, characteristic(meta, 'status')))

    raw = pd.read_csv(RAW / 'GSE45498_mRNA_non-normalized_data.txt.gz', sep='\t')
    raw_cols = list(raw.columns[3:])
    endo = raw[raw['Code Class'] == 'Endogenous'].set_index('Name')[raw_cols].astype(float)
    genes = endo.index.intersection(norm.index)
    endo, norm = endo.loc[genes], norm.loc[genes, gsms].astype(float)

    def zrank(mat):
        r = np.apply_along_axis(rankdata, 0, mat)
        return (r - r.mean(0)) / r.std(0)

    with np.errstate(invalid='ignore', divide='ignore'):
        rho = zrank(endo.values).T @ zrank(norm.values) / len(genes)
    for k in np.where(np.isnan(rho).any(axis=1))[0]:
        col = endo.iloc[:, k]
        ok = col.notna()
        for j, g in enumerate(gsms):
            rho[k, j] = spearmanr(col[ok].values, norm.loc[ok[ok].index, g].values).correlation
    rho = np.nan_to_num(rho, nan=-1.0)

    best_raw_for_gsm = rho.argmax(axis=0)
    rows = []
    for i, col in enumerate(raw_cols):
        order = np.argsort(-rho[i])
        j, j2 = order[0], order[1]
        col_order = np.argsort(-rho[:, j])
        row_margin = rho[i, j] - rho[i, j2]
        col_margin = rho[col_order[0], j] - rho[col_order[1], j]
        mutual = best_raw_for_gsm[j] == i
        confident = bool(mutual and rho[i, j] >= MIN_RHO and row_margin >= MIN_MARGIN and col_margin >= MIN_MARGIN)
        rows.append({
            'raw_pos': i + 1, 'raw_col': col.strip(),
            'best_gsm': gsms[j], 'best_title': title[gsms[j]], 'best_status': status[gsms[j]],
            'best_rho': round(rho[i, j], 4),
            'second_gsm': gsms[j2], 'second_title': title[gsms[j2]], 'second_rho': round(rho[i, j2], 4),
            'row_margin': round(row_margin, 4), 'gsm_col_margin': round(col_margin, 4),
            'mutual_best': bool(mutual), 'confident': confident,
            'positional_gsm': gsms[i], 'positional_title': title[gsms[i]], 'positional_status': status[gsms[i]],
        })
    df = pd.DataFrame(rows)
    df['positional_correct'] = df.positional_gsm == df.best_gsm
    df['positional_status_correct'] = df.positional_status == df.best_status
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    unmatched = sorted(title[g] for g in set(gsms) - set(df.loc[df.confident, 'best_gsm']))
    print(f'raw columns: {len(df)} | confident matches: {int(df.confident.sum())} | '
          f'positional column == true GSM: {int(df.positional_correct.sum())} | '
          f'positional tissue label correct: {int(df.positional_status_correct.sum())}')
    print(f'GSMs without a confident raw column (dropped downstream): {unmatched}')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
