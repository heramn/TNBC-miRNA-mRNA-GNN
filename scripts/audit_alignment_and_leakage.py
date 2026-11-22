#!/usr/bin/env python3
"""Audit the processed build: label/tissue consistency of the attached mRNA profile and
patient leakage across the train/test split under every available patient reading.

Checks (exit 1 on failure):
  1. every sample's mRNA profile comes from a GSM whose GEO status equals its label;
  2. no GEO-title patient id and no raw-header patient id appears in both train and test.
Limitation it cannot check: GEO carries no patient identity for the miRNA profiles.
Run from repo root after notebooks/01_preprocessing_DE.py.
"""
import gzip
import json
import re
import sys

import pandas as pd

RAW = 'data/raw/'
STATUS = {'Metastatic': 'metastatic', 'normal': 'normal', 'primary triple negative cancer': 'primary'}


def series(path):
    out = {}
    with gzip.open(path, 'rt') as fh:
        for line in fh:
            if line.startswith(('!Sample_title', '!Sample_geo_accession')):
                key, *vals = line.rstrip('\n').split('\t')
                out[key] = [v.strip('"') for v in vals]
            elif line.startswith('!Sample_characteristics_ch1') and 'status:' in line:
                out['status'] = [v.strip('"').split(': ', 1)[1] for v in line.rstrip('\n').split('\t')[1:]]
    return out


def pid(text):
    m = re.match(r'^(\d+)[NT]', text) or re.match(r'^(?:TAS09 12 |Normal_)(\d+)', text)
    if m:
        return 'P' + m.group(1)
    m = re.match(r'^TAS10-15-(\d+)', text)
    return 'MET' + m.group(1) if m else None


def main():
    mi = series(RAW + 'GSE45498-GPL16231_series_matrix.txt.gz')
    mr = series(RAW + 'GSE45498-GPL16299_series_matrix.txt.gz')
    mapping = pd.read_csv('metadata/mrna_raw_to_gsm_mapping.csv')
    mapping = mapping[mapping.confident].set_index('best_gsm')
    split = json.load(open('data/processed/split_indices.json'))
    labels = pd.concat([pd.read_csv('data/processed/train_labels.csv'),
                        pd.read_csv('data/processed/test_labels.csv')]).set_index('sample_id').tissue_group
    side = {s: 'train' for s in split['train_sample_ids']} | {s: 'test' for s in split['test_sample_ids']}

    rows = []
    for i, name in enumerate(mi['!Sample_title'][:len(mr['!Sample_title'])]):
        if name not in labels.index:
            continue
        gsm = mr['!Sample_geo_accession'][i]
        rows.append({'sample_id': name, 'split': side[name], 'label': labels[name],
                     'mrna_gsm': gsm, 'mrna_title': mr['!Sample_title'][i],
                     'mrna_status': STATUS[mr['status'][i]],
                     'title_pid': pid(mr['!Sample_title'][i]),
                     'rawheader_pid': pid(mapping.loc[gsm, 'raw_col'])})
    d = pd.DataFrame(rows)
    bad_label = int((d.label != d.mrna_status).sum())
    leaks = {}
    for col in ['title_pid', 'rawheader_pid']:
        sides = d.groupby(col).split.agg(set)
        leaks[col] = sorted(sides[sides.map(len) > 1].index)
    print(f'samples: {len(d)} | label != mRNA-GSM tissue: {bad_label}')
    for col, ids in leaks.items():
        print(f'patients crossing train/test by {col}: {len(ids)} {ids[:10]}')
    if bad_label or any(leaks.values()):
        sys.exit('AUDIT FAILED')
    print('AUDIT OK')


if __name__ == '__main__':
    main()
