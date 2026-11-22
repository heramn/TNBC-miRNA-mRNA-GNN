#!/usr/bin/env python3
"""Build the TCGA-BRCA TNBC cohort with patient-matched miRNA-seq and RNA-seq.

Inputs (UCSC Xena TCGA hub, TCGA.BRCA.sampleMap): HiSeqV2 (log2(RSEM norm+1)),
miRNA_HiSeq_gene (log2(RPM+1)), BRCA_clinicalMatrix. A sample is kept only if the
SAME TCGA sample barcode (patient + sample type) has both a miRNA and an mRNA profile,
so every pair is a true same-specimen pair.

TNBC = ER IHC Negative AND PR IHC Negative AND HER2 negative, where HER2 negative is
IHC Negative, or IHC Equivocal with ISH Negative. Indeterminate / missing -> excluded.
Normals: solid tissue normal (sample type 11) from TNBC patients, plus (cohort
'all_normals') normals from any BRCA patient; normal breast is not receptor-subtyped.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

XENA = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

clin = pd.read_csv(XENA / 'BRCA_clinicalMatrix', sep='\t', index_col=0, low_memory=False)
mrna = pd.read_csv(XENA / 'HiSeqV2.gz', sep='\t', index_col=0)
mir = pd.read_csv(XENA / 'miRNA_HiSeq_gene.gz', sep='\t', index_col=0)

def norm(s):
    return s.astype(str).str.strip().str.lower()

er = norm(clin['breast_carcinoma_estrogen_receptor_status'])
pr = norm(clin['breast_carcinoma_progesterone_receptor_status'])
her_ihc = norm(clin['lab_proc_her2_neu_immunohistochemistry_receptor_status'])
her_ish = norm(clin['lab_procedure_her2_neu_in_situ_hybrid_outcome_type'])
her2_neg = (her_ihc == 'negative') | ((her_ihc == 'equivocal') & (her_ish == 'negative')) | \
           ((her_ihc.isin(['nan', 'indeterminate', ''])) & (her_ish == 'negative'))
her2_pos = (her_ihc == 'positive') | (her_ish == 'positive')
tnbc_pat = set(clin.loc[(er == 'negative') & (pr == 'negative') & her2_neg & ~her2_pos, '_PATIENT'])

both = sorted(set(mrna.columns) & set(mir.columns))
stype = pd.Series({s: s[13:15] for s in both})
patient = pd.Series({s: s[:12] for s in both})

pat_clin = clin.drop_duplicates('_PATIENT').set_index('_PATIENT')
def node_status(p):
    v = str(pat_clin.at[p, 'pathologic_N']) if p in pat_clin.index else 'nan'
    if v.startswith('N0'):
        return 'N0'
    if v[:2] in ('N1', 'N2', 'N3'):
        return 'Npos'
    return 'NA'
def stage(p):
    v = str(pat_clin.at[p, 'pathologic_stage']) if p in pat_clin.index else 'nan'
    return v

rows = []
for s in both:
    p = patient[s]
    t = stype[s]
    if t == '01' and p in tnbc_pat:
        rows.append((s, p, 'tumor', node_status(p), stage(p), True))
    elif t == '11':
        rows.append((s, p, 'normal', 'NA', 'NA', p in tnbc_pat))
    elif t == '06' and p in tnbc_pat:
        rows.append((s, p, 'metastatic', node_status(p), stage(p), True))
meta = pd.DataFrame(rows, columns=['sample', 'patient', 'tissue', 'node', 'stage', 'tnbc_patient']).set_index('sample')

mir_keep = mir[meta.index]
frac_present = mir_keep.notna().mean(axis=1)
mir_keep = mir_keep.loc[frac_present >= 0.9].fillna(0.0)
mrna_keep = mrna[meta.index]
mrna_keep = mrna_keep.loc[(mrna_keep > 1).mean(axis=1) >= 0.5]

meta.to_csv(OUT / 'tcga_tnbc_meta.csv')
mir_keep.T.to_csv(OUT / 'tcga_tnbc_mirna.csv.gz')
mrna_keep.T.to_csv(OUT / 'tcga_tnbc_mrna.csv.gz')

summary = {
    'tnbc_patients_clinical_ihc': len(tnbc_pat),
    'samples_with_both_modalities_all_brca': len(both),
    'tnbc_tumors_matched': int((meta.tissue == 'tumor').sum()),
    'tnbc_metastatic_matched': int((meta.tissue == 'metastatic').sum()),
    'normals_matched_all_brca': int((meta.tissue == 'normal').sum()),
    'normals_matched_tnbc_patients': int(((meta.tissue == 'normal') & meta.tnbc_patient).sum()),
    'tnbc_tumor_N0': int(((meta.tissue == 'tumor') & (meta.node == 'N0')).sum()),
    'tnbc_tumor_Npos': int(((meta.tissue == 'tumor') & (meta.node == 'Npos')).sum()),
    'tnbc_tumor_N_unknown': int(((meta.tissue == 'tumor') & (meta.node == 'NA')).sum()),
    'mirna_features_kept': mir_keep.shape[0],
    'mrna_features_kept': mrna_keep.shape[0],
}
pd.Series(summary).to_csv(OUT / 'tcga_tnbc_cohort_summary.csv', header=['value'])
print(pd.Series(summary).to_string())
print(meta[meta.tissue == 'tumor'].stage.value_counts().to_string())
