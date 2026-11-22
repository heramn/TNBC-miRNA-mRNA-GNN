#!/usr/bin/env bash
# Reproduces results/tcga_matched/ (branch fix-multiomic-B). Run from the repo root after
# notebooks/01_preprocessing_DE.py has written data/processed (corrected GEO build) and
# data/raw/miRDB_v6.0_prediction_result.txt.gz exists (run_pipeline.sh downloads it).
set -euo pipefail
W=${WORK:-$HOME/.capy/work}
PY=${PY:-python}
mkdir -p "$W/xena" "$W/mirbase"
for f in HiSeqV2.gz miRNA_HiSeq_gene.gz BRCA_clinicalMatrix; do
  [[ -s "$W/xena/$f" ]] || curl -sSfL -o "$W/xena/$f" "https://tcga.xenahubs.net/download/TCGA.BRCA.sampleMap/$f"
done
[[ -s "$W/mirbase/mature.fa" ]] || curl -sSfL -o "$W/mirbase/mature.fa" https://www.mirbase.org/download/mature.fa
R=results/tcga_matched
$PY tcga_matched/00_build_tcga_tnbc.py "$W/xena" "$W/tcga_out"
$PY tcga_matched/01_tcga_matched_gat.py "$W/tcga_out" $R/tumor_vs_normal tumor_vs_normal 20 50
$PY tcga_matched/01_tcga_matched_gat.py "$W/tcga_out" $R/node node 20 100
$PY tcga_matched/02_geo_replication.py "$W/tcga_out" $R/geo_replication 20 50
$PY tcga_matched/03_edge_replication.py "$W/tcga_out" $R/edge_replication
$PY tcga_matched/04_geo_priors_in_tcga.py "$W/tcga_out" $R/geo_priors_in_tcga edges
for p in gat_real gat_noedge gat_rewired xgb; do $PY tcga_matched/04_geo_priors_in_tcga.py "$W/tcga_out" $R/geo_priors_in_tcga $p 20 20; done
$PY tcga_matched/04_geo_priors_in_tcga.py "$W/tcga_out" $R/geo_priors_in_tcga summarize
$PY tcga_matched/05_harmonize.py "$W/tcga_out" $R/harmonization diagnostics
for m in none combat_ref zscore quantile rank; do $PY tcga_matched/05_harmonize.py "$W/tcga_out" $R/harmonization_eval eval $m 10 20; done
$PY tcga_matched/05_harmonize.py "$W/tcga_out" $R/harmonization_eval summarize
