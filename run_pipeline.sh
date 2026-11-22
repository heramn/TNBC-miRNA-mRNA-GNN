#!/usr/bin/env bash
# Canonical end-to-end run. USE THIS instead of running notebooks ad hoc: it
# enforces dependency order and halts on a wrong data build.
#   bash run_pipeline.sh              # core pipeline (all paper numbers)
#   bash run_pipeline.sh --with-tcga  # also run TCGA transfer (needs internet)
# Environment: pip install -r requirements.txt (exact pins; scikit-learn must be 1.7.x).
set -euo pipefail
cd "$(dirname "$0")"
run(){ echo -e "\n=== $1 ==="; python "notebooks/$1.py"; }
# Reproduce the paper's reported GAT config (2 heads / hidden 32 / dropout 0.2), selected by
# the grid search in 5.6 and then fixed. Every GAT notebook honors this; unset it to re-tune.
export GAT_FIX_PARAMS="num_heads=2,hidden_size=32,dropout=0.2,lr=0.001"
mkdir -p data/processed results/tables results/figures results/networks models

# 0. INPUTS: miRDB v6.0 is used only to annotate edges; it is too large to commit.
MIRDB=data/raw/miRDB_v6.0_prediction_result.txt.gz
if [[ ! -s "$MIRDB" ]]; then
  echo "=== downloading miRDB v6.0 ==="
  curl -fSL -o "$MIRDB" https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz
fi
python - <<'PY'
import hashlib, sys
h = hashlib.sha256(open('data/raw/miRDB_v6.0_prediction_result.txt.gz', 'rb').read()).hexdigest()
expected = '446636d2182212f44ff91623f1429f7922f9575b82226962a6213d592a7e109d'
print(f'miRDB sha256 {h}')
if h != expected:
    sys.exit(f'miRDB file differs from the one used for the committed results (expected {expected})')
PY

# 0b. mRNA raw-column -> GSM mapping (fixes the positional misalignment; committed CSV is
#     regenerated and must be identical)
echo -e "\n=== mRNA raw-column -> GSM mapping ==="; python scripts/derive_mrna_gsm_mapping.py
git diff --quiet -- metadata/mrna_raw_to_gsm_mapping.csv 2>/dev/null || echo "WARNING: regenerated mapping differs from the committed one"

# 1. DATA: DE + patient-clean split + co-expression networks
for nb in 01_preprocessing_DE 02_pair_construction 03_network_construction; do run "$nb"; done

# 1b. BUILD GUARD: stop before modelling if the build is wrong
echo -e "\n=== build guard ==="; python scripts/verify_build.py
echo -e "\n=== leakage audit ==="; python scripts/audit_alignment_and_leakage.py

# 2. MODELS
for nb in 04_xgboost_baseline 05_gat_baseline 05b_gat_finetuning 06_tabnet_baseline 27_modality_ablation; do run "$nb"; done

# 3. GRAPH STRUCTURE  (29 MUST run before the attribution scripts that read its output)
for nb in 28_network_ablation_graphml 29_attention_stability_graphml 31_edge_stability 38_gat_graph_only 39_ablation_distribution_figure; do run "$nb"; done

# 4. ATTRIBUTION / CLASSIFICATION  (depend on 04 + 29)
for nb in 24_figure2_feature_importance 26_gat_shap_overlap_report 32_gat_vs_xgboost_differences 36_gat_shap 40_edge_attention \
          30_gat_seed_distribution 30b_naive_baselines 14_statistical_tests 15_threshold_robustness 16_primary_vs_metastatic \
          17_network_statistics 33_gat_tuning 33b_disease_relevance; do run "$nb"; done

# 5. TCGA external transfer (optional; needs internet)
if [[ "${1:-}" == "--with-tcga" ]]; then
  for nb in 07_tcga_validation 10_tcga_classifier_validation 18_tcga_mirna_feasibility \
            19_tcga_mirna_classifier_validation 37_tcga_multimodal_validation; do run "$nb"; done
fi

# 6. summary + pipeline figure
for nb in 08_model_performance_summary 23_methodology_flowchart; do run "$nb"; done
echo -e "\n✅ Pipeline complete."
