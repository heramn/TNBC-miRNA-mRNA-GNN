# Reproduction report: TNBC miRNA–mRNA GNN (GEO GSE45498)

Branch `repro-verify-2026-11`, base commit `d31566f` (main). Date: 2026-11-22.
Everything here was rerun from the raw GEO files in `data/raw/`. Numbers are copied from the
regenerated CSVs, not rounded toward the paper. "Committed" means the tables on `main` at
`d31566f`; "as-is" means the code on `main` run unchanged (scikit-learn pinned to 1.7.2,
committed seeds, `GAT_FIX_PARAMS` as exported by `run_pipeline.sh`, no tuning); "corrected"
means this branch (230 confidently aligned samples); "miRNA-only" means the corrected 230-sample
build with the mRNA half dropped (miRNA labels come from the miRNA sample names, so they do not
depend on the mRNA alignment).

## 1. Verdict

**The mRNA misalignment is confirmed.** `01_preprocessing_DE.py` pairs each miRNA sample with the
mRNA *raw-file column at the same position*, but the raw mRNA file is ordered by patient number with
N/T columns interleaved, while the GEO series matrix (which supplies the tissue labels and patient
IDs) is ordered mets → normals → tumors. Matching every raw column to the GSM data tables by
Spearman correlation, only **49/254** raw columns sit at their positional GSM and only **198/254**
carry the right tissue (the earlier run's 197 was off by one). Of the **235** samples the as-is code
uses, **186** take a non-intended mRNA profile, **51** have a tissue label that disagrees with the
tissue of the mRNA actually used (**28 of the 40 "normal" samples carry primary-tumor mRNA**; 10 of
the 51 mismatches are in the test set), and **9 patients (18 samples)** cross the 184/51 train/test
split once patients are identified by the mRNA actually used. The miRNA side is clean (282/282
columns match their GSM).

**No real miRNA↔mRNA patient key exists** in GEO GSE45498 (titles, characteristics, supplementary
files), the sister series GSE41970, or the source paper (Cascione et al. 2013, PLoS One 8:e55910,
PMID 23405235). Order-based evidence links miRNA and mRNA samples for metastases (the positional
pairing's cross-modal agreement is far above a permutation null) and shows a non-random monotone
order link for primary tumors (to a *different* order than the code's pairing). Normals show no
usable link. Individual pairs are not identifiable, so the corrected build guarantees that both
halves of a sample have the right **tissue**, not that they come from the same **patient**.

**Does the central negative result survive?** Partly:

* *Graph edges add nothing beyond node expression (network ablation, p = 0.39).* **Survives.** Corrected, the real graph (macro-F1 0.886 ± 0.027, 5 seeds) now scores above the
  no-edge control (0.852), the reverse of the as-is order (0.936 vs 0.959). But 1,000
  degree-preserving rewirings do about as well (null 0.860 ± 0.036, empirical p = 0.201), so the
  specific topology is still not shown to matter. The as-is rerun gives p = 0.446, which does not
  reproduce the paper's 0.39 exactly but reaches the same conclusion; miRNA-only gives p = 0.495.
  Graph-only GAT (expression bypass removed) reaches 0.803 ± 0.053 corrected, so the graph pathway
  carries class signal, just not topology-specific signal.
* *GAT is not significantly better than XGBoost.* **Survives** (McNemar p = 0.219 corrected,
  1.0 as-is, 1.0 miRNA-only), but the effect size changed: corrected single-seed GAT 0.905 vs
  XGBoost 0.809 macro-F1, and the 20-seed GAT mean is 0.868 ± 0.030. With 40 test samples
  (6 normals) the test is badly underpowered, so "no significant difference" is not evidence of
  equivalence.
* *mRNA is the weak modality and fails to transfer to TCGA.* **Does not survive. It was produced
  by the misalignment.** As-is, mRNA-only GAT reaches held-out macro-F1 0.546 and every TCGA
  mRNA-only path predicts the majority class (F1 0.380). Corrected, mRNA-only GAT reaches 0.851
  held-out, TCGA mRNA-only GAT 0.992, the TCGA overlap-gene XGBoost retrain 0.916, and the
  gene-signature AUROC 0.99995 (as-is 0.447).
* *SHAP and attention disagree (Jaccard 0.22).* **The 0.22 reproduces as-is** (GAT-SHAP vs
  GAT-attention, top-25: 0.220) **but falls to 0.064 corrected** (miRNA 0.220, mRNA 0.042). XGBoost
  SHAP vs GAT attention is 0.163 as-is and 0.064 corrected. The qualitative "low agreement"
  claim survives, and gets stronger.

Bottom line: the headline sample and feature numbers (184/51, 118 = 86 + 32, 0.9358 real-graph F1,
Jaccard 0.22) reproduce **only** on the misaligned build, and only under scikit-learn ≤ 1.7.
The committed tables come from yet another, older build (47-sample test set, 146 nodes). The
"graph topology doesn't help / GAT ≈ XGBoost" conclusions survive the correction. Every mRNA
conclusion has to be rewritten, and so does anything built on cross-modal (miRNA–mRNA)
correlations, because those need patient-matched pairs that GEO does not provide.

## 2. Comparison table

Test-set metrics are macro-F1 unless noted (accuracy in parentheses). Test n: committed 47,
as-is 51, corrected and miRNA-only 40.

| # | Item | Paper claim | Committed value (`main`) | As-is rerun | Corrected rerun | miRNA-only | Status | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | Samples / patients | 235 matched samples, 173 patients | test set 47 samples / 28 patients | 235 samples, 200 patient groups; 186/235 use a non-intended mRNA GSM, 51 tissue mismatches, 5 unmappable | 230 samples, 168 patient groups (union of GEO-title and raw-header IDs) | 230 / 168 | **Wrong**: 173 not reproduced; pairing misaligned | `repro/asis_235_sample_audit.csv`, `data/processed/patient_ids.csv` |
| 2 | Patient-clean split | 184 / 51, no patient in both | 47-sample test (older build) | 184 / 51 with sklearn 1.3.2–1.7.2 (188 / 47 with 1.9.1); 9 patients (18 samples) cross by the mRNA actually used | 190 / 40, 0 patients cross | 190 / 40 | **Reproduces only with pinned sklearn, and leaks** | `scripts/verify_build.py`, `scripts/audit_alignment_and_leakage.py` |
| 3 | DE features | 118 (86 miRNA + 32 mRNA) | networks built on 146 nodes (90 + 56) | 86 + 32 = 118 | 93 + 157 = 250 | 93 | As-is reproduces; **corrected differs** | `data/processed/de_*.csv` |
| 4 | Network edges | "999" vs "1,350" | 3 × 450 = 1,350 rows, 1,036 unique | 1,350 rows, **999 unique** | 1,350 rows, 1,106 unique | 450 rows, 349 unique | **Resolved**: 1,350 = per-tissue sum; 999 = unique edges across tissues (as-is build) | `results/networks/*_edges_top150.csv` |
| 5 | Cross-modal edges surviving FDR | none | 0 | 0 | 0 | n/a | Holds | `fdr_significant` column |
| 6 | Cross-modal edges in miRDB / miRTarBase | ~7% / ~3% | 7.4% / 2.7% from `edge_validation_db.csv` (336 edges), which does not match the committed networks; their own flags give 2/450 (0.4%) miRDB | 4/450 (0.9%) | 2/450 (0.4%) | n/a | **Stale**: no code regenerates 7%/3%; miRTarBase not reproducible | `scripts/summarize_edge_validation.py`, `miRDB_validated` |
| 7 | Regulatory pairs (miRDB > 70, \|r\| > 0.1) | n/a | 13 | 7 (5 negative) | 81 (25 negative, 56 positive) | 0 | Changed | `mirna_mrna_pairs.csv`, `threshold_robustness_summary.csv` |
| 8 | XGBoost test | n/a | 0.925 (0.936) | 0.916 (0.922) | 0.809 (0.825) | 0.809 (0.825) | Changed | `model_comparison.csv` |
| 9 | GAT test, single seed | n/a | 0.923 (0.936) | 0.940 (0.941) | 0.905 (0.925) | 0.840 (0.850) | Changed | `model_comparison.csv` |
| 10 | GAT mean ± SD, 20 seeds [min, max] | reported (value not available to me) | not committed | 0.918 ± 0.029 [0.855, 0.960]; acc 0.921 ± 0.029 | 0.868 ± 0.030 [0.802, 0.905]; acc 0.888 ± 0.030 | 0.836 ± 0.020 [0.815, 0.864]; acc 0.846 ± 0.020 | **Check against manuscript** | `gat_seed_distribution.csv` |
| 11 | TabNet test | n/a | 0.702 (0.787) in `model_comparison.csv` but 0.667 (0.723) in `bootstrap_metric_cis.csv` | 0.824 (0.843) | 0.725 (0.750) | 0.700 (0.725) | Committed tables inconsistent; changed | `model_comparison.csv` |
| 12 | 05b Optuna-finetuned GAT | n/a | 0.901 (0.915) | 0.958 (0.961) | 0.905 (0.925) | 0.815 (0.825) | Changed | `gat_finetuned_test_summary.csv` |
| 13 | Naive baselines (majority / stratified / uniform) | n/a | not committed | 0.242 / 0.328 / 0.313 | 0.243 / 0.328 / 0.306 | as corrected | New | `naive_baselines.csv` |
| 14 | Patient-cluster bootstrap 95% CI, macro-F1 | n/a | GAT [0.818, 1.0]; XGB [0.828, 1.0] | GAT [0.862, 1.0]; XGB [0.826, 0.982]; TabNet [0.697, 0.925] | GAT [0.756, 1.0]; XGB [0.655, 0.926]; TabNet [0.568, 0.853] | GAT [0.697, 0.951]; XGB [0.653, 0.925]; TabNet [0.520, 0.838] | Changed | `bootstrap_metric_cis.csv` |
| 15 | McNemar XGBoost vs GAT | n/a | p = 1.0 (b = 1, c = 1) | p = 1.0 (b = 3, c = 4) | p = 0.219 (b = 1, c = 5) | p = 1.0 (b = 1, c = 2) | Not significant in all | `mcnemar_tests.csv` |
| 16 | McNemar XGBoost vs TabNet | n/a | p = 0.00195 | p = 0.388 (BH 0.582) | p = 0.453 | p = 0.125 | Changed | `mcnemar_tests.csv` |
| 17 | Network-ablation permutation p | p = 0.39 | 0.048 (20 permutations, older build) | p = 0.446 (445/1,000 ≥ real) | p = 0.201 (200/1,000 ≥ real) | p = 0.495 (494/1,000 ≥ real) | 0.39 not reproduced exactly; "not significant" holds in all three | `network_ablation_graphml_permutations.csv` |
| 18 | Real graph vs no edges, macro-F1 | 0.9358 vs 0.9395 (fallbacks in 39) | 0.923 vs 0.902 (older build) | 0.9358 ± 0.0145 (n = 5) vs 0.9591 | 0.8855 ± 0.0268 vs 0.8517 | 0.8304 ± 0.0122 vs 0.8404 | Real-graph value reproduces as-is; no-edge value does not; **order flips when corrected** | `network_ablation_graphml.csv` |
| 19 | Rewired-graph null, macro-F1 | n/a | 0.881 ± 0.042 (20) | 0.926 ± 0.027 [0.757, 1.000] | 0.860 ± 0.036 [0.725, 0.934] | 0.828 ± 0.021 [0.717, 0.901] | New | `network_ablation_graphml_permutations.csv` |
| 20 | Graph-only GAT (expression bypass removed), 20 seeds | n/a | not committed | 0.765 ± 0.074 | 0.803 ± 0.053 | 0.807 ± 0.032 | New | `gat_graph_only_seed_distribution.csv` |
| 21 | Modality ablation (both / miRNA / mRNA) | mRNA weak | not committed | 0.940 / 0.901 / **0.546** | 0.905 / 0.840 / **0.851** | n/a (no mRNA) | **mRNA weakness was an artifact** | `modality_ablation.csv` |
| 22 | SHAP–attention Jaccard (top-25) | 0.22 | not committed | GAT-SHAP vs attention **0.220**; XGB-SHAP vs attention 0.163; GAT-SHAP vs XGB-SHAP 0.250 | 0.064 (miRNA 0.220, mRNA 0.042); 0.064; 0.220 | GAT-SHAP vs attention 0.190; XGB-SHAP vs attention 0.250; GAT-SHAP vs XGB-SHAP 0.389 | **Reproduces only as-is** | `gat_shap_comparison.csv`, `gat_vs_xgboost_agreement.csv` |
| 23 | Cross-modal edge bootstrap Jaccard (normal / primary / met) | n/a | not committed | 0.295 / 0.278 / 0.369 | 0.211 / 0.186 / 0.236 | n/a | New | `edge_stability.csv` |
| 24 | Primary vs metastatic, held-out | n/a | not committed | 0.933 (0.950), n = 40 | 0.788 (0.824), n = 34 | 0.817 (0.853), n = 34 | New | `primary_vs_metastatic_heldout.csv` |
| 25 | TCGA mRNA gene signature, AUROC | ~0.88 | 0.878 (8 genes) | 07/10 **crash** with the committed ComBat; with inmoose: 0.447 (2 genes) | **0.99995** (20 genes) | n/a | **Not reproduced; reverses when corrected** | `tcga_classifier_validation.csv` |
| 26 | TCGA mRNA overlap-gene XGBoost retrain | fails to transfer | 0.426 (0.577) | 0.402 (0.592) | **0.916 (0.923)** | n/a | **Reverses** | `tcga_classifier_validation.csv` |
| 27 | TCGA miRNA classifier, pretrained, AUROC | ~0.99 | 0.989 (F1 0.738) | 0.988 (F1 0.745) | 0.966 (F1 0.380: majority class at the default threshold) | n/a | AUROC holds; calibration does not | `tcga_mirna_classifier_validation.csv` |
| 28 | TCGA matched multimodal GAT (both / mRNA / miRNA) | n/a | not committed | 0.964 / 0.380 / 0.948 | 0.972 / 0.992 / 0.893 | n/a | New | `tcga_multimodal_validation.csv` |
| 29 | TCGA matched multimodal XGBoost (both / mRNA / miRNA) | n/a | not committed | 0.728 / 0.380 / 0.745 | 0.812 / 0.380 (AUROC 0.994) / 0.380 | n/a | New; 0.380 = majority class | `tcga_multimodal_validation.csv` |

Notes:
* The test sets are small. The corrected test set has 6 normals out of 40, so macro-F1 moves by
  about 0.03–0.05 per misclassified normal. The CIs are patient-cluster bootstraps with 10,000
  resamples. The patient-cluster AUROC CI is NaN in the corrected build because some resamples
  contain no normal sample.
* SDs are sample SDs (ddof = 1). Two values I quoted earlier in the thread used ddof = 0 (graph-only
  as-is 0.072, corrected 0.051); the table values come from the CSVs.
* Row 17: p = (1 + #{permuted F1 ≥ real mean}) / (1 + N) with N = 1,000 degree-preserving
  rewirings. Each rewiring is a single training run. The real mean is over 5 seeds
  (`N_REAL_SEEDS`).
* Rows 25–29 use two deviations from the committed procedure (see `4): a reconstructed gene panel,
  and `inmoose` ComBat in place of the broken `combat` package.
* `33_gat_tuning.py` is a 50-trial Optuna search, which conflicts with "don't tune". I started it for all three variants. After about 5 hours none had finished (Optuna logs nothing until the end), so I cancelled them; `gat_tuning_results.csv` is not regenerated.

## 3. Alignment evidence (details)

* 236 endogenous genes are shared between the raw mRNA file and the GSM data tables. Each of the 254
  raw columns was matched to all 254 GSMs by Spearman ρ. A match counts as confident when it is the
  mutual best, ρ ≥ 0.90, and both margins are ≥ 0.01. **249/254** raw columns are confident
  (median best ρ 0.9965).
* The 5 GSMs with no confident column are GSM1038973 TAS10-15-56, GSM1039014 112N, GSM1039133 5T,
  GSM1039163 82T and GSM1039165 87T. Raw column TAS10-15-56 has NaNs (233 non-NaN genes). They are
  dropped, which leaves 230 samples. Five as-is samples (Tumor_29/56/58/64/73) point at them and
  cannot be assigned.
* miRNA: 282/282 raw columns match their positional GSM and title (minimum best ρ 0.9652,
  maximum second-best 0.9388).
* Raw-header patient numbers differ from GEO titles for 37/40 normals. Patient grouping is
  therefore the union-find of both readings (conservative).
* Order-link tests between miRNA and mRNA (series-matrix order):
  * Positional pairing, cross-modal agreement vs a permutation null: mets ρ = 0.540
    (p < 0.001), normals −0.090, tumors 0.007.
  * Monotone DP alignment: mets z 3.5–5.9, tumors z 5.7–10.5, normals z 0.2–1.7
    (z ranges over gap penalties 0.25–1.0; output in `repro/dp_align.out`).
  * Stable inferred pairs: tumors 112/153, none agreeing with the code's positional pairing;
    mets 44/50, 32 agreeing; normals 27/39, 1 agreeing.
  * So the code's pairing is supported for mets only. Tumors show an order link to a different
    pairing that cannot be identified pair by pair. Normals show none.

Evidence files: `metadata/mrna_raw_to_gsm_mapping.csv` (from
`scripts/derive_mrna_gsm_mapping.py`), `repro/asis_235_sample_audit.csv`,
`repro/inferred_mirna_mrna_pairs_order_alignment.csv`.

## 4. Fixes on this branch

Code/data, in priority order:
1. `notebooks/01_preprocessing_DE.py`: fetches each GSM's own raw mRNA column through
   `metadata/mrna_raw_to_gsm_mapping.csv`, drops GSMs with no confident match, and groups patients
   by union-find over GEO-title and raw-header IDs.
2. `scripts/derive_mrna_gsm_mapping.py` (new): derives the mapping. `run_pipeline.sh` regenerates
   it and warns on any drift.
3. `scripts/audit_alignment_and_leakage.py` (new): checks label vs mRNA tissue and patient
   crossing. It runs right after the build guard.
4. `scripts/verify_build.py`: expects the corrected 93 + 157, 190/40 build.
5. `requirements.txt`: exact pins, including `scikit-learn==1.7.2`. With 1.9.1 the same seed
   gives 188/47 and 123 features.
6. `run_pipeline.sh`: miRDB v6.0 download plus sha256 check, mapping and audit steps, and an
   order in which 14 regenerates predictions before 15 reads them.
7. `notebooks/14_statistical_tests.py`: regenerates predictions unless `REUSE_PREDICTIONS=1`.
   It used to silently reuse the stale 47-sample committed prediction CSVs, and 15 read them.
8. `notebooks/07, 10, 19`: the `ref_batch` path of `combat` 0.3.3 raises IndexError, and the
   package does not import under numpy 2 / pandas 3. These notebooks now use
   `inmoose.pycombat.pycombat_norm` with the same reference-batch design. `combat` was also
   missing from `requirements.txt`.
9. `data/tcga_gene_panel.txt`: read by 07/10 but absent from the repo, with no generator. Rebuilt
   from the 231 gene columns of `data/tcga_brca_basal_mrna_expression.csv` (a reconstruction, not
   the authors' file).
10. `notebooks/37, 38`: removed hard-coded 118-node assumptions.
11. `results/`: every table and network was regenerated from the corrected build. Committed tables
    with no producer in the current code were removed: `model_agreement_biomarkers*.csv`,
    `top_features_per_model.csv`, `nanostring_background.txt`, `tcga_mirna_availability.txt`,
    `{normal,primary,metastatic}_edges.csv`, `network_ablation.csv`,
    `network_ablation_permutation_tests.csv`, `network_ablation_significance_summary.csv`,
    `edge_validation_db.csv` and `edge_validation_hypergeom.csv` (miRTarBase flags cannot be
    regenerated), and `tcga_validation_summary.csv`. `results/figures_draft3/` was replaced by
    the regenerated figures.

Needs the authors:
* The real miRNA↔mRNA patient key. Without it, cross-modal correlations and the "multi-omic"
  framing rest on tissue-level pseudo-pairs.
* The original `tcga_gene_panel.txt` and the miRTarBase annotation behind the ~3% figure.

Paper text: see `PAPER_REVISIONS.md`.

## 5. Commands and environment

```bash
git clone https://github.com/heramn/TNBC-miRNA-mRNA-GNN && cd TNBC-miRNA-mRNA-GNN
git checkout repro-verify-2026-11
python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
bash run_pipeline.sh --with-tcga
```

As-is reproduction:
1. `git checkout d31566f`
2. `pip install -r requirements.txt scikit-learn==1.7.2 inmoose==0.9.1`
3. Put miRDB v6.0 in `data/raw/`.
4. Run `bash run_pipeline.sh --with-tcga`.
5. Run 29, 32, 26, 36, 24, 14, 15, 16, 17, 27, 28, 30, 30b, 31, 33, 33b, 37, 38, 39 and 40, all with
   `GAT_FIX_PARAMS="num_heads=2,hidden_size=32,dropout=0.2,lr=0.001"`.

The 1,000 permutations of 28 were run in 3 shards by `permutation % 3` and merged. Each permutation
uses its own seed, so the result equals a sequential run.

Verified from a fresh copy of this branch:
* the miRDB download matches sha256 `446636d2182212f44ff91623f1429f7922f9575b82226962a6213d592a7e109d`;
* the regenerated mapping equals the committed CSV;
* 01–03 output is byte-identical to the corrected build (all 11 `data/processed` files and
  3 networks);
* `BUILD OK` and `AUDIT OK` both pass.

Environment: Python 3.12.3, CPU only, Linux x86_64. numpy 2.5.5, pandas 3.1.1, scipy 1.18.1,
scikit-learn 1.7.2, statsmodels 0.15.0, matplotlib 3.11.2, networkx 3.7, xgboost 3.4.1,
shap 0.54.1, optuna 5.1.0, pytorch-tabnet 4.1.0, torch 2.15.0+cpu, torch-geometric 2.8.1,
pydeseq2 0.6.0, anndata 0.13.4, inmoose 0.9.1, requests 2.34.2, gseapy 1.3.2. Full lock:
`repro/requirements.lock.txt`.
