# Paper revisions after the alignment fix (GSE45498)

Companion to `REPRO_REPORT.md`. Every number below comes from a CSV regenerated on branch
`repro-verify-2026-11`; the "source file" column names it.

**Scope caveat.** I did not have the manuscript text. The "old" values come from three places: the
claims you listed (235/173, 184/51, 118 = 86 + 32, 999 vs 1,350 edges, p = 0.39, Jaccard 0.22,
TCGA transfer), the README, and the paper values hard-coded in the code (39: real 0.9358 /
edge-removed 0.9395; 37: mRNA signature AUROC ~0.88, miRNA AUROC ~0.99;
`summarize_edge_validation.py`: ~7% miRDB / ~3% miRTarBase). Any other number in the manuscript has
to be checked against the regenerated tables before resubmission. Rows marked "verify wording"
take the old value from the committed tables, not from the manuscript.

## 1. Corrected abstract

The draft below is written for the recommended framing, option (b) in `7, which reports the
mRNA results as a separate unimodal analysis. A version for option (a) follows it.

> **Background.** Triple-negative breast cancer (TNBC) tissue states (normal, primary, metastatic)
> differ in microRNA expression, but whether graph neural networks built on co-expression structure
> add information beyond expression values is unclear.
> **Methods.** We reanalysed GEO GSE45498 (NanoString nCounter; Cascione et al. 2013). miRNA
> profiles were taken from 230 samples whose tissue label was confirmed against the GEO sample
> records. Patients were grouped conservatively (168 patient groups), and the data were split into
> patient-disjoint training (n = 190) and held-out test (n = 40) sets. Differential expression
> (pydeseq2) selected 93 miRNAs. We compared XGBoost, TabNet and a multi-network graph attention
> network (GAT) on tissue-specific top-150 co-expression networks, and tested graph structure with
> no-edge controls and 1,000 degree-preserving rewirings. We analysed mRNA profiles separately, after
> matching every raw mRNA column to its GEO sample by expression correlation; this matching
> corrected a column-order error in the public raw file.
> **Results.** On the held-out set, the miRNA GAT reached macro-F1 0.836 ± 0.020 over 20 seeds
> (single seed 0.840) and XGBoost 0.809; the difference was not significant (McNemar p = 1.0).
> Rewired networks performed as well as the real networks (empirical p = 0.495), and removing all
> edges did not lower performance (0.840 vs 0.830). Feature rankings from GAT attention and XGBoost
> SHAP overlapped little (top-25 Jaccard 0.25). In the separate mRNA analysis, a 157-gene mRNA GAT
> reached held-out macro-F1 0.851. A 20-gene mRNA signature separated TCGA-BRCA normal from basal
> tumours (AUROC 0.9999), and an XGBoost retrained on overlapping genes reached macro-F1 0.916
> (both n = 284: 113 normal, 171 basal).
> **Conclusions.** In this cohort, tissue state is predictable from miRNA or mRNA expression alone.
> Co-expression graph topology adds no detectable information beyond node expression. With a
> 40-sample test set, the study is underpowered to rank the models.

Option (a), the corrected multi-omic paper. Replace the Methods and Results sentences with:

> ...230 samples with tissue-concordant miRNA and mRNA profiles (93 miRNAs + 157 mRNAs selected by
> DE)... The combined GAT reached held-out macro-F1 0.868 ± 0.030 over 20 seeds (single seed 0.905)
> vs XGBoost 0.809 (McNemar p = 0.22, n = 40). Real networks were not significantly better than
> 1,000 degree-preserving rewirings (macro-F1 0.885 ± 0.027 vs 0.860 ± 0.036; empirical
> p = 0.20), although they scored above the no-edge control (0.852)...

Option (a) must also state the pairing limitation in the abstract: "miRNA and mRNA profiles could
be matched by tissue but not verifiably by patient."

## 2. Methods text: alignment fix and pairing limitation

> **Sample alignment.** The GSE45498 raw mRNA matrix (`GSE45498_mRNA_non-normalized_data.txt.gz`)
> orders columns by patient number with normal and tumour columns interleaved, whereas the series
> matrix that carries tissue labels is ordered metastases → normals → primaries. Pairing by column
> position therefore assigns the wrong profile to most samples: only 49 of 254 raw columns sit at
> their positional GSM, and only 198 of 254 carry the correct tissue. We matched each raw column to
> the 254 GEO sample tables by Spearman correlation over 236 shared endogenous genes. A match was
> accepted when it was the mutual best, ρ ≥ 0.90, and its margin over the runner-up was ≥ 0.01 in
> both directions. 249 of 254 columns met these criteria (median ρ 0.9965). Five samples without
> a confident match were excluded (GSM1038973, GSM1039014, GSM1039133, GSM1039163, GSM1039165).
> The miRNA raw matrix matched its GEO samples one to one (282/282). The mapping and the script that
> derives it are provided (`metadata/mrna_raw_to_gsm_mapping.csv`,
> `scripts/derive_mrna_gsm_mapping.py`).
>
> **miRNA–mRNA pairing.** GSE45498, its sister series GSE41970 and the original publication contain
> no key linking a miRNA sample to the mRNA sample from the same patient. miRNA and mRNA profiles
> were therefore paired within tissue by series order. We tested that order empirically. For
> metastases, positional pairing gives cross-modal agreement well above a permutation null
> (ρ = 0.54, p < 0.001; monotone-alignment z = 3.5–5.9). For primary tumours, a monotone alignment
> is strongly non-random (z = 5.7–10.5) but does not coincide with positional pairing, and
> individual pairs are not identifiable. For normals there is no detectable link (z = 0.2–1.7).
> Each combined sample is therefore guaranteed to be tissue-concordant but not patient-matched,
> except, to a degree, among metastases.
>
> **Patient grouping.** Patient numbers in the raw mRNA headers disagree with the GEO titles for
> 37 of 40 normals. We grouped samples by the union of both readings (union-find), which gives
> 168 conservative patient groups, and kept every group within one side of the train/test split.
> The split, StratifiedGroupKFold-based, reproduces exactly only with scikit-learn 1.7.x, which is
> pinned.

## 3. Find/replace table

| Old | New | Source file |
|---|---|---|
| 235 matched samples | 230 tissue-concordant samples (5 without a confident mRNA match excluded) | `data/processed/*_labels.csv`, `metadata/mrna_raw_to_gsm_mapping.csv` |
| 173 patients | 168 patient groups (conservative union of title and raw-header IDs) | `data/processed/patient_ids.csv` |
| 184/51 patient-clean split | 190/40 (train: 33 normal / 115 primary / 42 metastatic; test: 6 / 23 / 11) | `data/processed/train_labels.csv`, `test_labels.csv` |
| 118 DE features (86 miRNA, 32 mRNA) | 250 (93 miRNA, 157 mRNA); miRNA-only paper: 93 | `data/processed/de_mirnas.csv`, `de_mrnas.csv`, `scripts/verify_build.py` |
| 999 edges | 1,106 unique edges across the three tissue networks | `results/networks/*_edges_top150.csv` |
| 1,350 edges | 1,350 = 3 tissues × 450 edges (150 per edge type); keep, but label it as per-tissue total | `results/networks/*_edges_top150.csv` |
| network nodes 118 | 250 (miRNA-only: 93) | `results/tables/network_statistics.csv` |
| GAT mean ± SD over 20 seeds (as written) | 0.868 ± 0.030 macro-F1, accuracy 0.888 ± 0.030 (miRNA-only 0.836 ± 0.020 / 0.846 ± 0.020) | `results/tables/gat_seed_distribution.csv` |
| permutation p = 0.39 (text) and p = 0.426 (Figure 5 flowchart, hard-coded in `23_methodology_flowchart.py`) | p = 0.20 (1,000 degree-preserving rewirings; miRNA-only p = 0.49); the as-is rerun gives 0.446, so neither old value reproduces exactly | `results/tables/network_ablation_graphml_permutations.csv` |
| real-network macro-F1 0.9358 | 0.885 ± 0.027 (5 seeds; miRNA-only 0.830 ± 0.012) | `results/tables/network_ablation_graphml.csv` |
| edge-removed macro-F1 0.9395 | 0.852 (miRNA-only 0.840) | `results/tables/network_ablation_graphml.csv` |
| SHAP–attention Jaccard 0.22 | top-25 GAT-SHAP vs GAT attention 0.064 (miRNA 0.22, mRNA 0.04); XGBoost SHAP vs attention 0.064; miRNA-only 0.19 / 0.25 | `results/tables/gat_shap_comparison.csv`, `gat_vs_xgboost_agreement.csv` |
| cross-modal edges ~7% in miRDB | 2/450 (0.4%) of top cross-modal edges carry a miRDB v6.0 target flag | `results/networks/*_edges_top150.csv` (`miRDB_validated`) |
| ~3% in miRTarBase | remove (the annotation is not in the repo and cannot be regenerated) | n/a |
| TCGA mRNA signature AUROC ~0.88 | AUROC 0.99995, macro-F1 0.996 (20 genes, 113 normal / 171 basal) | `results/tables/tcga_classifier_validation.csv` |
| mRNA does not transfer to TCGA | overlap-gene XGBoost retrain macro-F1 0.916 (accuracy 0.923); matched-cohort mRNA-only GAT 0.992 | `tcga_classifier_validation.csv`, `tcga_multimodal_validation.csv` |
| TCGA miRNA classifier AUROC ~0.99 | AUROC 0.966; at the default threshold it predicts all-tumour (macro-F1 0.380); the top-10 miRNA retrain gives macro-F1 0.734 | `results/tables/tcga_mirna_classifier_validation.csv` |
| XGBoost test accuracy 0.936 / macro-F1 0.925 (verify wording) | 0.825 / 0.809 | `results/tables/model_comparison.csv` |
| GAT test accuracy 0.936 / macro-F1 0.923 (verify wording) | 0.925 / 0.905 (single seed); report the 20-seed mean instead | `results/tables/model_comparison.csv` |
| TabNet accuracy 0.787 / macro-F1 0.702 or 0.723 / 0.667 (verify wording) | 0.750 / 0.725 | `results/tables/model_comparison.csv` |
| McNemar XGBoost vs GAT p = 1.0 (verify wording) | p = 0.22 (b = 1, c = 5) | `results/tables/mcnemar_tests.csv` |
| McNemar XGBoost vs TabNet p = 0.002 (verify wording) | p = 0.45 | `results/tables/mcnemar_tests.csv` |
| bootstrap 95% CI (test n = 47, 28 patients) | patient-cluster CI, n = 40, 33 patients: GAT [0.756, 1.0], XGBoost [0.655, 0.926], TabNet [0.568, 0.853] macro-F1 | `results/tables/bootstrap_metric_cis.csv` |
| mRNA modality weak / near-noise | mRNA-only GAT macro-F1 0.851 vs miRNA-only 0.840 vs both 0.905 | `results/tables/modality_ablation.csv` |
| 13 miRNA–mRNA regulatory pairs (miRDB > 70, \|r\| > 0.1) (verify wording) | 81 pairs, of which 25 are anti-correlated | `data/processed/mirna_mrna_pairs.csv`, `results/tables/threshold_robustness_summary.csv` |
| graph-only GAT (if reported) | 0.803 ± 0.053 (20 seeds) | `results/tables/gat_graph_only_seed_distribution.csv` |
| naive baseline (if reported) | majority-class macro-F1 0.243; stratified 0.328 ± 0.075 | `results/tables/naive_baselines.csv` |

## 4. Edge count, resolved

Each tissue network keeps the top 150 edges of each of three types (miRNA–miRNA, mRNA–mRNA,
miRNA–mRNA), giving 450 edges per tissue and 1,350 across the three tissues. Many edges recur in
more than one tissue. In the build the paper describes (118 nodes), the number of **distinct**
edges across tissues is 999. So "999" and "1,350" are both correct, but they count different
things, and the text must say which one it means. The committed network files on `main` come from
an older 146-node build and give 1,036 distinct edges. The corrected build gives 1,106 distinct
edges (miRNA-only: 450 slots, 349 distinct).

## 5. Limitations paragraph

> The public GSE45498 deposit does not link miRNA and mRNA samples by patient. We repaired a
> column-order error in the raw mRNA file and confirmed every retained mRNA profile's tissue, but
> the miRNA–mRNA pairing within a tissue is only supported by order-based evidence for metastases
> and, more weakly, for primaries, and not for normals. Cross-modal (miRNA–mRNA) correlations, the
> cross-modal network layer, and any claim of miRNA–target regulation therefore rest on
> tissue-level pseudo-pairs and should be treated as exploratory; none of these correlations
> survive FDR correction, and only 0.4% of the top cross-modal edges are predicted miRDB targets.
> The held-out set is small (40 samples, 6 normals), so test metrics move by several points per
> misclassified sample and the model comparisons are underpowered. The patient-disjoint split
> depends on the scikit-learn version and reproduces only with the pinned 1.7.x. The TCGA transfer
> used a gene panel reconstructed from the shipped TCGA expression file and ComBat from the inmoose
> package, because the original panel file is absent and the original ComBat implementation fails.
> TCGA has no metastatic class, so the external check covers normal vs primary only.

## 6. What changes in the narrative

* "Graph topology does not add information beyond expression" **stays**: p = 0.20 corrected,
  0.49 miRNA-only. Do not reword the corrected real > no-edge gap (0.885 vs 0.852) as a positive
  graph effect: rewired graphs give the same gain (null mean 0.860).
* "GAT ≈ XGBoost" **stays**, as "no significant difference with n = 40", not as equivalence.
* "mRNA is weak and does not transfer" **must be removed**. Corrected, it is the opposite.
* "SHAP and attention disagree (0.22)" **stays qualitatively**, with the corrected numbers.
* Anything about miRNA→mRNA regulation or cross-modal edges as biology should be cut or relabelled
  as exploratory co-variation.

## 7. Recommendation: (a) corrected multi-omic paper vs (b) miRNA-only paper

**Recommend (b): a miRNA-only paper, with a separate unimodal mRNA analysis.** The miRNA data are
cleanly identified (282/282 columns match their GEO records, and labels come from the sample names),
so the miRNA results do not depend on the unresolved pairing. The central claims all hold there:
graph topology adds nothing detectable (p = 0.49), and GAT and XGBoost are statistically
indistinguishable (McNemar p = 1.0). The mRNA results that are actually strong (mRNA-only
classification, TCGA transfer) need no cross-modal pairing either, so they can be reported as a
second unimodal arm.

Option (a) is defensible only with the miRNA–mRNA patient key from the original lab (the NanoString
run sheets behind Cascione et al. 2013). Without it, the defining feature of (a), joint modelling
and cross-modal edges, rests on pseudo-pairs for primaries and normals, and the small gain from
combining modalities (0.905 vs 0.840 / 0.851, single seed, n = 40) is within the bootstrap CIs.
If the key is recovered, rerun this branch with the true pairing and (a) becomes viable.

A miRNA-only analysis could also use all 282 miRNA samples instead of the 230 kept for comparison
here, which would give more power. That variant was not run.
