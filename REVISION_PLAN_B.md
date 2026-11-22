# Revision plan B: GEO multi-omic GAT, validated on TCGA true same-patient pairs

Branch `fix-multiomic-B` (from `repro-verify-2026-11`). Date: 2026-11-22. Every number below comes
from a CSV under `results/tcga_matched/`; the source is named in each table. Code is in
`tcga_matched/` (`run_all.sh` reruns everything). Nothing was tuned: the GAT uses the repo's fixed
`GAT_FIX_PARAMS` (2 heads, hidden 32, dropout 0.2, lr 1e-3), the same 2-layer multi-network
architecture as `notebooks/05_gat_baseline.py`, the same early stopping (250 epochs, patience 35) and
the same network recipe as `notebooks/03` (top 150 edges per type per tissue).

## 1. Design (framing set by the author)

* **GEO GSE45498, corrected build, stays the primary cohort.** The paper stays multi-omic:
  miRNA + mRNA nodes, tissue-specific co-expression networks with miRNA–miRNA, mRNA–mRNA and
  miRNA–mRNA edges, and the multi-network GAT. Task, split (190/40, patient-disjoint), labels and DE
  features are unchanged from `repro-verify-2026-11`.
* **TCGA-BRCA TNBC is a supporting validation cohort** with true same-specimen miRNA-seq + RNA-seq
  (same TCGA sample barcode). It answers the two questions GEO alone cannot:
  1. *Do the cross-modal (miRNA–mRNA) relationships the networks encode hold on true pairs?*
  2. *Do GEO-derived network priors and the GEO-trained GAT transfer to an independent platform?*
* **Pooling GEO and TCGA into one training set is not used** for the headline results. The
  harmonization study (section 5) shows batch stays fully learnable within each tissue class after
  every correction we tried, so pooled results cannot be shown to be batch-free. Cross-cohort
  (leave-one-cohort-out) validation is used instead.

### TCGA cohort (source: `results/tcga_matched/*/run_info.json`, `00_build_tcga_tnbc.py` output)

| Quantity | n |
|---|---|
| TNBC patients by clinical IHC (ER−, PR−, HER2− by IHC, or IHC equivocal + ISH negative) | 142 |
| **TNBC primary tumors with matched miRNA + mRNA (same sample barcode)** | **102** (65 N0, 37 N1–3) |
| TNBC metastases with both modalities | 1 (not usable) |
| Solid-tissue normals with both modalities, any BRCA patient | 76 (6 from TNBC patients) |
| BRCA samples with both modalities (all subtypes) | 830 |
| Shared features with the GEO panels | 237 miRNAs mapped to miRBase arms, 230/230 mRNAs; 66 miRNA + 153 mRNA of the GEO DE nodes |

Class-by-batch table (`harmonization/class_by_batch.csv`): normal GEO 39 / TCGA 76; primary GEO 138 /
TCGA 102; metastatic GEO 53 / TCGA 0. Batch is partly confounded with class: metastasis exists only
in GEO, and 70 of the 76 TCGA normals come from non-TNBC patients (normal breast is not
receptor-subtyped, but the paper must say so).

## 2. New methods text (drop-in)

> **External validation cohort with true miRNA–mRNA pairs.** GEO GSE45498 provides no patient key
> linking miRNA and mRNA profiles, so cross-modal relationships in the GEO networks rest on
> tissue-level pairing. We therefore validated them in TCGA-BRCA, where miRNA-seq
> (log2 RPM + 1) and RNA-seq (log2 RSEM + 1) come from the same tumor specimen (UCSC Xena,
> TCGA.BRCA.sampleMap). TNBC was defined from clinical IHC (ER and PR negative; HER2 IHC negative, or
> equivocal with negative ISH), giving 142 TNBC patients, 102 of whom have a primary tumor profiled
> on both platforms. Solid-tissue normals profiled on both platforms (n = 76) served as normal
> tissue. Features were restricted to the molecules on the GEO NanoString panels; legacy miRNA names
> were mapped to the dominant miRBase arm.
>
> **Cross-modal edge validation.** For each edge in the GEO tissue networks we computed the Pearson
> correlation of the same two molecules across TCGA true pairs of the matching tissue (GEO normal →
> TCGA normal; GEO primary and metastatic → TCGA TNBC primary, the only TCGA TNBC tissue with n > 1)
> and compared the median signed correlation with 10,000 size-matched random edge sets of the same
> type. We also tested whether the agreement between TCGA true-pair and GEO cross-modal correlation
> patterns exceeds a null in which miRNA and mRNA profiles are randomly re-paired within each GEO
> tissue (200 permutations), and whether FDR-significant negative miRNA–mRNA correlations in TCGA
> are enriched for miRDB v6.0 predicted targets (score ≥ 80; Fisher's exact test, with positive
> correlations as a negative control).
>
> **Cross-cohort transfer.** The GAT trained on the GEO training split with the GEO networks was
> applied unchanged to all 178 TCGA samples; the score for tumor was 1 − P(normal). Each cohort was
> z-scored on its own samples (label-free). We compared the GAT with the same GAT after removing all
> edges, after degree-preserving rewiring of each edge type (20 rewirings), and with XGBoost on the
> same features, over 20 seeds, with patient-cluster bootstrap 95% CIs (2,000 resamples).
>
> **TCGA-internal analyses.** Within TCGA we rebuilt class-specific networks from training patients
> only (patient-grouped 75/25 split; feature selection, scaling and networks fit on training data)
> and trained the same GAT for tumor vs normal and for nodal status (N0 vs N1–3, a proxy for spread,
> because TCGA has no TNBC metastases with both modalities).
>
> **Harmonization.** We compared no correction, reference-batch ComBat (GEO as reference, tissue
> as protected covariate, fit on training samples of the classes present in both cohorts), per-cohort
> z-scoring, per-feature quantile mapping of TCGA onto GEO, and within-sample rank scoring. All
> parameters were fit on training patients only. Diagnostics used only normal and primary samples.

## 3. Results tables

### 3a. Primary cohort: GEO 3-class, held-out 40 samples, 20 seeds (`geo_replication/geo_replication_summary.csv`)

Nodes are the GEO DE features that also exist in TCGA (66 miRNA + 153 mRNA). The graph is the only
thing that changes between GAT rows. Note: `geo_all` here applies BH-FDR to cross-modal edges as for
the other edge types, so GEO normal and primary networks contain no cross-modal edges (none survive
FDR on GEO's tissue-level pairing; 16 negative ones survive in metastases).

| Model | macro-F1 mean ± SD | AUROC mean | 20-seed ensemble F1 [95% patient-bootstrap CI] |
|---|---|---|---|
| GAT, GEO-derived edges | 0.867 ± 0.037 | 0.972 | 0.905 [0.752, 1.000] |
| GAT, GEO within-modality + TCGA true-pair cross-modal edges ("hybrid") | 0.875 ± 0.027 | 0.973 | 0.905 [0.752, 1.000] |
| GAT, TCGA true-pair networks | 0.874 ± 0.032 | 0.973 | 0.905 [0.752, 1.000] |
| GAT, edges removed | 0.868 ± 0.037 | 0.971 | 0.905 [0.752, 1.000] |
| GAT, hybrid networks rewired (50 rewirings) | 0.870 ± 0.040 | – | empirical p (hybrid ≥ rewired) = 0.65 |
| XGBoost, same nodes | 0.840 ± 0.007 | 0.982 | 0.842 [0.706, 0.949] |

### 3b. Do the GEO networks hold on TCGA true pairs? (`geo_priors_in_tcga/geo_edges_on_tcga_true_pairs.csv`)

Signed r = r for co-expression edges, −r for miRNA–mRNA (anti-correlation) edges.

| GEO network → TCGA tissue | Edge type | GEO edges mapped | Same sign in TCGA | FDR-sig, same sign (random pairs) | Median signed r (random) | p vs random |
|---|---|---|---|---|---|---|
| normal → normal (n = 76) | miRNA–miRNA | 120 | 0.82 | 0.65 (0.34) | 0.39 (0.16) | < 1e-4 |
| normal → normal | mRNA–mRNA | 134 | 0.77 | 0.60 (0.28) | 0.35 (0.05) | < 1e-4 |
| normal → normal | miRNA–mRNA | 107 | 0.52 | 0.04 (0.09) | 0.01 (−0.01) | 0.21 |
| primary → TNBC tumor (n = 102) | miRNA–miRNA | 113 | 0.89 | 0.54 (0.26) | 0.26 (0.08) | < 1e-4 |
| primary → TNBC tumor | mRNA–mRNA | 133 | 0.96 | 0.81 (0.20) | 0.45 (0.04) | < 1e-4 |
| primary → TNBC tumor | miRNA–mRNA | 108 | 0.60 | 0.09 (0.08) | 0.06 (0.01) | 0.016 |
| metastatic → TNBC tumor | miRNA–miRNA | 97 | 0.92 | 0.75 (0.26) | 0.36 (0.08) | < 1e-4 |
| metastatic → TNBC tumor | mRNA–mRNA | 136 | 0.92 | 0.70 (0.20) | 0.37 (0.04) | < 1e-4 |
| metastatic → TNBC tumor | miRNA–mRNA | 120 | 0.68 | 0.05 (0.08) | 0.07 (0.01) | 0.007 |

Cross-modal pattern agreement against a GEO re-pairing null (`edge_replication/edge_replication_tcga_vs_geo.csv`):
Spearman between TCGA-tumor true-pair r and GEO r over 10,098 miRNA–mRNA pairs is 0.257 for GEO
metastases (null 0.009 ± 0.103, z = 2.41, p = 0.015) and 0.153 for GEO primaries (null −0.015 ± 0.122,
z = 1.37, p = 0.10). For normals it is −0.056 (p = 0.71). This matches the order-link evidence in
`REPRO_REPORT.md` (metastases linked, primaries partially, normals not).

True-pair cross-modal structure in TCGA (`tumor_vs_normal/run_info.json`,
`geo_replication/mirdb_support_true_pair_edges.csv`): among the 60 × 60 train-selected nodes,
621 (tumor) and 195 (normal) negative miRNA–mRNA correlations survive BH-FDR. GEO has 0 in normal
and primary. Over the whole shared panel (56,580 pairs), FDR-significant negative pairs in TNBC
tumors are enriched for miRDB targets (2.02% vs 1.47%, OR 1.38, Fisher p = 0.018). Positive
correlations (negative control) show no enrichment (OR 0.92, p = 0.73), and neither do normals
(OR 0.94, p = 0.70). The effect is modest and should be described as modest.

### 3c. GEO-trained model applied to TCGA (all 178 samples, tumor vs normal; `geo_priors_in_tcga/geo2tcga_summary.csv`)

| Model (trained on GEO only) | AUROC mean ± SD | macro-F1 mean ± SD | macro-F1 95% CI (ensemble) |
|---|---|---|---|
| GAT, GEO networks | 0.996 ± 0.001 | 0.969 ± 0.006 | [0.936, 0.989] |
| GAT, edges removed | 0.996 ± 0.001 | 0.968 ± 0.005 | [0.936, 0.989] |
| GAT, GEO networks rewired (20) | 0.995 ± 0.001 | 0.969 ± 0.006 | p (real ≥ rewired) = 0.57 |
| XGBoost | 0.994 ± 0.001 | 0.905 ± 0.010 | [0.858, 0.947] |

Reverse direction, TCGA-trained (TCGA networks) → all 230 GEO samples, normal vs tumor
(`geo_replication/transfer_tcga_to_geo_summary.csv`): GAT AUROC 0.993 ± 0.002, F1 0.767 ± 0.070;
logistic regression 0.990 / 0.885; XGBoost 0.966 / 0.860. Ranking transfers, calibration does not.

### 3d. TCGA-internal (`tumor_vs_normal/summary.csv`, `node/summary.csv`; 20 seeds)

| Task (test n) | GAT real | GAT edges removed | GAT graph-only | XGBoost | L2 logistic | Naive stratified | Rewired null p |
|---|---|---|---|---|---|---|---|
| Tumor vs normal (44: 17 N / 27 T), macro-F1 | 1.000 ± 0.000 | 0.999 ± 0.005 | 0.977 ± 0.024 | 0.976 | 1.000 | 0.522 | 1.0 (ceiling) |
| Node N0 vs N1–3 (26: 18 / 8), AUROC | 0.558 ± 0.024 | 0.547 ± 0.045 | 0.472 ± 0.035 | 0.444 ± 0.034 | 0.590 | 0.519 | 0.42 (100 rewirings) |
| Node, macro-F1 [ensemble 95% CI] | 0.518 ± 0.052 [0.35, 0.76] | 0.486 ± 0.058 | 0.445 ± 0.041 | 0.453 ± 0.025 | 0.561 | 0.512 | 0.40 |

Tumor vs normal is at ceiling for every model, so it is a sanity check, not a model comparison.
Nodal status is not predictable from these panels (no feature passes FDR on the training split;
every model's CI includes chance). Report this as a negative result.

## 4. Verdict on the GAT

**The GAT matches the baselines; it does not beat them in a way attributable to the network.**
On GEO it averages 0.87 macro-F1 vs XGBoost 0.84, and on GEO→TCGA transfer 0.969 vs 0.905
(bootstrap CIs barely overlap). But the edge-removed GAT and rewired-network GAT score the same in
every setting (GEO p = 0.65, GEO→TCGA p = 0.57, TCGA node p = 0.40), and an L2 logistic regression
is as good or better in several rows. So the advantage over XGBoost comes from the GAT's
expression pathway and smooth, calibrated outputs, not from graph topology. Replacing GEO's
unverifiable cross-modal edges with TCGA true-pair edges gives the same accuracy (0.875 vs 0.867).
That is honest support for keeping the multi-omic GAT. Its networks are biologically grounded and
validated on true pairs, and it transfers across platforms. But the paper must not claim that
network structure improves classification.

## 5. Harmonization and pooling (`harmonization/`, `harmonization_eval/`)

Setup: 219 shared nodes; GEO log2(count + 1), TCGA log2(RSEM/RPM + 1). Correction fit on training
patients only (GEO 190-sample split + TCGA patient-grouped 75%). Diagnostics on normal + primary only.
Evaluations: GAT 10 seeds, XGBoost 20 seeds, L2 logistic 1.

| Method | Silhouette batch | Silhouette class | kBET reject (normal / primary) | Batch AUROC within normal / primary | DE sign kept (TCGA) | Pooled-train DE sign = GEO |
|---|---|---|---|---|---|---|
| none | 0.82 | 0.13 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 | 0.41 |
| ComBat, GEO reference, tissue protected | 0.04 | 0.35 | 0.83 / 0.55 | 1.00 / 0.89 | 1.00 | 0.89 |
| per-cohort z-score | 0.02 | 0.30 | 0.70 / 0.78 | 0.97 / 0.95 | 1.00 | 0.90 |
| quantile map to GEO | 0.02 | 0.29 | 0.60 / 0.76 | 0.97 / 0.95 | 1.00 | 0.91 |
| within-sample rank | 0.36 | 0.24 | 1.00 / 1.00 | 1.00 / 1.00 | 0.94 | 0.83 |

Batch AUROC is a patient-grouped 5-fold L2 classifier predicting GEO vs TCGA within one tissue class,
so 0.5 would mean batch-free. TCGA-vs-GEO DE sign concordance on shared nodes is 0.83 (Spearman of
t-statistics 0.90) under every location/scale method. That is a real biological agreement and
the corrections do not distort it. The embedding panels (PCA and UMAP, by batch and by class) are in
`harmonization/harmonization_embeddings.png`.

Model results (means; `harmonization_eval/harmonization_eval_summary.csv`):

| Experiment | Method | GAT | XGBoost | L2 logistic |
|---|---|---|---|---|
| Pooled train → held-out patients, 3-class macro-F1 (GEO test) | none / ComBat / z / quantile / rank | 0.82 / 0.85 / 0.87 / 0.85 / 0.90 | 0.80 / 0.79 / 0.82 / 0.83 / 0.84 | 0.85 / 0.87 / 0.84 / 0.84 / 0.91 |
| GEO → TCGA held-out, normal-vs-tumor F1 | none / ComBat / z / quantile / rank | 1.00 / 1.00 / 1.00 / 0.99 / 0.99 | 0.38 / 0.98 / 0.93 / 0.83 / 0.38 | 0.38 / 1.00 / 0.98 / 0.95 / 0.38 |
| TCGA → GEO held-out, normal-vs-tumor AUROC | none / ComBat / z / quantile / rank | 0.99 / 0.99 / 0.99 / 0.99 / 0.98 | 0.51 / 0.98 / 0.97 / 0.97 / 0.97 | 0.98 / 0.98 / 0.98 / 0.98 / 0.98 |
| Pooled: batch AUROC of model's P(normal) within GEO+TCGA primaries | none / ComBat / z / quantile / rank | 0.17 / 0.34 / 0.13 / 0.14 / 0.37 | 0.40 / 0.29 / 0.14 / 0.09 / 0.07 | 0.03 / 0.24 / 0.04 / 0.05 / 0.73 |

The last row should be about 0.5 if predictions ignore batch. It is far from 0.5 for every method,
so pooled predictions for the same tissue differ systematically by cohort. Rank scoring raises
pooled GEO 3-class F1 to 0.90, but batch remains perfectly separable under it (AUROC 1.00), so that
gain cannot be credited to biology.

**Recommendation: do not pool.** No method makes batch unlearnable within class (best within-class
batch AUROC 0.89), the metastatic class exists only in GEO, and TCGA normals are mostly non-TNBC.
A pooled result would be confounded and a reviewer would be right to reject it. Use
leave-one-cohort-out validation with per-cohort z-scoring. It is label-free, needs no shared
classes, and gives the transfer numbers in 3c (GEO→TCGA GAT F1 0.97, AUROC 0.996). ComBat with GEO
as reference is a sensitivity analysis only; it needs labels for the protected covariate, which is
not available for a truly external cohort. Report the harmonization table above in the supplement
as the evidence for this choice.

## 6. Which sections change

| Section | Change |
|---|---|
| Abstract | Add the TCGA validation cohort (142 IHC-TNBC patients, 102 with matched miRNA + mRNA tumors, 76 normals). Add the cross-modal validation (GEO tumor-network cross-modal edges hold on true pairs above chance, p = 0.016 / 0.007; miRDB enrichment OR 1.38). Add GEO→TCGA transfer (AUROC 0.996). State that the GAT matches but is not shown to beat edge-removed or rewired controls. |
| Introduction | One paragraph: public miRNA/mRNA cohorts often lack a patient key; we validate on a cohort that has one. |
| Data | New subsection: TCGA-BRCA TNBC cohort, TNBC definition, matched-pair criterion, shared-feature mapping, class-by-batch table. |
| Methods | Insert section 2 text. Network construction unchanged for GEO. Add the edge-validation, transfer and harmonization procedures. |
| Results | Keep the GEO main results (3a). Add 3b (edge validation), 3c (transfer), 3d (TCGA-internal) and the harmonization table (supplement). |
| Discussion | The multi-omic network is biologically supported on true pairs, more for within-modality edges than cross-modal ones. Topology is not shown to drive accuracy. Pooling is unsafe. |
| Limitations | (1) GEO miRNA–mRNA pairing is tissue-level, not patient-verified. TCGA supports the cross-modal edges of tumor networks but not of the normal network. (2) TCGA has no TNBC metastases with both modalities, so the metastatic class is validated only through network structure; nodal status, the proxy, is not predictable (AUROC ≈ 0.56). (3) 70 of 76 TCGA normals are from non-TNBC patients. (4) The tumor-vs-normal transfer task is near ceiling. (5) Platform effects remain learnable after every harmonization tried. |
| Supplement | All CSVs in `results/tcga_matched/`, the embeddings figure, the miRNA name map (`*/geo_to_tcga_mirna_map.csv`). |

## 7. Caveats on these numbers

* Single fixed patient-grouped split per task. Seed SDs reflect model randomness, not split
  variance; the bootstrap CIs cover the test-set sampling.
* Rewired nulls use one seed per rewiring and compare against the 20-seed mean of the real graph.
* The GEO cross-modal-edge FDR filter in 3a (`02_geo_replication.py`) differs from `notebooks/03`,
  which keeps the top-150 anti-correlations without FDR. Section 3b and 3c use the committed
  `notebooks/03` networks, so both variants are covered.
* Legacy NanoString miRNA names map to the dominant miRBase arm (`map_geo_mirna_to_tcga`). Merged
  probes (e.g. miR-20a+miR-20b) and names with no miRBase match are dropped. The lists are in
  `geo_replication/geo_replication_info.json`.
