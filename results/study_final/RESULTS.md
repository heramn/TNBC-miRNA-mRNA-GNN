# GSE45498 pairing-free multi-omic GAT: final study results (study-final)

The study plan, code and TCGA edge list were **frozen after exploration**. They were not pre-registered.

| Item | Value |
|---|---|
| TCGA edge freeze | tag `tcga-edges-frozen-v1` → `9adec57`, 2026-11-22T23:09Z |
| Study freeze | tag `study-frozen-v1` → `515c096`, 2026-11-22T23:28:26Z (`study/FREEZE_STUDY.json`) |
| Final runs | 2026-11-22T23:30Z to 2026-11-23T05:30Z (UTC) |
| Frozen-file check | `pytest study/tests`: 11 passed on the results commit, including the sha256 test |

The plan is in `study/STUDY_PLAN.md`, and the edge freeze and its exploration timeline are in
`study/FREEZE_EDGES.md`. All numbers below come from `results/study_final/tables/*.csv`, which
`study/analysis.py analyze` writes from `predictions_final.csv.gz`.

**Task.** Each GSE45498 profile is classified as normal, primary TNBC or lymph-node metastasis. This
is a tissue-state task, not progression. There are 282 miRNA profiles and 249 mRNA profiles. Every
example has one observed modality. No sample is paired across modalities at training or test time.

## Bottom line

On this cohort, cross-modal miRNA→mRNA edges do not improve classification.

- **Edges vs no edges (H1):** miRDB-edge GAT vs the same GAT with no cross edges. The difference is
  within ±0.03 macro-F1 on both modalities, and the TOST equivalence test passes.
- **Real vs shuffled edges (H3):** miRDB edges vs degree-preserving rewired edges. Also equivalent
  within ±0.03.
- **TCGA edges (secondary arm):** density-matched TCGA edges behave the same as miRDB edges.

This matches the null outcome stated before the runs. Three caveats limit how far the null can be
read.

1. **The positive control failed.** The pipeline did not show that it can use planted edge structure,
   so the null is "no detectable benefit from this model on this data". It is not evidence that the
   edges carry no information.
2. **The joint-vs-routed comparison (H2) is inconclusive.** Its CIs are about ±0.04 wide, and power at
   the 0.03 margin is 0.35–0.42.
3. **Every GAT variant trails simple baselines.** Elastic net and TabPFN v2 beat all GAT variants, so
   the GAT is not the best model for this task either way.

## Claims and evidence

| Claim | Evidence | Status |
|---|---|---|
| miRDB cross edges do not change macro-F1 beyond ±0.03 (H1) | ΔF1 +0.001 (miRNA), +0.004 (mRNA); 90% corrected CIs inside ±0.03; TOST p ≤ 0.001; Holm p ≥ 0.29 | Supported (equivalence) |
| Real miRDB edges do no better than rewired edges (H3) | ΔF1 −0.0004 / −0.0015; 90% CI within ±0.005; TOST p < 0.0001; Holm p = 1 | Supported (equivalence) |
| TCGA-discovered edges match miRDB, rewired and no-cross edges (B) | all ΔF1 between −0.002 and +0.004; all TOST p ≤ 0.002 | Supported (equivalence); not in the Holm family |
| Joint training differs from routed single-modality GATs (H2) | ΔF1 −0.011 (miRNA, 95% CI −0.045 to 0.023), +0.001 (mRNA, −0.037 to 0.039); power 0.42 / 0.35 | Inconclusive |
| The pipeline can detect a planted edge signal (positive control) | δ=0.5, mRNA: mirdb − rewired ΔF1 −0.001 (−0.015 to 0.013), sign-flip p = 0.11; the success criterion was p < 0.05 | **Not shown (failed)** |
| The GAT beats simple baselines | elastic net routed 0.904 / 0.813 and TabPFN v2 routed 0.927 / 0.863, vs joint_mirdb 0.875 / 0.778 | Not supported. The GAT is worse than TabPFN (95% CI excludes 0 on both modalities). |
| LN-met vs primary is not explained by technical shortcuts alone | GAT AUROC beats control probes alone by +0.043 (miRNA, 95% CI 0.012–0.079) and +0.188 (mRNA, 0.103–0.275) | Supported, with caveats (batch section) |
| Single-modality models transfer zero-shot to the other modality | macro-F1 0.10–0.12, below chance (0.33) | Not supported |
| TCGA edges replicate within TCGA and are anti-correlated pairs consistent with repression | split-half overlap 0.40 vs 0.13 random; 96% negative in non-TNBC BRCA; miRDB OR 2.1–3.0 | Supported, within TCGA only |

## 1. Primary hypotheses (5 × 5 grouped CV, 20 seeds, Holm across H1–H3 × 2 modalities)

Δ is arm A minus arm B, in pooled out-of-fold macro-F1 averaged over repeats and seeds. The CIs are
Nadeau–Bengio corrected, over 25 fold-level seed-mean differences. TOST uses a ±0.03 margin.
Power is achieved power to detect Δ = 0.03 at α = 0.05. The primary test is the patient-level
sign-flip test on per-sample log-loss differences, with 20,000 flips.

| Hyp. | Contrast | Test mod. | F1 A | F1 B | ΔF1 | 95% CI | 90% CI | TOST p | Power | Δ log-loss | Sign-flip p | Holm p |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H1 | joint_mirdb − joint_nocross | miRNA | 0.875 | 0.874 | +0.001 | −0.012, 0.014 | −0.010, 0.012 | 0.0001 | 1.00 | −0.001 | 0.76 | 1.00 |
| H1 | joint_mirdb − joint_nocross | mRNA | 0.778 | 0.774 | +0.004 | −0.011, 0.019 | −0.008, 0.017 | 0.0008 | 0.98 | −0.010 | 0.049 | 0.29 |
| H2 | joint_nocross − single_mi_nocross | miRNA | 0.874 | 0.885 | −0.011 | −0.045, 0.023 | −0.039, 0.017 | 0.13 | 0.42 | −0.025 | 0.28 | 1.00 |
| H2 | joint_nocross − single_mr_nocross | mRNA | 0.774 | 0.774 | +0.001 | −0.037, 0.039 | −0.031, 0.032 | 0.06 | 0.35 | +0.017 | 0.55 | 1.00 |
| H3 | joint_mirdb − joint_rewired | miRNA | 0.875 | 0.875 | −0.000 | −0.003, 0.002 | −0.002, 0.001 | <0.0001 | 1.00 | −0.000 | 0.96 | 1.00 |
| H3 | joint_mirdb − joint_rewired | mRNA | 0.778 | 0.780 | −0.002 | −0.005, 0.002 | −0.005, 0.002 | <0.0001 | 1.00 | +0.001 | 0.68 | 1.00 |

**Reading.** The mRNA H1 log-loss difference has nominal p = 0.049. It favours miRDB edges by
0.010 nats, but it does not survive Holm correction (p = 0.29) and the F1 difference is equivalent to
zero. H2 cannot be decided at this sample size. The patient-cluster bootstrap CI for H2 on miRNA is
−0.022 to −0.001, which leans towards joint training being slightly worse on miRNA examples, but that
is descriptive only. For H3 on mRNA, the cluster bootstrap CI (−0.003 to −0.000) narrowly excludes zero
in favour of rewired edges, a difference far inside the margin. We do **not** claim that cross-modal edges improve classification.

**Variance.** For joint_mirdb, the fold-to-fold SD of seed-mean macro-F1 is 0.046 (miRNA) and 0.064
(mRNA). The seed SD within a fold is 0.025 and 0.029. Fold composition matters more than seed.

## 2. Secondary arm B: frozen TCGA edges (not in the Holm family)

| Contrast | Test mod. | ΔF1 | 95% CI | TOST p | Δ log-loss | Sign-flip p |
|---|---|---|---|---|---|---|
| joint_tcga − joint_mirdb | miRNA | +0.000 | −0.010, 0.011 | <0.0001 | +0.007 | 0.027 |
| joint_tcga − joint_mirdb | mRNA | −0.000 | −0.010, 0.010 | <0.0001 | +0.005 | 0.41 |
| joint_tcga − joint_rewired | miRNA | +0.000 | −0.010, 0.010 | <0.0001 | +0.007 | 0.029 |
| joint_tcga − joint_rewired | mRNA | −0.002 | −0.010, 0.007 | <0.0001 | +0.006 | 0.37 |
| joint_tcga − joint_nocross | miRNA | +0.002 | −0.010, 0.013 | <0.0001 | +0.006 | 0.21 |
| joint_tcga − joint_nocross | mRNA | +0.004 | −0.012, 0.020 | 0.001 | −0.005 | 0.51 |

The TCGA arm is equivalent to the other arms on macro-F1, as expected before the runs. On miRNA test
examples, its log-loss is about 0.007 nats worse than miRDB or rewired edges (nominal p ≈ 0.03). This
is uncorrected and small. The TCGA edges are anti-correlated pairs consistent with repression, not
validated targets.

## 3. All arms and baselines

The table shows macro-F1 with patient-cluster bootstrap 95% CIs (2,000 resamples for contrasts and
500 for arm CIs). Runs are repeat × seed combinations. Full columns are in `tables/arm_summary.csv`.

| Arm | miRNA F1 (95% CI) | mRNA F1 (95% CI) | Log-loss mi / mr | ECE mi / mr | Runs |
|---|---|---|---|---|---|
| joint_mirdb (primary) | 0.875 (0.842–0.904) | 0.778 (0.729–0.820) | 0.365 / 0.610 | 0.043 / 0.094 | 100 |
| joint_tcga (secondary) | 0.875 (0.840–0.905) | 0.778 (0.727–0.821) | 0.373 / 0.614 | 0.043 / 0.095 | 100 |
| joint_nocross | 0.874 (0.841–0.904) | 0.774 (0.725–0.813) | 0.366 / 0.615 | 0.042 / 0.096 | 100 |
| joint_rewired | 0.875 (0.840–0.902) | 0.780 (0.730–0.820) | 0.365 / 0.609 | 0.044 / 0.094 | 100 |
| joint_mirdb_full (all DE nodes) | 0.884 (0.846–0.912) | 0.782 (0.738–0.827) | 0.328 / 0.590 | 0.039 / 0.088 | 25 |
| gcn_mirdb | 0.878 (0.845–0.910) | 0.777 (0.727–0.818) | 0.359 / 0.607 | 0.040 / 0.092 | 25 |
| mlp_only (no graph) | 0.884 (0.850–0.914) | 0.774 (0.721–0.813) | 0.341 / 0.598 | 0.043 / 0.087 | 25 |
| graphonly_mirdb | 0.769 (0.729–0.800) | 0.703 (0.659–0.738) | 0.540 / 0.674 | 0.068 / 0.083 | 25 |
| routed GAT, nocross | 0.885 (0.853–0.917) | 0.774 (0.728–0.814) | 0.392 / 0.590 | 0.055 / 0.073 | 25 |
| routed GAT, mirdb | 0.884 (0.851–0.914) | 0.781 (0.729–0.820) | 0.373 / 0.576 | 0.049 / 0.071 | 25 |
| routed GAT, rewired | 0.883 (0.849–0.914) | 0.781 (0.733–0.820) | 0.378 / 0.574 | 0.050 / 0.068 | 25 |
| XGBoost joint (NaN for missing) | 0.888 (0.849–0.920) | 0.788 (0.741–0.831) | 0.284 / 0.476 | 0.026 / 0.052 | 25 |
| XGBoost joint, all DE nodes | 0.890 (0.856–0.923) | 0.791 (0.743–0.832) | 0.299 / 0.468 | 0.029 / 0.055 | 25 |
| XGBoost routed | 0.892 (0.856–0.922) | 0.782 (0.738–0.827) | 0.283 / 0.464 | 0.037 / 0.051 | 25 |
| Elastic net joint | 0.904 (0.871–0.933) | 0.811 (0.764–0.854) | 0.307 / 0.494 | 0.051 / 0.074 | 5 |
| Elastic net routed | 0.904 (0.870–0.930) | 0.813 (0.763–0.856) | 0.310 / 0.513 | 0.045 / 0.070 | 5 |
| Target-set elastic net (miRDB prior) | 0.902 (0.871–0.929) | 0.813 (0.763–0.856) | 0.309 / 0.517 | 0.048 / 0.066 | 5 |
| **TabPFN v2 routed** | **0.927 (0.895–0.952)** | **0.863 (0.822–0.900)** | 0.268 / 0.373 | 0.039 / 0.044 | 25 |
| Naive (prior / majority) | 0.250 | 0.258 | 0.951 / 0.919 | — | 5 |

Descriptive contrasts against joint_mirdb are in `tables/contrasts.csv`, family `descriptive`.

- **TabPFN v2 routed:** −0.052 (miRNA, 95% CI −0.102 to −0.002) and −0.084 (mRNA, −0.155 to −0.014).
- **Elastic net routed:** −0.029 and −0.035. The CIs include 0, but log-loss is significantly worse,
  with sign-flip p = 0.02 / 0.0002.
- **Target-set elastic net vs elastic net:** the miRDB prior adds nothing to the linear model (ΔF1 −0.002 / −0.000).
- **Graph branch alone vs full model:** the graph branch alone is much worse (+0.106 / +0.075 for the
  full model), so the expression bypass carries most of the signal.

## 4. Per-class breakdown (Δ per-class F1, `fig3_per_class_gain.png`)

| Contrast | Mod. | normal | primary | LN metastasis |
|---|---|---|---|---|
| H1 mirdb − nocross | miRNA | −0.002 | +0.002 | +0.003 |
| H1 mirdb − nocross | mRNA | +0.002 | +0.005 | +0.006 |
| H2 joint − routed | miRNA | −0.029 | −0.012 | +0.009 |
| H2 joint − routed | mRNA | −0.029 | +0.003 | +0.028 |
| H3 mirdb − rewired | miRNA | +0.000 | −0.000 | −0.001 |
| H3 mirdb − rewired | mRNA | −0.001 | −0.001 | −0.003 |
| B1 tcga − mirdb | miRNA | −0.000 | −0.001 | +0.002 |
| B1 tcga − mirdb | mRNA | −0.001 | −0.002 | +0.002 |

No class shows an edge effect larger than 0.006. The H2 pattern goes the opposite way for normal
tissue (−0.029) and LN metastasis on mRNA (+0.028), but H2 itself is inconclusive.

## 5. Positive control (repeat 0): failed

Labels were permuted within each modality. A class signal was then planted along miRDB edges:
+1.0 SD on one miRNA per class in miRNA examples, and −δ SD on all of that miRNA's miRDB targets in
mRNA examples. Three runs are shown: miRDB edges, no cross edges, and rewired edges.

| δ | Test mod. | mirdb F1 | nocross F1 | rewired F1 | mirdb − rewired ΔF1 (95% CI) | Sign-flip p | mirdb − nocross ΔF1 (95% CI) | Sign-flip p |
|---|---|---|---|---|---|---|---|---|
| 0.25 | miRNA | 0.811 | 0.801 | 0.811 | −0.000 (−0.005, 0.005) | 0.13 | +0.010 (−0.051, 0.071) | 0.11 |
| 0.25 | mRNA | 0.407 | 0.396 | 0.408 | −0.001 (−0.013, 0.011) | 0.27 | +0.011 (−0.038, 0.059) | 0.032 |
| 0.5 | miRNA | 0.897 | 0.891 | 0.896 | +0.001 (−0.005, 0.007) | 0.15 | +0.006 (−0.021, 0.032) | 0.036 |
| 0.5 | mRNA | 0.653 | 0.660 | 0.654 | −0.001 (−0.015, 0.013) | 0.11 | −0.007 (−0.011, −0.003) | 0.081 |

The pre-specified success criterion was that mirdb beats rewired on mRNA at δ = 0.5 with sign-flip
p < 0.05. It is **not met** (p = 0.11, ΔF1 −0.001). The planted signal is learned: on permuted labels,
mRNA macro-F1 rises from about 0.40 at δ = 0.25 to about 0.65 at δ = 0.5. But it is learned directly
from the shifted target genes, which are observed in mRNA examples, and the edges add nothing.

This control, as frozen, only partly isolates edge use. A signal readable without edges does not
force the model to route through them. The failure still means this study cannot show that the GAT
would exploit genuine cross-modal edge information if it existed. The null H1/H3/B results should
be read with that limit.

The plan text says the control used 5 seeds. The frozen code ran each arm's own seed count, 20 seeds,
so the control has more runs than planned.

## 6. Rewired null (descriptive)

Each joint_rewired run uses a fresh degree-preserving rewiring of the fold's miRDB edges. It is
paired with the real-graph run at the same fold and seed, giving 100 runs per arm.

| Test mod. | Real mean F1 | Null mean F1 | Null SD | P(null ≥ real mean) | Mann–Whitney p (real > null) |
|---|---|---|---|---|---|
| miRNA | 0.8749 | 0.8752 | 0.014 | 0.58 | 0.56 |
| mRNA | 0.7784 | 0.7800 | 0.015 | 0.54 | 0.81 |

## 7. Zero-shot cross-modality transfer

Single-modality-trained GATs were scored on the other modality's test examples. In those examples
the training modality's nodes are latent target nodes: they are unobserved and get representations
only from messages.

| Trained → tested | nocross F1 | mirdb F1 | rewired F1 | mirdb − nocross ΔF1 (95% CI) | mirdb − rewired ΔF1 (95% CI) |
|---|---|---|---|---|---|
| miRNA → mRNA | 0.117 | 0.122 | 0.122 | +0.004 (−0.022, 0.030) | −0.000 (−0.007, 0.006) |
| mRNA → miRNA | 0.098 | 0.100 | 0.099 | +0.002 (−0.011, 0.015) | +0.001 (−0.002, 0.003) |

There is no usable zero-shot transfer. All cells are below chance (0.33), with or without edges.
Some log-loss differences are nominally significant: miRNA→mRNA mirdb − nocross is −0.27 nats,
p = 0.0003. They are differences between near-useless predictors, and we do not interpret them as
edge-mediated transfer.

## 8. Shortcut ladder (LN metastasis vs primary, same CV, binary elastic net)

| Mod. | Feature set | Features | AUROC (mean over repeats) |
|---|---|---|---|
| miRNA | control probes only | 19 | 0.910 |
| miRNA | immune miRNAs | 6 | 0.786 |
| miRNA | all subset features | 70 | 0.974 |
| miRNA | model: joint_mirdb / joint_nocross / xgb_joint | — | 0.955 / 0.955 / 0.976 |
| mRNA | control probes only | 14 | 0.693 |
| mRNA | lymphoid/myeloid genes | 12 | 0.677 |
| mRNA | lymphoid score (mean) | 12 | 0.476 |
| mRNA | all subset features | 156 | 0.897 |
| mRNA | model: joint_mirdb / joint_nocross / xgb_joint | — | 0.881 / 0.877 / 0.872 |

Margin of the joint_mirdb GAT over the best restricted shortcut (control probes), using
per-sample averaged scores and a patient-cluster bootstrap:

- **miRNA:** +0.043, 95% CI 0.012–0.079.
- **mRNA:** +0.188, 95% CI 0.103–0.275.

On miRNA, technical control probes alone separate LN metastasis from primary at AUROC 0.91. Most of
the miRNA met-vs-primary signal is therefore reachable from technical features (see section 11). A
linear model on all subset features matches or beats the GAT on both modalities.

## 9. Learning curve (repeat 0)

| Model | Test mod. | 25% | 50% | 100% macro-F1 |
|---|---|---|---|---|
| joint_mirdb | miRNA | not run (compute budget) | not run (compute budget) | 0.886 |
| joint_mirdb | mRNA | not run (compute budget) | not run (compute budget) | 0.797 |
| joint_nocross | miRNA | not run (compute budget) | not run (compute budget) | 0.882 |
| joint_nocross | mRNA | not run (compute budget) | not run (compute budget) | 0.799 |
| XGBoost joint | miRNA / mRNA | not run (compute budget) | not run (compute budget) | 0.888 / 0.810 |
| XGBoost routed | miRNA / mRNA | not run (compute budget) | not run (compute budget) | 0.897 / 0.817 |
| Elastic net routed | miRNA / mRNA | not run (compute budget) | not run (compute budget) | 0.905 / 0.818 |

The 100% points are the repeat-0, seeds 0–4 subset of the main runs. The 25% and 50% points were
phase-2 tasks and were skipped on the author's instruction.

## 10. Sensitivity analyses (repeat 0, 5 seeds) and 190/40 continuity

| Analysis | Test mod. | ΔF1 mirdb − nocross (95% CI) | Δ log-loss | Sign-flip p |
|---|---|---|---|---|
| Main scheme, repeat 0 (reference) | miRNA | +0.004 (−0.013, 0.021) | −0.007 | 0.39 |
| Main scheme, repeat 0 (reference) | mRNA | −0.002 (−0.021, 0.017) | +0.006 | 0.68 |
| Conservative grouping (miRNA linked to order-alignment partner) | miRNA | −0.005 (−0.027, 0.017) | +0.014 | 0.077 |
| Conservative grouping | mRNA | −0.007 (−0.049, 0.036) | −0.023 | 0.099 |
| 230-sample subset | miRNA | +0.003 (−0.018, 0.024) | −0.001 | 0.91 |
| 230-sample subset | mRNA | +0.017 (−0.014, 0.048) | +0.002 | 0.84 |
| TCGA normal-derived edges | — | not run (compute budget) | | |
| ESTIMATE-purity TCGA edges | — | not run (compute budget) | | |
| Train-side random within-tissue pairing augmentation | — | not run (compute budget) | | |

The conservative scheme lowers absolute miRNA F1 (mirdb 0.822 vs 0.886 on the main scheme at
repeat 0). That is consistent with some cross-modal same-patient leakage in the main scheme, or with
the harder split. The edge contrast stays null.

**190/40 continuity.** These runs use A's split, the paper configuration and 20 seeds. The test set
is 40 samples per modality.

| Row | miRNA F1 (boot 95% CI) | mRNA F1 (boot 95% CI) |
|---|---|---|
| joint_mirdb | 0.834 (0.691–0.936) | 0.792 (0.664–0.888) |
| joint_nocross | 0.835 (0.686–0.939) | 0.792 (0.661–0.888) |
| joint_rewired | 0.834 (0.664–0.950) | 0.798 (0.673–0.893) |
| joint_tcga | 0.835 (0.652–0.942) | 0.792 (0.636–0.886) |
| mirdb − nocross | −0.000 (−0.006, 0.005) | +0.000 (−0.007, 0.006) |
| mirdb − rewired | 0.000 (0.000, 0.000) | −0.006 (−0.016, 0.000) |
| tcga − mirdb | +0.000 (−0.005, 0.007) | −0.001 (−0.014, 0.011) |

With only 40 test examples, argmax predictions rarely flip between arms, so the miRNA mirdb and
rewired rows are identical. Their probabilities differ by up to 0.02 per sample. The earlier
fix-multiomic-A estimate of +0.009 (miRNA) for miRDB vs no-cross is not reproduced here
(−0.000, 0.000).

## 11. TCGA edge validation (TCGA only; no GEO data used)

- **Split-half replication:** 100 stratified 47/47 splits of the 94 purity-adjusted IHC-TNBC
  tumours. Top-K overlap between halves is 0.404 (95% range 0.322–0.491), vs 0.131 for random pairs.
  Sign agreement is 0.877 vs 0.530. Top-K beat random in 100% of splits.
- **Non-TNBC BRCA, 625 tumours, purity-adjusted:** 96.2% of the 394 reference pairs are negative and
  82.7% are negative at p < 0.05. For random pairs the negative fraction is 0.49 (0.44–0.55);
  empirical p = 0.001.
- **Prior enrichment, q < 0.05 negative edges over the 63,012-pair universe:**
  - miRDB: OR 2.14, p = 4×10⁻⁸
  - miRTarBase strong evidence: OR 1.49, p = 0.007
  - miRTarBase any evidence: OR 1.15, p = 0.13
- **Prior enrichment on the 190-sample reference subset:**
  - top-K vs miRDB: OR 2.95, p = 2×10⁻⁷
  - top-K vs miRTarBase strong: OR 1.58, p = 0.03
- **Purity-adjusted check of miRDB pairs:** miRDB pairs are more negative than other pairs. Over the
  universe, mean r is −0.021 vs −0.003 (Mann–Whitney p = 0.0002). On the reference subset it is
  −0.059 vs −0.013 (p < 10⁻⁴).

The TCGA edges are reproducible anti-correlated pairs consistent with repression, and they are
modestly enriched for curated targets. That they do not help the GEO classifier is therefore not
explained by the edges being noise in TCGA.

## 12. Batch audit (from GEO files only)

- **Separate project batch.** All 54 LN metastases are lymph-node tissue. Their mRNA raw headers
  carry a different project prefix (`TAS10-15`) from normal and primary (`TAS09 12`), so in mRNA,
  class and project batch are confounded.
- **miRNA technical features.** The miRNA positive spike-in geometric mean separates LN metastasis
  from primary at AUROC 0.78. Other features also differ by class:
  - negative-background mean: AUROC 0.71
  - endogenous total counts: AUROC 0.66
  - Kruskal p < 10⁻³ for each
- **mRNA technical features.** Endogenous total counts give AUROC 0.27, which is 0.73 in the
  reverse direction. The fraction of non-integer values gives 0.28.
- **What can't be checked.** NanoString RCC files (cartridge, lane, scan date) are not deposited, so
  a real batch analysis is impossible. All classifier results here are for tissue state, potentially
  including site and batch differences, not for biology alone.

## 13. Pre-specified analyses not run

| Analysis | Status |
|---|---|
| Learning curve, 25% and 50% of inner-fit groups (joint_mirdb, joint_nocross, baselines) | not run (compute budget) |
| Sensitivity: TCGA normal-derived edges (`sens_tcga_normal`) | not run (compute budget) |
| Sensitivity: ESTIMATE-purity TCGA edges (`sens_tcga_estimate`) | not run (compute budget) |
| Sensitivity: train-side random within-tissue pairing augmentation (`sens_aug_pairs`) | not run (compute budget) |

All of these were phase-2 tasks. No phase-2 run was started before the author asked on 2026-11-23
to skip them, so no partial sensitivity results exist. The conservative-grouping, 230-subset and
190/40 continuity rows were phase-1 tasks; they ran in full and are reported above. TabPFN ran with
its full 16-configuration grid, so no reduced-tuning flag applies.

The plan's own disclosed deviations also stand: scGLUE and miRTarBase/TargetScan secondary-prior
GAT arms were not run, and there is no scGLUE baseline.

## 14. Deviations and execution notes

1. **Rewired null.** Instead of 200 rewirings × 5 seeds, joint_rewired uses one fresh rewiring per
   run (100 runs), paired with the real-graph run at the same fold and seed. H3 is tested on that arm.
2. **Seeds.** Secondary rows use 5 seeds per fold. Sensitivities use repeat 0 only. The elastic net
   uses 1 seed per repeat. The positive control ran 20 seeds per arm rather than the plan text's 5,
   because the frozen code uses each arm's seed count.
3. **Training.** Training was capped at 100 epochs with patience 15, set before the freeze.
4. **TabPFN.** TabPFN uses the open v2 classifier weights (tabpfn 2.0.9; TabPFN ≥ 2.5 needs a
   license token). Its separate environment pins scikit-learn < 1.7.
5. **Data sources.**
   - miRTarBase 9.0 came from the Zenodo CyTargetLinker mirror (record 13789211), because the
     official server timed out.
   - The miRBase aliases file returned HTTP 500, so the probe mapping uses probe → MIMAT → v22 names.
6. **scGLUE.** No scGLUE baseline was run.
7. **Thread setting.** The only execution change after the freeze is the environment setting
   `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`, which fixes BLAS oversubscription.
   No frozen file changed; the sha256 test passes.
8. **Sharding.** Phase 1 was split into 8 deterministic shards (`run_gat.py shard`). Shard 5 ran on
   the coordinating machine. Shards 0, 2 and 7 were rebalanced across machines with helper drivers
   that call the frozen `run_task` / `run_one` unchanged. Several runs were computed on two machines;
   every such duplicate was identical apart from the wall-clock column, and runs are deterministic.
9. **Analysis code.** `study/analysis.py` and `study/figures.py` were written after the freeze and
   before the runs finished. They were then debugged on partial results.
   - One bug was fixed in `analysis.py`: the baseline glob also matched GAT run files.
   - One label was fixed in `figures.py`: the H2 row in fig1.
   - Partial positive-control and contrast numbers were seen during that debugging. No configuration,
     arm, run or statistic was changed as a result.
10. **Seen before the freeze.** One seed-0 timing run each of A's joint and miRNA-only GAT on the
    190/40 split was seen before the freeze (FREEZE_EDGES.md). So was the fix-multiomic-B "hybrid"
    TCGA-edge result, which is a prior look at nearly the same comparison as B.

## 15. Exploration timeline (UTC)

| When | Event |
|---|---|
| 2026-11-22 12:40–18:42 | Branches repro-verify, fix-multiomic-A and fix-multiomic-B were explored and their results seen (FREEZE_EDGES.md) |
| 2026-11-22 ~23:00 | Seed-0 timing runs of A's model on the 190/40 split |
| 2026-11-22 23:09 | TCGA edge list frozen (`tcga-edges-frozen-v1`) |
| 2026-11-22 23:10–23:28 | Study code written; smoke tests ran with 2 epochs and 2 configs, and were not scored |
| 2026-11-22 23:28 | Study frozen (`study-frozen-v1`) |
| 2026-11-22 23:30 → 2026-11-23 05:30 | Final phase-1 GAT runs, baselines and TabPFN |
| 2026-11-23 ~04:50 | Analysis code dry-run on partial results (see deviation 9) |
| 2026-11-23 ~05:10 | Author instruction: skip all phase-2 runs (compute budget) |
| 2026-11-23 05:50 | Final analysis on complete phase-1 results |

## Files

- `results/study_final/predictions_final.csv.gz`: every selected-config test prediction. The
  columns are scheme, arm, repeat, fold, config, seed, sample id, modality, label, group, class
  probabilities and number of cross edges.
- `results/study_final/selected_configs.csv` and `runs/selection/*.json`: inner-validation config
  selection for each tuned arm × fold.
- `results/study_final/gat_runs_summary.csv.gz`: inner-validation loss for every tuning run.
- `results/study_final/tables/`: all tables above.
- `results/study_final/figures/`, nine figures:
  - fig1: hypotheses forest plot
  - fig2: arms and baselines
  - fig3: per-class differences
  - fig4: shortcut ladder
  - fig5: learning curve, 100% points only
  - fig6: positive control
  - fig7: reliability
  - fig8: zero-shot transfer
  - fig9: TCGA split-half replication
- `results/study_final/edge_validation/`, `batch_audit/`, `shortcut_predictions.csv`,
  `mirtarbase_pairs.csv`. The TabPFN fold inputs (`tabpfn_inputs/`) are git-ignored and are
  regenerated by `study/baselines.py`.
- The per-run CSVs (`results/study_final/runs/<scheme>/<arm>/`, 8,383 files, 156 MB) are not
  committed; the selection JSONs are. Everything needed for the tables is in `predictions_final.csv.gz`, and the runs can be
  regenerated deterministically with `study/run_gat.py`.
- `study/make_repro_package.sh` builds a local reproducibility archive. It is not uploaded or
  published.
