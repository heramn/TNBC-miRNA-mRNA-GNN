# REVISION_PLAN_A: a pairing-free multi-omic GAT for GSE45498

Branch: `fix-multiomic-A` (from `repro-verify-2026-11`). Date: 2026-11-22.
Every number below comes from a CSV in `results/fix_multiomic_A/`, produced by
`notebooks/41_unpaired_multiomic_gat.py` and `notebooks/42_unpaired_baselines_stats.py` on the corrected
230-sample build (190 train / 40 test, patient-disjoint, 93 DE miRNAs + 157 DE mRNAs).
Nothing was tuned on the test set, and no number was chosen to match the old manuscript.

Framing (from Heram): GEO GSE45498 is the primary cohort and the pairing-free GAT below is the main
method. TCGA-BRCA is supporting validation and is handled in a separate run; nothing in this
document depends on it.

## 1. Verdict

* **The paper can stay multi-omic and keep the GAT without any miRNA-mRNA patient pairing.** The
  model below never combines a miRNA profile with an mRNA profile at the sample level. Cross-modal
  structure comes from prior knowledge (miRDB v6.0 target predictions), and the two omics are
  integrated through one shared graph and shared GAT weights. Every training and test example
  carries the label from its own GEO record.
* **GAT vs XGBoost: the GAT matches XGBoost and does not significantly beat it.** On miRNA test
  samples the joint GAT is nominally higher (macro-F1 0.854 ± 0.018 over 20 seeds vs 0.809; paired
  patient-cluster bootstrap difference +0.045, 95% CI [−0.000, 0.124]; McNemar p = 0.50). On mRNA
  test samples they are tied (0.797 vs 0.793; difference +0.005, CI [−0.135, 0.144]). With 40 test
  samples (6 normals) the comparison is underpowered, so "not significantly different" is not
  evidence of equivalence.
* **What the multi-omic design buys is measurable, but it comes from joint training, not from the
  specific miRDB edges.** Training one GAT on both modalities raises miRNA-sample macro-F1 from
  0.823 (miRNA-only GAT) to 0.854 (difference +0.031, CI [0.005, 0.068]); for mRNA samples the
  gain is +0.017, CI [−0.023, 0.053]. Most of the miRNA gain is already present without cross
  edges (+0.022, CI [0.003, 0.051]). The miRDB cross edges add +0.009 (CI [0.000, 0.024]) on miRNA
  samples and +0.002 on mRNA samples. Against 100 degree-preserving rewirings of those edges the
  empirical p is 0.34 (miRNA), 0.52 (mRNA) and 0.44 (all test samples). So "biological target
  topology improves classification" is **not** supported, consistent with the network-ablation
  result already in the paper.
* **Specimen-level fusion is reported as a bounded sensitivity analysis, not as the headline.**
  Late fusion of the joint GAT's two per-modality predictions gives 0.869 ± 0.025 under the
  order-based pairing and 0.925 (95% range 0.851–0.978) under 1,000 random within-tissue
  pairings. The random-pairing number is optimistic by construction (see 5) and must not be the
  headline.

These contrasts are 11 unadjusted comparisons on one 40-sample test set. Present them as
estimates with intervals and describe the significant ones as suggestive.

## 2. Design

**Candidates considered.** (1) Prior-knowledge miRNA-mRNA edges with per-modality GATs and late
fusion. (2) Tissue-level or order-supported pairing with sensitivity analysis over random
within-tissue pairings. (3) A two-tower GAT with cross-modal attention through prior-knowledge
target edges. The chosen design merges (1) and (3) into one model. (2) is kept only as
sensitivity analyses (the paired_random / paired_order inputs and specimen-level late fusion),
because pairing by tissue uses the label to build test inputs and cannot be the primary analysis.

**Graph.** One node set: 93 DE miRNAs + 157 DE mRNAs (250 nodes), as in the corrected build. Three
tissue networks (normal, primary, metastatic), as in the paper's multi-network GAT:
* Intra-modal edges: the top 150 miRNA-miRNA and top 150 mRNA-mRNA edges by |Pearson r| in each
  tissue's training samples, computed **within one modality**. This needs no cross-modal pairing.
* Cross-modal edges: miRDB v6.0 predicted targets with score > 70 between a DE miRNA and a DE mRNA:
  **574 edges, 82 miRNAs, 119 mRNAs**, median score 85.6
  (`results/fix_multiomic_A/mirdb_cross_edges.csv`). They are the same in all three tissue
  networks and are weighted by score / 100. NanoString probe names (miRBase v14 era) are mapped to
  miRDB (miRBase v22) mature names by removing the probe note, splitting merged probes
  (`hsa-miR-20a+hsa-miR-20b`), and trying the bare name, -5p/-3p arms, and an "a" paralog letter.
  An exact match covers only 23 of 93 miRNAs; the mapping covers 82.
  No cross-modal edge is derived from a cross-sample correlation.

**Examples.** Each GEO sample is one example with one observed modality: 190 miRNA + 190 mRNA
training examples, 40 + 40 test examples. Node input = [value × observed, observed flag, learned
16-d node embedding, 4-d modality embedding]. Unobserved nodes get value 0 and flag 0, so their
GAT representations are built only from attention-weighted messages, including messages along the
miRDB edges from the observed modality.

**Model.** The paper's MultiNetworkGAT architecture, unchanged except for the observed-flag input:
GATConv(2 heads × 32, edge_dim = 1) → ELU → dropout 0.2 → GATConv(1 × 32), mean+max pooling per
tissue network, a per-network branch classifier, an expression bypass over [masked values, flags],
and an MLP head. The paper's fixed configuration (2 heads, hidden 32, dropout 0.2, lr 1e-3) is
kept; it was not re-tuned. AdamW (weight decay 1e-4), batch 32, class-weighted cross-entropy, at most
200 epochs, early stopping (patience 25) on macro-F1 of an inner patient-grouped validation split
(1/6 of training patients, stratified by class × modality). The test set is never used for
selection. 20 seeds (0–19).

**Conditions.**

| Condition | What it isolates |
|---|---|
| joint (main) | both modalities trained jointly, miRDB cross edges |
| joint_nocross | cross-modal edges removed (edge ablation) |
| joint_rewired | miRDB edges replaced by a degree-preserving bipartite rewiring (100 permutations, one run each, seeds 1000–1099) |
| single_mirna / single_mrna | a GAT trained on one modality only (no cross edges) |
| paired_random | sensitivity: both modalities observed, miRNA and mRNA samples re-paired at random within tissue (train and test), new pairing per seed |
| paired_order | sensitivity: the order-based pairing of the 01 build (patient-supported for metastases only) |

**Baselines.** XGBoost (grid of max_depth {3,5,7} × n_estimators {100,300,500} × learning_rate
{0.01,0.05,0.1}, 5-fold StratifiedGroupKFold on training patients, chosen once per condition with
seed 0; subsample and colsample 0.8 so that seeds vary) on identical inputs: single-modality
features, the joint input with the missing modality as NaN (handled natively by XGBoost), and the two
paired inputs. Naive baselines are in `results/tables/naive_baselines.csv` (majority macro-F1 0.243).

**Statistics.**
* Patient-cluster bootstrap (2,000 resamples of test patient groups) of the 20-seed mean macro-F1;
  paired examples are clustered by the miRNA sample's patient group.
* Paired patient-cluster bootstrap of differences between models on the same test samples.
* Exact McNemar test on seed-majority predictions.
* Cross-edge permutation test: p = (1 + #{rewired run ≥ real 20-seed mean}) / (1 + 100), the same
  convention as notebook 28. A Mann-Whitney test of the 20 real runs against the 100 null runs
  is also given; it treats the test set as fixed and captures training randomness only.

## 3. Why a TMLR reviewer should accept it

1. **No fabricated pairs.** GEO has no miRNA↔mRNA patient key. The main model never needs one:
   every example is a single GEO sample with its own label. This is unpaired ("diagonal")
   multi-omic integration, the setting of GLUE (Cao & Gao 2022, *Nat. Biotechnol.* 40:1458),
   which links unpaired omics through a prior-knowledge guidance graph. Our graph plays that role
   with miRDB target edges.
2. **The cross-modal claim is falsifiable and was tested.** Edge removal, degree-preserving
   rewiring, and single-modality training each isolate a different part of "multi-omic". The
   paper reports what survives (joint training helps the miRNA arm) and what doesn't (topology
   specificity).
3. **Calibrated comparisons.** 20 seeds, bootstrap CIs over patients, paired differences, and
   McNemar tests, with the same inputs for the GAT and XGBoost. No test-set tuning, and the GAT
   hyperparameters are the paper's frozen ones.
4. **Pairing-dependent numbers are fenced off.** Every analysis that combines modalities per
   specimen is labelled as a sensitivity analysis, with its optimism stated.

## 4. Number table (held-out test, 40 samples per modality, 20 seeds)

Macro-F1 mean ± SD over seeds [min, max]; CI = patient-cluster bootstrap 95% CI of the seed mean.

| Model | Condition | Test examples | Macro-F1 | 95% CI | Accuracy |
|---|---|---|---|---|---|
| **GAT** | **joint (main)** | miRNA | **0.854 ± 0.018** [0.836, 0.885] | [0.711, 0.950] | 0.867 |
| **GAT** | **joint (main)** | mRNA | **0.797 ± 0.036** [0.730, 0.872] | [0.658, 0.900] | 0.818 |
| GAT | joint, cross edges removed | miRNA | 0.845 ± 0.021 | [0.702, 0.944] | 0.860 |
| GAT | joint, cross edges removed | mRNA | 0.795 ± 0.034 | [0.647, 0.892] | 0.813 |
| GAT | miRNA only | miRNA | 0.823 ± 0.023 | [0.672, 0.930] | 0.841 |
| GAT | mRNA only | mRNA | 0.781 ± 0.049 | [0.645, 0.872] | 0.791 |
| XGBoost | joint (NaN for missing modality) | miRNA | 0.808 ± 0.005 | [0.653, 0.928] | 0.824 |
| XGBoost | joint (NaN for missing modality) | mRNA | 0.807 ± 0.023 | [0.660, 0.906] | 0.815 |
| XGBoost | miRNA only | miRNA | 0.809 ± 0.000 | [0.662, 0.927] | 0.825 |
| XGBoost | mRNA only | mRNA | 0.793 ± 0.022 | [0.641, 0.900] | 0.804 |
| GAT | sensitivity: order pairing, both observed | pairs | 0.852 ± 0.030 | [0.693, 0.950] | 0.876 |
| XGBoost | sensitivity: order pairing, both observed | pairs | 0.843 ± 0.006 | [0.711, 0.946] | 0.851 |
| GAT | sensitivity: random within-tissue pairing | pairs | 0.886 ± 0.037 | [0.796, 0.941] | 0.894 |
| XGBoost | sensitivity: random within-tissue pairing | pairs | 0.880 ± 0.034 | [0.780, 0.957] | 0.885 |

Paired differences (patient-cluster bootstrap; "frac ≤ 0" = share of resamples with no gain):

| Contrast | Test examples | Δ macro-F1 | 95% CI | frac ≤ 0 |
|---|---|---|---|---|
| GAT joint − XGBoost single-modality | miRNA | +0.045 | [−0.000, 0.124] | 0.032 |
| GAT joint − XGBoost single-modality | mRNA | +0.005 | [−0.135, 0.144] | 0.455 |
| GAT joint − XGBoost joint | miRNA | +0.046 | [0.001, 0.120] | 0.020 |
| GAT joint − XGBoost joint | mRNA | −0.010 | [−0.144, 0.120] | 0.581 |
| GAT joint − GAT single-modality | miRNA | +0.031 | [0.005, 0.068] | 0.005 |
| GAT joint − GAT single-modality | mRNA | +0.017 | [−0.023, 0.053] | 0.185 |
| GAT joint − GAT joint without cross edges | miRNA | +0.009 | [0.000, 0.024] | 0.025 |
| GAT joint − GAT joint without cross edges | mRNA | +0.002 | [−0.015, 0.017] | 0.380 |
| GAT joint without cross edges − GAT single-modality | miRNA | +0.022 | [0.003, 0.051] | 0.007 |
| GAT joint without cross edges − GAT single-modality | mRNA | +0.014 | [−0.014, 0.043] | 0.161 |
| GAT − XGBoost, order pairing | pairs | +0.008 | [−0.165, 0.148] | 0.456 |

McNemar (seed-majority predictions), GAT vs XGBoost: joint miRNA b/c = 2/0, p = 0.50; joint mRNA
5/5, p = 1.0; single miRNA 1/0, p = 1.0; single mRNA 5/5, p = 1.0; order pairing 3/2, p = 1.0.

Cross-edge permutation test (100 degree-preserving rewirings of the 574 miRDB edges):

| Test examples | Real (20-seed mean) | No cross edges | Rewired null mean ± SD | Empirical p | Mann-Whitney p (runs vs null) |
|---|---|---|---|---|---|
| miRNA | 0.854 | 0.845 | 0.841 ± 0.023 | 0.34 | 0.005 |
| mRNA | 0.797 | 0.795 | 0.798 ± 0.036 | 0.52 | 0.57 |
| all 80 | 0.825 | 0.819 | 0.818 ± 0.022 | 0.44 | 0.13 |

Read the permutation result as: real miRDB edges are not distinguishable from rewired edges at the
level the paper already uses (empirical p). On miRNA samples, the real edges beat rewired ones
consistently across training runs (Mann-Whitney p = 0.005), but the effect is about 0.013
macro-F1, well inside the test-set uncertainty. Report it as "a small, modality-specific
advantage that does not reach significance under the primary test".

Specimen-level late fusion (sum of log-probabilities of a test miRNA sample and a test mRNA sample of
the same tissue; 20 seeds × 50 random pairings = 1,000 draws, or 20 seeds for the order pairing):

| Model | Pairing | Fused macro-F1 mean ± SD | 2.5–97.5% range |
|---|---|---|---|
| GAT joint | order (01 build) | 0.869 ± 0.025 | 0.835–0.924 |
| GAT joint | random within tissue | 0.925 ± 0.035 | 0.851–0.978 |
| GAT single-modality pair | order | 0.861 ± 0.025 | 0.822–0.910 |
| GAT single-modality pair | random within tissue | 0.902 ± 0.036 | 0.829–0.977 |
| XGBoost joint | order | 0.813 ± 0.018 | 0.793–0.844 |
| XGBoost joint | random within tissue | 0.904 ± 0.034 | 0.842–0.955 |
| XGBoost single-modality pair | order | 0.800 ± 0.013 | 0.793–0.829 |
| XGBoost single-modality pair | random within tissue | 0.886 ± 0.034 | 0.823–0.953 |

## 5. New Methods text (drop-in)

> **Pairing-free multi-omic integration.** GSE45498 deposits miRNA and mRNA profiles as separate
> GEO samples with no key linking the two assays of one patient. We therefore treated the two
> omics as unpaired. Each GEO sample is one example carrying one observed modality and the tissue
> label of its own GEO record (230 miRNA and 230 mRNA samples; 190 + 190 training, 40 + 40 test,
> patient-disjoint as in Section X). All examples share one graph over the 93 differentially
> expressed miRNAs and 157 differentially expressed mRNAs. For each tissue we built a network
> from the top 150 miRNA–miRNA and top 150 mRNA–mRNA edges by absolute Pearson correlation among
> that tissue's training samples, computed within one modality so that no cross-assay pairing is
> required. Cross-modal edges were taken from prior knowledge: miRDB v6.0 predicted targets with
> score > 70 (574 edges between 82 miRNAs and 119 mRNAs, weighted by score/100); NanoString probe
> names were mapped to miRBase v22 mature names. Node inputs were the standardized expression
> value, an observed/unobserved indicator, a learned node embedding and a modality embedding;
> nodes of the unobserved modality received a value of zero, so their representations are formed
> only from attention-weighted messages, including messages along miRNA–target edges. A single
> multi-network GAT (two GATConv layers, 2 heads × 32 units, dropout 0.2, mean+max pooling per
> tissue network, an expression bypass, and an MLP head) was trained on miRNA and mRNA examples
> jointly with shared weights. This follows the guidance-graph approach to unpaired multi-omic
> integration (Cao & Gao, 2022). Hyperparameters were fixed to the values selected in Section Y;
> training used AdamW (lr 1e-3, weight decay 1e-4), class-weighted cross-entropy, and early
> stopping on an inner patient-grouped validation split. The test set was used only for the final
> evaluation.
>
> **Ablations and nulls.** We compared the joint model with (i) the same model without cross-modal
> edges, (ii) the same model with the miRDB edges replaced by 100 degree-preserving bipartite
> rewirings, and (iii) GATs trained on one modality only. XGBoost was trained on identical inputs
> (single-modality features, and the joint input with the missing modality encoded as missing).
> Each GAT configuration was trained with 20 seeds. We report macro-F1 with patient-cluster
> bootstrap 95% CIs (2,000 resamples), paired bootstrap CIs for model differences, exact McNemar
> tests on seed-majority predictions, and an empirical permutation p-value for the cross-modal
> edges.
>
> **Specimen-level sensitivity analyses.** To show how much a per-specimen combination of the two
> assays could add, we (a) fused the joint model's miRNA and mRNA test predictions (sum of
> log-probabilities) and (b) trained models with both modalities observed. Both require a
> miRNA–mRNA pairing; we used the series-order pairing (supported by order statistics for
> metastases only; Supplementary Section Z) and 1,000 random pairings within tissue. Random
> within-tissue pairing makes the two assays' errors conditionally independent given the tissue,
> which real same-patient pairs need not be, so it bounds the achievable fusion gain from above
> and is not used for model comparison.

## 6. Paper sections that change

| Section | Change |
|---|---|
| Abstract | Replace the "matched samples" sentence with: "Because GSE45498 provides no miRNA–mRNA patient key, we integrate the two omics without pairing: a single graph attention network is trained jointly on 230 miRNA and 230 mRNA profiles over a shared graph whose cross-omic edges are miRDB target predictions." Results sentence: joint GAT macro-F1 0.854 (miRNA) / 0.797 (mRNA) vs XGBoost 0.809 / 0.793, not significant; joint training improves the miRNA arm over a miRNA-only GAT (+0.031, 95% CI 0.005–0.068); the target topology is not distinguishable from rewired graphs (p = 0.34 / 0.52). Add the limitation sentence: "miRNA and mRNA profiles could not be linked by patient." |
| Introduction | One paragraph on unpaired multi-omic integration and guidance graphs (GLUE). The contribution becomes "pairing-free multi-omic GAT with falsification tests for the cross-omic graph", not "patient-matched miRNA–mRNA network". |
| Data | Replace "235 matched samples" with the alignment fix and the statement that miRNA and mRNA samples are treated as separate, correctly labelled samples (PAPER_REVISIONS.md 2, "Sample alignment"). Keep the 190/40 patient-disjoint split. |
| Methods: networks | Cross-modal edges now come from miRDB, not cross-sample correlation. Remove the miRNA–mRNA correlation layer, its FDR gate, and the "regulatory pairs (miRDB > 70, \|r\| > 0.1)" analysis, which all need paired samples. Intra-modal co-expression networks stay. |
| Methods: model | Section 5 text above (observed-flag input, joint training). Architecture and fixed hyperparameters otherwise unchanged. |
| Methods: evaluation | 20 seeds, patient-cluster bootstrap, paired-difference bootstrap, McNemar, rewiring permutation test (Section 5 text). |
| Results: classification | Table 4 (main rows + XGBoost). Headline: GAT matches XGBoost; nominal miRNA advantage, not significant. |
| Results: multi-omic contribution | New subsection: joint vs single-modality vs no-cross-edge vs rewired. Joint training helps the miRNA arm, topology specificity is not shown. This replaces the old "network ablation" result and reaches the same conclusion. |
| Results: specimen-level fusion | New short subsection, explicitly labelled sensitivity analysis (order pairing 0.869; random within tissue 0.925 as an upper bound). |
| Results: interpretation (attention/SHAP) | Recompute attention on the joint model before reporting anything; old Jaccard values do not apply to this model (not done in this branch). |
| TCGA | Kept as supporting external validation from the separate run. TCGA has patient-matched miRNA and mRNA, so it is also where a paired-vs-unpaired comparison of this same model can be made. |
| Discussion / Limitations | Section 7 table. |
| Figures | Replace the cross-modal correlation network panel with a miRDB guidance-graph schematic; add a forest plot of the paired differences in Section 4. |

## 7. Limitations reviewers will raise, and how the paper answers each

| Likely reviewer objection | Response in the paper |
|---|---|
| "Multi-omic without paired samples isn't multi-omic integration." | It is unpaired (diagonal) integration, an established setting (GLUE). Integration happens through shared weights and a prior-knowledge graph. The paper shows it does something: joint training improves the miRNA arm over a miRNA-only GAT (+0.031, CI 0.005–0.068). It states plainly that patient-level cross-omic covariance cannot be learned from this cohort. |
| "The GAT doesn't beat XGBoost." | Agreed, and stated: matched performance (miRNA +0.045, CI −0.000 to 0.124; mRNA +0.005; all McNemar p ≥ 0.5). The paper's claim is about honest integration and falsification of graph structure, not SOTA accuracy. |
| "Do the miRDB edges matter?" | Not detectably beyond chance under the primary permutation test (p = 0.34 / 0.52 / 0.44). Most of the joint-training gain remains without cross edges. The paper says so and does not interpret attention on target edges as regulation. |
| "miRDB is a prediction, not validated biology." | Stated. miRDB predictions (score > 70) serve as a prior for message passing, not as ground truth. A miRTarBase (experimentally supported) graph is a natural robustness check (not run here; the annotation is not in the repository). |
| "Probe-to-miRBase name mapping could be wrong." | The mapping rule is given in Methods and the edge list is released (`mirdb_cross_edges.csv`). Merged NanoString probes inherit targets from all members, which is the conservative choice. |
| "40 test samples (6 normals) are too few." | Stated. All results come with patient-cluster CIs; per-sample metric swings are noted (one normal ≈ 0.03–0.05 macro-F1); claims are restricted to what the CIs support. TCGA provides external support. |
| "Many comparisons, no multiplicity correction." | The paper names the primary comparison (joint GAT vs XGBoost per modality) and labels the ablation contrasts as secondary and exploratory. |
| "Fusion by tissue-matched pairing leaks the label." | Late fusion and paired inputs are sensitivity analyses only. Random within-tissue pairing is stated as an upper bound because it makes errors conditionally independent. The order pairing, the more realistic one, gives 0.869. |
| "miRNA patient identity is unknown, so the split may leak for miRNA samples." | Stated. The split is patient-disjoint by the mRNA-derived patient groups; miRNA samples inherit the group of their order-paired mRNA. For normal/primary miRNA samples from the same patient, crossing the split cannot be ruled out. Any leak would put the same patient in different classes, so it is unlikely to inflate accuracy, but it is listed. |
| "Hyperparameters were tuned on the old, misaligned build." | The paper's fixed GAT configuration was reused unchanged, with no re-tuning, and early stopping used an inner validation split. If anything, this disadvantages the GAT. |
| "Results depend on DE feature selection." | DE used training samples only (pydeseq2), as in the corrected build. Same features for all models. |
| "Why did mRNA-only GAT drop vs the earlier 0.851?" | 0.851 was a single seed of the old paired architecture. The 20-seed mean for this architecture is 0.781 ± 0.049 (mRNA-only) and 0.797 ± 0.036 (joint). Report the seed distributions. |
| "Is this reproducible?" | Deterministic preprocessing (md5-checked across machines), seeds listed, scripts 41/42, pinned environment (scikit-learn 1.7.2). |

## 8. Reproduce

```bash
git checkout fix-multiomic-A
pip install -r requirements.txt
bash run_pipeline.sh            # or at least: miRDB download, derive mapping, 01_preprocessing_DE
python notebooks/41_unpaired_multiomic_gat.py --seeds 20 --tag main          # 120 runs, about 26 min on 4 cores
python notebooks/41_unpaired_multiomic_gat.py --conditions '' --perms 100 --tag perm   # rewiring null
python notebooks/42_unpaired_baselines_stats.py                             # XGBoost + all statistics
```

The committed permutation runs were executed in two shards of 50 (perms 0–49 and 50–99, seeds
1000 + perm) on separate machines from the same commit, after checking that the 01 outputs were
byte-identical (md5). They are stored as `predictions_perm_0.csv` and `predictions_perm_50.csv`;
`--tag perm` regenerates the same runs in one file.

Outputs in `results/fix_multiomic_A/`: `predictions_main.csv`, `predictions_perm_*.csv`,
`xgb_predictions.csv`, `summary.csv`, `contrasts.csv`, `mcnemar.csv`,
`permutation_test.csv`, `late_fusion.csv`, `mirdb_cross_edges.csv`.

## 9. Not done in this branch

* Attention and SHAP on the joint model (the old interpretation section must be recomputed).
* A miRTarBase-only cross-edge graph (the annotation is not available in the repository).
* TCGA (separate run, per the framing above).
* Using all 282 miRNA and 249 mRNA GEO samples. Unpaired training makes this possible, since
  samples no longer need a partner. It would add about 50 miRNA and about 20 mRNA samples, but it
  needs a new patient-disjoint split, so the 230-sample build was kept here for comparability.
