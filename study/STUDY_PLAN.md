# Final study plan (study-final): frozen after exploration

Frozen at the git tag \`study-frozen-v1\`, before any final GEO run. It was frozen **after exploration**
(see the timeline in \`FREEZE_EDGES.md\` and below). It is **not pre-registered**. \`FREEZE_STUDY.json\` lists the
sha256 of every study script, and \`tests/test_study.py\` fails if any of them changes.

## Task and data
GEO GSE45498 is the only training and evaluation cohort. Labels are normal / primary TNBC /
**lymph-node metastasis**, taken from each GSM's own GEO record. All 54 metastases are lymph-node
tissue on a separate mRNA project batch (raw IDs TAS10 vs TAS09), so the task is **tissue state**,
not progression.

Every confident profile is one example with **one** observed modality:
- 282 miRNA profiles (59 / 169 / 54)
- 249 mRNA profiles (39 / 157 / 53); 5 mRNA profiles have no confident raw column and are excluded

No sample is ever paired across modalities, at training or at test time. The only exception is
\`sens_aug_pairs\`, which pairs within training data only.

## Evaluation
**Outer CV.** Repeated grouped cross-validation, 5 repeats × 5 folds. Each modality is split by
\`StratifiedGroupKFold(5, shuffle, random_state = 1000 + repeat)\`, and fold k is the union of the two
modalities' fold k.

**Groups.** The grouping guarantee differs by modality.
- *mRNA:* groups are patient IDs (union of the GEO-title ID and the raw-header ID). The split is patient-disjoint.
- *miRNA:* GEO gives no patient ID, and there is no near-duplicate structure (max nearest-neighbour
  Spearman 0.946 with no gap), so miRNA profiles are singletons.
- *Cross-modal:* same-patient leakage across modalities cannot be excluded. The \`conservative\` scheme
  links each miRNA profile to the patient of its order-alignment partner.

**Inner split.** This is the first split of \`StratifiedGroupKFold(5)\` on class × modality. All models,
baselines included, are fit on the inner-fit portion and early-stopped or selected on class-weighted
inner-validation cross-entropy. No model is refit on the whole outer-training fold.

**Fold-local preprocessing.** Each outer fold refits, from its outer-training samples only:
- pydeseq2 DE (3 pairwise contrasts; padj < 0.05, |log2FC| > 1 for miRNA and > 0.5 for mRNA)
- log2(x+1) z-scoring
- the top-150 |r| co-expression edges per modality and tissue
- the miRDB and TCGA edge subsets

## Model
Architecture is A's multi-network GAT, with two changes:
- edge features [weight, is_cross]
- cross edges are binary

Unobserved-modality nodes are **latent target nodes** (value 0, flag 0) whose representations come
only from messages. They are not imputations, and the method is not called "GLUE-style".

**Tuning budget.** Every model gets the same budget: 16 configurations. The first is the paper's or a
conventional default; the other 15 come from a fixed-seed random draw.
- GAT space: heads {1,2,4} × hidden {16,32,64} with heads × hidden ≤ 128 × dropout {.1,.2,.3,.5}
  × lr {3e-4,1e-3,3e-3} × wd {1e-5,1e-4,1e-3}
- Training: max 100 epochs, patience 15, batch 32, AdamW, class-weighted cross-entropy

## Arms (all on the fold's TCGA-mappable DE subset unless stated)
| Arm | Role | Seeds per fold |
|---|---|---|
| joint_mirdb | miRDB v6.0 score > 70 cross edges, **primary** | 20 |
| joint_tcga | frozen TCGA edges, K-matched to miRDB, **pre-specified secondary** | 20 |
| joint_nocross | no cross edges | 20 |
| joint_rewired | degree-preserving rewiring of the miRDB edges, a new rewiring per run; config of joint_mirdb (disclosed) | 20 |
| joint_mirdb_full | miRDB on all fold DE nodes (separate row) | 5 |
| gcn_mirdb | GCN swap | 5 |
| mlp_only | expression bypass alone (no graph) | 5 |
| graphonly_mirdb | graph branch alone (mechanism check) | 5 |
| single_{mi,mr}_{nocross,mirdb,rewired} | factorial single-modality cells; their other-modality test predictions are the **zero-shot transfer** | 5 |
| XGBoost joint (NaN for missing) and routed; elastic net routed and joint; TabPFN v2 routed; target-set elastic net; naive | baselines | 5 (EN: 1) |

**Repeat-0 sensitivities (5 folds × 5 seeds):**
- TCGA normal-derived edges
- ESTIMATE-purity edges
- train-side random within-tissue pairing augmentation
- conservative grouping scheme (paper config)
- 230-sample subset (paper config)

**190/40 continuity.** On A's split: joint_mirdb, nocross, tcga and rewired, paper config, 20 seeds.

## Hypotheses (3 × 2 modalities, Holm-corrected)
- **H1:** joint_mirdb vs joint_nocross
- **H2:** joint_nocross vs routed single-modality GAT (single_mi_nocross on miRNA test, single_mr_nocross on mRNA test)
- **H3:** joint_mirdb vs joint_rewired

**Secondary (B), not in the Holm family:** joint_tcga vs joint_mirdb, vs joint_rewired, and vs joint_nocross.
The pre-stated expected result for H1, H3 and B is **null**.

## Statistics
**Primary statistic.** Patient-level sign-flip test, two-sided, 20,000 flips. It runs on paired
per-sample log-loss differences, where each sample's log-loss is averaged over repeats and seeds,
and is then aggregated to the group mean within each modality.

**Headline metric.** Macro-F1 of the pooled out-of-fold predictions per repeat and seed, averaged.

**Difference CIs.** Nadeau–Bengio corrected repeated-CV t intervals, with variance × (1/25 + n_test/n_train),
over fold-level seed-mean macro-F1 differences.
- **TOST:** fixed margin ±0.03. Equivalence is declared when the 90% corrected CI lies inside (−0.03, 0.03).
- **Achieved power:** power to detect Δ = 0.03 at α = 0.05, given the corrected SE.

**Also reported:**
- patient-cluster bootstrap CIs (2,000 resamples) for each arm's macro-F1
- fold and seed variance, reported separately
- log-loss, Brier score, ECE and a pooled reliability plot
- per-class (normal / primary / LN-met) breakdown of every hypothesis

## Other pre-specified analyses
- **Shortcut ladder (met vs primary, per modality, same CV and budget).** Feature sets: NanoString
  control probes only, immune miRNAs (miR-142-3p/5p, 146a, 150, 155, 223), lymphoid/myeloid genes
  and their mean score, and all subset features. Report AUROC and the joint GAT's margin over the
  best shortcut.
- **Positive control (repeat 0).** Labels are permuted within modality, then a class signal is planted
  along miRDB edges: +1.0 SD on one miRNA per class in miRNA examples, and −δ SD on all of its miRDB
  targets in mRNA examples, with δ ∈ {0.25, 0.5}. Arms are mirdb, nocross and rewired, with paper
  config and 5 seeds. **Success** means mirdb beats rewired on mRNA examples at δ = 0.5 (sign-flip p < 0.05).
- **Learning curve (repeat 0).** 25%, 50% and 100% of inner-fit groups, sampled per class × modality.
  The DE and graph come from the full outer-training fold.
- **Zero-shot cross-modality transfer.** Single-modality-trained arms are scored on the other
  modality's test examples, comparing miRDB vs nocross vs rewired edges.
- **TCGA edge validation (TCGA only).**
  - 100 stratified 47/47 split-half replications vs random pairs
  - sign agreement in non-TNBC BRCA tumours (purity-adjusted)
  - miRTarBase strong-evidence enrichment, if the database can be obtained
  - the purity-adjusted check of the miRDB pairs vs random pairs
- **Batch audit from the GEO files.** Series-matrix fields, raw headers, and control-probe and
  spike-in profiles by class. NanoString RCC files (cartridge, lane, date) would allow a real batch
  analysis; they are not deposited.

## Deviations from the requested design (compute or data driven, disclosed)
1. **Rewired null.** The 200 rewirings × 5 seeds at full CV scale would need about 25,000 GAT fits.
   Instead, the rewired arm uses a fresh rewiring for each of its 500 runs, paired with the real-graph
   run at the same fold and seed. H3 is tested on that paired arm.
2. **Seeds.** Secondary rows get 5 seeds per fold, and sensitivity analyses run on repeat 0 only.
3. **scGLUE and secondary-prior GAT arms not run.** scGLUE embeddings and miRTarBase / TargetScan
   secondary-prior GAT arms were not run (compute). Prior-without-GNN is covered by the target-set
   elastic net.
4. **TabPFN.** TabPFN uses the open v2 classifier weights (tabpfn 2.0.9). TabPFN ≥ 2.5 needs a
   license token.
5. **Shared tuned configuration.** joint_rewired and single_*_rewired reuse the configuration tuned
   for the matching miRDB arm.

## Timeline (UTC)
| When | Event |
|---|---|
| 2026-11-22 12:40–18:42 | branches repro-verify, fix-multiomic-A and fix-multiomic-B; their results were seen (see FREEZE_EDGES.md) |
| 2026-11-22 23:09 | TCGA edge list frozen (tag tcga-edges-frozen-v1) |
| 2026-11-22 23:10–23:50 | study code written. Smoke tests ran on fold r0/f0 with 2 epochs and 2 configs, plus timing runs; their predictions were not scored |
| see tag time | study frozen (tag study-frozen-v1) |
