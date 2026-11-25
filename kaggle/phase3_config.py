"""Phase 3 settings: post-hoc, added after the frozen study (study-final, tag study-frozen-v1).

Everything here was written and committed before any phase-3 study fit ran. Earlier, on the development
machine, only one-fold smoke tests (to check the code; their scores were not inspected or used) and, for the
kaggle-phase3r retry, the design gate (graph-free models only) ran. Nothing in study/ is changed;
the frozen code is imported, never edited.

A. planted-signal positive control (does the architecture CAN use miRNA-mRNA edges?)
B. edge-forcing GAT variants: selection on (A-style planted task, inner validation) + real-data inner
   validation only, then a frozen choice (kaggle/phase3_frozen/selected_variant.json, committed by hand
   between notebook 1 and notebook 2), then one outer-fold evaluation
C. learning-curve repeats 1-4 for the 25% / 50% points
"""

# ---------------------------------------------------------------- A. planted-signal control (redesign)
# Phase 3a's first design (an XOR of two hub set means, planted as mean shifts) failed: joint_nocross read it
# as well as joint_mirdb (0.976 vs 0.982 miRNA at delta 1), because a fixed set mean is a linear projection
# that any flexible joint model learns without the graph. This redesign (approved by Heram as one bounded
# retry; post-hoc) makes the label depend on WHICH features are grouped together by miRDB edges.
#
# Design (post-hoc). Real labels are permuted within modality (real signal destroyed); then, per example,
# four latent values u_1..u_4 ~ N(0, 1) are drawn, one per hub set, and delta * PC3_AMPLITUDE * u_i is added
# to every member of hub set i:
#   mRNA examples : four hub miRNAs (most miRDB targets in the fold's node subset, target sets made disjoint,
#                   >= PC3_MIN_SET genes each); members = their target genes
#   miRNA examples: four hub genes (most miRDB regulators, regulator sets disjoint); members = their regulators
#   the class fixes only the SIGNS of u1*u2 and u3*u4: class 0 (+, +), class 1 (-, -), class 2 (+, -);
#   each pair's common sign and all magnitudes are random
# So in every class each feature has the same mean and spread, each set mean has the same distribution, and
# the within-set co-expression is the same; only the sign of the association between the two sets of a pair
# differs, which the frozen co-expression networks (|Pearson r|) do not record. Reading the class requires
# knowing which features form each miRDB set (the edge identity); degree-preserving rewiring scrambles the sets.
# delta = 0 is pure noise. The labels and u do not depend on delta.
#
# Design history (all on planted data only, no edge-using GAT fit): (1) relabelling by the signs of products
# of the REAL hub set means failed its own check: real set means correlate 0.6-0.94, so classes came out about
# 70/8/22 and per-gene class variances differed up to 18x. (2) This latent design at amplitude 1, 2, 3 (z units):
# graph-free XGBoost mean inner-validation balanced accuracy 0.34-0.41 / 0.36-0.40 / 0.38-0.46, edge-aware
# oracle (logistic on the true set-mean products) 0.45-0.69 / 0.66-0.77 / 0.76-0.82 (miRNA-mRNA). Amplitude 2
# is the largest tried that keeps every graph-free model under the gate, so delta = 1 means amplitude 2.
PC3_DELTAS = [0.0, 0.5, 0.75, 1.0]
PC3_N_HUBS = 4
PC3_AMPLITUDE = 2.0             # z-score units added per unit latent value at delta = 1 (see the gate notes)
PC3_MIN_SET = 5
PC3_SEED = 4242
PC3_REPEAT = 0
PC3_SEEDS = [0, 1, 2]           # GAT seeds per fold
PC3_GAT_ARMS = ['joint_mirdb', 'joint_nocross', 'joint_rewired']   # frozen arms, paper configuration (config 0)
PC3_BASELINES = ['xgb_routed_mi', 'xgb_routed_mr', 'enet_routed_mi', 'enet_routed_mr']   # 16-config budget, as phase 1
# pass/fail (Heram's decision, consistent with phase 3a and projects/tnbc-paper.md): the frozen rule at
# strength RULE_DELTA = 1 ONLY. joint_mirdb must beat BOTH joint_nocross (no edges) and joint_rewired (shuffled
# edges) by mean fold-level dF1 >= B_PASS_MARGIN with one-sided paired t-test p < 0.05 over the 5 folds on at
# least one test modality, and at strength 0 every mean |dF1| < B_PASS_MARGIN. This is the same rule (same
# code, phase3.py fold_rule / passes_rule) as the variant selection in part B. Strengths 0.5 and 0.75 are
# diagnostics only: reported as a detection curve, never used for pass/fail or selected_variant.json.
RULE_DELTA = 1.0

# design gate (run BEFORE any edge-using GAT fit; inner-validation split of repeat 0, 5 folds, delta = 1):
# the graph-free models must sit near chance: mean inner-validation balanced accuracy <= GATE_MAX_BALACC
# on both test modalities for each of: logistic regression (elastic net, config 0) per modality, XGBoost per
# modality and XGBoost joint on the concatenated features (config 0), and joint_nocross (frozen GAT without
# cross edges, config 0, seed 0). Edge-aware references (reported, not gated): XGBoost with the miRDB
# target-set means added, and the oracle (logistic on the two true set-mean products).
GATE_DELTA = 1.0
GATE_MAX_BALACC = 0.43          # chance 0.33 + 0.10
GATE_SCHEME = 'p3rgate_d1.0'

# ---------------------------------------------------------------- B. edge-forcing variants
# candidates (variant, parameter); see kaggle/phase3_model.py
B_CANDIDATES = [
    ('xfirst', None),          # first GAT layer passes messages along cross (miRNA-mRNA) edges only, then the
                               # full tissue graph; readout pools miRNA and mRNA nodes separately
    ('gated', None),           # frozen 2-layer GAT plus a gated residual cross-edge-only GAT path; split readout
    ('selfdrop', 0.25),        # self-feature dropout: each observed node is hidden with prob q during training
    ('selfdrop', 0.5),
]
B_SEL_REPEAT = 0
B_SEL_SEEDS = [0, 1]
B_SEL_CONFIG = 0               # paper configuration for every selection fit
B_SEL_DELTAS = [0.0, 1.0]      # planted task for selection: strength 0 and RULE_DELTA only; INNER-VALIDATION split
# step 1 (planted, inner validation, fold-level paired differences over 5 folds x 2 seeds):
#   pass if at delta = 1.0 the variant with miRDB edges beats the same variant with no cross edges AND with
#   rewired edges by mean dF1 >= B_PASS_MARGIN with one-sided paired t-test p < 0.05 over the 5 folds, on at
#   least one modality, and at delta = 0 both mean |dF1| < B_PASS_MARGIN
B_PASS_MARGIN = 0.03
# step 2 (real data, inner validation): among passing candidates, the lowest mean class-weighted
#   inner-validation cross-entropy with miRDB edges (paper config, 5 folds x 2 seeds); ties -> list order.
#   If no candidate passes, nothing is evaluated on the outer folds.
# outer evaluation (notebook 2), all 5 repeats x 5 folds, same splits and the same 16-configuration budget:
B_EVAL_SEEDS = 10              # selected variant with miRDB edges: 16 configs (seed 0) + seeds 1..9 of the best
B_EVAL_SECONDARY_SEEDS = 5     # same variant with rewired edges / no cross edges, configuration of the miRDB arm
# primary contrast: variant_mirdb - joint_nocross (frozen phase-1 runs, 20 seeds); also variant_mirdb -
# variant_rewired and variant_mirdb - variant_nocross. Holm over these 3 x 2 modalities (sign-flip p).

# ---------------------------------------------------------------- C. learning-curve repeats
C_REPEATS = [1, 2, 3, 4]       # repeat 0 comes from the phase-2 output (attach it) or is recomputed
C_FRACTIONS = [0.25, 0.5]
C_ARMS = ['joint_mirdb', 'joint_nocross']
C_BASELINES = ['xgb_joint', 'xgb_routed_mi', 'xgb_routed_mr', 'enet_routed_mi', 'enet_routed_mr']
