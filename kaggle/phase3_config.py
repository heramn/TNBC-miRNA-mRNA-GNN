"""Phase 3 settings: post-hoc, added after the frozen study (study-final, tag study-frozen-v1).

Everything here was written and committed before any phase-3 study fit ran (only a one-fold smoke test on
the development machine ran earlier, to check the code; its fits are not used). Nothing in study/ is changed;
the frozen code is imported, never edited.

A. planted-signal positive control (does the architecture CAN use miRNA-mRNA edges?)
B. edge-forcing GAT variants: selection on (A-style planted task, inner validation) + real-data inner
   validation only, then a frozen choice (kaggle/phase3_frozen/selected_variant.json, committed by hand
   between notebook 1 and notebook 2), then one outer-fold evaluation
C. learning-curve repeats 1-4 for the 25% / 50% points
"""

# ---------------------------------------------------------------- A. planted-signal control
# Labels are the real labels permuted within modality (signal destroyed), then a class signal is planted
# that can only be read by grouping features along miRDB edges:
#   mRNA examples : two hub miRNAs (most miRDB targets in the fold's node subset, target sets made
#                   disjoint); every target gene of hub h gets +s_h * delta (z-score units)
#   miRNA examples: two hub genes (most miRDB regulators, regulator sets disjoint); every regulator of
#                   hub h gets +s_h * delta
#   class 0: s_1 = s_2 = +-1 (same sign, random), class 1: s_1 = -s_2 = +-1, class 2: s_1 = s_2 = 0
# Every single feature has zero mean shift in every class, so no feature is marginally informative;
# the class is the XOR of two set means (and their magnitude for class 2).
PC3_DELTAS = [0.0, 0.5, 1.0, 2.0]   # set from the oracle check (phase3.py check-plant, no model fit): balanced accuracy
                                     # of a logistic model on the two true hub set means, folds 0-4, miRNA / mRNA:
                                     # 0.27-0.34 / 0.26-0.43 at 0, 0.46-0.49 / 0.57-0.70 at 0.5,
                                     # 0.64-0.75 / 0.81-0.84 at 1, 0.82-0.90 / 0.93-0.97 at 2 (chance 0.33)
PC3_MIN_SET = 5                 # a hub needs >= 5 members after making the two sets disjoint
PC3_SEED = 4242
PC3_REPEAT = 0
PC3_SEEDS = [0, 1, 2]           # GAT seeds per fold
PC3_GAT_ARMS = ['joint_mirdb', 'joint_nocross', 'joint_rewired']   # frozen arms, paper configuration (config 0)
PC3_BASELINES = ['xgb_routed_mi', 'xgb_routed_mr', 'enet_routed_mi', 'enet_routed_mr']   # 16-config budget, as phase 1
# success (stated before any GAT run): at delta 1.0 and 2.0, joint_mirdb beats BOTH joint_nocross and
# joint_rewired on at least one test modality with the corrected 95% CI of dF1 above 0 and sign-flip p < 0.05,
# and at delta = 0 no mirdb contrast has a corrected 95% CI excluding 0
PC3_SUCCESS_DELTAS = [1.0, 2.0]

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
B_SEL_DELTAS = [0.0, 1.0]      # planted task used for selection, scored on the INNER-VALIDATION split
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
