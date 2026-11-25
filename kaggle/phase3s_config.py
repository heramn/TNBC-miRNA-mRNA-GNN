"""Phase 3s settings: post-hoc follow-up testing, added after the frozen study and after phase 3r.

Not a change to the paper or to its frozen analysis (study/, tag study-frozen-v1, is imported, never edited).
Phase 3s follows up on Prof. Ahmet Ay's two suggestions (Nov 2026) about the phase-3r planted control, where
joint_mirdb matched joint_rewired and both were slightly above joint_nocross:

 1. study/edges.py rewire_bipartite is a DEGREE-PRESERVING double-edge swap (20 swaps per edge) on the miRDB
    pairs inside the fold's node subset: every miRNA and every gene keeps its exact cross-edge degree, so the
    rewired graph touches exactly the same nodes as miRDB and only the pairings change. Phase 3s adds a
    uniform random-pair null ('random'): the same number of cross edges placed on miRNA-mRNA pairs drawn
    uniformly without replacement from all (miRNA, gene) pairs of the fold's node subset (not degree-preserving;
    a new draw per run, seeded like the rewired arm).
 2. The phase-3r effect was the size of seed noise and the edge-aware oracle reached only 0.66-0.77, so phase 3s
    uses a much stronger planted signal and more simulated examples (below), with more seeds per fold.

Everything here is fixed before any phase-3s GAT fit with cross edges. The only fits run while choosing the
strength / sample size were graph-free (calibrate: elastic net, XGBoost, XGBoost joint, the oracle; and the
frozen joint_nocross GAT for the gate), exactly as the phase-3r design gate was chosen.
"""

# ---------------------------------------------------------------- planted task (same latent design as phase 3r)
# Same latent design and random draws as phase 3r's plant() (kaggle/phase3.py): real labels permuted within
# modality; four miRDB hub sets per modality; per-example latent u_1..u_4; the class fixes only the signs of u1*u2
# and u3*u4. Changes (phase3s.plant_s / augment):
#   AMPLITUDE    z units added per unit latent value at strength 1, per example modality (phase 3r: 2.0 for both)
#   TEST_COPIES  simulated test examples: each outer-test sample is used TEST_COPIES times, each copy with its own
#                permuted label and latent draw (same background row, same patient group). Training data are
#                unchanged in size.
# Conditions (z units per unit latent value, per example modality; 'zero' = no signal, the null check):
CONDITIONS = {
    'zero':   {'miRNA': 0.0, 'mRNA': 0.0},
    'base':   {'miRNA': 2.0, 'mRNA': 2.0},   # phase 3r's strength 1, now with the random-pair null and more seeds
    'strong': {'miRNA': 2.0, 'mRNA': 8.0},   # Ay's strong-signal test (see the calibration notes below)
}
SIGNAL_CONDITIONS = ['strong', 'base']
TEST_COPIES = 4                 # simulated copies per outer-test sample (training data unchanged)
AUG_SEED = 5151                 # generator for the simulated test copies only
REPEAT = 0
FOLDS = [0, 1, 2, 3, 4]         # outer folds of repeat 0, scored on the (augmented) outer test fold, as phase-3r part A
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]   # GAT seeds per fold (phase 3r: 3 in part A, 2 in part B)
CONFIG = 0                      # paper configuration
CROSSES = ['mirdb', 'none', 'rewired', 'random']
# (variant, parameter): None = the frozen paper GAT (study/model.py, unchanged); selfdrop 0.25 = phase-3r's
# strongest variant (self-feature dropout, kaggle/phase3_model.py)
VARIANTS = [(None, None), ('selfdrop', 0.25)]

# Calibration (graph-free models and the oracle only, 6 label/latent draws, inner validation, repeat 0; see
# projects notes / tables/p3s_calibration*.csv):
#  * mRNA-example amplitude 8 gives oracle balanced accuracy 0.92 (0.90-0.94 over draws; 0.898 on the runs' draw)
#    while XGBoost per modality / joint stay at 0.38 / 0.38 (max 0.42) and elastic net at 0.30. Amplitude 12
#    adds little (oracle 0.94) and brings XGBoost to 0.43.
#  * miRNA-example sets are smaller (5-9 of 70 miRNAs): at amplitude 2 XGBoost already averages 0.45 over draws
#    (0.40 on the runs' draw), so the miRNA side cannot carry a strong signal without becoming readable
#    feature by feature. It stays at phase 3r's amplitude 2.
#  * More simulated TRAINING examples make the signal readable without edges (amplitude 8, 4x examples: XGBoost
#    0.80-0.89 vs oracle 0.85-0.93), leaving no headroom for an edge effect. So the extra simulated examples
#    are put in the outer test folds only (TEST_COPIES), which cuts evaluation noise without helping any model
#    learn the planted sets.
#  * The frozen joint_nocross GAT (no cross edges, but with the per-class co-expression networks) reads the
#    strong mRNA signal (inner-validation balanced accuracy 0.71 mRNA / 0.52 miRNA, seed 0): at amplitude 8 the
#    within-set co-expression is so strong that the top-150 co-expression edges recover the planted sets.
#    So at 'strong' the 0.43 gate is applied to the graph-free tabular models only and joint_nocross is
#    reported as the co-expression-only reference; at 'base' the full phase-3r gate (including joint_nocross)
#    applies, as before.

# ---------------------------------------------------------------- pass rule (unchanged structure, phase 3a / 3r)
# For each variant and each signal condition separately: on at least one test modality, the miRDB arm beats
# EVERY null (no cross edges, degree-preserving rewired, uniform random pairs) by mean fold-level dF1 >=
# PASS_MARGIN with one-sided paired t-test p < 0.05 over the 5 folds (macro-F1 per (fold, seed), averaged over
# the seeds within each fold); and at 'zero' every |dF1| < PASS_MARGIN. Same code as phase 3r
# (phase3.fold_rule / passes_rule). The decisive condition for Ay's question is 'strong'; the miRDB - random
# contrast is also reported on its own.
PASS_MARGIN = 0.03
DECISIVE = 'strong'

# ---------------------------------------------------------------- design gate (unchanged threshold)
# Run before any edge-using fit, inner-validation split of repeat 0, on the runs' exact data. Graph-free
# tabular models (elastic net and XGBoost per modality, XGBoost joint; config 0): mean balanced accuracy
# <= GATE_MAX_BALACC on both modalities at every signal condition. joint_nocross GAT (config 0, seed 0): gated at
# 'base' (as phase 3r), reported only at 'strong' (see the calibration notes).
GATE_MAX_BALACC = 0.43
GATE_NOCROSS_CONDITIONS = ['base']
ORACLE_TARGET = 0.90            # design target for the oracle at 'strong' (reported, not gated)
