"""Frozen configuration of the final study (study-final). Changing anything here after the
study-frozen-v1 tag invalidates the final runs; tests/test_frozen.py checks the hash."""
import itertools

import numpy as np

CLASSES = ['normal', 'primary', 'ln_metastasis']   # "lymph-node metastasis" = tissue state, not progression
MODS = ['miRNA', 'mRNA']

# cross-validation
N_REPEATS, N_FOLDS = 5, 5
FOLD_SEED_BASE = 1000          # StratifiedGroupKFold(random_state=FOLD_SEED_BASE + repeat)
INNER_SEED_BASE = 2000         # inner split: first split of StratifiedGroupKFold(5, random_state=INNER_SEED_BASE + 10*repeat + fold)

# features
MIRNA_FC, MRNA_FC, DE_PADJ = 1.0, 0.5, 0.05   # as notebooks/01 (pydeseq2, 3 pairwise contrasts, union)
TOP_N_COEXPR = 150             # per modality per tissue, |Pearson r| on that tissue's outer-training samples
MIRDB_MIN_SCORE = 70

# training (all models are fit on the inner-fit split and selected / early-stopped on the inner-validation split)
MAX_EPOCHS, PATIENCE, BATCH = 100, 15, 32
N_CONFIGS = 16
SEEDS_PRIMARY, SEEDS_SECONDARY = 20, 5
PAPER_CONFIG = dict(heads=2, hidden=32, dropout=0.2, lr=1e-3, wd=1e-4)


def gat_configs():
    """16 configs: the paper's configuration first, then 15 drawn without replacement (fixed RNG)."""
    grid = [dict(heads=h, hidden=d, dropout=p, lr=lr, wd=wd)
            for h, d, p, lr, wd in itertools.product([1, 2, 4], [16, 32, 64], [0.1, 0.2, 0.3, 0.5],
                                                     [3e-4, 1e-3, 3e-3], [1e-5, 1e-4, 1e-3])
            if h * d <= 128]
    grid = [g for g in grid if g != PAPER_CONFIG]
    pick = np.random.default_rng(12345).choice(len(grid), N_CONFIGS - 1, replace=False)
    return [PAPER_CONFIG] + [grid[i] for i in pick]


# GAT-family arms. nodes: 'subset' = fold DE nodes that are TCGA-mappable, 'full' = all fold DE nodes.
# train: modalities whose examples are used for training (test always scores both modalities).
ARMS = {
    # primary edge arms (H1, H3, B)
    'joint_mirdb':      dict(model='gat', nodes='subset', cross='mirdb', train=(0, 1), seeds=SEEDS_PRIMARY, tune=True),
    'joint_tcga':       dict(model='gat', nodes='subset', cross='tcga', train=(0, 1), seeds=SEEDS_PRIMARY, tune=True),
    'joint_nocross':    dict(model='gat', nodes='subset', cross='none', train=(0, 1), seeds=SEEDS_PRIMARY, tune=True),
    'joint_rewired':    dict(model='gat', nodes='subset', cross='rewired', train=(0, 1), seeds=SEEDS_PRIMARY, config_from='joint_mirdb'),
    # secondary rows
    'joint_mirdb_full': dict(model='gat', nodes='full', cross='mirdb', train=(0, 1), seeds=SEEDS_SECONDARY, tune=True),
    'gcn_mirdb':        dict(model='gcn', nodes='subset', cross='mirdb', train=(0, 1), seeds=SEEDS_SECONDARY, tune=True),
    'mlp_only':         dict(model='mlp', nodes='subset', cross='none', train=(0, 1), seeds=SEEDS_SECONDARY, tune=True),
    'graphonly_mirdb':  dict(model='graphonly', nodes='subset', cross='mirdb', train=(0, 1), seeds=SEEDS_SECONDARY, tune=True),
    # factorial single-modality cells (H2); their other-modality test predictions are the zero-shot transfer
    'single_mi_nocross': dict(model='gat', nodes='subset', cross='none', train=(0,), seeds=SEEDS_SECONDARY, tune=True),
    'single_mi_mirdb':   dict(model='gat', nodes='subset', cross='mirdb', train=(0,), seeds=SEEDS_SECONDARY, tune=True),
    'single_mi_rewired': dict(model='gat', nodes='subset', cross='rewired', train=(0,), seeds=SEEDS_SECONDARY, config_from='single_mi_mirdb'),
    'single_mr_nocross': dict(model='gat', nodes='subset', cross='none', train=(1,), seeds=SEEDS_SECONDARY, tune=True),
    'single_mr_mirdb':   dict(model='gat', nodes='subset', cross='mirdb', train=(1,), seeds=SEEDS_SECONDARY, tune=True),
    'single_mr_rewired': dict(model='gat', nodes='subset', cross='rewired', train=(1,), seeds=SEEDS_SECONDARY, config_from='single_mr_mirdb'),
}
# repeat-0 sensitivity arms (5 folds, 5 seeds, configuration taken from the named main arm of the same fold)
SENS_ARMS = {
    'sens_tcga_normal':   dict(model='gat', nodes='subset', cross='tcga_normal', train=(0, 1), seeds=SEEDS_SECONDARY, config_from='joint_tcga'),
    'sens_tcga_estimate': dict(model='gat', nodes='subset', cross='tcga_estimate', train=(0, 1), seeds=SEEDS_SECONDARY, config_from='joint_tcga'),
    'sens_aug_pairs':     dict(model='gat', nodes='subset', cross='mirdb', train=(0, 1), seeds=SEEDS_SECONDARY, config_from='joint_mirdb', augment_pairs=True),
}
# alternative fold schemes / sample sets (repeat 0 only, paper configuration, 5 seeds)
SCHEME_ARMS = ['joint_mirdb', 'joint_nocross']
SCHEMES = ['conservative', 'subset230']

# hypotheses (Holm over 3 x 2 modalities); primary statistic: patient-level sign-flip test on paired
# per-sample log-loss differences; headline: macro-F1 difference with Nadeau-Bengio corrected CI + TOST
HYPOTHESES = {
    'H1_mirdb_vs_nocross':        ('joint_mirdb', 'joint_nocross'),
    'H2_joint_vs_single':         ('joint_nocross', {'miRNA': 'single_mi_nocross', 'mRNA': 'single_mr_nocross'}),
    'H3_mirdb_vs_rewired':        ('joint_mirdb', 'joint_rewired'),
}
SECONDARY = {
    'B1_tcga_vs_mirdb':   ('joint_tcga', 'joint_mirdb'),
    'B2_tcga_vs_rewired': ('joint_tcga', 'joint_rewired'),
    'B3_tcga_vs_nocross': ('joint_tcga', 'joint_nocross'),
}
TOST_MARGIN = 0.03
ALPHA = 0.05
N_SIGNFLIP = 20000
N_BOOT = 2000

# shortcut ladder (met vs primary)
IMMUNE_MIRNAS = ['hsa-miR-142-3p', 'hsa-miR-142-5p', 'hsa-miR-146a', 'hsa-miR-150', 'hsa-miR-155', 'hsa-miR-223']
LYMPHOID_GENES = ['LCK', 'SYK', 'LYN', 'BCL6', 'LMO2', 'BCL2A1', 'CXCL9', 'IL4', 'HCK', 'FGR', 'SPI1', 'CSF1R']

# positive control (repeat 0): labels permuted within modality, signal planted along miRDB edges
PC_DELTA_MIRNA = 1.0
PC_DELTA_MRNA = [0.25, 0.5]
PC_ARMS = ['joint_mirdb', 'joint_nocross', 'joint_rewired']
# learning curve (repeat 0)
LC_FRACTIONS = [0.25, 0.5]
LC_ARMS = ['joint_mirdb', 'joint_nocross']
