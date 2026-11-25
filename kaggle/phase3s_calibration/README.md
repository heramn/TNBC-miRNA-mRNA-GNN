# Phase 3s calibration (post-hoc follow-up testing; design only)

Run on the development box before the design was frozen. Only graph-free models, the edge-aware oracle and the
frozen joint_nocross GAT (no cross edges) were fitted. No model with cross edges was fitted on phase-3s data
before the freeze commit.

* `p3s_calibration.csv` / `_folds.csv`: `phase3s.py calibrate --amps 2:2,2:4,2:8,2:12,1:8 --copies 4 --draws 0,1,2,3,4,5`.
  Inner-validation balanced accuracy (repeat 0, 5 folds) for 6 label / latent draws (draw 0 = the runs' draw,
  identical to phase 3r's for the real samples), per (miRNA amplitude, mRNA amplitude).
* `p3s_gate_local_precheck.csv`: the full gate (tabular models plus joint_nocross, seed 0) on the runs' exact
  data at `strong` and `base`. `base` reproduces phase 3r's gate exactly (joint_nocross 0.378 mRNA / 0.428 miRNA).

Earlier exploratory calibration (an earlier version of `augment` that copied **all** splits, i.e. more
simulated training examples; draw 0 only, mean inner-validation balanced accuracy, mRNA / miRNA):

| amplitude (both) | copies | oracle | XGBoost per modality | XGBoost joint |
|---|---|---|---|---|
| 2 | 1 | 0.77 / 0.66 | 0.36 / 0.40 | 0.40 / 0.40 |
| 2 | 4 | 0.74 / 0.71 | 0.64 / 0.69 | 0.57 / 0.64 |
| 6 | 4 | 0.91 / 0.90 | 0.85 / 0.87 | 0.74 / 0.80 |
| 8 | 4 | 0.93 / 0.91 | 0.89 / 0.90 | 0.79 / 0.83 |
| 8 | 8 | 0.92 / 0.91 | 0.91 / 0.90 | 0.87 / 0.89 |

With more simulated training examples, graph-free XGBoost learns the planted interaction and approaches the
oracle, so no headroom is left for an edge effect. Hence phase 3s adds simulated examples to the outer test
folds only (lower evaluation noise) and gets its strength from the amplitude.
