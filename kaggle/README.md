# Phase 2 on Kaggle

`phase2.ipynb` runs the phase-2 work that `study-final` (commit `59ca5e0`) skipped for compute budget,
using the frozen code, settings, seeds and phase-1 configuration selections. It changes nothing in
`study/` (`pytest study/tests` runs inside the notebook and checks the frozen-file hashes).

| Item | Work |
|---|---|
| Learning curve 25% / 50% (joint_mirdb, joint_nocross; repeat 0, 5 folds × 5 seeds) | 100 GAT fits |
| `sens_tcga_normal`, `sens_tcga_estimate`, `sens_aug_pairs` (repeat 0, 5 folds × 5 seeds) | 75 GAT fits |
| Learning-curve baselines (`study/baselines.py lc`) | 50 fold files, minutes |

`phase2.py` drives it: it enumerates exactly `run_gat.tasks(2)` and calls the frozen `run_gat.run_one` per fit
with the arguments `run_task` would pass, then reuses `analysis.collect` / `analysis.analyze` / `figures.py`.

## Run it

1. Kaggle → *New Notebook* → *File → Import Notebook* → paste
   `https://github.com/heramn/TNBC-miRNA-mRNA-GNN/blob/kaggle-phase2/kaggle/phase2.ipynb`.
2. Session options: **Internet on**, accelerator **None (CPU)** (GPU T4 x2 also works but is not used:
   the frozen model is CPU-only and single-threaded per fit). Run all with `SMOKE_TEST = True`: 3 real
   fits, and it prints a runtime estimate for the rest.
3. Set `SMOKE_TEST = False`, then *Save Version → Save & Run All (Commit)*. It runs all fits, the
   baselines and the analysis, and finishes inside the 12 h limit (no new fit starts after 10 h).
4. Open the version's *Output*: `tnbc_phase2.zip`, `phase2_runs/` (every per-run CSV) and
   `phase2_output/RESULTS_phase2_rows.md` (rows for RESULTS.md sections 9, 10 and 13), also printed at the
   end of the log.

**Resume.** Each fit writes one CSV atomically and finished fits are never rerun. Re-running cells in the
same session continues where it stopped. For a new session, attach the previous version's output
(*Add Input → Your Work → this notebook*) and run again; `phase2.py restore` copies the finished fits
from `phase2_runs/` or `tnbc_phase2.zip` under `/kaggle/input`.

## Data

* **GEO GSE45498**: the four files phase 1 used are committed in `data/raw/` and verified by sha256.
  Nothing is fetched from GEO, because GEO's current series-matrix gzips differ byte-wise from them.
* **miRDB v6.0**: downloaded from `https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz` and
  checked against sha256 `446636d2182212f44ff91623f1429f7922f9575b82226962a6213d592a7e109d`. If mirdb.org is
  unreachable from Kaggle, create a Kaggle Dataset containing that one file (59.5 MB, same name) and attach
  it; the notebook looks for it anywhere under `/kaggle/input`.
* **TCGA**: not needed. The sensitivity arms use the frozen edge lists in `study/frozen/`
  (`tcga_normal_edges_ranked_sensitivity.csv`, `tcga_tumor_edges_ranked_estimate_sensitivity.csv`).

## Environment

An isolated Python 3.12 venv (uv) with phase 1's pins: torch 2.15.0+cpu, torch_geometric 2.8.1,
scikit-learn 1.7.2 (1.8 changes the grouped splits), numpy 2.5.5, pandas 3.1.1, scipy 1.18.1, xgboost 3.4.1,
pydeseq2 0.6.0. Kaggle's preinstalled CUDA torch is left alone; PyG 2.8.1 is pure Python and needs no
CUDA-specific extension wheels. Threads are pinned to 1 per process, as in phase 1.

# Phase 3 on Kaggle (post-hoc, added after the frozen study)

Three add-ons that were **not** in the plan frozen after exploration (tag `study-frozen-v1`). Settings and decision
rules are in `phase3_config.py`, committed before any phase-3 study fit ran. `study/` is imported, never edited.
`phase3.py` drives everything; `phase3_model.py` holds the edge-forcing variants (subclasses of the frozen `Net`).

| Notebook | Part | Work | GAT fits | Wall clock, 4 CPUs (from the smoke test) |
|---|---|---|---|---|
| `phase3a.ipynb` | A. Planted-signal positive control | frozen joint_mirdb / joint_nocross / joint_rewired (paper config) + routed XGBoost / elastic net, δ ∈ {0, 0.5, 1, 2}, repeat 0, 5 folds × 3 seeds | 180 | ≈ 1.3 h |
| | B. Edge-forcing variant selection | xfirst, gated, selfdrop q = 0.25 / 0.5; planted task on the inner-validation split + real-data inner-validation loss; writes `selected_variant.json`; no outer test fold | 280 | ≈ 2.3 h |
| | C. Learning-curve repeats | 25% / 50%, joint_mirdb + joint_nocross (+ baselines), repeats 0–4 × 5 folds × 5 seeds (repeat 0 is reused from the phase-2 output if attached) | 500 (400 with phase 2 attached) | ≈ 1.4 h |
| `phase3b.ipynb` | Outer evaluation of the selected variant | 16 configs at seed 0, then 10 seeds (miRDB) and 5 seeds (rewired, no-cross) of the chosen config, 5 repeats × 5 folds | 875 | ≈ 4–10 h depending on the variant (35–160 s per fit) |

Notebook 1 totals about 5 h of GAT fits plus installs and baselines, inside one 12 h session. Notebook 2 may need a
second session for the slower variants; no new fit starts after 10.5 h, and `restore` resumes it.

## Run it

1. Import `kaggle/phase3a.ipynb` from branch `kaggle-phase3` (as for phase 2). Internet **on**, accelerator **None**.
   Optional: attach the phase-2 notebook's output so the 100 repeat-0 learning-curve fits are reused.
2. Run all with `SMOKE_TEST = True` (one fold, one seed per arm type, then a runtime estimate). Then set
   `SMOKE_TEST = False` and *Save Version → Save & Run All*.
3. Download `phase3_output/selected_variant.json` and commit it as `kaggle/phase3_frozen/selected_variant.json` on
   `kaggle-phase3`. If it says `"selected": false`, no variant passed the planted control and notebook 2 is not run.
4. Import `kaggle/phase3b.ipynb` and run it the same way. `check-frozen` stops it unless the selection is committed
   and unmodified, and the clone is taken after that commit.
5. Each notebook writes `phase3_output/RESULTS_phase3_rows.md` (rows and captions for RESULTS.md) and zips
   `phase3_runs/` and `phase3_output/`.

The resume rules are the same as phase 2: attach the previous version's output and run again.


# Phase 3 retry on Kaggle: kaggle-phase3r (post-hoc, added after the frozen study)

Phase 3a (branch `kaggle-phase3`, commit `5a5bec9`) failed its planted control: joint_nocross read the planted XOR of
two set means as well as joint_mirdb (0.976 vs 0.982 miRNA at δ = 1), so the control did not require edges, and no
variant passed. Heram approved one bounded retry that changes only the control. The variants (xfirst, gated,
selfdrop 0.25 / 0.5) and the pass rule are unchanged. Part C is finished and is not rerun.

`phase3r.ipynb` replaces `phase3a.ipynb` on this branch:

| Step | GAT fits | Wall clock, 4 CPUs |
|---|---|---|
| Design gate: graph-free models must be near chance at δ = 1 (inner validation); the notebook stops on FAIL | 5 | minutes |
| A. Redesigned planted control, δ ∈ {0, 0.5, 0.75, 1}, 5 folds × 3 seeds | 180 | see the smoke-test estimate |
| B. Variant selection on the redesigned task (inner validation only) | 280 (240 with phase 3a's output attached) | see the smoke-test estimate |

The planted signal and the gate are described in `phase3_config.py`. `phase3b.ipynb` (the outer evaluation) is
unchanged apart from cloning `kaggle-phase3r`; it runs only if a variant passes and its `selected_variant.json` is
committed as `kaggle/phase3_frozen/selected_variant.json` on this branch.
