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
