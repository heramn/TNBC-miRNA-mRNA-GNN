#!/usr/bin/env python3
"""Phase-2 driver for the Kaggle notebook (kaggle/phase2.ipynb). Nothing in study/ is changed.

Runs exactly the phase-2 work that study-final skipped, through the frozen study code:

* 175 GAT fits = every fit of study/run_gat.py tasks(phase=2):
  - sens_tcga_normal, sens_tcga_estimate (configuration of joint_tcga) and sens_aug_pairs
    (configuration of joint_mirdb): repeat 0, 5 folds, 5 seeds -> 75 fits
  - learning curve, 25% and 50% of the inner-fit groups, joint_mirdb and joint_nocross with their
    phase-1 selected configuration: repeat 0, 5 folds, 5 seeds -> 100 fits
  Each fit calls run_gat.run_one with the arguments run_gat.run_task passes, so file names, seeds,
  configurations and outputs are the ones the frozen runner would produce. A fit's CSV is written
  atomically and an existing CSV is never refit, which is the per-fit checkpoint.
* learning-curve baselines: study/baselines.py lc (XGBoost joint and routed, elastic net routed).
* analysis: analysis.collect on the phase-2 run files, appended to the committed phase-1
  predictions_final.csv.gz, then analysis.analyze and figures.py (frozen-plan analysis, no retuning).

  python kaggle/phase2.py data                        # verify / fetch inputs
  python kaggle/phase2.py restore                     # copy finished fits from an attached earlier output
                                                      # (a phase2_runs/ folder or a zip holding phase2_runs/)
  python kaggle/phase2.py plan
  python kaggle/phase2.py run [--smoke 3] [--workers 4] [--deadline EPOCH]
  python kaggle/phase2.py baselines [--workers 4]
  python kaggle/phase2.py analyze                     # needs every fit; then prints the RESULTS.md rows
  python kaggle/phase2.py report                      # RESULTS.md rows from the rebuilt tables

Environment: STUDY_RUNS (run files = checkpoint; default /kaggle/working/phase2_runs),
STUDY_CACHE (fold preprocessing cache), PHASE2_OUT (outputs; default /kaggle/working/phase2_output).
"""
import argparse
import glob
import hashlib
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
REPO = Path(__file__).resolve().parents[1]
KAGGLE = Path('/kaggle/working').exists()
os.environ.setdefault('STUDY_RUNS', '/kaggle/working/phase2_runs' if KAGGLE else str(REPO / 'results/study_final/runs_phase2'))
os.environ.setdefault('STUDY_CACHE', '/tmp/study_cache' if KAGGLE else str(Path.home() / '.capy/work/study_cache'))
OUT = Path(os.environ.get('PHASE2_OUT', '/kaggle/working/phase2_output' if KAGGLE else REPO / 'results/study_final/phase2_output'))
RUNS = Path(os.environ['STUDY_RUNS'])
RUNS.mkdir(parents=True, exist_ok=True)
Path(os.environ['STUDY_CACHE']).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'study'))

import pandas as pd  # noqa: E402

import config as C  # noqa: E402

RES = REPO / 'results/study_final'
SEL_SRC = RES / 'runs/selection'
RAW = REPO / 'data/raw'
INPUTS = {
    # GEO files committed in the repository and used by phase 1 (GEO's current series-matrix gzips
    # differ byte-wise, so the committed copies are the inputs)
    'GSE45498_raw_data.txt.gz': 'b15687c693a5117c30595d12409a08679ae5e823005b87b6d918db60ae2569b7',
    'GSE45498_mRNA_non-normalized_data.txt.gz': '4ab75d356d141bed9a189866f43a64e3f8c4618867fb893045c0c5a20784c69f',
    'GSE45498-GPL16231_series_matrix.txt.gz': '993f76c1f0ce70829c5c0886c9498eb19cee22ab76699f6f89e16c2b1b472f49',
    'GSE45498-GPL16299_series_matrix.txt.gz': '5c9b67aa500661425c6784ebe2faf281f3bade35959180fde1056b965b22f2df',
    'miRDB_v6.0_prediction_result.txt.gz': '446636d2182212f44ff91623f1429f7922f9575b82226962a6213d592a7e109d',
}
MIRDB_URL = 'https://mirdb.org/download/miRDB_v6.0_prediction_result.txt.gz'
LC_BASELINES = ('xgb_joint', 'xgb_routed_mi', 'xgb_routed_mr', 'enet_routed_mi', 'enet_routed_mr')
WEIGHT = {'sens_aug_pairs': 1.6, 'sens_tcga_normal': 1.3, 'sens_tcga_estimate': 1.3, 'joint_mirdb': 1.0, 'joint_nocross': 0.55}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ inputs and checkpoint restore
def data():
    for name, want in INPUTS.items():
        p = RAW / name
        if not p.exists() and name.startswith('miRDB'):
            try:
                print('downloading', MIRDB_URL, flush=True)
                urllib.request.urlretrieve(MIRDB_URL, p)
            except Exception as e:
                print('  mirdb.org download failed:', e)
                p.unlink(missing_ok=True)
                hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True))
                if not hits:
                    sys.exit(f'{name} unavailable: attach a Kaggle Dataset containing it (see kaggle/README.md)')
                shutil.copy(hits[0], p)
        if not p.exists():
            sys.exit(f'missing {p}: it is committed in the repository; check the clone')
        got = sha(p)
        if got != want:
            sys.exit(f'{name}: sha256 {got} differs from the file phase 1 used ({want})')
        print(f'ok  {name}')
    sel = RUNS / 'selection'
    sel.mkdir(parents=True, exist_ok=True)
    for f in SEL_SRC.glob('*.json'):
        dst = sel / f.name
        if dst.exists():
            assert dst.read_bytes() == f.read_bytes(), f'{dst} differs from the committed selection'
        else:
            shutil.copy(f, dst)
    print(f'ok  {len(list(sel.glob("*.json")))} phase-1 configuration selections in {sel}')


def restore():
    """Copy finished run files from an earlier version's output attached under /kaggle/input."""
    n = 0
    for d in glob.glob('/kaggle/input/**/phase2_runs', recursive=True):
        for f in Path(d).rglob('*.csv'):
            dst = RUNS / f.relative_to(d)
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
                n += 1
    for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
        with zipfile.ZipFile(z) as zf:
            for m in zf.namelist():
                if m.endswith('.csv') and m.startswith('phase2_runs/'):
                    dst = RUNS / m[len('phase2_runs/'):]
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zf.read(m))
                        n += 1
    print(f'restored {n} run files from attached inputs into {RUNS}')


# ------------------------------------------------------------------ GAT fits
def fits():
    import run_gat as R
    out = []
    for kind, scheme, r, k, arms, opt in R.tasks(2):
        assert kind in ('main', 'lc'), kind
        for arm in arms:
            a = R.ALL_ARMS[arm]
            if kind == 'lc':                       # run_task: c = select('main', arm, r, k), SEEDS_SECONDARY seeds
                src, seeds = arm, range(C.SEEDS_SECONDARY)
            else:                                  # run_task: c = select(scheme, config_from, r, k), a['seeds'] seeds
                assert not a.get('tune') and scheme == 'main'
                src, seeds = a['config_from'], range(a['seeds'])
            assert R.sel_file('main', src, r, k).exists(), f'missing phase-1 selection for {src} r{r} f{k}; run "data" first'
            c = R.select('main', src, r, k)
            out += [dict(kind=kind, scheme=scheme, repeat=r, fold=k, arm=arm, config=c, seed=s, frac=opt.get('frac'))
                    for s in seeds]
    out.sort(key=lambda f: (-WEIGHT[f['arm']] * (f['frac'] or 1.0), f['scheme'], f['arm'], f['fold'], f['seed']))
    return out


def fit_path(f):
    import run_gat as R
    return R.run_file(f['scheme'], f['arm'], f['repeat'], f['fold'], f['config'], f['seed'])


def baseline_paths():
    return [RUNS / f'lc_{fr}' / n / f'r0_f{k}.csv' for fr in C.LC_FRACTIONS for k in range(C.N_FOLDS) for n in LC_BASELINES]


def smoke_pick(fs, n):
    """One fit from each of the most different kinds: augmentation, TCGA-normal edges, learning curve."""
    order = [('sens_aug_pairs', 'main'), ('sens_tcga_normal', 'main'), ('joint_mirdb', 'lc_0.25'), ('sens_tcga_estimate', 'main')]
    out = []
    for arm, sch in order:
        out += [f for f in fs if f['arm'] == arm and f['scheme'] == sch and f['fold'] == 0 and f['seed'] == 0][:1]
    return out[:n]


_DEADLINE = float('inf')


def _init(deadline):
    global _DEADLINE
    _DEADLINE = deadline


def _fit(f):
    import prep as P
    import run_gat as R
    out = fit_path(f)
    if out.exists():
        return f, 'cached', 0.0
    if time.time() > _DEADLINE:
        return f, 'deferred', 0.0
    t0 = time.time()
    fd = P.prep(f['scheme'], f['repeat'], f['fold'])          # run_task preps the task's own scheme
    fit_ids = R.learning_subset(fd, f['frac']) if f['kind'] == 'lc' else None
    R.run_one(fd, f['scheme'], f['arm'], f['config'], f['seed'], None, None, fit_ids, None)
    return f, 'ran', time.time() - t0


def plan():
    fs = fits()
    done = sum(fit_path(f).exists() for f in fs)
    print(pd.DataFrame(fs).groupby(['scheme', 'arm']).agg(fits=('seed', 'size'), configs=('config', lambda c: sorted(set(c)))).to_string())
    print(f'{len(fs)} GAT fits, {done} already done; {sum(p.exists() for p in baseline_paths())}/{len(baseline_paths())} baseline fold files done')


def run(smoke, workers, deadline):
    from multiprocessing import Pool
    import prep as P
    for t in RUNS.rglob('*.tmp'):
        t.unlink()
    fs = fits()
    if smoke:
        fs = smoke_pick(fs, smoke)
    todo = [f for f in fs if not fit_path(f).exists()]
    print(f'{len(fs)} fits selected, {len(fs) - len(todo)} already done, {len(todo)} to run on {workers} workers', flush=True)
    if not todo:
        return
    folds = sorted({(f['scheme'], f['repeat'], f['fold']) for f in todo})
    t0 = time.time()
    with Pool(min(workers, len(folds))) as pool:
        pool.starmap(P.prep, folds)
    print(f'[{time.time() - t0:6.0f}s] fold preprocessing ready ({len(folds)} folds)', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / 'fit_log.csv'
    n_ran = n_def = 0
    with Pool(workers, initializer=_init, initargs=(deadline,), maxtasksperchild=1) as pool:
        for i, (f, status, secs) in enumerate(pool.imap_unordered(_fit, todo)):
            n_ran += status == 'ran'
            n_def += status == 'deferred'
            print(f'[{time.time() - t0:6.0f}s] {i + 1}/{len(todo)} {status:8s} {f["scheme"]:8s} {f["arm"]:18s} '
                  f'f{f["fold"]} s{f["seed"]} c{f["config"]:02d} {secs:6.1f}s', flush=True)
            if status == 'ran':
                row = f | {'secs': round(secs, 1), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                pd.DataFrame([row]).to_csv(log, mode='a', header=not log.exists(), index=False)
    left = sum(not fit_path(f).exists() for f in fits())
    print(f'ran {n_ran} fits in {(time.time() - t0) / 60:.1f} min; deferred by the time budget: {n_def}; '
          f'GAT fits still missing overall: {left}')
    if smoke and n_ran:
        per = pd.read_csv(log).groupby('arm').secs.mean().to_dict()
        rest = [f for f in fits() if not fit_path(f).exists()]
        est = sum(per.get(f['arm'], max(per.values())) * (f['frac'] or 1.0) for f in rest)
        print(f'rough estimate for the {len(rest)} remaining fits on {workers} workers: {est / workers / 3600:.1f} h '
              f'(mean seconds per fit seen here: { {k: round(v) for k, v in per.items()} })')


def baselines(workers):
    for p in baseline_paths():
        if p.exists():
            try:
                ok = bool(pd.read_csv(p).selected.any())
            except Exception:
                ok = False
            if not ok:
                print('removing incomplete', p)
                p.unlink()
    t0 = time.time()
    subprocess.run([sys.executable, str(REPO / 'study/baselines.py'), 'lc'], check=True,
                   env=os.environ | {'WORKERS': str(workers)}, cwd=REPO)
    print(f'learning-curve baselines: {sum(p.exists() for p in baseline_paths())}/{len(baseline_paths())} fold files '
          f'({(time.time() - t0) / 60:.1f} min)')


# ------------------------------------------------------------------ analysis and report
def analyze():
    missing = [fit_path(f) for f in fits() if not fit_path(f).exists()] + [p for p in baseline_paths() if not p.exists()]
    if missing:
        print(f'{len(missing)} phase-2 outputs missing (first: {missing[0]}); rerun the notebook with this output attached')
        sys.exit(2)
    import analysis as A
    assert Path(A.RUNS) == RUNS
    phase1 = RES / 'predictions_phase1.csv.gz'
    if not phase1.exists():
        shutil.copy(RES / 'predictions_final.csv.gz', phase1)
    p1 = pd.read_csv(phase1)
    assert not (p1.scheme.str.startswith('lc_') | p1.arm.isin(list(C.SENS_ARMS))).any()
    tmp = OUT / 'collect'
    tmp.mkdir(parents=True, exist_ok=True)
    A.RES = tmp                          # collect writes its consolidated files here, not over phase 1
    A.collect()
    A.RES = RES
    p2 = pd.read_csv(tmp / 'predictions_final.csv.gz')
    assert (p2.scheme.str.startswith('lc_') | p2.arm.isin(list(C.SENS_ARMS))).all()
    pd.concat([p1, p2], ignore_index=True).to_csv(RES / 'predictions_final.csv.gz', index=False, float_format='%.5g')
    t0 = time.time()
    A.analyze()
    print(f'analysis done in {(time.time() - t0) / 60:.1f} min')
    subprocess.run([sys.executable, str(REPO / 'study/figures.py')], check=True, cwd=REPO)
    for sub in ('tables', 'figures'):
        shutil.copytree(RES / sub, OUT / sub, dirs_exist_ok=True)
    shutil.copy(tmp / 'predictions_final.csv.gz', OUT / 'predictions_phase2.csv.gz')
    shutil.copy(RES / 'predictions_final.csv.gz', OUT / 'predictions_final_with_phase2.csv.gz')
    report()


def _n(x, d=3, sign=True):
    return (f'{x:+.{d}f}' if sign else f'{x:.{d}f}').replace('-', '−')


def _p(p):
    return '<0.0001' if p < 1e-4 else f'{p:.2g}'


def report():
    tab = OUT / 'tables' if (OUT / 'tables/learning_curve.csv').exists() else RES / 'tables'
    lc = pd.read_csv(tab / 'learning_curve.csv')
    sens = pd.read_csv(tab / 'sensitivity_repeat0.csv')

    def v(m, mod, fr):
        s = lc[(lc.model == m) & (lc.modality == mod) & (lc.fraction == fr)].macro_f1
        return f'{s.iloc[0]:.3f}' if len(s) else 'missing'

    L = ['## Updated RESULTS.md rows (phase 2, run on Kaggle from branch kaggle-phase2)', '',
         '### Section 9, learning curve (repeat 0)', '',
         '| Model | Test mod. | 25% | 50% | 100% macro-F1 |', '|---|---|---|---|---|']
    for m in ('joint_mirdb', 'joint_nocross'):
        for mod in ('miRNA', 'mRNA'):
            L.append(f'| {m} | {mod} | {v(m, mod, 0.25)} | {v(m, mod, 0.5)} | {v(m, mod, 1.0)} |')
    for m, lab in (('xgb_joint', 'XGBoost joint'), ('xgb_routed', 'XGBoost routed'), ('enet_routed', 'Elastic net routed')):
        L.append(f'| {lab} | miRNA / mRNA | ' + ' | '.join(f'{v(m, "miRNA", fr)} / {v(m, "mRNA", fr)}' for fr in (0.25, 0.5, 1.0)) + ' |')
    L += ['', 'Caption: the 100% points are the repeat-0, seeds 0–4 subset of the main runs. The 25% and 50% points '
          'were run after the phase-1 analysis, on Kaggle, with the frozen code and the phase-1 selected configurations.',
          '', '### Section 10, sensitivity rows (repeat 0, 5 seeds; Δ is A − B)', '',
          '| Analysis | Test mod. | ΔF1 A − B (95% CI) | Δ log-loss | Sign-flip p |', '|---|---|---|---|---|']
    labels = {'main:sens_tcga_normal - main:joint_tcga': 'TCGA normal-derived edges − TCGA tumour edges',
              'main:sens_tcga_normal - main:joint_mirdb': 'TCGA normal-derived edges − miRDB edges',
              'main:sens_tcga_estimate - main:joint_tcga': 'ESTIMATE-purity TCGA edges − ABSOLUTE-purity TCGA edges',
              'main:sens_aug_pairs - main:joint_mirdb': 'Train-side random within-tissue pairing augmentation − joint_mirdb'}
    for key, lab in labels.items():
        for _, r in sens[sens.contrast == key].iterrows():
            L.append(f'| {lab} | {r.modality} | {_n(r.delta_macro_f1)} ({_n(r.corrected_ci95_lo, sign=False)}, '
                     f'{_n(r.corrected_ci95_hi, sign=False)}) | {_n(r.delta_logloss)} | {_p(r.signflip_p_logloss)} |')
    L += ['', 'F1 A / F1 B behind these rows:', '']
    for key, lab in labels.items():
        for _, r in sens[sens.contrast == key].iterrows():
            L.append(f'- {lab}, {r.modality}: {r.f1_a:.3f} / {r.f1_b:.3f}')
    L += ['', '### Section 13, status of the formerly skipped analyses', '',
          '| Analysis | Status |', '|---|---|',
          '| Learning curve, 25% and 50% of inner-fit groups (joint_mirdb, joint_nocross, baselines) | run in phase 2 (Kaggle, branch kaggle-phase2); section 9 |',
          '| Sensitivity: TCGA normal-derived edges (`sens_tcga_normal`) | run in phase 2 (Kaggle); section 10 |',
          '| Sensitivity: ESTIMATE-purity TCGA edges (`sens_tcga_estimate`) | run in phase 2 (Kaggle); section 10 |',
          '| Sensitivity: train-side random within-tissue pairing augmentation (`sens_aug_pairs`) | run in phase 2 (Kaggle); section 10 |',
          '', 'Note: analysis.py draws every bootstrap and sign-flip test from one RNG stream and the new sensitivity rows come '
          'first in that section, so the sign-flip p of the phase-1 reference rows in section 10 can move in the last digit. '
          'Their ΔF1 and corrected CIs are deterministic and unchanged; keep the phase-1 values for those rows.']
    txt = '\n'.join(L)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RESULTS_phase2_rows.md').write_text(txt + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['data', 'restore', 'plan', 'run', 'baselines', 'analyze', 'report'])
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--deadline', type=float, default=float('inf'), help='epoch seconds after which no new fit starts')
    a = ap.parse_args()
    {'data': data, 'restore': restore, 'plan': plan, 'analyze': analyze, 'report': report,
     'run': lambda: run(a.smoke, a.workers, a.deadline), 'baselines': lambda: baselines(a.workers)}[a.cmd]()


if __name__ == '__main__':
    main()
