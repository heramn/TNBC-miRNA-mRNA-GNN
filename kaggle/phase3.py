#!/usr/bin/env python3
"""Phase-3 driver for the Kaggle notebooks kaggle/phase3r.ipynb (retry of phase3a.ipynb on branch kaggle-phase3)
and kaggle/phase3b.ipynb.

POST-HOC, ADDED AFTER THE FROZEN STUDY. Nothing in study/ is changed; the frozen code is imported. The
settings and decision rules are in kaggle/phase3_config.py, committed before any phase-3 study fit ran.

  A  planted-signal positive control: frozen GAT arms (joint_mirdb / joint_nocross / joint_rewired, paper
     configuration) and routed baselines on a semi-synthetic task whose label can only be read by grouping
     features along miRDB edges, at several strengths including 0; outer folds of repeat 0
     (retry: redesigned so the label depends on which features form each miRDB set; a design gate checks that
     graph-free models stay near chance before any edge-using GAT fit)
  B  edge-forcing variants (kaggle/phase3_model.py)
       select   planted task scored on the INNER-VALIDATION split + real-data inner-validation loss;
                writes selected_variant.json (never touches outer test folds)
       eval     notebook 2 only, after selected_variant.json is committed as
                kaggle/phase3_frozen/selected_variant.json: the selected variant on all 5 x 5 outer folds
  C  learning curve 25% / 50% for joint_mirdb, joint_nocross and the baselines on repeats 1-4 (repeat 0 is
     restored from an attached phase-2 output or recomputed)

  python kaggle/phase3.py data | restore | plan | check-plant
  python kaggle/phase3.py run --part gate && python kaggle/phase3.py gate   # retry: design gate, before part A/B
  python kaggle/phase3.py run --part A|B|C|eval [--smoke] [--workers 4] [--deadline EPOCH]
  python kaggle/phase3.py baselines --part A|C [--workers 4]
  python kaggle/phase3.py select                 # after run --part B
  python kaggle/phase3.py check-frozen           # notebook 2: the committed selection
  python kaggle/phase3.py analyze --part A|C|eval
  python kaggle/phase3.py report                 # RESULTS_phase3_rows.md from whatever tables exist
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[v] = '1'
REPO = Path(__file__).resolve().parents[1]
KAGGLE = Path('/kaggle/working').exists()
HOME_WORK = Path.home() / '.capy/work'
os.environ.setdefault('STUDY_RUNS', '/kaggle/working/phase3_runs' if KAGGLE else str(HOME_WORK / 'phase3_runs'))
os.environ.setdefault('STUDY_CACHE', '/tmp/study_cache' if KAGGLE else str(HOME_WORK / 'study_cache'))
OUT = Path(os.environ.get('PHASE3_OUT', '/kaggle/working/phase3_output' if KAGGLE else HOME_WORK / 'phase3_output'))
RUNS = Path(os.environ['STUDY_RUNS'])
RUNS.mkdir(parents=True, exist_ok=True)
Path(os.environ['STUDY_CACHE']).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / 'study'))
sys.path.insert(0, str(REPO / 'kaggle'))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import phase3_config as K  # noqa: E402

RES = REPO / 'results/study_final'
FROZEN_SEL = REPO / 'kaggle/phase3_frozen/selected_variant.json'
LABEL = 'Post-hoc, added after the frozen study'
MODN = {0: 'miRNA', 1: 'mRNA'}


def vname(v, q):
    return v if q is None else f'{v}{q}'


# ------------------------------------------------------------------ inputs and checkpoint restore
def data():
    import phase2 as P2          # same input checks and phase-1 selection copy as phase 2
    P2.data()


def restore():
    """Copy finished run files from attached earlier outputs: phase-3 runs, and the phase-2 learning-curve
    repeat-0 runs (part C uses them)."""
    n = 0
    for tag in ('phase3_runs', 'phase2_runs'):
        def take(rel):
            return tag == 'phase3_runs' or rel.parts[0].startswith('lc_')
        for d in glob.glob(f'/kaggle/input/**/{tag}', recursive=True):
            for f in Path(d).rglob('*'):
                rel = f.relative_to(d)
                if f.suffix not in ('.csv', '.json') or not take(rel):
                    continue
                dst = RUNS / rel
                if not dst.exists():
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(f, dst)
                    n += 1
        for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():
                    if not m.startswith(tag + '/') or not m.endswith(('.csv', '.json')):
                        continue
                    rel = Path(m[len(tag) + 1:])
                    if not take(rel):
                        continue
                    dst = RUNS / rel
                    if not dst.exists():
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        dst.write_bytes(zf.read(m))
                        n += 1
    print(f'restored {n} files from attached inputs into {RUNS}')


# ------------------------------------------------------------------ A: planted task (edge identity)
def hubs(md, hub_col, member_col):
    """PC3_N_HUBS hubs, taken greedily from the largest down: a hub is kept if, after removing every member
    shared with another kept hub, each kept hub still has >= PC3_MIN_SET members. Sets are then disjoint."""
    sets = md.groupby(hub_col)[member_col].apply(lambda s: set(s))
    order = sorted(sets.index, key=lambda h: (-len(sets[h]), h))
    excl = lambda hs: [(h, sorted(sets[h] - set().union(*(sets[o] for o in hs if o != h)))) for h in hs]
    picked = []
    for h in order:
        if all(len(m) >= K.PC3_MIN_SET for _, m in excl(picked + [h])):
            picked.append(h)
        if len(picked) == K.PC3_N_HUBS:
            return excl(picked)
    raise RuntimeError(f'only {len(picked)} disjoint hubs with >= {K.PC3_MIN_SET} members')


def fold_hubs(fd):
    import edges as E
    import prep as P
    mi, mr = fd['nodes']['subset']
    md = E.pairs_in(P.mirdb(), mi, mr)
    return {1: hubs(md, 'probe', 'gene'),      # mRNA examples: target sets of hub miRNAs
            0: hubs(md, 'gene', 'probe')}      # miRNA examples: regulator sets of hub genes


PATTERNS = {0: (1.0, 1.0), 1: (-1.0, -1.0), 2: (1.0, -1.0)}   # class -> signs of (u1*u2, u3*u4)


def set_means(Zm, ids, hb):
    return np.column_stack([Zm.loc[ids, mem].values.mean(1) for _, mem in hb])


def plant(fd, delta):
    """Real labels permuted within modality, then a latent signal along miRDB edges: per example, four latent
    values u_i ~ N(0, 1), one per hub set, are added (times delta * PC3_AMPLITUDE) to every member of hub set i. The class
    fixes only the signs of u1*u2 and u3*u4 (class 0: +,+; class 1: -,-; class 2: +,-); each pair's common
    sign and the magnitudes are random. The labels and u do not depend on delta."""
    t = fd['samples']
    rng = np.random.default_rng(K.PC3_SEED + 100 * fd['repeat'] + fd['fold'])
    y = t.y.copy()
    hb = fold_hubs(fd)
    Z = {m: fd['Z'][m].copy() for m in (0, 1)}
    for m in (0, 1):
        ids = list(t.index[t.modality == m])
        y[ids] = rng.permutation(t.loc[ids, 'y'].values)
        mag = np.abs(rng.standard_normal((len(ids), 4)))
        lead = rng.choice([-1.0, 1.0], (len(ids), 2))
        pat = np.array([PATTERNS[c] for c in y[ids].values])
        u = mag * np.column_stack([lead[:, 0], lead[:, 0] * pat[:, 0], lead[:, 1], lead[:, 1] * pat[:, 1]])
        if delta:
            for i, (_, mem) in enumerate(hb[m]):
                Z[m].loc[ids, mem] = Z[m].loc[ids, mem].values + (delta * K.PC3_AMPLITUDE * u[:, [i]]).astype(np.float32)
    extra = {'delta': delta, 'design': 'edge_identity_latent',
             'hubs': {MODN[m]: [[h, len(mem)] for h, mem in hb[m]] for m in (0, 1)}}
    return y, Z, extra


def check_plant():
    """No model fit. Repeat-0 folds: hub set sizes; the largest single-feature class signal in the mean
    (eta^2) and in the spread (largest / smallest class variance, the largest over features); and the 5-fold
    balanced accuracy of a logistic model given the two products of the true hub set means, the ceiling for
    a model that knows the edges (chance 0.33)."""
    import prep as P
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    for k in range(C.N_FOLDS):
        fd = P.prep('main', K.PC3_REPEAT, k)
        hb = fold_hubs(fd)
        for delta in K.PC3_DELTAS:
            y, Z, extra = plant(fd, delta)
            t = fd['samples']
            msg = []
            for m in (0, 1):
                ids = list(t.index[t.modality == m])
                cols = list(fd['nodes']['subset'][m])
                X = Z[m].loc[ids, cols]
                yy = y[ids].values
                eta = max(float(np.var([X[c][yy == j].mean() for j in range(3)]) / max(X[c].var(), 1e-9)) for c in cols)
                vr = max(float(max(X[c][yy == j].var() for j in range(3)) / max(min(X[c][yy == j].var() for j in range(3)), 1e-9))
                         for c in cols)
                S = set_means(Z[m], ids, hb[m])
                feats = np.column_stack([S[:, 0] * S[:, 1], S[:, 2] * S[:, 3]])
                acc = cross_val_score(LogisticRegression(max_iter=2000, class_weight='balanced'), feats, yy, cv=5,
                                      scoring='balanced_accuracy').mean()
                msg.append(f'{MODN[m]} hub sets {[n for _, n in extra["hubs"][MODN[m]]]} max eta2 {eta:.3f} '
                           f'max var ratio {vr:.2f} oracle bal.acc {acc:.2f}')
            print(f'fold {k} delta {delta}: ' + ' | '.join(msg), flush=True)


def gate_tabular():
    """Design gate, tabular part (no GAT): inner-fit -> inner-validation split of the repeat-0 folds at
    GATE_DELTA. Graph-free: elastic net and XGBoost per modality, XGBoost joint (both modalities' columns,
    unobserved = NaN), all at configuration 0 of the frozen baseline grids. Edge-aware references: XGBoost
    per modality with the miRDB target-set means added, and the oracle on the true set-mean products."""
    import baselines as B
    import prep as P
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rows = []
    for k in range(C.N_FOLDS):
        fd = P.prep('main', K.PC3_REPEAT, k)
        y, Z, _ = plant(fd, K.GATE_DELTA)
        fdp = dict(fd, Z=Z)
        hb = fold_hubs(fd)
        t = fd['samples']
        mods = lambda ids, ms: [i for i in ids if t.at[i, 'modality'] in ms]
        preds = {}
        fit, val = mods(fd['fit'], (0, 1)), mods(fd['val'], (0, 1))
        p = B.fit_proba('xgb', B.xgb_configs()[0], 0, B.features(fdp, fit, 'subset', 'joint_nan'), y[fit].values,
                        B.features(fdp, val, 'subset', 'joint_nan'))
        preds['xgb_joint'] = dict(zip(val, p.argmax(1)))
        for m, mode in ((0, 'mi'), (1, 'mr')):
            fit, val = mods(fd['fit'], (m,)), mods(fd['val'], (m,))
            yf = y[fit].values
            for name, est, cfg, md in (('enet', 'enet', B.enet_configs()[0], mode), ('xgb', 'xgb', B.xgb_configs()[0], mode),
                                       ('xgb_targetset (edge-aware)', 'xgb', B.xgb_configs()[0], mode + '_ts')):
                p = B.fit_proba(est, cfg, 0, B.features(fdp, fit, 'subset', md), yf, B.features(fdp, val, 'subset', md))
                preds.setdefault(name, {}).update(zip(val, p.argmax(1)))
            Sf, Sv = set_means(Z[m], fit, hb[m]), set_means(Z[m], val, hb[m])
            prod = lambda S: np.column_stack([S[:, 0] * S[:, 1], S[:, 2] * S[:, 3]])
            o = LogisticRegression(max_iter=2000, class_weight='balanced').fit(prod(Sf), yf)
            preds.setdefault('oracle (edge-aware)', {}).update(zip(val, o.predict(prod(Sv))))
        for name, pr in preds.items():
            for m in (0, 1):
                ids = [i for i in pr if t.at[i, 'modality'] == m]
                yt, yp = y[ids].values, np.array([pr[i] for i in ids])
                rows.append({'model': name, 'fold': k, 'modality': MODN[m], 'n': len(ids),
                             'bal_acc': balanced_accuracy_score(yt, yp), 'macro_f1': f1_score(yt, yp, average='macro')})
    return pd.DataFrame(rows)


def gate():
    """Design gate: the tabular models of gate_tabular() plus joint_nocross (frozen GAT without cross edges,
    config 0, seed 0, run with 'run --part gate'), all scored on the inner-validation split at GATE_DELTA.
    PASS if every graph-free model's mean balanced accuracy over the 5 folds is <= GATE_MAX_BALACC on both
    modalities. Exits 1 on FAIL (the notebook stops before any edge-using GAT fit)."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    import run_gat as R
    d = gate_tabular()
    rows = []
    for k in range(C.N_FOLDS):
        p = R.run_file(K.GATE_SCHEME, 'joint_nocross', K.PC3_REPEAT, k, 0, 0)
        if not p.exists():
            raise SystemExit(f'gate fit missing for fold {k}: run "run --part gate" first')
        x = pd.read_csv(p)
        for m in (0, 1):
            xm = x[x.modality == m]
            yp = xm[['p0', 'p1', 'p2']].values.argmax(1)
            rows.append({'model': 'joint_nocross (GAT)', 'fold': k, 'modality': MODN[m], 'n': len(xm),
                         'bal_acc': balanced_accuracy_score(xm.y, yp), 'macro_f1': f1_score(xm.y, yp, average='macro')})
    d = pd.concat([d, pd.DataFrame(rows)])
    tab = d.groupby(['model', 'modality']).agg(bal_acc=('bal_acc', 'mean'), bal_acc_min=('bal_acc', 'min'),
                                                bal_acc_max=('bal_acc', 'max'), macro_f1=('macro_f1', 'mean'),
                                                n_val=('n', 'sum')).reset_index()
    tab['graph_free'] = ~tab.model.str.contains('edge-aware')
    ok = bool((tab[tab.graph_free].bal_acc <= K.GATE_MAX_BALACC).all())
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    tab.assign(verdict='PASS' if ok else 'FAIL').to_csv(OUT / 'tables/p3r_gate.csv', index=False)
    print(tab.round(3).to_string(index=False))
    print(f'design gate (graph-free mean inner-validation balanced accuracy <= {K.GATE_MAX_BALACC} at delta '
          f'{K.GATE_DELTA}):', 'PASS' if ok else 'FAIL')
    if not ok:
        raise SystemExit(1)


# ------------------------------------------------------------------ fit specifications
def specs(part):
    import run_gat as R
    out = []
    folds = range(C.N_FOLDS)
    if part == 'A':
        for d in K.PC3_DELTAS:
            for k in folds:
                for arm in K.PC3_GAT_ARMS:
                    out += [dict(part='A', kind='frozen', scheme=f'p3ra_d{d}', repeat=K.PC3_REPEAT, fold=k, arm=arm, config=0,
                                 seed=s, delta=d, frac=None, w=0.55 if arm == 'joint_nocross' else 1.0) for s in K.PC3_SEEDS]
    elif part == 'B':
        for v, q in K.B_CANDIDATES:
            for d in K.B_SEL_DELTAS:
                for cross in ('mirdb', 'none', 'rewired'):
                    for k in folds:
                        out += [dict(part='B', kind='variant', scheme=f'p3rbsel_d{d}', repeat=K.B_SEL_REPEAT, fold=k,
                                     arm=f'{vname(v, q)}_{cross}', variant=v, vparam=q, cross=cross, config=K.B_SEL_CONFIG,
                                     seed=s, delta=d, frac=None, eval='val', w=1.2) for s in K.B_SEL_SEEDS]
            for k in folds:
                out += [dict(part='B', kind='variant', scheme='p3bsel_real', repeat=K.B_SEL_REPEAT, fold=k,
                             arm=f'{vname(v, q)}_mirdb', variant=v, vparam=q, cross='mirdb', config=K.B_SEL_CONFIG,
                             seed=s, delta=None, frac=None, eval='val', w=1.2) for s in K.B_SEL_SEEDS]
    elif part == 'gate':
        out += [dict(part='gate', kind='variant', scheme=K.GATE_SCHEME, repeat=K.PC3_REPEAT, fold=k, arm='joint_nocross',
                     variant=None, vparam=None, cross='none', config=0, seed=0, delta=K.GATE_DELTA, frac=None, eval='val', w=1.0)
                for k in folds]
    elif part == 'C':
        for r in [0] + list(K.C_REPEATS):
            for f in K.C_FRACTIONS:
                for k in folds:
                    for arm in K.C_ARMS:
                        assert R.sel_file('main', arm, r, k).exists(), f'missing phase-1 selection {arm} r{r} f{k}; run data first'
                        c = R.select('main', arm, r, k)
                        out += [dict(part='C', kind='frozen', scheme=f'lc_{f}', repeat=r, fold=k, arm=arm, config=c, seed=s,
                                     delta=None, frac=f, w=f * (0.55 if arm == 'joint_nocross' else 1.0))
                                for s in range(C.SEEDS_SECONDARY)]
    elif part == 'eval':
        sel = frozen_selection()
        v, q, vn = sel['variant'], sel['vparam'], sel['name']
        for r in range(C.N_REPEATS):
            for k in folds:
                arm = f'{vn}_mirdb'
                best = select_v(arm, r, k, required=False)
                if best is None:            # stage 1: the 16 seed-0 configurations
                    out += [dict(part='eval', kind='variant', scheme='p3b', repeat=r, fold=k, arm=arm, variant=v, vparam=q,
                                 cross='mirdb', config=c, seed=0, delta=None, frac=None, eval='test', w=1.2)
                            for c in range(C.N_CONFIGS)]
                    continue
                out += [dict(part='eval', kind='variant', scheme='p3b', repeat=r, fold=k, arm=arm, variant=v, vparam=q,
                             cross='mirdb', config=best, seed=s, delta=None, frac=None, eval='test', w=1.2)
                        for s in range(K.B_EVAL_SEEDS)]
                for cross in ('rewired', 'none'):
                    out += [dict(part='eval', kind='variant', scheme='p3b', repeat=r, fold=k, arm=f'{vn}_{cross}', variant=v,
                                 vparam=q, cross=cross, config=best, seed=s, delta=None, frac=None, eval='test',
                                 w=1.3 if cross == 'rewired' else 0.7)
                            for s in range(K.B_EVAL_SECONDARY_SEEDS)]
    out.sort(key=lambda f: (-f['w'], f['scheme'], f['arm'], f['repeat'], f['fold'], f['seed'], f['config']))
    return out


def fit_path(f):
    import run_gat as R
    return R.run_file(f['scheme'], f['arm'], f['repeat'], f['fold'], f['config'], f['seed'])


def select_v(arm, r, k, required=True):
    """Configuration selection for the tuned variant arm: the frozen rule (lowest inner-validation loss over
    the 16 seed-0 configurations, ties -> lower index), stored like the phase-1 selection files."""
    import run_gat as R
    f = R.sel_file('p3b', arm, r, k)
    if f.exists():
        return json.load(open(f))['config']
    files = [R.run_file('p3b', arm, r, k, c, 0) for c in range(C.N_CONFIGS)]
    if not all(p.exists() for p in files):
        if required:
            raise RuntimeError(f'configuration runs missing for {arm} r{r} f{k}')
        return None
    vls = [(float(pd.read_csv(p, nrows=1).val_loss[0]), c) for c, p in enumerate(files)]
    best = min(vls)[1]
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'config': best, 'val_losses': [v for v, _ in vls], 'params': R.CONFIGS[best]}, open(f, 'w'))
    return best


# ------------------------------------------------------------------ one fit
def run_v(fd, f, labels=None, Z=None, extra=None):
    """study/run_gat.run_one for a phase-3 variant: same graph building, examples, file layout and
    rewiring seed; scored on the outer test fold (eval='test') or on the inner-validation split ('val')."""
    import prep as P
    import run_gat as R
    from phase3_model import fit_predict_v
    scheme, arm, r, k, c, seed = f['scheme'], f['arm'], f['repeat'], f['fold'], f['config'], f['seed']
    out = R.run_file(scheme, arm, r, k, c, seed)
    if out.exists():
        return
    rng = np.random.default_rng(R.arm_seed(scheme, arm, r, k, seed))
    nodes, nt, nets, pairs = P.build_graph(fd, 'subset', f['cross'], rng, labels=labels, Z=Z)
    mi, mr = fd['nodes']['subset']
    idx = {g: i for i, g in enumerate(mi)} | {g: len(mi) + i for i, g in enumerate(mr)}
    cross_idx = [(idx[a], idx[b]) for a, b in pairs]
    n_mi = len(mi)
    t = fd['samples']
    y = t.y if labels is None else labels
    fit, val = list(fd['fit']), list(fd['val'])
    test = val if f['eval'] == 'val' else list(fd['test'])
    Xf, Mf = P.examples(fd, fit, nodes, n_mi, Z)
    Xv, Mv = P.examples(fd, val, nodes, n_mi, Z)
    Xt, Mt = P.examples(fd, test, nodes, n_mi, Z)
    t0 = time.time()
    if f['variant'] is None:          # frozen network (design gate: joint_nocross scored on inner validation)
        import model as M
        vl, ep, p = M.fit_predict(Xf, Mf, y[fit].values, Xv, Mv, y[val].values, Xt, Mt, len(nodes), nt, nets,
                                  R.CONFIGS[c], 'gat', seed)
    else:
        vl, ep, p = fit_predict_v(Xf, Mf, y[fit].values, Xv, Mv, y[val].values, Xt, Mt, len(nodes), nt, nets,
                                  R.CONFIGS[c], seed, f['variant'], f['vparam'], cross_idx)
    df = pd.DataFrame({'id': test, 'modality': t.loc[test, 'modality'].values, 'y': y[test].values,
                       'group': t.loc[test, 'group'].values, 'p0': p[:, 0], 'p1': p[:, 1], 'p2': p[:, 2]})
    df = df.assign(scheme=scheme, arm=arm, repeat=r, fold=k, config=c, seed=seed, val_loss=vl, epochs=ep,
                   n_nodes_mi=n_mi, n_nodes_mr=len(nodes) - n_mi, n_cross=len(pairs), n_fit=len(fit),
                   secs=round(time.time() - t0, 1), variant=f['variant'], vparam=f['vparam'], scored_on=f['eval'],
                   extra=json.dumps(extra or {}))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    df.to_csv(tmp, index=False, float_format='%.6g')
    tmp.rename(out)


_DEADLINE = float('inf')


def _init(deadline):
    global _DEADLINE
    _DEADLINE = deadline


def _fit(f):
    import prep as P
    import run_gat as R
    if fit_path(f).exists():
        return f, 'cached', 0.0
    if time.time() > _DEADLINE:
        return f, 'deferred', 0.0
    t0 = time.time()
    fd = P.prep('main', f['repeat'], f['fold'])
    labels = Z = extra = None
    if f.get('delta') is not None:
        labels, Z, extra = plant(fd, f['delta'])
    if f['kind'] == 'frozen':
        fit_ids = R.learning_subset(fd, f['frac']) if f.get('frac') else None
        R.run_one(fd, f['scheme'], f['arm'], f['config'], f['seed'], labels, Z, fit_ids, extra)
    else:
        run_v(fd, f, labels, Z, extra)
    return f, 'ran', time.time() - t0


def smoke_pick(fs, part):
    first = lambda pred: [f for f in fs if pred(f) and f['fold'] == 0 and f['seed'] == 0][:1]
    if part == 'A':
        return sum((first(lambda f, a=a: f['arm'] == a and f['scheme'] == 'p3ra_d1.0') for a in K.PC3_GAT_ARMS), [])
    if part == 'B':
        out = sum((first(lambda f, a=f'{vname(v, q)}_mirdb': f['arm'] == a and f['scheme'] == 'p3rbsel_d1.0')
                   for v, q in K.B_CANDIDATES), [])
        return out + sum((first(lambda f, a=a: f['arm'] == a and f['scheme'] == 'p3rbsel_d1.0') for a in ('xfirst_none', 'gated_rewired')), [])
    if part == 'gate':
        return first(lambda f: True)
    if part == 'C':
        return sum((first(lambda f, a=a: f['arm'] == a and f['repeat'] == 1 and f['scheme'] == 'lc_0.25') for a in K.C_ARMS), [])
    return first(lambda f: f['repeat'] == 0)


def plan():
    rows = []
    for part in ('gate', 'A', 'B', 'C'):
        fs = specs(part)
        rows.append({'part': part, 'fits': len(fs), 'done': sum(fit_path(f).exists() for f in fs)})
    print(pd.DataFrame(rows).to_string(index=False))
    print('baseline fold files: A', sum(baseline_done('A', j) for j in baseline_jobs('A')), '/', len(baseline_jobs('A')),
          '| C', sum(baseline_done('C', j) for j in baseline_jobs('C')), '/', len(baseline_jobs('C')))
    print('frozen selection committed:', FROZEN_SEL.exists())


def run(part, smoke, workers, deadline):
    from multiprocessing import Pool
    import prep as P
    for t in RUNS.rglob('*.tmp'):
        t.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / 'fit_log.csv'
    for stage in range(2 if part == 'eval' else 1):
        fs = specs(part)
        if smoke:
            fs = smoke_pick(fs, part)
        todo = [f for f in fs if not fit_path(f).exists()]
        print(f'part {part}{" stage " + str(stage + 1) if part == "eval" else ""}: {len(fs)} fits, '
              f'{len(fs) - len(todo)} done, {len(todo)} to run on {workers} workers', flush=True)
        if not todo:
            continue
        folds = sorted({(f['repeat'], f['fold']) for f in todo})
        t0 = time.time()
        with Pool(min(workers, len(folds))) as pool:
            pool.starmap(P.prep, [('main', r, k) for r, k in folds])
        print(f'[{time.time() - t0:6.0f}s] fold preprocessing ready ({len(folds)} folds)', flush=True)
        n_ran = n_def = 0
        with Pool(workers, initializer=_init, initargs=(deadline,), maxtasksperchild=1) as pool:
            for i, (f, status, secs) in enumerate(pool.imap_unordered(_fit, todo)):
                n_ran += status == 'ran'
                n_def += status == 'deferred'
                print(f'[{time.time() - t0:6.0f}s] {i + 1}/{len(todo)} {status:8s} {f["scheme"]:12s} {f["arm"]:22s} '
                      f'r{f["repeat"]} f{f["fold"]} s{f["seed"]} c{f["config"]:02d} {secs:6.1f}s', flush=True)
                if status == 'ran':
                    row = {k: f.get(k) for k in ('part', 'kind', 'scheme', 'arm', 'repeat', 'fold', 'config', 'seed', 'delta', 'frac')}
                    row |= {'secs': round(secs, 1), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                    pd.DataFrame([row]).to_csv(log, mode='a', header=not log.exists(), index=False)
        print(f'ran {n_ran} fits in {(time.time() - t0) / 60:.1f} min; deferred by the time budget: {n_def}', flush=True)
        if n_def or smoke:
            break


def _family(f):
    return f['arm'].rsplit('_', 1)[0] if f['kind'] == 'variant' else f['arm']


def estimate(workers):
    """Remaining wall-clock from the fit log: mean seconds per arm family, scaled by the training fraction."""
    lg = pd.read_csv(OUT / 'fit_log.csv')
    lg['family'] = [a.rsplit('_', 1)[0] if k == 'variant' else a for a, k in zip(lg.arm, lg.kind)]
    lg['full_secs'] = lg.secs / lg.frac.fillna(1.0)
    per = lg.groupby('family').full_secs.mean().to_dict()
    fallback = float(lg.full_secs.mean())
    out = {}
    for part in ('gate', 'A', 'B'):
        fs = [f for f in specs(part) if not fit_path(f).exists()]
        out[part] = sum(per.get(_family(f), fallback) * (f['frac'] or 1.0) for f in fs) / workers / 3600
    print(f'GAT wall-clock still to run on {workers} workers (h):', {k: round(v, 2) for k, v in out.items()},
          '| seconds per full-data fit seen:', {k: round(v) for k, v in per.items()})


# ------------------------------------------------------------------ baselines (A planted, C learning curve)
def _base_planted(args):
    import baselines as B
    import prep as P
    name, d, k = args
    out = RUNS / f'p3ra_d{d}' / name / f'r{K.PC3_REPEAT}_f{k}.csv'
    if out.exists():
        return out
    est, mode, ns, mods, n_seeds = B.MODELS[name]
    fd = P.prep('main', K.PC3_REPEAT, k)
    y, Z, _ = plant(fd, d)
    fd = dict(fd, Z=Z)
    t = fd['samples']
    sel = lambda ids: [i for i in ids if t.at[i, 'modality'] in mods]
    fit, val, test = sel(fd['fit']), sel(fd['val']), sel(fd['test'])
    Xf, Xv, Xt = (B.features(fd, ids, ns, mode) for ids in (fit, val, test))
    yf, yv = y[fit].values, y[val].values
    w = B.class_weights(yf)
    cfgs = B.xgb_configs() if est == 'xgb' else B.enet_configs()
    rows, vls = [], []
    for c, cfg in enumerate(cfgs):
        p = B.fit_proba(est, cfg, 0, Xf, yf, np.vstack([Xv, Xt]))
        vls.append(B.wlogloss(yv, p[:len(val)], w))
        rows.append((c, 0, vls[-1], p[len(val):]))
    best = int(np.argmin(vls))
    for s in range(1, n_seeds):
        rows.append((best, s, vls[best], B.fit_proba(est, cfgs[best], s, Xf, yf, Xt)))
    dfs = [pd.DataFrame({'id': test, 'modality': t.loc[test, 'modality'].values, 'y': y[test].values,
                         'group': t.loc[test, 'group'].values, 'p0': p[:, 0], 'p1': p[:, 1], 'p2': p[:, 2],
                         'scheme': f'p3ra_d{d}', 'arm': name, 'repeat': K.PC3_REPEAT, 'fold': k, 'config': c, 'seed': s,
                         'val_loss': vl, 'selected': c == best}) for c, s, vl, p in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    pd.concat(dfs).to_csv(tmp, index=False, float_format='%.6g')
    tmp.rename(out)
    return out


def _base_lc(args):
    import baselines as B
    import prep as P
    import run_gat as R
    name, r, k, f = args
    fit_ids = R.learning_subset(P.prep('main', r, k), f)
    return B.run_model_fold((name, r, k, fit_ids, f'lc_{f}'))


def baseline_jobs(part):
    if part == 'A':
        return [(n, d, k) for d in K.PC3_DELTAS for k in range(C.N_FOLDS) for n in K.PC3_BASELINES]
    return [(n, r, k, f) for r in [0] + list(K.C_REPEATS) for f in K.C_FRACTIONS for k in range(C.N_FOLDS) for n in K.C_BASELINES]


def baseline_done(part, j):
    if part == 'A':
        n, d, k = j
        return (RUNS / f'p3ra_d{d}' / n / f'r{K.PC3_REPEAT}_f{k}.csv').exists()
    n, r, k, f = j
    return (RUNS / f'lc_{f}' / n / f'r{r}_f{k}.csv').exists()


def baselines(part, workers, smoke=False):
    from multiprocessing import Pool
    jobs = [j for j in baseline_jobs(part) if not baseline_done(part, j)]
    if smoke:
        jobs = jobs[:2]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_base_planted if part == 'A' else _base_lc, jobs)):
            if (i + 1) % 20 == 0:
                print(f'{i + 1}/{len(jobs)} baseline fold files', flush=True)
    print(f'part {part} baselines: {sum(baseline_done(part, j) for j in baseline_jobs(part))}/{len(baseline_jobs(part))} '
          f'fold files ({(time.time() - t0) / 60:.1f} min)')


# ------------------------------------------------------------------ analysis
def _load(scheme, arms):
    fs = []
    for a in arms:
        fs += glob.glob(str(RUNS / scheme / a / 'r*_f*_c*_s*.csv'))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def _load_base(scheme, names):
    fs = []
    for n in names:
        fs += [f for f in glob.glob(str(RUNS / scheme / n / 'r*_f*.csv')) if Path(f).stem.count('_') == 1]
    if not fs:
        return pd.DataFrame()
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return d[d.selected]


def _route(d, mi_name, mr_name, new):
    d = d[((d.arm == mi_name) & (d.modality == 0)) | ((d.arm == mr_name) & (d.modality == 1))]
    return d.assign(arm=new)


def _f1(x):
    return 'n/a' if pd.isna(x) else f'{x:.3f}'


def _d(x):
    return f'{x:+.3f}'.replace('-', '−')


def _p(p):
    return '<0.0001' if p < 1e-4 else f'{p:.2g}'


def analyze_A():
    import analysis as A
    rows = []
    for d in K.PC3_DELTAS:
        sch = f'p3ra_d{d}'
        g = _load(sch, K.PC3_GAT_ARMS)
        need = len(K.PC3_GAT_ARMS) * C.N_FOLDS * len(K.PC3_SEEDS)
        have = g.groupby(['arm', 'fold', 'seed']).ngroups if len(g) else 0
        if have < need:
            print(f'part A delta {d}: {have}/{need} GAT fits present; row skipped')
            continue
        b = _load_base(sch, K.PC3_BASELINES)
        base = pd.concat([_route(b, 'xgb_routed_mi', 'xgb_routed_mr', 'xgb_routed'),
                          _route(b, 'enet_routed_mi', 'enet_routed_mr', 'enet_routed')]) if len(b) else pd.DataFrame()
        for mod in (0, 1):
            X = {a: A.Arm(g[g.arm == a], mod) for a in K.PC3_GAT_ARMS}
            row = {'delta': d, 'modality': MODN[mod], **{f'f1_{a}': x.per_run_f1().mean() for a, x in X.items()}}
            for bn in ('xgb_routed', 'enet_routed'):
                bb = base[base.arm == bn] if len(base) else base
                row[f'f1_{bn}'] = A.Arm(bb, mod).per_run_f1().mean() if len(bb) and len(set(bb.fold)) == C.N_FOLDS else np.nan
            for tag, other in (('vs_nocross', 'joint_nocross'), ('vs_rewired', 'joint_rewired')):
                c = A.contrast(X['joint_mirdb'], X[other], f'mirdb - {other}', mod)
                row |= {f'{tag}_{k}': c[k] for k in ('delta_macro_f1', 'corrected_ci95_lo', 'corrected_ci95_hi',
                                                    'delta_logloss', 'signflip_p_logloss')}
            rows.append(row)
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit('part A: no strength has all its GAT fits yet; no table written')
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / 'tables/p3a_planted_control.csv', index=False)
    print(tab.round(4).to_string())


def verdict_A(tab):
    if tab.empty or set(K.PC3_DELTAS) - set(tab.delta):
        return 'incomplete'
    beats = lambda r, tag: r[f'{tag}_corrected_ci95_lo'] > 0 and r[f'{tag}_signflip_p_logloss'] < 0.05
    ok = all(any(beats(r, 'vs_nocross') and beats(r, 'vs_rewired') for _, r in tab[tab.delta == d].iterrows())
             for d in K.PC3_SUCCESS_DELTAS)
    tie = all(r[f'{tag}_corrected_ci95_lo'] <= 0 <= r[f'{tag}_corrected_ci95_hi']
              for _, r in tab[tab.delta == 0.0].iterrows() for tag in ('vs_nocross', 'vs_rewired'))
    if ok and tie:
        return 'PASS'
    return 'FAIL: no clear miRDB-edge advantage at a nonzero strength' if not ok else 'FAIL: a difference at zero strength'


def select():
    """Step 1: planted task on inner validation; step 2: real-data inner-validation loss. Writes
    OUT/selected_variant.json, to be committed as kaggle/phase3_frozen/selected_variant.json."""
    from scipy import stats
    import analysis as A
    rows = []
    for v, q in K.B_CANDIDATES:
        vn = vname(v, q)
        rec = {'variant': v, 'vparam': q, 'name': vn}
        ok_any, zero_ok = False, True
        for d in K.B_SEL_DELTAS:
            g = _load(f'p3rbsel_d{d}', [f'{vn}_{c}' for c in ('mirdb', 'none', 'rewired')])
            need = 3 * C.N_FOLDS * len(K.B_SEL_SEEDS)
            if (g.groupby(['arm', 'fold', 'seed']).ngroups if len(g) else 0) < need:
                raise SystemExit(f'selection fits missing for {vn} delta {d}; run part B first')
            for mod in (0, 1):
                gm = g[g.modality == mod]
                f1 = {key: A.macro_f1(x.y.values, x[['p0', 'p1', 'p2']].values.argmax(1)) for key, x in gm.groupby(['arm', 'fold', 'seed'])}
                fold_f1 = pd.Series(f1).groupby(level=[0, 1]).mean()
                for other in ('none', 'rewired'):
                    diff = (fold_f1[f'{vn}_mirdb'] - fold_f1[f'{vn}_{other}']).values
                    p = float(stats.ttest_1samp(diff, 0, alternative='greater').pvalue) if diff.std() > 0 else (0.0 if diff.mean() > 0 else 1.0)
                    rec[f'd{d}_{MODN[mod]}_vs_{other}_dF1'] = float(diff.mean())
                    rec[f'd{d}_{MODN[mod]}_vs_{other}_p'] = p
                if d == 0.0:
                    zero_ok &= all(abs(rec[f'd0.0_{MODN[mod]}_vs_{o}_dF1']) < K.B_PASS_MARGIN for o in ('none', 'rewired'))
                else:
                    ok_any |= all(rec[f'd{d}_{MODN[mod]}_vs_{o}_dF1'] >= K.B_PASS_MARGIN and rec[f'd{d}_{MODN[mod]}_vs_{o}_p'] < 0.05
                                  for o in ('none', 'rewired'))
        rec['passes_planted'] = bool(ok_any and zero_ok)
        real = _load('p3bsel_real', [f'{vn}_mirdb'])
        if real.groupby(['fold', 'seed']).ngroups < C.N_FOLDS * len(K.B_SEL_SEEDS):
            raise SystemExit(f'real-data selection fits missing for {vn}; run part B first')
        rec['real_inner_val_loss'] = float(real.drop_duplicates(['fold', 'seed']).val_loss.mean())
        rows.append(rec)
    tab = pd.DataFrame(rows)
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / 'tables/p3b_selection.csv', index=False)
    print(tab.round(4).to_string())
    passing = tab[tab.passes_planted]
    choice = None
    if len(passing):
        r = passing.sort_values('real_inner_val_loss', kind='stable').iloc[0]
        choice = {'variant': r.variant, 'vparam': None if pd.isna(r.vparam) else float(r.vparam), 'name': r['name']}
    sel = {'label': LABEL, 'selected': choice is not None, **(choice or {'variant': None, 'vparam': None, 'name': None}),
           'rule': ('kaggle/phase3_config.py, part B: planted task on the inner-validation split (at delta 1.0 the miRDB '
                    f'version beats the no-cross and rewired versions by >= {K.B_PASS_MARGIN} macro-F1, one-sided paired '
                    f't-test p < 0.05 over 5 folds, on >= 1 modality; |dF1| < {K.B_PASS_MARGIN} at delta 0), then the lowest '
                    'real-data inner-validation loss. No outer test fold was used.'),
           'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'code_commit': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip(),
           'candidates': json.loads(tab.to_json(orient='records'))}
    (OUT / 'selected_variant.json').write_text(json.dumps(sel, indent=1) + '\n')
    print('selection:', choice or 'NONE: no candidate passed the planted control, so the outer evaluation is skipped')
    print('commit this file as kaggle/phase3_frozen/selected_variant.json before running notebook 2:', OUT / 'selected_variant.json')


def frozen_selection():
    if not FROZEN_SEL.exists():
        raise SystemExit(f'{FROZEN_SEL} is not in the checkout: commit notebook 1\'s selected_variant.json first')
    tracked = subprocess.run(['git', '-C', str(REPO), 'ls-files', '--error-unmatch', str(FROZEN_SEL)], capture_output=True).returncode == 0
    clean = subprocess.run(['git', '-C', str(REPO), 'diff', '--quiet', 'HEAD', '--', str(FROZEN_SEL)]).returncode == 0
    if not (tracked and clean):
        raise SystemExit(f'{FROZEN_SEL} must be committed and unmodified')
    sel = json.load(open(FROZEN_SEL))
    if not sel.get('selected'):
        raise SystemExit('the frozen selection says no variant passed the planted control: nothing to evaluate')
    return sel


def check_frozen():
    sel = frozen_selection()
    git = lambda *a: subprocess.run(['git', '-C', str(REPO), *a], capture_output=True, text=True).stdout.strip()
    print('frozen selection:', sel['name'], '| last commit touching it:', git('log', '-1', '--format=%H %cI', '--', str(FROZEN_SEL)),
          '| checkout HEAD:', git('rev-parse', 'HEAD'))


def analyze_C():
    import analysis as A
    rows = []
    p1 = pd.read_csv(RES / 'predictions_final.csv.gz')
    full = p1[(p1.scheme == 'main') & (p1.seed < C.SEEDS_SECONDARY)]
    for f in K.C_FRACTIONS + [1.0]:
        if f == 1.0:
            g, b = full[full.arm.isin(K.C_ARMS)], full[full.arm.isin(K.C_BASELINES)]
        else:
            g, b = _load(f'lc_{f}', K.C_ARMS), _load_base(f'lc_{f}', K.C_BASELINES)
        if g.empty:
            continue
        have = g.groupby(['arm', 'repeat', 'fold', 'seed']).ngroups
        need = len(K.C_ARMS) * C.N_REPEATS * C.N_FOLDS * C.SEEDS_SECONDARY
        if f != 1.0 and have < need:
            print(f'part C fraction {f}: {have}/{need} GAT fits present (repeats 0-4); row skipped')
            continue
        reps = sorted(set(g.repeat))
        base = pd.concat([b[b.arm == 'xgb_joint'], _route(b, 'xgb_routed_mi', 'xgb_routed_mr', 'xgb_routed'),
                          _route(b, 'enet_routed_mi', 'enet_routed_mr', 'enet_routed')]) if len(b) else pd.DataFrame()
        for mod in (0, 1):
            X = {a: A.Arm(g[g.arm == a], mod) for a in K.C_ARMS}
            c = A.contrast(X['joint_mirdb'], X['joint_nocross'], f'lc {f}: mirdb - nocross', mod)
            row = {'fraction': f, 'modality': MODN[mod], 'repeats': ','.join(map(str, reps)), 'n_fold_pairs': c['n_fold_pairs'],
                   'f1_joint_mirdb': c['f1_a'], 'f1_joint_nocross': c['f1_b'], 'delta_macro_f1': c['delta_macro_f1'],
                   'ci95_lo': c['corrected_ci95_lo'], 'ci95_hi': c['corrected_ci95_hi'], 'delta_logloss': c['delta_logloss'],
                   'signflip_p': c['signflip_p_logloss'], 'cluster_boot_lo': c['cluster_boot_ci95_lo'], 'cluster_boot_hi': c['cluster_boot_ci95_hi']}
            for bn in ('xgb_joint', 'xgb_routed', 'enet_routed'):
                bb = base[(base.arm == bn) & base.repeat.isin(reps)] if len(base) else base
                row[f'f1_{bn}'] = A.Arm(bb, mod).per_run_f1().mean() if len(bb) else np.nan
            rows.append(row)
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise SystemExit('part C: no learning-curve fits yet; no table written')
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / 'tables/p3c_learning_curve.csv', index=False)
    print(tab.round(4).to_string())


def analyze_eval():
    import analysis as A
    sel = frozen_selection()
    vn = sel['name']
    g = _load('p3b', [f'{vn}_mirdb', f'{vn}_rewired', f'{vn}_none'])
    keep = []
    for (arm, r, k), x in g.groupby(['arm', 'repeat', 'fold']):
        c = select_v(f'{vn}_mirdb', r, k)
        n = K.B_EVAL_SEEDS if arm.endswith('_mirdb') else K.B_EVAL_SECONDARY_SEEDS
        keep.append(x[(x.config == c) & (x.seed < n)])
    g = pd.concat(keep)
    for a, n in ((f'{vn}_mirdb', K.B_EVAL_SEEDS), (f'{vn}_rewired', K.B_EVAL_SECONDARY_SEEDS), (f'{vn}_none', K.B_EVAL_SECONDARY_SEEDS)):
        have = g[g.arm == a].groupby(['repeat', 'fold', 'seed']).ngroups
        if have < n * C.N_REPEATS * C.N_FOLDS:
            raise SystemExit(f'{a}: {have}/{n * C.N_REPEATS * C.N_FOLDS} runs present; rerun notebook 2 with its output attached')
    p1 = pd.read_csv(RES / 'predictions_final.csv.gz')
    noc = p1[(p1.scheme == 'main') & (p1.arm == 'joint_nocross')]
    rows = []
    for mod in (0, 1):
        V = A.Arm(g[g.arm == f'{vn}_mirdb'], mod)
        for name, other in (('joint_nocross (frozen phase-1 arm, 20 seeds)', noc), (f'{vn}_rewired', g[g.arm == f'{vn}_rewired']),
                            (f'{vn}_none', g[g.arm == f'{vn}_none'])):
            rows.append(A.contrast(V, A.Arm(other, mod), f'{vn}_mirdb − {name}', mod))
    tab = pd.DataFrame(rows)
    tab['holm_p'] = A.holm(tab.signflip_p_logloss.values)
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / 'tables/p3b_outer_eval.csv', index=False)
    print(tab[['contrast', 'modality', 'f1_a', 'f1_b', 'delta_macro_f1', 'corrected_ci95_lo', 'corrected_ci95_hi',
               'tost_p', 'power_at_margin', 'delta_logloss', 'signflip_p_logloss', 'holm_p']].round(4).to_string())


def analyze(part):
    {'A': analyze_A, 'C': analyze_C, 'eval': analyze_eval}[part]()
    report()


# ------------------------------------------------------------------ report
def report():
    T = OUT / 'tables'
    L = [f'## Phase 3 rows ({LABEL.lower()}; run on Kaggle from branch kaggle-phase3r, the one approved retry of part A)', '',
         'None of these analyses was in the plan frozen after exploration (tag study-frozen-v1). They were designed after the '
         'frozen results were seen; their settings and decision rules were committed in `kaggle/phase3_config.py` before any '
         'phase-3 study fit ran. They do not change the frozen primary analysis.', '']
    if (T / 'p3r_gate.csv').exists():
        g = pd.read_csv(T / 'p3r_gate.csv')
        L += ['### P3-A design gate (post-hoc, added after the frozen study)', '',
              '| Model | Uses miRDB sets | Test mod. | Balanced accuracy, mean (range over 5 folds) | Macro-F1 |', '|---|---|---|---|---|']
        for _, r in g.iterrows():
            L.append(f'| {r.model.replace(" (edge-aware)", "")} | {"no" if r.graph_free else "yes"} | {r.modality} | '
                     f'{r.bal_acc:.2f} ({r.bal_acc_min:.2f}–{r.bal_acc_max:.2f}) | {r.macro_f1:.2f} |')
        L += ['', f'Gate: **{g.verdict.iloc[0]}**.', '',
              f'Caption: run before any edge-using GAT fit, on the inner-validation split of repeat 0 (5 folds) at δ = {K.GATE_DELTA}. '
              f'Graph-free models must have mean balanced accuracy ≤ {K.GATE_MAX_BALACC} (chance 0.33) on both modalities: elastic net and '
              'XGBoost per modality and XGBoost joint (configuration 0 of the frozen baseline grids), and the frozen joint_nocross GAT '
              '(configuration 0, seed 0). The edge-aware rows (XGBoost with the miRDB target-set means added; the oracle, a logistic '
              'model on the two true set-mean products) show the signal is readable once the sets are known.', '']
    if (T / 'p3a_planted_control.csv').exists():
        a = pd.read_csv(T / 'p3a_planted_control.csv')
        L += ['### P3-A. Planted-signal positive control, redesigned (post-hoc, added after the frozen study)', '',
              '| δ | Test mod. | joint_mirdb | joint_nocross | joint_rewired | XGBoost routed | Elastic net routed | '
              'mirdb − nocross ΔF1 (95% CI), p | mirdb − rewired ΔF1 (95% CI), p |', '|---|---|---|---|---|---|---|---|---|']
        for _, r in a.iterrows():
            cell = lambda tag: (f'{_d(r[f"{tag}_delta_macro_f1"])} ({_d(r[f"{tag}_corrected_ci95_lo"])}, '
                                f'{_d(r[f"{tag}_corrected_ci95_hi"])}), {_p(r[f"{tag}_signflip_p_logloss"])}')
            L.append(f'| {r.delta} | {r.modality} | {_f1(r.f1_joint_mirdb)} | {_f1(r.f1_joint_nocross)} | {_f1(r.f1_joint_rewired)} | '
                     f'{_f1(r.get("f1_xgb_routed"))} | {_f1(r.get("f1_enet_routed"))} | {cell("vs_nocross")} | {cell("vs_rewired")} |')
        L += ['', f'Verdict against the rule committed in `phase3_config.py` before the runs: **{verdict_A(a)}**.', '',
              'Caption: the real labels are permuted within each modality, so the real signal is gone. Each example then gets four '
              'latent values, one per miRDB hub set (the targets of four hub miRNAs in mRNA examples, the regulators of four hub genes '
              f'in miRNA examples), added at δ × {K.PC3_AMPLITUDE:g} SD to every member of that set. The class fixes only the signs of the '
              'two within-pair products, so every feature, every set mean and the within-set co-expression have the same distribution '
              'in all classes; the class can be read only by knowing which features form each miRDB set. This replaces the first '
              'design (an XOR of two set-mean shifts), which joint_nocross read as well as joint_mirdb. Frozen GAT arms in the paper '
              'configuration on repeat 0, 5 outer folds × 3 seeds; routed baselines with the phase-1 16-configuration budget. p is the '
              'patient-level sign-flip test on per-sample log-loss, and the CIs are Nadeau–Bengio corrected over 5 folds. Success '
              f'(committed before the runs): at δ = {" and ".join(map(str, K.PC3_SUCCESS_DELTAS))}, joint_mirdb beats both joint_nocross and '
              'joint_rewired on at least one modality (CI above 0 and p < 0.05), and no mirdb contrast differs from 0 at δ = 0.', '']
    if (T / 'p3b_selection.csv').exists():
        s = pd.read_csv(T / 'p3b_selection.csv')
        L += ['### P3-B. Edge-forcing GAT variants: selection (post-hoc, added after the frozen study)', '',
              '| Variant | Planted δ=1.0: ΔF1 vs no-cross (miRNA / mRNA) | ΔF1 vs rewired (miRNA / mRNA) | Planted δ=0: max abs ΔF1 | '
              'Passes planted control | Real inner-val loss |', '|---|---|---|---|---|---|']
        for _, r in s.iterrows():
            z = max(abs(r[f'd0.0_{m}_vs_{o}_dF1']) for m in ('miRNA', 'mRNA') for o in ('none', 'rewired'))
            L.append(f'| {r["name"]} | {_d(r["d1.0_miRNA_vs_none_dF1"])} / {_d(r["d1.0_mRNA_vs_none_dF1"])} | '
                     f'{_d(r["d1.0_miRNA_vs_rewired_dF1"])} / {_d(r["d1.0_mRNA_vs_rewired_dF1"])} | {z:.3f} | '
                     f'{"yes" if r.passes_planted else "no"} | {r.real_inner_val_loss:.3f} |')
        selj = OUT / 'selected_variant.json'
        chosen = json.load(open(selj)).get('name') if selj.exists() else None
        L += ['', f'Selected: **{chosen or "none: no variant passed the redesigned planted control. Reported as an architecture limitation: none of these GAT variants showed that it can use miRDB edge identity, and the outer evaluation is skipped"}**.', '',
              'Caption: each variant was fitted with miRDB, no-cross and rewired edges on the redesigned planted task (repeat 0, 5 folds × 2 seeds, '
              'paper configuration) and scored on the inner-validation split only. A variant passes if, at δ = 1.0, miRDB edges beat '
              f'both controls by ≥ {K.B_PASS_MARGIN} macro-F1 (one-sided paired t-test p < 0.05 over folds) on at least one modality, '
              f'with |ΔF1| < {K.B_PASS_MARGIN} at δ = 0. Among passing variants, the one with the lowest real-data inner-validation '
              'loss is selected. No outer test fold was used for the selection.', '']
    if (T / 'p3b_outer_eval.csv').exists():
        e = pd.read_csv(T / 'p3b_outer_eval.csv')
        L += ['### P3-B. Selected variant on the outer folds (post-hoc, added after the frozen study)', '',
              '| Contrast | Test mod. | F1 A | F1 B | ΔF1 | 95% CI | TOST p | Power | Δ log-loss | Sign-flip p | Holm p |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for _, r in e.iterrows():
            L.append(f'| {r.contrast} | {r.modality} | {_f1(r.f1_a)} | {_f1(r.f1_b)} | {_d(r.delta_macro_f1)} | '
                     f'{_d(r.corrected_ci95_lo)}, {_d(r.corrected_ci95_hi)} | {_p(r.tost_p)} | {r.power_at_margin:.2f} | '
                     f'{_d(r.delta_logloss)} | {_p(r.signflip_p_logloss)} | {_p(r.holm_p)} |')
        L += ['', 'Caption: the variant was chosen and committed (`kaggle/phase3_frozen/selected_variant.json`) before this evaluation '
              'ran. Same 5 × 5 grouped splits and the same 16-configuration inner-validation tuning as phase 1. The miRDB arm has 10 '
              'seeds; the variant with rewired or no cross edges has 5 seeds and uses the miRDB arm\'s configuration. joint_nocross is '
              'the frozen phase-1 arm with 20 seeds. Holm correction covers the 3 contrasts × 2 modalities. A gain over joint_nocross '
              'alone could come from the architecture change rather than the edges, so the edges are credited only if the variant also '
              'beats its own rewired and no-cross versions.', '']
    if (T / 'p3c_learning_curve.csv').exists():
        c = pd.read_csv(T / 'p3c_learning_curve.csv')
        L += ['### P3-C. Learning curve with repeats (post-hoc, added after the frozen study)', '',
              '| Fraction | Test mod. | Repeats | joint_mirdb | joint_nocross | ΔF1 (95% CI) | Sign-flip p | XGBoost joint | XGBoost routed | Elastic net routed |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for _, r in c.iterrows():
            L.append(f'| {r.fraction:.0%} | {r.modality} | {r.repeats} | {_f1(r.f1_joint_mirdb)} | {_f1(r.f1_joint_nocross)} | '
                     f'{_d(r.delta_macro_f1)} ({_d(r.ci95_lo)}, {_d(r.ci95_hi)}) | {_p(r.signflip_p)} | '
                     f'{_f1(r.get("f1_xgb_joint"))} | {_f1(r.get("f1_xgb_routed"))} | {_f1(r.get("f1_enet_routed"))} |')
        L += ['', 'Caption: 25% and 50% of the inner-fit groups, sampled per class × modality with the frozen `learning_subset`; DE and '
              'graph come from the full outer-training fold; phase-1 selected configurations; 5 seeds. ΔF1 is joint_mirdb − '
              'joint_nocross with a Nadeau–Bengio corrected CI over repeat × fold pairs, and p is the sign-flip test on per-sample '
              'log-loss. The 100% row is the seeds 0–4 subset of the phase-1 runs. Repeat 0 of the 25% and 50% points is the phase-2 '
              'run; repeats 1–4 are new. Not Holm-corrected; descriptive.', '']
    txt = '\n'.join(L)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RESULTS_phase3_rows.md').write_text(txt + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['data', 'restore', 'plan', 'check-plant', 'run', 'baselines', 'select', 'check-frozen',
                                    'analyze', 'report', 'estimate', 'gate'])
    ap.add_argument('--part', default='A', choices=['gate', 'A', 'B', 'C', 'eval'])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--deadline', type=float, default=float('inf'))
    a = ap.parse_args()
    {'data': data, 'restore': restore, 'plan': plan, 'check-plant': check_plant, 'select': select,
     'check-frozen': check_frozen, 'report': report, 'gate': gate, 'estimate': lambda: estimate(a.workers),
     'run': lambda: run(a.part, a.smoke, a.workers, a.deadline),
     'baselines': lambda: baselines(a.part, a.workers, a.smoke),
     'analyze': lambda: analyze(a.part)}[a.cmd]()


if __name__ == '__main__':
    main()
