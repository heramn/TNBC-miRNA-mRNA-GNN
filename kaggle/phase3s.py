#!/usr/bin/env python3
"""Phase-3s driver (kaggle/phase3s.ipynb). POST-HOC FOLLOW-UP TESTING, added after the frozen study and phase 3r.

Settings and the pass rule: kaggle/phase3s_config.py. Reuses phase 3r's planted task and fit code
(kaggle/phase3.py); study/ is imported, never edited.

  python kaggle/phase3s.py calibrate [--amps 2,3,4] [--copies 1,2,4]   # design only: graph-free models + oracle
  python kaggle/phase3s.py check-edges                                  # miRDB / rewired / random graph stats
  python kaggle/phase3s.py run --part gate|main [--variant gat|selfdrop0.25] [--smoke] [--workers 4] [--deadline EPOCH]
  python kaggle/phase3s.py gate
  python kaggle/phase3s.py analyze [--variant ...]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3 as Q  # noqa: E402  (sets STUDY_RUNS / STUDY_CACHE / PHASE3_OUT and sys.path)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config as C  # noqa: E402
import phase3s_config as S  # noqa: E402

K = Q.K
OUT, RUNS, MODN = Q.OUT, Q.RUNS, Q.MODN
LABEL = 'Post-hoc follow-up testing (phase 3s), added after the frozen study'


def plant_s(fd, delta, amp):
    """phase3.plant with a per-modality amplitude, amp = {0: miRNA-example, 1: mRNA-example amplitude} (z units per
    unit latent value at strength 1). The real samples get exactly phase 3r's labels and latent draws (same seed,
    same order). Simulated test copies ('<id>#a<j>', see augment) get their labels (a permutation of the real
    test-fold labels of that modality, per copy) and latent values from a separate generator, so adding them does
    not change any real sample's label or signal."""
    t = fd['samples']
    copy_ids = [i for i in t.index if '#a' in i]
    real = t.drop(index=copy_ids)
    rng = np.random.default_rng(K.PC3_SEED + 100 * fd['repeat'] + fd['fold'])
    rng_c = np.random.default_rng(S.AUG_SEED + 100 * fd['repeat'] + fd['fold'])
    y = t.y.copy()
    hb = Q.fold_hubs(fd)
    Z = {m: fd['Z'][m].copy() for m in (0, 1)}
    test_real = set(fd['test'])

    def latent(ids, g):
        mag = np.abs(g.standard_normal((len(ids), 4)))
        lead = g.choice([-1.0, 1.0], (len(ids), 2))
        pat = np.array([Q.PATTERNS[c] for c in y[ids].values])
        return mag * np.column_stack([lead[:, 0], lead[:, 0] * pat[:, 0], lead[:, 1], lead[:, 1] * pat[:, 1]])

    for m in (0, 1):
        ids = list(real.index[real.modality == m])
        y[ids] = rng.permutation(real.loc[ids, 'y'].values)
        u = latent(ids, rng)
        groups = [(ids, u)]
        cm = [i for i in copy_ids if t.at[i, 'modality'] == m]
        if cm:
            src = [i for i in ids if i in test_real]
            for j in sorted({c.rsplit('#a', 1)[1] for c in cm}, key=int):
                cj = [f'{i}#a{j}' for i in src]
                y[cj] = rng_c.permutation(y[src].values)
                groups.append((cj, latent(cj, rng_c)))
        if delta:
            for gids, gu in groups:
                for i, (_, mem) in enumerate(hb[m]):
                    Z[m].loc[gids, mem] = Z[m].loc[gids, mem].values + (delta * amp[m] * gu[:, [i]]).astype(np.float32)
    extra = {'delta': delta, 'design': 'edge_identity_latent_3s', 'amplitude': {MODN[m]: amp[m] for m in (0, 1)},
             'test_copies': fd.get('test_copies', 1), 'hubs': {MODN[m]: [[h, len(mem)] for h, mem in hb[m]] for m in (0, 1)}}
    return y, Z, extra


def amp_of(cond):
    a = S.CONDITIONS[cond]
    return {0: a['miRNA'], 1: a['mRNA']}


def delta_of(cond):
    return 0.0 if cond == 'zero' else 1.0


# ------------------------------------------------------------------ random-pair null
def _install_random_cross():
    import edges as E
    import prep as P
    if getattr(P, '_p3s_patched', False):
        return
    orig = P.cross_pairs

    def cross_pairs(fd, node_set, cross, rng=None):
        if cross != 'random':
            return orig(fd, node_set, cross, rng)
        mi, mr = fd['nodes'][node_set]
        m = len(E.pairs_in(P.mirdb(), mi, mr))
        pick = rng.choice(len(mi) * len(mr), size=m, replace=False)
        return [(mi[i // len(mr)], mr[i % len(mr)]) for i in sorted(pick)]

    P.cross_pairs = cross_pairs
    P._p3s_patched = True


# ------------------------------------------------------------------ simulated examples
def augment(fd, copies):
    """Simulated TEST examples: each outer-test sample is used `copies` times. Copy j > 0 of sample i has id
    'i#a{j}', the same modality and patient group, and the same background expression row; its label and latent
    signal are drawn independently (plant_s). Copies exist only in the outer test fold, so training, early
    stopping, the co-expression networks and the inner-validation gate see exactly the real-size data (adding
    simulated TRAINING examples makes the planted signal readable without edges; see the calibration table).
    Z is restricted to the fold's DE columns (all models use the subset)."""
    if copies == 1:
        return fd
    t = fd['samples'].loc[list(fd['test'])]
    new = [fd['samples']]
    for j in range(1, copies):
        tj = t.copy()
        tj.index = [f'{i}#a{j}' for i in t.index]
        new.append(tj)
    t2 = pd.concat(new)
    ex = lambda ids: [i if j == 0 else f'{i}#a{j}' for j in range(copies) for i in ids]
    Z = {}
    for m in (0, 1):
        cols = list(dict.fromkeys(list(fd['nodes']['full'][m]) + list(fd['nodes']['subset'][m])))
        z = fd['Z'][m][cols]
        zt = z.loc[[i for i in fd['test'] if i in z.index]]
        parts = [z]
        for j in range(1, copies):
            zj = zt.copy()
            zj.index = [f'{i}#a{j}' for i in zt.index]
            parts.append(zj)
        Z[m] = pd.concat(parts)
    return dict(fd, samples=t2, Z=Z, test=ex(fd['test']), test_copies=copies)


def planted(fd, delta, amp, copies=None):
    fda = augment(fd, copies or S.TEST_COPIES)
    y, Z, extra = plant_s(fda, delta, amp)
    return fda, y, Z, extra


# ------------------------------------------------------------------ calibration (design only; no edge-using fit)
def gate_tabular(amp, copies, folds=range(5)):
    """amp = {0: miRNA, 1: mRNA}; copies = test copies (only changes the label / latent draws, as in the runs)."""
    import baselines as B
    import prep as P
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    rows = []
    for k in folds:
        fd0 = P.prep('main', S.REPEAT, k)
        hb = Q.fold_hubs(fd0)
        fd, y, Z, _ = planted(fd0, 1.0, amp, copies)
        fdp = dict(fd, Z=Z)
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
            Sf, Sv = Q.set_means(Z[m], fit, hb[m]), Q.set_means(Z[m], val, hb[m])
            prod = lambda X: np.column_stack([X[:, 0] * X[:, 1], X[:, 2] * X[:, 3]])
            o = LogisticRegression(max_iter=2000, class_weight='balanced').fit(prod(Sf), yf)
            preds.setdefault('oracle (edge-aware)', {}).update(zip(val, o.predict(prod(Sv))))
        for name, pr in preds.items():
            for m in (0, 1):
                ids = [i for i in pr if t.at[i, 'modality'] == m]
                yt, yp = y[ids].values, np.array([pr[i] for i in ids])
                rows.append({'amp_miRNA': amp[0], 'amp_mRNA': amp[1], 'copies': copies, 'model': name, 'fold': k, 'modality': MODN[m], 'n': len(ids),
                             'bal_acc': balanced_accuracy_score(yt, yp), 'macro_f1': f1_score(yt, yp, average='macro')})
    return pd.DataFrame(rows)


def _cal_job(a):
    amp, copies, folds, off = a
    K.PC3_SEED = 4242 + off             # calibration only: other label / latent draws (the runs use 4242)
    return gate_tabular(amp, copies, folds).assign(draw=off)


def calibrate(amps, copies, workers, draws=(0,)):
    """amps: list of (miRNA amplitude, mRNA amplitude); draws: offsets to the planting seed (0 = the runs' draw)."""
    from multiprocessing import Pool
    jobs = [({0: a[0], 1: a[1]}, c, [k], o) for a in amps for c in copies for o in draws for k in range(5)]
    with Pool(workers) as pool:
        d = pd.concat(pool.map(_cal_job, jobs))
    per = d.groupby(['amp_miRNA', 'amp_mRNA', 'copies', 'draw', 'model', 'modality']).bal_acc.mean()
    tab = per.groupby(['amp_miRNA', 'amp_mRNA', 'copies', 'model', 'modality']).agg(['mean', 'min', 'max']).unstack('model')
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    d.to_csv(OUT / 'tables/p3s_calibration_folds.csv', index=False)
    tab.to_csv(OUT / 'tables/p3s_calibration.csv')
    pd.set_option('display.width', 250)
    print(tab.round(3).to_string())


# ------------------------------------------------------------------ graph stats
def check_edges():
    import prep as P
    import run_gat as R
    _install_random_cross()
    rows = []
    for k in S.FOLDS:
        fd = P.prep('main', S.REPEAT, k)
        md = set(P.cross_pairs(fd, 'subset', 'mirdb'))
        deg = lambda ps, j: pd.Series([p[j] for p in ps]).value_counts().sort_index()
        for cross in ('rewired', 'random'):
            ps = P.cross_pairs(fd, 'subset', cross, np.random.default_rng(R.arm_seed('check', cross, k)))
            same_deg = deg(ps, 0).equals(deg(list(md), 0)) and deg(ps, 1).equals(deg(list(md), 1))
            nodes = lambda e: {p[0] for p in e} | {p[1] for p in e}
            rows.append({'fold': k, 'null': cross, 'n_edges': len(ps), 'n_mirdb': len(md), 'overlap_with_mirdb': len(md & set(ps)),
                         'degree_sequence_identical': same_deg, 'same_node_set': nodes(ps) == nodes(md),
                         'nodes_touched': len(nodes(ps)), 'mirdb_nodes_touched': len(nodes(md))})
    tab = pd.DataFrame(rows)
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    tab.to_csv(OUT / 'tables/p3s_edge_nulls.csv', index=False)
    print(tab.to_string(index=False))


# ------------------------------------------------------------------ fits
def vn(v, q):
    return 'gat' if v is None else Q.vname(v, q)


def specs(part, variant=None):
    out = []
    if part == 'gate':
        out += [dict(part='gate', kind='variant', scheme=f'p3sgate_{c}', repeat=S.REPEAT, fold=k, arm='gat_none', cond=c,
                     variant=None, vparam=None, cross='none', config=S.CONFIG, seed=0, delta=1.0, frac=None,
                     eval='val', w=1.0) for c in S.SIGNAL_CONDITIONS for k in S.FOLDS]
    else:
        for v, q in S.VARIANTS:
            if variant and vn(v, q) != variant:
                continue
            for c in S.CONDITIONS:
                for cross in S.CROSSES:
                    for k in S.FOLDS:
                        out += [dict(part='main', kind='variant', scheme=f'p3s_{c}', repeat=S.REPEAT, fold=k, cond=c,
                                     arm=f'{vn(v, q)}_{cross}', variant=v, vparam=q, cross=cross, config=S.CONFIG, seed=s,
                                     delta=delta_of(c), frac=None, eval='test', w=1.0) for s in S.SEEDS]
    # decisive condition first, then interleaved so a time-limited run finishes whole (fold, seed) blocks of all arms
    rank = {c: i for i, c in enumerate([S.DECISIVE, 'zero'] + [c for c in S.CONDITIONS if c not in (S.DECISIVE, 'zero')])}
    out.sort(key=lambda f: (rank[f['cond']], f['seed'], f['fold'], f['arm']))
    return out


def _fit(f):
    import prep as P
    _install_random_cross()
    if Q.fit_path(f).exists():
        return f, 'cached', 0.0
    if time.time() > Q._DEADLINE:
        return f, 'deferred', 0.0
    t0 = time.time()
    fd, y, Z, extra = planted(P.prep('main', f['repeat'], f['fold']), f['delta'], amp_of(f['cond']))
    Q.run_v(fd, f, y, Z, extra)
    return f, 'ran', time.time() - t0


def run(part, variant, smoke, workers, deadline):
    from multiprocessing import Pool
    import prep as P
    for t in RUNS.rglob('*.tmp'):
        t.unlink()
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / 'fit_log.csv'
    fs = specs(part, variant)
    if smoke:
        fs = [f for f in fs if f['fold'] == 0 and f['seed'] == 0 and f['cond'] in (S.DECISIVE, 'base')][:len(S.CROSSES) * len(S.VARIANTS) * 2]
    todo = [f for f in fs if not Q.fit_path(f).exists()]
    print(f'part {part} {variant or ""}: {len(fs)} fits, {len(fs) - len(todo)} done, {len(todo)} to run on {workers} workers', flush=True)
    if not todo:
        return
    t0 = time.time()
    folds = sorted({(f['repeat'], f['fold']) for f in todo})
    with Pool(min(workers, len(folds))) as pool:
        pool.starmap(P.prep, [('main', r, k) for r, k in folds])
    print(f'[{time.time() - t0:6.0f}s] fold preprocessing ready ({len(folds)} folds)', flush=True)
    n_ran = n_def = 0
    with Pool(workers, initializer=Q._init, initargs=(deadline,), maxtasksperchild=1) as pool:
        for i, (f, status, secs) in enumerate(pool.imap_unordered(_fit, todo)):
            n_ran += status == 'ran'
            n_def += status == 'deferred'
            print(f'[{time.time() - t0:6.0f}s] {i + 1}/{len(todo)} {status:8s} {f["scheme"]:12s} {f["arm"]:22s} '
                  f'f{f["fold"]} s{f["seed"]} {secs:6.1f}s', flush=True)
            if status == 'ran':
                row = {k: f.get(k) for k in ('part', 'scheme', 'cond', 'arm', 'repeat', 'fold', 'config', 'seed', 'delta')}
                row |= {'secs': round(secs, 1), 'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
                pd.DataFrame([row]).to_csv(log, mode='a', header=not log.exists(), index=False)
    print(f'ran {n_ran} fits in {(time.time() - t0) / 60:.1f} min; deferred by the time budget: {n_def}', flush=True)


# ------------------------------------------------------------------ gate
def gate():
    """Design gate on the runs' exact data (inner validation, repeat 0). Writes tables/p3s_gate.csv and
    p3s_gate.json ({condition: PASS/FAIL}); exits 1 if any condition fails."""
    from sklearn.metrics import balanced_accuracy_score, f1_score
    import run_gat as R
    tabs, verdict = [], {}
    for c in S.SIGNAL_CONDITIONS:
        d = gate_tabular(amp_of(c), S.TEST_COPIES, S.FOLDS)
        rows = []
        for k in S.FOLDS:
            p = R.run_file(f'p3sgate_{c}', 'gat_none', S.REPEAT, k, S.CONFIG, 0)
            if not p.exists():
                raise SystemExit(f'gate fit missing for {c} fold {k}: run "run --part gate" first')
            x = pd.read_csv(p)
            for m in (0, 1):
                xm = x[x.modality == m]
                yp = xm[['p0', 'p1', 'p2']].values.argmax(1)
                rows.append({'model': 'joint_nocross (GAT)', 'fold': k, 'modality': MODN[m], 'n': len(xm),
                             'bal_acc': balanced_accuracy_score(xm.y, yp), 'macro_f1': f1_score(xm.y, yp, average='macro')})
        d = pd.concat([d.drop(columns=['amp_miRNA', 'amp_mRNA', 'copies']), pd.DataFrame(rows)])
        tab = d.groupby(['model', 'modality']).agg(bal_acc=('bal_acc', 'mean'), bal_acc_min=('bal_acc', 'min'),
                                                    bal_acc_max=('bal_acc', 'max'), macro_f1=('macro_f1', 'mean'),
                                                    n_val=('n', 'sum')).reset_index()
        gated = ~tab.model.str.contains('edge-aware')
        if c not in S.GATE_NOCROSS_CONDITIONS:
            gated &= ~tab.model.str.contains('joint_nocross')
        tab['gated'] = gated
        ok = bool((tab[gated].bal_acc <= S.GATE_MAX_BALACC).all())
        verdict[c] = 'PASS' if ok else 'FAIL'
        tabs.append(tab.assign(condition=c, verdict=verdict[c]))
        print(f'--- condition {c} amplitude {S.CONDITIONS[c]}')
        print(tab.round(3).to_string(index=False))
        print(f'design gate ({c}):', verdict[c])
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    pd.concat(tabs).to_csv(OUT / 'tables/p3s_gate.csv', index=False)
    (OUT / 'p3s_gate.json').write_text(json.dumps(verdict) + '\n')
    if any(v != 'PASS' for v in verdict.values()):
        raise SystemExit(1)


# ------------------------------------------------------------------ analysis
def _bal_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in range(3) if (y == c).any()]))


def analyze(variant=None):
    """Per variant and condition: arm means (macro-F1, balanced accuracy, within-fold seed SD) and the frozen rule
    contrasts (fold_rule: miRDB minus each null). Verdict per variant and signal condition with passes_rule."""
    import analysis as A
    K.B_PASS_MARGIN = S.PASS_MARGIN
    rule_rows, mean_rows, verdicts = [], [], {}
    for v, q in S.VARIANTS:
        name = vn(v, q)
        if variant and name != variant:
            continue
        arms = [f'{name}_{c}' for c in S.CROSSES]
        res, complete = {}, {}
        for c in S.CONDITIONS:
            g = Q._load(f'p3s_{c}', arms)
            need = len(arms) * len(S.FOLDS) * len(S.SEEDS)
            have = g.groupby(['arm', 'fold', 'seed']).ngroups if len(g) else 0
            complete[c] = have >= need
            print(f'{name} {c}: {have}/{need} fits present')
            if not have or g.groupby('arm').fold.nunique().min() < 2 or g.arm.nunique() < len(arms):
                continue
            res[c] = Q.fold_rule(g, f'{name}_mirdb', arms[1:])
            for mod in (0, 1):
                gm = g[g.modality == mod]
                f1 = pd.Series({key: A.macro_f1(x.y.values, x[['p0', 'p1', 'p2']].values.argmax(1))
                                for key, x in gm.groupby(['arm', 'fold', 'seed'])})
                ba = pd.Series({key: _bal_acc(x.y.values, x[['p0', 'p1', 'p2']].values.argmax(1))
                                for key, x in gm.groupby(['arm', 'fold', 'seed'])})
                for a in arms:
                    fa = f1[a]
                    mean_rows.append({'variant': name, 'condition': c, 'modality': MODN[mod], 'arm': a, 'macro_f1': fa.mean(),
                                      'bal_acc': ba[a].mean(), 'n_fits': len(fa),
                                      'seed_sd_within_fold': float(fa.groupby(level=0).std().mean()),
                                      'n_test_per_fold': float(gm[gm.arm == a].groupby(['fold', 'seed']).size().mean())})
            for (m, o), (df, p) in res[c].items():
                rule_rows.append({'variant': name, 'condition': c, 'modality': m, 'contrast': f'{name}_mirdb - {o}',
                                  'fold_mean_dF1': df, 'one_sided_p': p, 'complete': complete[c]})
        if 'zero' in res:
            for c in S.SIGNAL_CONDITIONS:
                if c not in res:
                    continue
                ok_all, ok, zero = Q.passes_rule(res[c], res['zero'])
                rnd = f'{name}_random'
                verdicts[f'{name}|{c}'] = {
                    'passes_rule': ok_all, 'signal_ok': ok, 'zero_ok': zero, 'complete': complete[c] and complete['zero'],
                    'beats_random': any(res[c][(m, rnd)][0] >= S.PASS_MARGIN and res[c][(m, rnd)][1] < 0.05 for m in MODN.values())}
    (OUT / 'tables').mkdir(parents=True, exist_ok=True)
    rt, mt = pd.DataFrame(rule_rows), pd.DataFrame(mean_rows)
    sfx = f'_{variant}' if variant else ''
    rt.to_csv(OUT / f'tables/p3s_rule{sfx}.csv', index=False)
    mt.to_csv(OUT / f'tables/p3s_arms{sfx}.csv', index=False)
    (OUT / f'p3s_verdict{sfx}.json').write_text(json.dumps({'label': LABEL, 'conditions': S.CONDITIONS, 'test_copies': S.TEST_COPIES,
                                                           'seeds': len(S.SEEDS), 'verdicts': verdicts}, indent=1) + '\n')
    pd.set_option('display.width', 250)
    print(mt.round(4).to_string(index=False))
    print(rt.round(4).to_string(index=False))
    print(json.dumps(verdicts, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd')
    ap.add_argument('--part', default='main')
    ap.add_argument('--variant', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--deadline', type=float, default=float('inf'))
    ap.add_argument('--amps', default='2:2,2:8')   # miRNA:mRNA amplitude pairs
    ap.add_argument('--copies', default='4')
    ap.add_argument('--draws', default='0')
    a = ap.parse_args()
    if a.cmd == 'data':
        Q.data()
    elif a.cmd == 'calibrate':
        calibrate([tuple(float(v) for v in x.split(':')) for x in a.amps.split(',')], [int(x) for x in a.copies.split(',')], a.workers,
                  [int(x) for x in a.draws.split(',')])
    elif a.cmd == 'check-edges':
        check_edges()
    elif a.cmd == 'run':
        run(a.part, a.variant, a.smoke, a.workers, a.deadline)
    elif a.cmd == 'gate':
        gate()
    elif a.cmd == 'analyze':
        analyze(a.variant)
    else:
        raise SystemExit(f'unknown command {a.cmd}')


if __name__ == '__main__':
    main()
