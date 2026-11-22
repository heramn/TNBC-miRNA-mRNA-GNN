#!/usr/bin/env python3
"""Run GAT-family jobs for the final study, sharded across machines.

  python study/run_gat.py work --phase 1 --shard 0 --nshards 9 --workers 4
  python study/run_gat.py list --phase 1 --nshards 9          # print the plan

A task is (scheme, repeat, fold, [arms]). A tuned arm runs its 16 configurations with seed 0 on the
inner split, selects the configuration with the lowest class-weighted inner-validation loss
(ties -> lower index), then runs the remaining seeds with it. Arms with config_from reuse the
selected configuration of the named arm in the same fold. Fixed-configuration tasks (alternative
fold schemes, positive control) use the paper configuration (config 0).
Each run writes one CSV of test-fold predictions (both modalities) to RUNS/<scheme>/<arm>/.
"""
import argparse
import copy
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C
import prep as P
from model import fit_predict

RUNS = Path(os.environ.get('STUDY_RUNS', Path(__file__).resolve().parents[1] / 'results/study_final/runs'))
CONFIGS = C.gat_configs()
ALL_ARMS = C.ARMS | C.SENS_ARMS
UNIT = {'gat': 1.0, 'gcn': 0.7, 'graphonly': 0.9, 'mlp': 0.1}


def arm_seed(*parts):
    return [zlib.crc32('|'.join(map(str, parts)).encode())]


def run_file(scheme, arm, r, k, c, s):
    return RUNS / scheme / arm / f'r{r}_f{k}_c{c:02d}_s{s:02d}.csv'


def sel_file(scheme, arm, r, k):
    return RUNS / 'selection' / f'{scheme}__{arm}__r{r}_f{k}.json'


def positive_control(fd, delta_mr):
    """Labels permuted within modality (real signal destroyed) and a class signal planted along
    miRDB edges: class c gets +PC_DELTA_MIRNA on miRNA m_c in miRNA examples and -delta_mr on every
    miRDB target of m_c (inside the node subset) in mRNA examples."""
    t = fd['samples']
    rng = np.random.default_rng(777 + fd['fold'])
    y = t.y.copy()
    for m in (0, 1):
        ids = t.index[t.modality == m]
        y[ids] = rng.permutation(t.loc[ids, 'y'].values)
    mi, mr = fd['nodes']['subset']
    md = P.mirdb()
    md = md[md.probe.isin(mi) & md.gene.isin(mr)]
    tg = md.groupby('probe').gene.apply(list)
    cand = sorted(tg.index[tg.str.len() >= 5])
    planted = list(np.random.default_rng(999 + fd['fold']).choice(cand, 3, replace=False))
    Z = {m: fd['Z'][m].copy() for m in (0, 1)}
    for c, m_c in enumerate(planted):
        mi_ids = [i for i in Z[0].index if y[i] == c]
        mr_ids = [i for i in Z[1].index if y[i] == c]
        Z[0].loc[mi_ids, m_c] += C.PC_DELTA_MIRNA
        Z[1].loc[mr_ids, tg[m_c]] -= delta_mr
    return y, Z, {'planted': planted, 'n_targets': [len(tg[m]) for m in planted]}


def learning_subset(fd, frac):
    t = fd['samples']
    fit = t.loc[fd['fit']]
    gc = P.group_col(fd['scheme'])
    rng = np.random.default_rng(555 + fd['fold'] + int(frac * 100))
    keep = []
    for _, s in fit.groupby(['y', 'modality']):
        g = sorted(s[gc].unique())
        n = max(1, int(round(frac * len(g))))
        keep += list(rng.choice(g, n, replace=False))
    return [i for i in fd['fit'] if fit.at[i, gc] in set(keep)]


def run_one(fd, scheme, arm, c, seed, labels=None, Z=None, fit_ids=None, extra=None):
    a = ALL_ARMS[arm]
    r, k = fd['repeat'], fd['fold']
    out = run_file(scheme, arm, r, k, c, seed)
    if out.exists():
        return pd.read_csv(out)
    rng = np.random.default_rng(arm_seed(scheme, arm, r, k, seed))
    nodes, nt, nets, pairs = P.build_graph(fd, a['nodes'], a['cross'], rng, labels=labels, Z=Z)
    n_mi = len(fd['nodes'][a['nodes']][0])
    t = fd['samples']
    y = t.y if labels is None else labels
    sel = lambda ids: [i for i in ids if t.at[i, 'modality'] in a['train']]
    fit, val, test = sel(fit_ids or fd['fit']), sel(fd['val']), list(fd['test'])
    Xf, Mf = P.examples(fd, fit, nodes, n_mi, Z)
    yf = y[fit].values
    if a.get('augment_pairs'):
        pr = np.random.default_rng(arm_seed('aug', r, k, seed))
        Xa, Ma, ya = [], [], []
        for cl in range(3):
            mi_ids = [i for i in fit if t.at[i, 'modality'] == 0 and y[i] == cl]
            mr_ids = [i for i in fit if t.at[i, 'modality'] == 1 and y[i] == cl]
            n = min(len(mi_ids), len(mr_ids))
            for i, j in zip(pr.permutation(mi_ids)[:n], pr.permutation(mr_ids)[:n]):
                xi, mi_ = P.examples(fd, [i], nodes, n_mi, Z)
                xj, mj = P.examples(fd, [j], nodes, n_mi, Z)
                Xa.append(xi[0] + xj[0]); Ma.append(mi_[0] + mj[0]); ya.append(cl)
        Xf, Mf, yf = np.vstack([Xf, Xa]), np.vstack([Mf, Ma]), np.concatenate([yf, ya])
    Xv, Mv = P.examples(fd, val, nodes, n_mi, Z)
    Xt, Mt = P.examples(fd, test, nodes, n_mi, Z)
    assert (Mt.reshape(len(test), -1)[:, :n_mi].any(1) != Mt[:, n_mi:].any(1)).all(), 'test example with both modalities'
    t0 = time.time()
    vl, ep, p = fit_predict(Xf, Mf, yf, Xv, Mv, y[val].values, Xt, Mt, len(nodes), nt, nets,
                            CONFIGS[c], a['model'], seed)
    df = pd.DataFrame({'id': test, 'modality': t.loc[test, 'modality'].values, 'y': y[test].values,
                       'group': t.loc[test, P.group_col(scheme)].values,
                       'p0': p[:, 0], 'p1': p[:, 1], 'p2': p[:, 2]})
    df = df.assign(scheme=scheme, arm=arm, repeat=r, fold=k, config=c, seed=seed, val_loss=vl, epochs=ep,
                   n_nodes_mi=n_mi, n_nodes_mr=len(nodes) - n_mi, n_cross=len(pairs), n_fit=len(fit),
                   secs=round(time.time() - t0, 1), extra=json.dumps(extra or {}))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    df.to_csv(tmp, index=False, float_format='%.6g')
    tmp.rename(out)
    return df


def select(scheme, arm, r, k):
    f = sel_file(scheme, arm, r, k)
    if f.exists():
        return json.load(open(f))['config']
    vls = [(float(pd.read_csv(run_file(scheme, arm, r, k, c, 0), nrows=1).val_loss[0]), c) for c in range(C.N_CONFIGS)]
    best = min(vls)[1]
    f.parent.mkdir(parents=True, exist_ok=True)
    json.dump({'config': best, 'val_losses': [v for v, _ in vls], 'params': CONFIGS[best]}, open(f, 'w'))
    return best


def run_task(task):
    kind, scheme, r, k, arms, opt = task
    fd = P.prep(scheme if kind != 'pc' else 'main', r, k)
    labels = Z = fit_ids = extra = None
    if kind == 'pc':
        labels, Z, extra = positive_control(fd, opt['delta'])
    if kind == 'lc':
        fit_ids = learning_subset(fd, opt['frac'])
    for arm in arms:
        a = ALL_ARMS[arm]
        if kind in ('pc', 'fixed'):
            c, seeds = 0, range(a['seeds'] if kind == 'pc' else opt.get('seeds', C.SEEDS_SECONDARY))
        elif kind == 'lc':
            c, seeds = select('main', arm, r, k), range(C.SEEDS_SECONDARY)
        elif a.get('tune'):
            for cc in range(C.N_CONFIGS):
                run_one(fd, scheme, arm, cc, 0)
            c, seeds = select(scheme, arm, r, k), range(a['seeds'])
        else:
            c, seeds = select(scheme, a['config_from'], r, k), range(a['seeds'])
        for s in seeds:
            run_one(fd, scheme, arm, c, s, labels, Z, fit_ids, extra)
    return task


def tasks(phase):
    out = []
    if phase == 1:
        groups = [['joint_mirdb', 'joint_rewired'], ['joint_tcga'], ['joint_nocross'], ['joint_mirdb_full'], ['gcn_mirdb'],
                  ['mlp_only'], ['graphonly_mirdb'], ['single_mi_mirdb', 'single_mi_rewired'], ['single_mi_nocross'],
                  ['single_mr_mirdb', 'single_mr_rewired'], ['single_mr_nocross']]
        for r in range(C.N_REPEATS):
            for k in range(C.N_FOLDS):
                out += [('main', 'main', r, k, g, {}) for g in groups]
        for sch in C.SCHEMES:
            for k in range(C.N_FOLDS):
                out.append(('fixed', sch, 0, k, C.SCHEME_ARMS, {'seeds': C.SEEDS_SECONDARY}))
        out.append(('fixed', 'asplit', 0, 0, ['joint_mirdb', 'joint_nocross', 'joint_tcga', 'joint_rewired'], {'seeds': C.SEEDS_PRIMARY}))
        for d in C.PC_DELTA_MRNA:
            for k in range(C.N_FOLDS):
                out.append(('pc', f'pc_d{d}', 0, k, C.PC_ARMS, {'delta': d}))
    else:
        for k in range(C.N_FOLDS):
            out.append(('main', 'main', 0, k, list(C.SENS_ARMS), {}))
            for f in C.LC_FRACTIONS:
                out.append(('lc', f'lc_{f}', 0, k, C.LC_ARMS, {'frac': f}))
    return out


def cost(task):
    kind, scheme, r, k, arms, opt = task
    tot = 0
    for arm in arms:
        a = ALL_ARMS[arm]
        unit = UNIT[a['model']] * (0.45 if len(a['train']) == 1 else 1.0)
        if kind == 'fixed':
            n = opt['seeds']
        elif kind == 'pc':
            n = a['seeds']
        elif kind == 'lc':
            n = C.SEEDS_SECONDARY * opt['frac']
        else:
            n = (C.N_CONFIGS + a['seeds'] - 1) if a.get('tune') else a['seeds']
        tot += n * unit
    return tot


def shard(phase, i, n):
    """Longest-processing-time-first assignment of tasks to n machines (deterministic)."""
    ts = sorted(tasks(phase), key=lambda t: (-cost(t), str(t)))
    load = [0.0] * n
    mine = []
    for t in ts:
        j = int(np.argmin(load))
        load[j] += cost(t)
        if j == i:
            mine.append(t)
    return mine, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['work', 'list', 'prep'])
    ap.add_argument('--phase', type=int, default=1)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    a = ap.parse_args()
    mine, load = shard(a.phase, a.shard, a.nshards)
    if a.cmd == 'list':
        print(len(tasks(a.phase)), 'tasks; machine loads (joint-run units):', [round(x) for x in load])
        for t in mine:
            print(round(cost(t), 1), t[:5])
        return
    folds = sorted({(t[1] if t[0] not in ('pc', 'lc') else 'main', t[2], t[3]) for t in mine})
    with Pool(a.workers) as pool:
        pool.starmap(P.prep, folds)
    if a.cmd == 'prep':
        return
    t0 = time.time()
    with Pool(a.workers, maxtasksperchild=1) as pool:
        for i, t in enumerate(pool.imap_unordered(run_task, mine)):
            print(f'[{time.time() - t0:7.0f}s] done {i + 1}/{len(mine)} {t[:5]}', flush=True)


if __name__ == '__main__':
    main()
