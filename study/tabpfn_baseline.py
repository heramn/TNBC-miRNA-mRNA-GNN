#!/usr/bin/env python3
"""TabPFN v2 (open v2 classifier weights, tabpfn==2.0.9, CPU) routed per modality, same budget as
every other model: 16 configurations (n_estimators x softmax_temperature), fit on the inner-fit
split, selected by class-weighted inner-validation log-loss, best configuration run with 5 seeds.
Runs in its own venv (tabpfn 2.0.9 needs scikit-learn < 1.7); reads matrices from baselines.py export.
  python tabpfn_baseline.py <tabpfn_inputs dir> <out csv>
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier

inp, out = Path(sys.argv[1]), Path(sys.argv[2])
CFGS = [dict(n_estimators=n, softmax_temperature=t) for n, t in itertools.product([1, 2, 4, 8], [0.5, 0.75, 0.9, 1.0])]
SEEDS = 5


def wll(y, p):
    cnt = np.bincount(y, minlength=3).astype(float)
    w = len(y) / (3 * np.maximum(cnt, 1))
    p = np.clip(p, 1e-7, 1)
    return float(np.sum(w[y] * -np.log(p[np.arange(len(y)), y])) / np.sum(w[y]))


def proba(cfg, seed, Xf, yf, Xo):
    m = TabPFNClassifier(device='cpu', random_state=seed, ignore_pretraining_limits=True, **cfg)
    m.fit(Xf, yf)
    p = np.zeros((len(Xo), 3))
    p[:, m.classes_] = m.predict_proba(Xo)
    return p


done = pd.read_csv(out) if out.exists() else None
rows = [] if done is None else [done]
for f in sorted(inp.glob('*.npz')):
    r, k, m = (int(x[1:]) for x in f.stem.split('_'))
    if done is not None and ((done.repeat == r) & (done.fold == k) & (done.modality == m)).any():
        continue
    d = np.load(f, allow_pickle=True)
    vls, ps = [], []
    for cfg in CFGS:
        p = proba(cfg, 0, d['Xf'], d['yf'], np.vstack([d['Xv'], d['Xt']]))
        vls.append(wll(d['yv'], p[:len(d['yv'])]))
        ps.append(p[len(d['yv']):])
    best = int(np.argmin(vls))
    res = [(c, 0, ps[c]) for c in range(len(CFGS))]
    res += [(best, s, proba(CFGS[best], s, d['Xf'], d['yf'], d['Xt'])) for s in range(1, SEEDS)]
    for c, s, p in res:
        rows.append(pd.DataFrame({'id': d['ids'], 'modality': m, 'y': d['yt'], 'group': d['groups'], 'p0': p[:, 0], 'p1': p[:, 1],
                                  'p2': p[:, 2], 'scheme': 'main', 'arm': f'tabpfn_routed_{"mi" if m == 0 else "mr"}',
                                  'repeat': r, 'fold': k, 'config': c, 'seed': s, 'val_loss': vls[c], 'selected': c == best}))
    pd.concat(rows).to_csv(out, index=False, float_format='%.6g')
    print(f.stem, 'best', CFGS[best], flush=True)
