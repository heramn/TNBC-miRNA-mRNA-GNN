"""Frozen-study tests: run before any final run (pytest study/tests)."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

S = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(S))
import config as C  # noqa: E402
import edges as E  # noqa: E402
import geo_data  # noqa: E402
import prep as P  # noqa: E402

sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()


def test_frozen_edge_files_unchanged():
    info = json.load(open(S / 'frozen/tcga_edges_info.json'))
    for name, h in info['output_sha256'].items():
        assert sha(S / 'frozen' / name) == h, name


def test_study_code_matches_freeze_manifest():
    f = S / 'FREEZE_STUDY.json'
    if not f.exists():
        pytest.skip('manifest written at freeze time')
    for name, h in json.load(open(f))['sha256'].items():
        assert sha(S / name) == h, name


def test_labels_come_from_own_geo_record():
    t = geo_data.load()['samples']
    for plat, m in (('GPL16231', 0), ('GPL16299', 1)):
        sm = geo_data.series_matrix(geo_data.RAW / f'GSE45498-{plat}_series_matrix.txt.gz').set_index('geo_accession')
        tm = t[t.modality == m]
        assert (sm.loc[tm.gsm, 'status'].map(geo_data.STATUS).values == tm.y.values).all()
    assert t.groupby('modality').size().to_dict() == {0: 282, 1: 249}


@pytest.mark.parametrize('scheme', ['main', 'conservative', 'subset230'])
def test_outer_folds_partition_and_group_disjoint(scheme):
    t = P.samples(scheme)
    gc = P.group_col(scheme)
    for r in range(C.N_REPEATS if scheme == 'main' else 1):
        folds = P.outer_folds(scheme, r)
        allids = sum(folds, [])
        assert sorted(allids) == sorted(t.index)
        for te in folds:
            tr = t.index.difference(te)
            for m in (0, 1):
                a = set(t.loc[[i for i in tr if t.at[i, 'modality'] == m], gc])
                b = set(t.loc[[i for i in te if t.at[i, 'modality'] == m], gc])
                assert not (a & b)
            if scheme == 'conservative':
                assert not (set(t.loc[tr, gc]) & set(t.loc[te, gc]))


def test_inner_split_group_disjoint():
    t = P.samples('main')
    te = P.outer_folds('main', 0)[0]
    tr = t.loc[t.index.difference(te)]
    fit, val = P.inner_split(tr, 'main', 0, 0)
    assert not (set(tr.loc[fit, 'group']) & set(tr.loc[val, 'group']))
    assert set(fit) | set(val) == set(tr.index)


def test_preprocessing_ignores_test_data(tmp_path, monkeypatch):
    """Scramble every test-fold profile: DE features, training z-scores and edges must not change."""
    fd = P.prep('main', 0, 0)
    d = geo_data.load()
    rng = np.random.default_rng(0)
    scr = {'counts': {}, 'controls': d['controls'], 'samples': d['samples']}
    for m in (0, 1):
        c = d['counts'][m].copy()
        ids = [i for i in fd['test'] if i in c.index]
        c.loc[ids] = rng.permutation(c.loc[ids].values.ravel()).reshape(len(ids), -1)
        scr['counts'][m] = c
    monkeypatch.setattr(geo_data, 'load', lambda: scr)
    monkeypatch.setattr(P, 'CACHE', tmp_path)
    fd2 = P.prep('main', 0, 0)
    assert fd2['de'] == fd['de'] and fd2['nodes'] == fd['nodes']
    for m in (0, 1):
        tr = [i for i in fd['train'] if i in fd['Z'][m].index]
        assert np.allclose(fd2['Z'][m].loc[tr].values, fd['Z'][m].loc[tr].values)
    for cross in ('mirdb', 'tcga', 'none'):
        assert P.build_graph(fd2, 'subset', cross)[2] == P.build_graph(fd, 'subset', cross)[2]


def test_test_examples_are_single_modality():
    fd = P.prep('main', 0, 0)
    nodes, _, _, _ = P.build_graph(fd, 'subset', 'mirdb')
    n_mi = len(fd['nodes']['subset'][0])
    X, M = P.examples(fd, fd['test'], nodes, n_mi)
    assert ((M[:, :n_mi].max(1) > 0) != (M[:, n_mi:].max(1) > 0)).all()
    assert (X[M == 0] == 0).all()


def test_tcga_arm_density_matched_and_rewiring_preserves_degrees():
    fd = P.prep('main', 0, 0)
    md = P.cross_pairs(fd, 'subset', 'mirdb')
    tc = P.cross_pairs(fd, 'subset', 'tcga')
    avail = len(E.pairs_in(E.tcga_ranked(), *fd['nodes']['subset']))
    assert len(tc) == min(len(md), avail)
    rw = P.cross_pairs(fd, 'subset', 'rewired', np.random.default_rng(1))
    deg = lambda ps, i: pd.Series([p[i] for p in ps]).value_counts().sort_index()
    assert deg(md, 0).equals(deg(rw, 0)) and deg(md, 1).equals(deg(rw, 1))
    assert len(set(rw)) == len(rw) and set(rw) != set(md)


def test_model_ignores_unobserved_values():
    import torch
    from model import Net
    nets = {c: [(0, 3, 1.0, 1.0), (1, 2, 0.5, 0.0)] for c in range(3)}
    torch.manual_seed(0)
    net = Net(4, np.array([0, 0, 1, 1]), nets, C.PAPER_CONFIG, 'gat').eval()
    x = torch.randn(2, 4)
    m = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1]], dtype=torch.float32)
    x2 = x.clone()
    x2[m == 0] = 99.0
    assert torch.allclose(net(x, m), net(x2, m))
