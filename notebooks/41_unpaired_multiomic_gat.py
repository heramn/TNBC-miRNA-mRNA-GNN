"""Pairing-free multi-omic GAT on GSE45498 (design A, branch fix-multiomic-A).

GSE45498 has no miRNA<->mRNA patient key, so the primary analysis never combines a
miRNA profile with an mRNA profile at the sample level.

* Every GEO sample is one example with ONE observed modality (miRNA or mRNA). Its
  own GEO record supplies the tissue label.
* One shared node set (DE miRNAs + DE mRNAs). Intra-modal edges are
  tissue-specific co-expression computed on that modality's own training samples
  (no pairing needed). Cross-modal miRNA-mRNA edges come from prior knowledge
  (miRDB v6.0, score > 70), never from cross-sample correlation.
* The GAT sees the observed modality's values on its nodes and an "unobserved"
  flag on the other modality's nodes; messages along the miRDB target edges carry
  information into (and back out of) the unobserved modality's nodes. All weights
  are shared across modalities.

Conditions:
  joint            joint training on both modalities, miRDB cross edges
  joint_nocross    same, cross-modal edges removed (edge ablation)
  joint_rewired    same, cross-modal edges replaced by a degree-preserving
                   bipartite rewiring (permutation null)
  single_mirna     GAT trained on miRNA samples only, no cross edges
  single_mrna      GAT trained on mRNA samples only, no cross edges
  paired_random    sensitivity arm: both modalities observed, miRNA and mRNA
                   samples re-paired uniformly at random WITHIN tissue (train and
                   test), a new pairing per seed
  paired_order     both observed, the order-based pairing of the 01 build
"""
import argparse
import os
import random
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.nn import GATConv

PROCESSED = 'data/processed/'
RAW = 'data/raw/'
OUT = 'results/fix_multiomic_A/'
CLASSES = ['normal', 'primary', 'metastatic']
TISSUES = CLASSES
TOP_N = 150
MIRDB_MIN_SCORE = 70
HEADS, HIDDEN, DROPOUT, LR, WD = 2, 32, 0.2, 1e-3, 1e-4
MAX_EPOCHS, PATIENCE, BATCH = 200, 25, 32


def load():
    d = {}
    for part in ('train', 'test'):
        d[f'{part}_mirna'] = pd.read_csv(f'{PROCESSED}{part}_mirna.csv', index_col=0)
        d[f'{part}_mrna'] = pd.read_csv(f'{PROCESSED}{part}_mrna.csv', index_col=0)
        d[f'{part}_y'] = pd.read_csv(f'{PROCESSED}{part}_labels.csv', index_col=0)['tissue_group']
    d['groups'] = pd.read_csv(f'{PROCESSED}patient_ids.csv').set_index('sample_id')['patient_id']
    return d


def mirdb_edges(mirnas, mrnas):
    raw = pd.read_csv(f'{RAW}GSE45498_mRNA_non-normalized_data.txt.gz', sep='\t', compression='gzip')
    ref2sym = dict(zip(raw['Accession'].str.split('.').str[0], raw['Name']))
    m = pd.read_csv(f'{RAW}miRDB_v6.0_prediction_result.txt.gz', sep='\t', header=None,
                    compression='gzip', names=['miRNA', 'refseq', 'score'])
    m = m[m['miRNA'].str.startswith('hsa-') & (m['score'] > MIRDB_MIN_SCORE)].copy()
    m['gene'] = m['refseq'].astype(str).str.split('.').str[0].map(ref2sym)
    m = m[m['gene'].isin(set(mrnas))]
    known = set(m['miRNA'])
    # NanoString probe names (miRBase ~v14) -> miRDB v6.0 (miRBase v22) mature names:
    # strip the probe note, split merged probes ("a+b"), then try the name as is, with
    # -5p/-3p arm suffixes, and with an "a" paralog letter.
    rows = []
    for probe in mirnas:
        for part in probe.split(' (')[0].split('+'):
            for cand in (part, part + '-5p', part + '-3p', part + 'a', part + 'a-5p', part + 'a-3p'):
                if cand in known:
                    rows.append((cand, probe))
    alias = pd.DataFrame(rows, columns=['mirdb_name', 'miRNA']).drop_duplicates()
    m = m.rename(columns={'miRNA': 'mirdb_name'}).merge(alias, on='mirdb_name')
    return m.groupby(['miRNA', 'gene'], as_index=False)['score'].max()


def coexpr_edges(X, offset, top_n=TOP_N):
    r = np.nan_to_num(np.corrcoef(X.values.T))
    np.fill_diagonal(r, 0)
    iu = np.triu_indices_from(r, k=1)
    vals = np.abs(r[iu])
    keep = np.argsort(-vals)[:top_n]
    return [(int(iu[0][k] + offset), int(iu[1][k] + offset), float(vals[k])) for k in keep]


def build_graph(d, cross):
    """Node order: miRNAs then mRNAs. Returns {tissue: [(a, b, w)]}, n_mi, n_mr."""
    mirnas, mrnas = list(d['train_mirna'].columns), list(d['train_mrna'].columns)
    n_mi = len(mirnas)
    idx = {g: i for i, g in enumerate(mirnas)} | {g: n_mi + i for i, g in enumerate(mrnas)}
    cross_edges = [(idx[a], idx[b], s / 100.0) for a, b, s in cross[['miRNA', 'gene', 'score']].itertuples(index=False)]
    nets = {}
    for t in TISSUES:
        ids = d['train_y'].index[d['train_y'] == t]
        nets[t] = coexpr_edges(d['train_mirna'].loc[ids], 0) + coexpr_edges(d['train_mrna'].loc[ids], n_mi) + cross_edges
    return nets, n_mi, len(mrnas)


def is_cross(e, n_mi):
    return (e[0] < n_mi) != (e[1] < n_mi)


def to_tensors(edges, keep_cross, n_mi):
    src, dst, w = [], [], []
    for e in edges:
        if is_cross(e, n_mi) and not keep_cross:
            continue
        src += [e[0], e[1]]; dst += [e[1], e[0]]; w += [e[2], e[2]]
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(w, dtype=torch.float32).unsqueeze(1)


def rewire_bipartite(cross_pairs, rng, n_swaps_factor=20):
    """Degree-preserving rewiring of a bipartite edge list by double-edge swaps."""
    edges = [list(e) for e in cross_pairs]
    present = set((a, b) for a, b, _ in edges)
    m = len(edges)
    for _ in range(n_swaps_factor * m):
        i, j = rng.integers(m), rng.integers(m)
        (a, b, s1), (c, e, s2) = edges[i], edges[j]
        if a == c or b == e or (a, e) in present or (c, b) in present:
            continue
        present -= {(a, b), (c, e)}
        present |= {(a, e), (c, b)}
        edges[i], edges[j] = [a, e, s1], [c, b, s2]
    return [tuple(e) for e in edges]


class UnpairedMultiOmicGAT(nn.Module):
    def __init__(self, n_nodes, node_types, nets, emb=16, mod_emb=4, expr_hidden=64):
        super().__init__()
        self.n = n_nodes
        self.names = list(nets)
        self.register_buffer('node_ids', torch.arange(n_nodes))
        self.register_buffer('node_types', node_types)
        for k, (ei, ea) in nets.items():
            self.register_buffer(f'{k}_ei', ei)
            self.register_buffer(f'{k}_ea', ea)
        self.node_emb = nn.Embedding(n_nodes, emb)
        self.mod_emb = nn.Embedding(2, mod_emb)
        self.conv1 = GATConv(2 + emb + mod_emb, HIDDEN, heads=HEADS, concat=True, dropout=DROPOUT, edge_dim=1)
        self.conv2 = GATConv(HIDDEN * HEADS, HIDDEN, heads=1, concat=False, dropout=DROPOUT, edge_dim=1)
        self.branch = nn.Linear(HIDDEN * 2, len(CLASSES))
        self.expr = nn.Sequential(nn.LayerNorm(2 * n_nodes), nn.Linear(2 * n_nodes, expr_hidden), nn.ELU(), nn.Dropout(DROPOUT))
        comb = expr_hidden + len(self.names) * (HIDDEN * 2 + len(CLASSES))
        self.head = nn.Sequential(nn.Linear(comb, 96), nn.ELU(), nn.Dropout(DROPOUT), nn.Linear(96, len(CLASSES)))
        self._cache = {}

    def _edges(self, k, b):
        if (k, b) not in self._cache:
            ei, ea = getattr(self, f'{k}_ei'), getattr(self, f'{k}_ea')
            off = (torch.arange(b) * self.n).view(-1, 1, 1)
            self._cache[(k, b)] = ((ei.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1), ea.repeat(b, 1))
        return self._cache[(k, b)]

    def forward(self, x, mask):
        b = x.size(0)
        xv = x * mask
        node_in = torch.cat([xv.unsqueeze(-1), mask.unsqueeze(-1),
                             self.node_emb(self.node_ids).expand(b, -1, -1),
                             self.mod_emb(self.node_types).expand(b, -1, -1)], dim=2).reshape(b * self.n, -1)
        parts = []
        for k in self.names:
            ei, ea = self._edges(k, b)
            h = F.elu(self.conv1(node_in, ei, edge_attr=ea))
            h = F.dropout(h, DROPOUT, self.training)
            h = F.elu(self.conv2(h, ei, edge_attr=ea)).reshape(b, self.n, -1)
            pooled = torch.cat([h.mean(1), h.max(1).values], 1)
            parts += [pooled, self.branch(pooled)]
        return self.head(torch.cat([self.expr(torch.cat([xv, mask], 1))] + parts, 1))


def make_examples(d, part, modalities, n_mi, n_mr, pairing=None):
    N = n_mi + n_mr
    y = d[f'{part}_y']
    mi, mr = d[f'{part}_mirna'], d[f'{part}_mrna']
    X, M, Y, MOD, G, IDS = [], [], [], [], [], []
    if pairing is not None:
        for s_mi, s_mr in pairing:
            X.append(np.concatenate([mi.loc[s_mi].values, mr.loc[s_mr].values])); M.append(np.ones(N))
            Y.append(CLASSES.index(y[s_mi])); MOD.append(2)
            G.append(d['groups'][s_mi] + '|' + d['groups'][s_mr]); IDS.append(f'{s_mi}|{s_mr}')
    else:
        for mod in modalities:
            for s in y.index:
                x, m = np.zeros(N), np.zeros(N)
                if mod == 0:
                    x[:n_mi] = mi.loc[s].values; m[:n_mi] = 1
                else:
                    x[n_mi:] = mr.loc[s].values; m[n_mi:] = 1
                X.append(x); M.append(m); Y.append(CLASSES.index(y[s])); MOD.append(mod)
                G.append(d['groups'][s]); IDS.append(f'{["mirna", "mrna"][mod]}:{s}')
    return (torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(np.array(M), dtype=torch.float32),
            np.array(Y), np.array(MOD), np.array(G), IDS)


def within_tissue_pairing(y, rng):
    pairs = []
    for t in TISSUES:
        ids = list(y.index[y == t])
        pairs += list(zip(ids, rng.permutation(ids)))
    return pairs


def macro_f1(y, p):
    return f1_score(y, p, average='macro', labels=[0, 1, 2], zero_division=0)


def train_eval(cfg):
    torch.set_num_threads(1)
    seed, cond = cfg['seed'], cfg['condition']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    d = load()
    cross = mirdb_edges(d['train_mirna'].columns, d['train_mrna'].columns)
    nets, n_mi, n_mr = build_graph(d, cross)
    N = n_mi + n_mr
    rng = np.random.default_rng(seed)

    keep_cross = cond not in ('joint_nocross', 'single_mirna', 'single_mrna')
    if cond == 'joint_rewired':
        rew = rewire_bipartite([e for e in nets['normal'] if is_cross(e, n_mi)], rng)
        nets = {t: [e for e in es if not is_cross(e, n_mi)] + rew for t, es in nets.items()}
    tensors = {t: to_tensors(es, keep_cross, n_mi) for t, es in nets.items()}
    node_types = torch.tensor([0] * n_mi + [1] * n_mr)

    if cond in ('paired_random', 'paired_order'):
        if cond == 'paired_random':
            tr_pairs, te_pairs = within_tissue_pairing(d['train_y'], rng), within_tissue_pairing(d['test_y'], rng)
        else:
            tr_pairs, te_pairs = [(s, s) for s in d['train_y'].index], [(s, s) for s in d['test_y'].index]
        tr = make_examples(d, 'train', None, n_mi, n_mr, tr_pairs)
        te = make_examples(d, 'test', None, n_mi, n_mr, te_pairs)
    else:
        mods = {'single_mirna': [0], 'single_mrna': [1]}.get(cond, [0, 1])
        tr = make_examples(d, 'train', mods, n_mi, n_mr)
        te = make_examples(d, 'test', mods, n_mi, n_mr)

    Xtr, Mtr, ytr, modtr, gtr, _ = tr
    # inner patient-grouped validation split for early stopping; the test set is never used for selection
    sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=seed)
    fit_idx, val_idx = next(sgkf.split(np.zeros(len(ytr)), ytr * 3 + modtr, gtr))
    model = UnpairedMultiOmicGAT(N, node_types, tensors)
    counts = np.bincount(ytr[fit_idx], minlength=3).astype(float)
    w = torch.tensor(len(fit_idx) / (3 * counts), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    yt = torch.tensor(ytr)
    best, best_state, bad, epochs = -1, None, 0, 0
    for epoch in range(MAX_EPOCHS):
        epochs = epoch + 1
        model.train()
        perm = np.random.permutation(fit_idx)
        for i in range(0, len(perm), BATCH):
            bi = perm[i:i + BATCH]
            opt.zero_grad()
            F.cross_entropy(model(Xtr[bi], Mtr[bi]), yt[bi], weight=w).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            f = macro_f1(ytr[val_idx], model(Xtr[val_idx], Mtr[val_idx]).argmax(1).numpy())
        if f > best:
            best, bad = f, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    Xte, Mte, yte, modte, gte, ids = te
    with torch.no_grad():
        prob = torch.softmax(model(Xte, Mte), 1).numpy()
    rows = pd.DataFrame({'id': ids, 'modality': modte, 'group': gte, 'y': yte,
                         'p_normal': prob[:, 0], 'p_primary': prob[:, 1], 'p_metastatic': prob[:, 2]})
    rows['pred'] = prob.argmax(1)
    rows['seed'] = seed
    rows['condition'] = cond
    rows['perm'] = cfg.get('perm', -1)
    rows['epochs'] = epochs
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conditions', default='joint,joint_nocross,single_mirna,single_mrna,paired_random,paired_order')
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--perms', type=int, default=0, help='joint_rewired permutations (one seed each)')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--tag', default='main')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    cfgs = [{'seed': s, 'condition': c} for c in a.conditions.split(',') if c for s in range(a.seeds)]
    cfgs += [{'seed': 1000 + p, 'condition': 'joint_rewired', 'perm': p} for p in range(a.perms)]
    t0 = time.time()
    out = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(train_eval, cfgs)):
            out.append(r)
            if (i + 1) % 4 == 0:
                print(f'{i + 1}/{len(cfgs)} runs, {time.time() - t0:.0f}s', flush=True)
                pd.concat(out).to_csv(f'{OUT}predictions_{a.tag}.csv', index=False)
    pd.concat(out).to_csv(f'{OUT}predictions_{a.tag}.csv', index=False)
    print(f'done {len(cfgs)} runs in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
