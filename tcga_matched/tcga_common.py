"""Shared helpers for the TCGA patient-matched multi-omic analysis (tcga_matched/)."""
import copy
import gzip
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from statsmodels.stats.multitest import multipletests
from torch_geometric.nn import GATConv

REPO = Path(__file__).resolve().parents[1]
TOP_N_PER_EDGE_TYPE = 150
FDR_ALPHA = 0.05
GAT_PARAMS = dict(num_heads=2, hidden_size=32, dropout=0.2, lr=0.001)
MAX_EPOCHS = 250
PATIENCE = 35
BATCH_SIZE = 16
BRANCH_LOSS_WEIGHT = 0.25


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mimat_to_name(fasta):
    out = {}
    with open(fasta) as fh:
        for line in fh:
            if line.startswith('>hsa-'):
                parts = line[1:].split()
                out[parts[1]] = parts[0]
    return out


def geo_panels():
    with gzip.open(REPO / 'data/raw/GSE45498_raw_data.txt.gz', 'rt') as fh:
        mir = [l.split('\t')[1] for l in fh if l.startswith('Endogenous')]
    with gzip.open(REPO / 'data/raw/GSE45498_mRNA_non-normalized_data.txt.gz', 'rt') as fh:
        mr = [l.split('\t')[1] for l in fh if l.startswith('Endogenous')]
    return mir, mr


def map_geo_mirna_to_tcga(geo_names, tcga_mir):
    """Map legacy NanoString names (e.g. hsa-miR-21) to TCGA MIMAT columns (renamed to miRBase
    names). Exact name first; otherwise the -5p/-3p arm, and if both arms exist the arm with the
    higher mean TCGA expression (the dominant arm, which legacy unsuffixed names referred to)."""
    cols = set(tcga_mir.columns)
    means = tcga_mir.mean()
    mapping = {}
    for g in geo_names:
        base = g.split('|')[0].split(' ')[0].strip()
        if base in cols:
            mapping[g] = base
            continue
        arms = [a for a in (base + '-5p', base + '-3p') if a in cols]
        if arms:
            mapping[g] = max(arms, key=lambda a: means[a])
    return mapping


def welch_select(X, y, k):
    pos, neg = X[y == 1], X[y == 0]
    t, p = stats.ttest_ind(pos, neg, equal_var=False, axis=0)
    p = np.nan_to_num(p, nan=1.0)
    q = multipletests(p, method='fdr_bh')[1]
    order = np.argsort(p)[:k]
    return list(X.columns[order]), pd.DataFrame({'t': t, 'p': p, 'q': q}, index=X.columns)


def _corr(a, b):
    az = (a - a.mean(0)) / (a.std(0) + 1e-12)
    bz = (b - b.mean(0)) / (b.std(0) + 1e-12)
    return (az.T @ bz) / a.shape[0]


def _pvals(r, n):
    r = np.clip(r, -0.999999, 0.999999)
    t = r * np.sqrt((n - 2) / (1 - r ** 2))
    return 2.0 * stats.t.sf(np.abs(t), df=n - 2)


def build_network(mir, mr, top_n=TOP_N_PER_EDGE_TYPE):
    """Same recipe as notebooks/03: per edge type, BH-FDR on Pearson r, keep the top_n strongest
    (positive for within-modality co-expression, most negative for miRNA-mRNA). Here the
    cross-modal correlations are computed on TRUE same-specimen pairs."""
    n = mir.shape[0]
    edges = []
    stats_out = {}
    for kind, a, b in (('miRNA-miRNA', mir, mir), ('mRNA-mRNA', mr, mr), ('miRNA-mRNA', mir, mr)):
        r = _corr(a.values, b.values)
        if kind != 'miRNA-mRNA':
            iu = np.triu_indices(r.shape[0], 1)
            rr, ii, jj = r[iu], iu[0], iu[1]
        else:
            rr = r.ravel()
            ii, jj = np.unravel_index(np.arange(rr.size), r.shape)
        ok = np.isfinite(rr)
        rr, ii, jj = rr[ok], ii[ok], jj[ok]
        q = multipletests(_pvals(rr, n), method='fdr_bh')[1]
        sig = q < FDR_ALPHA
        if kind == 'miRNA-mRNA':
            sel = sig & (rr < 0)
            order = np.argsort(rr)
            stats_out['cross_fdr_sig_total'] = int(sig.sum())
            stats_out['cross_fdr_sig_negative'] = int((sig & (rr < 0)).sum())
            stats_out['cross_tested'] = int(rr.size)
        else:
            sel = sig & (rr > 0)
            order = np.argsort(-rr)
        order = [o for o in order if sel[o]][:top_n]
        for o in order:
            edges.append((a.columns[ii[o]], b.columns[jj[o]], kind, float(rr[o]), float(q[o])))
    return pd.DataFrame(edges, columns=['source', 'target', 'edge_type', 'r', 'q']), stats_out


def edges_to_tensor(edges, nodes):
    idx = {n: i for i, n in enumerate(nodes)}
    e = edges[edges.source.isin(idx) & edges.target.isin(idx)]
    src = [idx[s] for s in e.source] + [idx[t] for t in e.target]
    dst = [idx[t] for t in e.target] + [idx[s] for s in e.source]
    w = list(np.abs(e.r.values)) * 2
    if not src:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0, 1))
    return torch.tensor([src, dst], dtype=torch.long), torch.tensor(w, dtype=torch.float32).view(-1, 1)


def rewire(edges, rng, n_swaps_factor=10):
    """Degree-preserving double-edge swaps within each edge type (the permutation null)."""
    out = []
    for kind, g in edges.groupby('edge_type'):
        s, t = list(g.source), list(g.target)
        existing = set(zip(s, t)) | set(zip(t, s))
        m = len(s)
        for _ in range(n_swaps_factor * m):
            if m < 2:
                break
            i, j = rng.integers(0, m, 2)
            if i == j:
                continue
            a, b, c, d = s[i], t[i], s[j], t[j]
            if kind == 'miRNA-mRNA':
                new1, new2 = (a, d), (c, b)
            else:
                new1, new2 = (a, d), (c, b)
            if new1[0] == new1[1] or new2[0] == new2[1] or new1 in existing or new2 in existing:
                continue
            existing -= {(a, b), (b, a), (c, d), (d, c)}
            existing |= {new1, new1[::-1], new2, new2[::-1]}
            s[i], t[i], s[j], t[j] = a, d, c, b
        gg = g.copy()
        gg['source'], gg['target'] = s, t
        out.append(gg)
    return pd.concat(out, ignore_index=True)


class MultiNetworkGAT(nn.Module):
    """Same architecture as notebooks/05_gat_baseline.py (2-layer GATConv per class-specific
    network + expression bypass + per-network branch heads), parameterised by class count.
    use_expression=False gives the graph-only variant."""

    def __init__(self, num_nodes, node_types, nets, n_classes, num_heads=2, hidden_size=32,
                 dropout=0.2, node_embedding_size=16, modality_embedding_size=4,
                 expression_hidden_size=64, use_expression=True):
        super().__init__()
        self.network_names = list(nets.keys())
        self.num_nodes = num_nodes
        self._dropout = dropout
        self.n_classes = n_classes
        self.use_expression = use_expression
        self._edge_cache = {}
        self.register_buffer('node_ids', torch.arange(num_nodes))
        self.register_buffer('node_types', node_types.long())
        for k, (ei, ea) in nets.items():
            self.register_buffer(f'{k}_ei', ei)
            self.register_buffer(f'{k}_ea', ea)
        in_ch = 1 + node_embedding_size + modality_embedding_size
        self.node_embedding = nn.Embedding(num_nodes, node_embedding_size)
        self.modality_embedding = nn.Embedding(2, modality_embedding_size)
        self.conv1 = GATConv(in_ch, hidden_size, heads=num_heads, concat=True, dropout=dropout, edge_dim=1)
        self.conv2 = GATConv(hidden_size * num_heads, hidden_size, heads=1, concat=False, dropout=dropout, edge_dim=1)
        self.branch_classifier = nn.Linear(hidden_size * 2, n_classes)
        self.expression_head = nn.Sequential(nn.LayerNorm(num_nodes), nn.Linear(num_nodes, expression_hidden_size),
                                             nn.ELU(), nn.Dropout(dropout))
        comb = (expression_hidden_size if use_expression else 0) + len(self.network_names) * (hidden_size * 2 + n_classes)
        self.classifier = nn.Sequential(nn.Linear(comb, 96), nn.ELU(), nn.Dropout(dropout), nn.Linear(96, n_classes))

    def _batched(self, k, bs):
        key = (k, bs)
        if key not in self._edge_cache:
            ei, ea = getattr(self, f'{k}_ei'), getattr(self, f'{k}_ea')
            off = torch.arange(bs).view(-1, 1, 1) * self.num_nodes
            bei = (ei.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1)
            self._edge_cache[key] = (bei, ea.repeat(bs, 1))
        return self._edge_cache[key]

    def forward(self, x):
        bs = x.size(0)
        xin = torch.cat([x.unsqueeze(-1),
                         self.node_embedding(self.node_ids).unsqueeze(0).expand(bs, -1, -1),
                         self.modality_embedding(self.node_types).unsqueeze(0).expand(bs, -1, -1)], 2)
        parts, blog = [], []
        for k in self.network_names:
            ei, ea = self._batched(k, bs)
            h = self.conv1(xin.reshape(bs * self.num_nodes, -1), ei, edge_attr=ea)
            h = F.dropout(F.elu(h), p=self._dropout, training=self.training)
            h = F.elu(self.conv2(h, ei, edge_attr=ea)).reshape(bs, self.num_nodes, -1)
            pooled = torch.cat([h.mean(1), h.max(1).values], 1)
            parts.append(pooled)
            blog.append(self.branch_classifier(pooled))
        if self.use_expression:
            parts = [self.expression_head(x)] + parts
        return self.classifier(torch.cat(parts + blog, 1)), torch.stack(blog, 1)


def metrics(y, prob):
    pred = prob.argmax(1)
    out = {'macro_f1': f1_score(y, pred, average='macro'), 'accuracy': accuracy_score(y, pred)}
    try:
        out['auroc'] = roc_auc_score(y, prob[:, 1]) if prob.shape[1] == 2 else roc_auc_score(y, prob, multi_class='ovr')
    except ValueError:
        out['auroc'] = float('nan')
    return out


def train_gat(Xtr, ytr, Xva, yva, Xte, nets, node_types, n_classes, seed, use_expression=True):
    set_seed(seed)
    model = MultiNetworkGAT(Xtr.shape[1], node_types, nets, n_classes,
                            num_heads=GAT_PARAMS['num_heads'], hidden_size=GAT_PARAMS['hidden_size'],
                            dropout=GAT_PARAMS['dropout'], use_expression=use_expression)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    w = len(ytr) / (n_classes * np.maximum(counts, 1))
    w = torch.tensor(w / w.mean(), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=GAT_PARAMS['lr'], weight_decay=1e-4)
    Xtr_t, ytr_t = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr)
    Xva_t, Xte_t = torch.tensor(Xva, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
    best, best_f1, bad = copy.deepcopy(model.state_dict()), -1, 0
    rng = np.random.default_rng(seed)
    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = rng.permutation(len(ytr))
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i:i + BATCH_SIZE]
            opt.zero_grad()
            logits, bl = model(Xtr_t[b])
            yb = ytr_t[b]
            loss = F.cross_entropy(logits, yb, weight=w) + BRANCH_LOSS_WEIGHT * F.cross_entropy(
                bl.reshape(-1, n_classes), yb.unsqueeze(1).expand(-1, bl.size(1)).reshape(-1), weight=w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pv = F.softmax(model(Xva_t)[0], 1).numpy()
        f1 = f1_score(yva, pv.argmax(1), average='macro')
        if f1 > best_f1 + 1e-8:
            best_f1, best, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best)
    model.eval()
    with torch.no_grad():
        return F.softmax(model(Xte_t)[0], 1).numpy()


def cluster_bootstrap_ci(y, prob, groups, n_boot=2000, seed=0, key='macro_f1'):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.where(groups == g)[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        ii = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[ii])) < 2:
            continue
        vals.append(metrics(y[ii], prob[ii])[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
