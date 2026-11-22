"""Pairing-free multi-omic GAT (and GCN / MLP-only / graph-only variants), study-final.

Same architecture as notebooks/41 (2 graph-conv layers shared across the three tissue networks,
mean+max pooling per network, a shared per-network branch classifier, an expression bypass over
[masked values, observed flags], MLP head). Changes, all from the stress test: edge features are
[weight, is_cross] (edge-type indicator; cross edges are binary, weight 1), and the three tissue
networks are run separately in chunks of CHUNK examples (identical computation; smaller graphs per
call are about 2x faster on CPU).
"""
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

import config as C

CHUNK = 8


class Net(nn.Module):
    def __init__(self, n_nodes, node_types, nets, cfg, kind='gat', emb=16, mod_emb=4, expr_hidden=64):
        super().__init__()
        self.n, self.kind, self.T = n_nodes, kind, len(nets)
        self.dropout = cfg['dropout']
        hid, heads = cfg['hidden'], cfg['heads']
        for t, es in enumerate(nets.values()):
            src, dst, ea = [], [], []
            for a, b, w, x in es:
                src += [a, b]
                dst += [b, a]
                ea += [[w, x], [w, x]]
            self.register_buffer(f'ei{t}', torch.tensor([src, dst], dtype=torch.long).reshape(2, -1))
            self.register_buffer(f'ea{t}', torch.tensor(ea, dtype=torch.float32).reshape(-1, 2))
        self.register_buffer('node_ids', torch.arange(n_nodes))
        self.register_buffer('node_types', torch.as_tensor(node_types, dtype=torch.long))
        parts = 0
        if kind != 'mlp':
            self.node_emb = nn.Embedding(n_nodes, emb)
            self.mod_emb = nn.Embedding(2, mod_emb)
            cin = 2 + emb + mod_emb
            if kind == 'gcn':
                self.conv1 = GCNConv(cin, hid * heads)
                self.conv2 = GCNConv(hid * heads, hid)
            else:
                self.conv1 = GATConv(cin, hid, heads=heads, concat=True, dropout=self.dropout, edge_dim=2)
                self.conv2 = GATConv(hid * heads, hid, heads=1, concat=False, dropout=self.dropout, edge_dim=2)
            self.branch = nn.Linear(2 * hid, 3)
            parts += self.T * (2 * hid + 3)
        if kind != 'graphonly':
            self.expr = nn.Sequential(nn.LayerNorm(2 * n_nodes), nn.Linear(2 * n_nodes, expr_hidden), nn.ELU(), nn.Dropout(self.dropout))
            parts += expr_hidden
        self.head = nn.Sequential(nn.Linear(parts, 96), nn.ELU(), nn.Dropout(self.dropout), nn.Linear(96, 3))
        self._cache = {}

    def _edges(self, t, b):
        if (t, b) not in self._cache:
            ei, ea = getattr(self, f'ei{t}'), getattr(self, f'ea{t}')
            off = (torch.arange(b) * self.n).view(-1, 1, 1)
            self._cache[(t, b)] = ((ei.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1), ea.repeat(b, 1))
        return self._cache[(t, b)]

    def _graph(self, node_in, t):
        b = node_in.size(0)
        ei, ea = self._edges(t, b)
        h = node_in.reshape(b * self.n, -1)
        if self.kind == 'gcn':
            h = F.elu(self.conv1(h, ei, edge_weight=ea[:, 0]))
            h = F.dropout(h, self.dropout, self.training)
            h = F.elu(self.conv2(h, ei, edge_weight=ea[:, 0]))
        else:
            h = F.elu(self.conv1(h, ei, edge_attr=ea))
            h = F.dropout(h, self.dropout, self.training)
            h = F.elu(self.conv2(h, ei, edge_attr=ea))
        h = h.reshape(b, self.n, -1)
        return torch.cat([h.mean(1), h.max(1).values], 1)

    def forward(self, x, mask):
        b = x.size(0)
        xv = x * mask
        parts = []
        if self.kind != 'graphonly':
            parts.append(self.expr(torch.cat([xv, mask], 1)))
        if self.kind != 'mlp':
            node_in = torch.cat([xv.unsqueeze(-1), mask.unsqueeze(-1),
                                 self.node_emb(self.node_ids).expand(b, -1, -1),
                                 self.mod_emb(self.node_types).expand(b, -1, -1)], 2)
            for t in range(self.T):
                pooled = torch.cat([self._graph(node_in[i:i + CHUNK], t) for i in range(0, b, CHUNK)], 0)
                parts += [pooled, self.branch(pooled)]
        return self.head(torch.cat(parts, 1))


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fit_predict(Xf, Mf, yf, Xv, Mv, yv, Xt, Mt, n_nodes, node_types, nets, cfg, kind, seed):
    """Train on the inner-fit split with early stopping on class-weighted inner-validation
    cross-entropy; return (best validation loss, epochs run, test probabilities)."""
    torch.set_num_threads(1)
    set_seed(seed)
    model = Net(n_nodes, node_types, nets, cfg, kind)
    counts = np.bincount(yf, minlength=3).astype(float)
    w = torch.tensor(len(yf) / (3 * np.maximum(counts, 1)), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    Xf, Mf, Xv, Mv, Xt, Mt = (torch.as_tensor(a) for a in (Xf, Mf, Xv, Mv, Xt, Mt))
    yf_t, yv_t = torch.as_tensor(yf), torch.as_tensor(yv)
    rng = np.random.default_rng(seed)
    best, best_state, bad, ep = np.inf, None, 0, 0
    for ep in range(1, C.MAX_EPOCHS + 1):
        model.train()
        perm = rng.permutation(len(yf))
        for i in range(0, len(perm), C.BATCH):
            bi = torch.as_tensor(perm[i:i + C.BATCH])
            opt.zero_grad()
            F.cross_entropy(model(Xf[bi], Mf[bi]), yf_t[bi], weight=w).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(model(Xv, Mv), yv_t, weight=w))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        p = torch.softmax(model(Xt, Mt), 1).numpy()
    return best, ep, p
