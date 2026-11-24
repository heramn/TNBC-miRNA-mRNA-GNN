"""Edge-forcing GAT variants for phase 3 (post-hoc, added after the frozen study).

VNet extends the frozen study/model.py Net without changing it:

* xfirst   - the first GAT layer passes messages along cross (miRNA-mRNA) edges only, so a latent hub node
             first aggregates its observed targets/regulators; the second layer runs on the full tissue
             graph (co-expression + cross edges). Readout pools miRNA nodes and mRNA nodes separately.
* gated    - the frozen two-layer GAT plus a parallel cross-edge-only GAT path added through a learned
             per-node-type sigmoid gate (gated residual). Readout pools the two modalities separately.
* selfdrop - the frozen network; during training each observed node is hidden (value and observed flag set
             to 0, in the graph and in the expression bypass) with probability q, so predictions have to
             draw on neighbours.
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'study'))
import config as C  # noqa: E402
import model as M  # noqa: E402

VARIANTS = ('xfirst', 'gated', 'selfdrop')


class VNet(M.Net):
    def __init__(self, n_nodes, node_types, nets, cfg, variant, vparam, cross_idx, emb=16, mod_emb=4, expr_hidden=64):
        super().__init__(n_nodes, node_types, nets, cfg, 'gat', emb, mod_emb, expr_hidden)
        assert variant in VARIANTS, variant
        self.variant, self.q = variant, vparam
        hid, heads = cfg['hidden'], cfg['heads']
        src = [a for a, b in cross_idx] + [b for a, b in cross_idx] + list(range(n_nodes))
        dst = [b for a, b in cross_idx] + [a for a, b in cross_idx] + list(range(n_nodes))
        ea = [[1.0, 1.0]] * (2 * len(cross_idx)) + [[0.0, 0.0]] * n_nodes   # explicit self-loops, so an empty cross set is valid
        self.register_buffer('xei', torch.tensor([src, dst], dtype=torch.long))
        self.register_buffer('xea', torch.tensor(ea, dtype=torch.float32))
        self.split = variant in ('xfirst', 'gated')
        cin = 2 + emb + mod_emb
        if variant == 'gated':
            self.xconv = GATConv(cin, hid, heads=1, concat=False, dropout=self.dropout, edge_dim=2)
            self.gate = nn.Parameter(torch.zeros(2))
        if self.split:
            self.branch = nn.Linear(4 * hid, 3)
            parts = self.T * (4 * hid + 3) + expr_hidden
            self.head = nn.Sequential(nn.Linear(parts, 96), nn.ELU(), nn.Dropout(self.dropout), nn.Linear(96, 3))
        self._xcache = {}

    def _xedges(self, b):
        if b not in self._xcache:
            off = (torch.arange(b) * self.n).view(-1, 1, 1)
            self._xcache[b] = ((self.xei.unsqueeze(0) + off).permute(1, 0, 2).reshape(2, -1), self.xea.repeat(b, 1))
        return self._xcache[b]

    def _graph(self, node_in, t):
        if self.variant == 'selfdrop':
            return super()._graph(node_in, t)
        b = node_in.size(0)
        ei, ea = self._edges(t, b)
        xei, xea = self._xedges(b)
        h0 = node_in.reshape(b * self.n, -1)
        if self.variant == 'xfirst':
            h = F.elu(self.conv1(h0, xei, edge_attr=xea))
        else:
            h = F.elu(self.conv1(h0, ei, edge_attr=ea))
        h = F.dropout(h, self.dropout, self.training)
        h = F.elu(self.conv2(h, ei, edge_attr=ea))
        if self.variant == 'gated':
            hx = F.elu(self.xconv(h0, xei, edge_attr=xea))
            g = torch.sigmoid(self.gate)[self.node_types].repeat(b).unsqueeze(1)
            h = h + g * hx
        h = h.reshape(b, self.n, -1)
        mi = self.node_types == 0
        out = []
        for sel in (mi, ~mi):
            hs = h[:, sel] if bool(sel.any()) else torch.zeros_like(h[:, :1])
            out += [hs.mean(1), hs.max(1).values]
        return torch.cat(out, 1)

    def forward(self, x, mask):
        if self.variant == 'selfdrop' and self.training and self.q:
            keep = (torch.rand_like(mask) >= self.q).float()
            x, mask = x * keep, mask * keep
        return super().forward(x, mask)


def fit_predict_v(Xf, Mf, yf, Xv, Mv, yv, Xt, Mt, n_nodes, node_types, nets, cfg, seed, variant, vparam, cross_idx):
    """The frozen study/model.py fit_predict loop (same optimiser, batching, early stopping and class
    weights), with VNet in place of Net."""
    torch.set_num_threads(1)
    M.set_seed(seed)
    model = VNet(n_nodes, node_types, nets, cfg, variant, vparam, cross_idx)
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
