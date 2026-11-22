"""Cross-modal edge sets for the GEO study (frozen with the TCGA edge list).

Every arm is undirected and binary (edge weight 1, plus an edge-type indicator added by the
graph builder). Arms on the TCGA-mappable subset of a fold's DE nodes:
  mirdb     miRDB v6.0 score > 70 pairs inside the subset                        (PRIMARY)
  tcga      top-K_f TCGA tumour anti-correlated pairs inside the subset,
            K_f = number of mirdb pairs inside the same subset                     (SECONDARY)
  nocross   no cross-modal edges
  rewired   degree-preserving bipartite rewiring of the mirdb pairs (new rewiring per run)
Separate row: mirdb on the fold's full DE node set.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FROZEN = Path(__file__).resolve().parent / 'frozen'


def tcga_ranked(kind='tumor'):
    f = {'tumor': 'tcga_tumor_edges_ranked.csv',
         'normal': 'tcga_normal_edges_ranked_sensitivity.csv',
         'estimate': 'tcga_tumor_edges_ranked_estimate_sensitivity.csv'}[kind]
    return pd.read_csv(FROZEN / f)


def tcga_mappable(kind='tumor'):
    """GEO probe / gene names that exist (after expression filters) in the TCGA matrices."""
    t = pd.read_csv(FROZEN / 'tcga_mappable_nodes.csv')
    return set(t[t.modality == 'miRNA'].node), set(t[t.modality == 'mRNA'].node)


def pairs_in(df, mi_nodes, mr_nodes):
    return df[df.probe.isin(set(mi_nodes)) & df.gene.isin(set(mr_nodes))]


def select_tcga(mi_nodes, mr_nodes, k, kind='tumor'):
    """Top-k ranked TCGA anti-correlated pairs inside the node subset (fewer if fewer exist)."""
    r = pairs_in(tcga_ranked(kind), mi_nodes, mr_nodes).sort_values('rank')
    return list(zip(r.probe[:k], r.gene[:k]))


def rewire_bipartite(pairs, rng, swaps_per_edge=20):
    """Degree-preserving rewiring of a bipartite edge list by double-edge swaps."""
    e = [list(p) for p in pairs]
    present = set(map(tuple, e))
    m = len(e)
    if m < 2:
        return [tuple(x) for x in e]
    for _ in range(swaps_per_edge * m):
        i, j = rng.integers(m), rng.integers(m)
        (a, b), (c, d) = e[i], e[j]
        if a == c or b == d or (a, d) in present or (c, b) in present:
            continue
        present -= {(a, b), (c, d)}
        present |= {(a, d), (c, b)}
        e[i], e[j] = [a, d], [c, b]
    return [tuple(x) for x in e]
