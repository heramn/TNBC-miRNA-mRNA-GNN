#!/usr/bin/env python3
"""Cross-modal edge / miRNA-target overlap for the current top-150 networks.

Counts how many of the cross-modal (miRNA-mRNA) edges in
results/networks/{normal,primary,metastatic}_edges_top150.csv are predicted
miRDB v6.0 targets, using the per-edge miRDB_validated flag that
03_network_construction.py writes from data/raw/miRDB_v6.0_prediction_result.txt.gz.

The earlier ~7% miRDB / ~3% miRTarBase figures came from a shipped table
(edge_validation_db.csv) that described an older edge set and cannot be regenerated
(the miRTarBase annotation is not in the repo), so it was removed.

Usage:  python scripts/summarize_edge_validation.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import pandas as pd

NETWORKS = Path(__file__).resolve().parent.parent / 'results' / 'networks'


def main() -> None:
    rows = []
    for tissue in ['normal', 'primary', 'metastatic']:
        path = NETWORKS / f'{tissue}_edges_top150.csv'
        if not path.exists():
            sys.exit(f'{path} not found; run notebooks/03_network_construction.py first.')
        df = pd.read_csv(path)
        cross = df[df.source_type != df.target_type]
        rows.append((tissue, len(cross), int(cross.miRDB_validated.sum())))
    total = sum(n for _, n, _ in rows)
    hits = sum(h for _, _, h in rows)
    if total == 0:
        print('No cross-modal edges (miRNA-only build).')
        return
    print(f'Cross-modal miRNA-mRNA edges checked: {total}')
    print(f'Predicted miRDB v6.0 targets: {hits} ({100 * hits / total:.1f}%)')
    for tissue, n, h in rows:
        print(f'  {tissue:<10} n={n:4d}  miRDB {h} ({100 * h / n:.1f}%)')


if __name__ == '__main__':
    main()
