#!/usr/bin/env python3
"""Parse the miRTarBase 9.0 human linkset (CyTargetLinker XGMML, Zenodo 10.5281/zenodo.13789210) into
GEO probe x panel gene pairs. Strong evidence = reporter assay, western blot or qPCR (miRTarBase's
own strong-evidence methods). Usage: 04_mirtarbase_pairs.py <xgmml>"""
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mappings as mp

xg = Path(sys.argv[1])
ent2sym, edges = {}, []
node, sym = None, None
cur = None
STRONG = re.compile(r'reporter|western|qrt-pcr|qpcr|rt-pcr', re.I)
with open(xg) as fh:
    for line in fh:
        if '<node ' in line:
            node = re.search(r'id="([^"]+)"', line).group(1)
        elif node and 'name="Target Gene" value=' in line:
            ent2sym[node] = re.search(r'value="([^"]*)"', line).group(1)
        elif '<edge ' in line:
            m = re.search(r'source="([^"]+)" target="([^"]+)"', line)
            cur = [m.group(1), m.group(2), '', '']
        elif cur is not None and 'name="Experiments"' in line:
            cur[2] = re.search(r'value="([^"]*)"', line).group(1)
        elif cur is not None and 'name="Support Type"' in line:
            cur[3] = re.search(r'value="([^"]*)"', line).group(1)
        elif cur is not None and '</edge>' in line:
            edges.append(cur)
            cur = None
e = pd.DataFrame(edges, columns=['mirna', 'entrez', 'experiments', 'support'])
e['symbol'] = e.entrez.map(ent2sym)
h = pd.read_csv(mp.EXT / 'hgnc_complete_set.txt', sep='\t', usecols=['symbol', 'entrez_id'], dtype=str).dropna()
ent2appr = dict(zip(h.entrez_id, h.symbol))
sm = pd.read_csv(mp.FROZEN / 'mrna_symbol_map.csv', dtype=str)
appr2gene = dict(zip(sm.hgnc_approved, sm.gene))
e['gene'] = e.entrez.map(ent2appr).map(appr2gene)
e = e.dropna(subset=['gene'])
e['strong'] = e.experiments.str.contains(STRONG) | (e.support == 'Functional MTI')
pt = mp.mirna_probe_table()
pt = pt[pt.mirbase_v22 != '']
j = e.merge(pt[['probe', 'mirbase_v22']], left_on='mirna', right_on='mirbase_v22')
out = j.groupby(['probe', 'gene']).agg(mirtarbase_strong=('strong', 'any')).reset_index()
out['mirtarbase_any'] = True
dst = mp.REPO / 'results/study_final/mirtarbase_pairs.csv'
out.to_csv(dst, index=False)
print(len(edges), 'MTIs parsed;', len(e), 'on panel genes;', len(out), 'probe-gene pairs;', int(out.mirtarbase_strong.sum()), 'strong')
