"""Feature mappings shared by every study script (study-final).

miRNA: NanoString probe (GPL16231 Accession, nmiR...) -> MIMAT accession(s) from the platform
annotation -> miRBase v22 mature name (mature.fa). MIMAT accessions are stable across miRBase
versions, so this fixes one arm per probe (no -5p/-3p guessing). Merged probes ("a+b") map to
every member MIMAT; downstream they are handled by SUMMING member arms (TCGA, linear scale) and by
the UNION of member targets (miRDB, max score per gene).

mRNA: panel symbol -> HGNC approved symbol (via previous/alias symbols); TCGA HiSeqV2 symbols are
mapped the same way, so legacy names such as CDC2 -> CDK1 and PCTK1 -> CDK16 are recovered.
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / 'data/raw'
EXT = Path.home() / '.capy/work/ext'
FROZEN = REPO / 'study/frozen'


def geo_mirna_probes():
    """Endogenous1 probes of the raw miRNA file: DataFrame[name, accession]."""
    df = pd.read_csv(RAW / 'GSE45498_raw_data.txt.gz', sep='\t', usecols=[0, 1, 2])
    df.columns = ['code_class', 'name', 'accession']
    return df[df.code_class == 'Endogenous1'][['name', 'accession']].reset_index(drop=True)


def geo_mrna_panel():
    df = pd.read_csv(RAW / 'GSE45498_mRNA_non-normalized_data.txt.gz', sep='\t', usecols=[0, 1, 2])
    df.columns = ['code_class', 'name', 'accession']
    return df[df.code_class == 'Endogenous'][['name', 'accession']].reset_index(drop=True)


def gpl16231(path=None):
    path = path or (FROZEN / 'GPL16231_annotation.tsv')
    if str(path).endswith('.tsv'):
        return pd.read_csv(path, sep='\t', dtype=str).fillna('')
    rows, header, on = [], None, False
    with open(path, encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.rstrip('\r\n')
            if line.startswith('!platform_table_begin'):
                on = True
                continue
            if line.startswith('!platform_table_end'):
                break
            if on:
                f = line.split('\t')
                if header is None:
                    header = f
                else:
                    rows.append(f + [''] * (len(header) - len(f)))
    return pd.DataFrame(rows, columns=header)


def mimat_names(path=EXT / 'mirbase/mature.fa'):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.startswith('>hsa-'):
                p = line[1:].split()
                out[p[1]] = p[0]
    return out


def build_mirna_probe_table(gpl_path, mature_path):
    """One row per (probe, member MIMAT): probe name, accession, MIMAT, miRBase v22 name, merged flag."""
    probes = geo_mirna_probes()
    gpl = gpl16231(gpl_path)
    acc2mimat = dict(zip(gpl['Accession'].str.strip(), gpl['miRNA_NAME'].str.strip()))
    names = mimat_names(mature_path)
    rows = []
    for name, acc in probes.itertuples(index=False):
        mimats = [m.strip() for m in acc2mimat.get(acc, '').split('+') if m.strip()]
        for m in mimats or ['']:
            rows.append({'probe': name, 'accession': acc, 'mimat': m, 'mirbase_v22': names.get(m, ''),
                         'n_members': len(mimats), 'merged_probe': len(mimats) > 1})
    return pd.DataFrame(rows)


def mirna_probe_table():
    return pd.read_csv(FROZEN / 'mirna_probe_map.csv', dtype={'mirbase_v22': str}).fillna({'mirbase_v22': '', 'mimat': ''})


def hgnc_approved(path=EXT / 'hgnc_complete_set.txt'):
    """{any symbol (approved, previous, alias): approved symbol}; approved symbols win, then
    previous symbols, then aliases."""
    h = pd.read_csv(path, sep='\t', usecols=['symbol', 'prev_symbol', 'alias_symbol'], dtype=str).fillna('')
    out = {}
    for col in ('prev_symbol', 'alias_symbol'):
        for sym, other in zip(h.symbol, h[col]):
            for o in other.split('|'):
                if o:
                    out.setdefault(o, sym)
    for sym in h.symbol:
        out[sym] = sym
    return out


def build_mrna_symbol_table(tcga_symbols, hgnc_path=EXT / 'hgnc_complete_set.txt'):
    """Panel gene -> TCGA HiSeqV2 column: exact symbol, else via the HGNC approved symbol of the
    panel RefSeq accession (falling back to previous/alias symbols)."""
    panel = geo_mrna_panel()
    appr = hgnc_approved(hgnc_path)
    h = pd.read_csv(hgnc_path, sep='\t', usecols=['symbol', 'refseq_accession'], dtype=str).fillna('')
    ref2appr = {r: s for s, rr in zip(h.symbol, h.refseq_accession) for r in rr.split('|') if r}
    tset = set(tcga_symbols)
    tcga_by_appr = {}
    for t in tcga_symbols:
        tcga_by_appr.setdefault(appr.get(t, t), t)
    rows = []
    for name, acc in panel.itertuples(index=False):
        a = ref2appr.get(acc.split('.')[0], appr.get(name, name))
        if name in tset:
            hit, how = name, 'exact'
        elif a in tcga_by_appr:
            hit, how = tcga_by_appr[a], 'hgnc_alias'
        else:
            hit, how = '', 'unmapped'
        rows.append({'gene': name, 'refseq': acc.split('.')[0], 'hgnc_approved': a, 'tcga_symbol': hit, 'how': how})
    return pd.DataFrame(rows)


def mrna_symbol_table():
    return pd.read_csv(FROZEN / 'mrna_symbol_map.csv', dtype=str).fillna('')


def mirdb_pairs(min_score=70):
    """miRDB v6.0 score > min_score pairs on the GEO panels: DataFrame[probe, gene, score].
    Panel genes are matched by RefSeq accession, probes by their member miRBase v22 names
    (merged probes: union of member targets, max score per gene)."""
    panel = geo_mrna_panel()
    ref2gene = dict(zip(panel.accession.str.split('.').str[0], panel.name))
    m = pd.read_csv(RAW / 'miRDB_v6.0_prediction_result.txt.gz', sep='\t', header=None,
                    names=['mirna', 'refseq', 'score'])
    m = m[m.mirna.str.startswith('hsa-') & (m.score > min_score)].copy()
    m['gene'] = m.refseq.astype(str).str.split('.').str[0].map(ref2gene)
    m = m.dropna(subset=['gene'])
    pt = mirna_probe_table()
    pt = pt[pt.mirbase_v22 != '']
    j = m.merge(pt[['probe', 'mirbase_v22']], left_on='mirna', right_on='mirbase_v22')
    return j.groupby(['probe', 'gene'], as_index=False)['score'].max()
