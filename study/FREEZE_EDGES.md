# Frozen TCGA edge list (study-final, design B)

Frozen at **2026-11-22T23:09Z**, before any GEO run of the final study. This file, the code in
\`study/01_tcga_edges.py\`, \`study/mappings.py\`, \`study/edges.py\` and everything in \`study/frozen/\`
are committed together and tagged \`tcga-edges-frozen-v1\`. Later commits must not change these
files; \`study/tests/test_frozen.py\` checks the sha256 values in \`tcga_edges_info.json\`.

## What was frozen

| Item | Value |
|---|---|
| Discovery cohort | TCGA-BRCA IHC-TNBC primary tumours with miRNA-seq + RNA-seq on the same barcode: 102; with an ABSOLUTE purity call: **94** (used) |
| Statistic | partial Spearman, adjusted for ABSOLUTE purity (rank residuals, t-test with n − 3 df) |
| Universe | 267 GEO miRNA probes present in TCGA (≥ 90% non-missing) × 236 GEO panel mRNAs = 63,012 pairs |
| Selection | negative pairs with BH q < 0.05 over the universe: **1,434**, ranked by \|r\| (\`tcga_tumor_edges_ranked.csv\`) |
| Per-fold density rule | K_f = number of miRDB v6.0 (score > 70) pairs on the fold's TCGA-mappable DE subset; the TCGA arm takes the top-K_f ranked pairs inside that subset |
| Edge semantics | undirected, binary, one edge set used in all three tissue graphs; described as "anti-correlated pairs consistent with repression", not targets |
| Sensitivity sets (not primary) | ESTIMATE-derived purity on all 102 tumours (2,167 pairs); TCGA normals, 76 (70 from non-TNBC patients), plain Spearman (4,300 pairs) |

Mapping (\`mirna_probe_map.csv\`, \`mrna_symbol_map.csv\`): NanoString probe → MIMAT from the GPL16231
annotation → miRBase v22 name, one arm per probe. 8 merged probes (e.g. miR-20a+miR-20b) are kept
by **summing member arms** on the linear RPM scale in TCGA and by the **union of member targets** in
miRDB. 32 probes have no v22 name (retired entries such as miR-1274a). mRNA: RefSeq → HGNC approved
symbol, which recovers CDC2 → CDK1 and PCTK1 → CDK16; all 236 panel genes map.

Reference count on the existing 190-sample build's DE nodes (names only, no GEO values or labels
read): 74 of 93 DE miRNAs and 157 of 157 DE mRNAs are TCGA-mappable (B's arm-guessing mapper kept
66 + 153). On that subset miRDB gives K = 394 pairs and the TCGA list has 457 candidates, so the
top-394 cut is at |r| = 0.318; 35 of the 394 are also miRDB pairs. miRDB on the full 250 nodes
gives 419 pairs (A's heuristic mapper gave 574 because it attached both arms to unsuffixed probes).

## Pre-stated expectation

The expected result for the TCGA arm on GEO is **null**: no difference from density-matched miRDB
edges and no gain over no-cross or rewired edges beyond the ±0.03 macro-F1 margin. The top TCGA
pairs (miR-141 / miR-19a / miR-200 family against TIMP2, TIMP3, PDGFRB) sit on an epithelial vs
stromal axis that survives purity adjustment, so many are likely cell-mix rather than targeting.

## Exploration timeline (disclosed; nothing here is pre-registered)

| When (UTC) | What was seen before this freeze |
|---|---|
| 2026-11-22 12:40 | \`repro-verify-2026-11\` (ce95634): mRNA misalignment fix, corrected 230-sample build |
| 2026-11-22 17:34–18:42 | \`fix-multiomic-B\`: TCGA true-pair networks; **"hybrid" GEO GAT with TCGA cross edges: macro-F1 0.875 vs 0.868 no edges vs 0.867 GEO edges, identical 20-seed ensemble F1 0.905, rewired p = 0.65** (different architecture, unadjusted Pearson, 150 edges). This is a prior look at almost the same comparison. |
| 2026-11-22 17:37–18:11 | \`fix-multiomic-A\`: pairing-free GAT on the 190/40 split; miRDB vs no-cross +0.009 (miRNA) / +0.002 (mRNA), rewired p = 0.34 / 0.52 |
| 2026-11-22 ~23:00 | this study: one timing run each of A's joint and miRNA-only GAT on A's 190/40 split (seed 0), outputs seen (macro-F1 0.86 / 0.76 and 0.80). No other GEO model was run. |
| 2026-11-22 23:09 | TCGA edge list frozen (this commit) |
