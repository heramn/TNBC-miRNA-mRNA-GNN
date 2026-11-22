# Reproduction evidence

Files supporting REPRO_REPORT.md. None of them are read by the pipeline.

- asis_235_sample_audit.csv: for each of the 235 samples in the as-is build, the mRNA GSM it
  was meant to use, the one it actually used, and whether the tissue and patient match.
- inferred_mirna_mrna_pairs_order_alignment.csv: stable miRNA-to-mRNA pairs inferred by
  monotone order alignment. These are evidence for an order link, not a patient key.
- dp_align.out: z-scores of the monotone alignment against a permutation null, by tissue and
  gap penalty.
- asis_network_ablation_graphml.csv, asis_network_ablation_permutations.csv: notebook 28 on
  the as-is build (code at d31566f, 1,000 permutations).
- mirna_only_network_ablation_graphml.csv, mirna_only_network_ablation_permutations.csv: the
  same for the miRNA-only variant.
- requirements.lock.txt: full pip freeze of the environment used.
