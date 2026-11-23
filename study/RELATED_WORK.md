# Related work notes (for the paper's related-work section)

Each entry covers what the method does, what it needs, and how it differs from this study. Only
claims supported by the original papers are listed; check the final wording against them before
submission.

**MOGONET.** Wang et al., *Nat Commun* 12:3445, 2021.
- *Method:* one GCN per omics type over a patient-similarity graph, plus a cross-omics "view
  correlation discovery network" over the per-omics label distributions, for supervised classification.
- *Needs:* every patient profiled on every omics type (paired samples).
- *Difference:* GSE45498 has no miRNA–mRNA patient key, so MOGONET's fusion step cannot be applied
  as designed. Our graph is over molecules, not patients, and the examples are unpaired.

**SUPREME.** Kesimoglu & Bozdag, *NAR Genomics and Bioinformatics*, 2023.
- *Method:* GCNs on patient-similarity networks per datatype, with embeddings concatenated with
  features for subtype prediction.
- *Needs:* paired samples across datatypes.
- *Difference:* the same as for MOGONET. We use it only as conceptual context.

**P-NET.** Elmarakeby et al., *Nature* 598:348, 2021.
- *Method:* a sparse, biologically informed network whose connections follow gene → pathway
  (Reactome) membership, applied to prostate cancer genomics.
- *Difference:* like P-NET, our model injects prior biological structure, but as message-passing
  edges between molecules rather than as a sparse layer hierarchy.
- *Shared caution:* P-NET's interpretability claims rest on the prior. We make no mechanism claims,
  and test whether the prior helps against rewired and edge-free controls.

**MOFA / MOFA+.** Argelaguet et al., *Mol Syst Biol* 14:e8124, 2018; *Genome Biol* 21:111, 2020.
- *Method:* unsupervised group factor analysis that tolerates missing views per sample.
- *Needs:* shared factors require at least partially matched samples to link views.
- *Difference:* it is unsupervised. A fully unpaired design like ours gives MOFA no shared samples
  to anchor factors across modalities.

**scGLUE.** Cao & Gao, *Nat Biotechnol* 40:1458, 2022.
- *Method:* unpaired ("diagonal") single-cell multi-omics integration. Per-modality VAEs are tied by
  a prior-knowledge guidance graph over features, and their cell embeddings are aligned adversarially.
- *Closest to ours:* it is the nearest conceptual relative because it uses no pairing and a prior
  feature graph.
- *Differences:* ours is supervised, has no reconstruction or alignment loss, and the unobserved
  modality's nodes are latent message-passing nodes, not imputations. So we do not call our model
  "GLUE-style".
- *Not run here:* an scGLUE baseline (see the deviations in \`STUDY_PLAN.md\`).

**Prior-graph GNN caution.** Results showing that rewired or random graphs often match
"biological" graphs in omics GNNs motivate our degree-preserving rewired control and the
edge-free / MLP-only arms. Cite the specific benchmark papers the authors rely on. Attention weights
are not importance (Jain & Wallace, NAACL 2019), so no attention-based mechanism claims are made.

**Evaluation methodology.**
- Nadeau & Bengio, *Mach Learn* 52:239, 2003: corrected resampled t for repeated CV.
- Bouthillier et al., *MLSys* 2021: separating seed variance from data variance.
- Lakens, *Soc Psychol Personal Sci* 8:355, 2017: TOST equivalence testing.
