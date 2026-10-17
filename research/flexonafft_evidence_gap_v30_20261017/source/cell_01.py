'''Credits, attribution and limitations
Direct source: user-uploaded casmi26-v22-rel150-e2r2-clean.ipynb; SHA256 13b8e5f80e016a487382906aabd2e4f83677265f4e3187fe0f2d832e505b9463. The file does not identify its uploader; provide its direct URL for a precise author link.
@antriksharya — original rank2 hedge: https://www.kaggle.com/code/antriksharya/casmi26-lb-0-443-v7c-engine-2-rank-2-insert
Omer Faruk Merey — metric-exact hedge, attributed by the uploaded V22 header; profile not verified.
@seyitkaangunes — rel150 sweep, attributed by V22: https://www.kaggle.com/seyitkaangunes
@sergiomalv — BIO structures: https://www.kaggle.com/datasets/sergiomalv/casmi-chebi-lmsd-structures
@ahmedberatozer — engines/resources and v7c fragment-gap integration: https://www.kaggle.com/code/ahmedberatozer/casmi26-v7c-e006-fraggap
Huseyin Emre Aksoy — reused fragment kernel: https://www.kaggle.com/huseyinemreaksoy
@canqiang — E006; @samanyu1808 — ICE fallback; @flexonafft — metric fusion; @skomuro — promotion; @prvsiyan/@megayak — engine2 assets. Full inherited credits follow. Their wording is provenance, not verified performance/licence evidence. The inherited ahmedberatoze source line points to obstacledeveloper; use the explicit v7c link above. Retain upstream licences and ATTRIBUTION.md from BIO. Our contribution: orchestration, BIO ambiguity guard, run audit and optional runner-up policy, not the original models.'''
'''Original V22 notes and inherited credits (quote delimiters normalized)
# CASMI26 V22 (rel > 150 PubChem Promo + CC-BY ChEBI/LMSD + Metric-Exact Engine-2 Rank-2 Hedge + Frag-Gap)

**V22 Synthesis Champion**:
1. **PubChem Promotion Margin**: Lowered from `rel > 250` to `rel > 150.0` (validated by Seyit Kaan Gunes sweep, LB 0.446).
2. **Metric-Exact Engine-2 Rank-2 Hedge**: Re-implemented with official `metric_identity` (tautomer-canonical InChIKey-14, stereo stripped) so twins of rank 1 never waste rank 2 (Omer Faruk Merey, LB 0.445).
3. **CC-BY ChEBI / LIPID MAPS Database**: Upgraded to `sergiomalv/casmi-chebi-lmsd-structures` (+0.001 to +0.003).
4. **Frag-Gap**: MetFrag-lite fragment re-ranking for molecules without forward-model simulation coverage (E006 + v7c).
5. **Runtime**: Pinned Python 3.12 T4 image with native RDKit 2025.3.6 and RDKit 2026.03.3 offline metric wheel.

## Credits & Attribution
- @ahmedberatoze — direct source: casmi26-v7c-e006-fraggap, version 1, scriptVersionId 357736980. Source: https://www.kaggle.com/code/obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail?scriptVersionId=356393001
The source page displays public score 0.441
- @canqiang — direct source: CASMI26 E006 Window 40 ICE Fallback True, version 1, Source: https://www.kaggle.com/code/canqiang/casmi26-e006-window-40-ice-fallback-true 
- @obstacledeveloper — direct source: CASMI26 V1.1 Metric Audit and Safe Tail, version 1, scriptVersionId 356393001. Source: https://www.kaggle.com/code/obstacledeveloper/casmi26-v1-1-metric-audit-and-safe-tail?scriptVersionId=356393001
The source page displays public score 0.433
- @bobthebot369 — inherited Sovereign Zenith ensemble and confidence-promotion ideas: https://www.kaggle.com/code/bobthebot369/enveda-casmi-2026-v18-sovereign-zenith/notebook
- @ahmedberatozer — engine, ranker, pool, PubChem, full FPNet and simulator packages: https://www.kaggle.com/ahmedberatozer
- @prvsiyan — analog propagation, spectral matching and fingerprint models: https://www.kaggle.com/code/prvsiyan/analog-propagation-casmi-2026-baseline
- @dmitriigluzdov — PubChem popularity prior: https://www.kaggle.com/datasets/dmitriigluzdov/casmi26-pubchem-popularity-prior
- @hengck23 — inherited hengck23-dreams-enveda-casmi26 structural candidate resource: https://www.kaggle.com/hengck23
- Official metric implementation, version 14: https://www.kaggle.com/code/metric/casmi-mean-reciprocal-rank
- ICEBERG/GLACIER research tooling: https://github.com/coleygroup/ms-pred
- Competition: https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra
RDKit official metric wheel: https://www.kaggle.com/datasets/metric/rdkit-2026-3-3-wheel
ChEBI/LIPID MAPS and COCONUT resources remain credited in inherited code and input metadata.
The public notebook declares Apache-2.0; upstream datasets/components retain their own licences. Substantial code, including embedded modules, is reused verbatim. Our adaptation changes presentation and adds provenance, not authorship of the engines. Upstream comments are retained as provenance, not independently established component-performance claims.’ ’ ’
'''
'''
Additional experiment attribution
@sergiomalv — new BIO structures; @prvsiyan — old BIO candidates and fingerprint code. Our contribution: explicit BIO union/recalculated additions or early metric-unique export, not the original engines, metric definition or candidate databases. Keep the sergiomalv BIO entries first and add old prvsiyan BIO structures absent by engine2 source key. Recalculate added fingerprints and masses with engine2 native fp_and_mass and the same selected bits. No assumptions about old fingerprint-bit ordering. Source-key overlaps keep the new version; full official metric deduplication is NOT introduced into the pool. Candidate discovery is pinned to sergiomalv; the old dataset is read only by the extension. A 1200s preparation deadline raises rather than silently accepting partial coverage. More candidates may reduce accuracy or exceed runtime.'''
