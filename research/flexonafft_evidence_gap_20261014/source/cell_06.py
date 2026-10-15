'''5. Run the controlled experiment
This runs the existing neural/model recipe once, then reconstructs routing without another neural pass. Source-parity mismatches stop execution before writing the new submission. Full inference and chart outputs require the Kaggle inputs/GPU environment.'''

stages = [
    InputRegistry('Runtime and verified inputs', [(i, SOURCE_CELLS[i]) for i in (7, 8, 10)]),
    CandidateRetriever('Retrieval, fingerprints and base rankings', [(i, SOURCE_CELLS[i]) for i in (12, 14, 16, 17, 19)]),
    ForwardEvidence('Simulators, PubChem gate and inherited fusion', [(i, SOURCE_CELLS[i]) for i in (21, 23, 25)]),
    MetricAudit('Official identity audit and same-run source observer', [(i, SOURCE_CELLS[i]) for i in (27, 28, 29, 30)]),
]
fragments = FragmentEvidenceScorer('MetFrag-lite molecule-gap reference', [(32, SOURCE_CELLS[32])])
reference_route = SourceStage('Unmodified v7c routing oracle', [(33, SOURCE_CELLS[33])])
config = ExperimentConfig(group_fragment_fallback=False)
experiment = CasmiExperiment(config, stages, fragments, reference_route)
result = experiment.run()
