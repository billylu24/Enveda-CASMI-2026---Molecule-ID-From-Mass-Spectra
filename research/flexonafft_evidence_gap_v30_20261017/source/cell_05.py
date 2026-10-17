'''Run and inspect
The retrieval report is eng/bio_union_audit.json or eng/shortlist_audit.json. submission_v22_control.csv here is the adapted source output, not an unmodified 0.451 control. v22_adaptation_audit.json reports this explicitly. Inspect actual additions/list changes before spending a submission; score gains are unverified.'''
experiment = V22Experiment(SOURCE_CELLS,runner_up=False)
result = experiment.run()
