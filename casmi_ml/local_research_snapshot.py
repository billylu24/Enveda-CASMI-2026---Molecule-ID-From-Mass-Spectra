"""Refresh safe local aggregate progress from authoritative controller state."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller, now

ACTIVE = {"registered", "prepared", "running", "external_running", "queued"}


def refresh():
    c = Controller()
    state = c.read()
    if not c.local_only():
        raise ValueError("Local-only snapshot requires persistent local mode")
    active = [
        {
            "id": r["id"],
            "direction": r["direction"],
            "status": r["status"],
            "report": r["report"],
        }
        for r in state["rounds"]
        if r["status"] in ACTIVE
    ]
    active = [r for r in active if not Path(r["report"]).exists()]
    value = {
        "updated_at": now(),
        "local_only": True,
        "remote_publication": False,
        "best": state.get("development_best"),
        "local_verified_inference": state.get("local_verified_inference"),
        "active_rounds": active,
        "local_candidate_runtime": state.get("local_candidate_runtime"),
        "local_array_runtime": state.get("local_array_runtime"),
        "local_object_candidate_runtime": state.get("local_object_candidate_runtime"),
        "local_blas_runtime": state.get("local_blas_runtime"),
        "local_validation_runtime": state.get("local_validation_runtime"),
        "local_proposal_runtime": state.get("local_proposal_runtime"),
        "local_query_zero_control": state.get("local_query_zero_control"),
        "drivers": [
            "diagnostic_query_multispectrum_followup.log",
            "diagnostic_query_chemical_only_followup.log",
            "diagnostic_query_balanced_followup.log",
            "object_candidate_followup.log",
            "diagnostic_query_strong_peak_followup.log",
            "structure_key_followup_v2.log",
            "diagnostic_query_admission_followup_v2.log",
            "diagnostic_query_all_spectra_followup.log",
            "candidate_compatibility_followup.log",
            "extended_candidate_followup.log",
            "intensity_candidate_followup.log",
            "max_candidate_followup.log",
            "full_gap_candidate_followup_v2.log",
            "candidate_count_followup.log",
            "connected_candidate_followup.log",
            "aromatic_h_candidate_followup.log",
            "high_slot_depth_followup.log",
            "streaming_prefetch_control.log",
            "finish_streaming_once236.log",
            "candidate_gate_trace_v2.log",
            "individual_gate_followup.log",
            "finish_reference_once236.log",
            "mass_capacity_followup.log",
            "mass_candidate_followup.log",
            "fingerprint_alias_followup.log",
            "alias_radius_followup.log",
            "alias_radius_source_followup.log",
            "forward_diagnostic_followup.log",
            "forward_candidate_capacity_followup.log",
            "forward_augmented_capacity_followup.log",
            "forward_augmented_followup.log",
            "resume_forward_augmented_cycle.log",
            "diagnostic_energy_stability_followup.log",
            "energy_conditioned_diagnostic_followup.log",
            "energy_conditioned_candidate_followup.log",
            "balanced_cpu_followup.log",
            "finish_cpubal_once236.log",
            "finish_cpubal_once236_v2.log",
            "finish_overlap_once236.log",
            "energy_conditioned_capacity_followup.log",
            "energy_conditioned_followup.log",
            "energy_conditioned_precision_followup.log",
            "bonus_generalization_followup.log",
            "generation_stream_control.log",
            "overlap_local400_corrected_control_v2.log",
            "overlap_full_followup.log",
            "conditioned_gate_impact_followup.log",
            "conditioned_gate_impact_corrected_followup.log",
            "neighborhood_diagnostic_followup.log",
            "catalog_scoring_funnel_followup.log",
            "catalog_scoring_funnel_corrected_followup.log",
            "local_overlap_request_observation.log",
            "generation_cpu_worker400_control.log",
            "generation_cpu_cold400_control.log",
            "overlap_cpu_full_followup.log",
            "finish_overlap_cpu_once236.log",
            "mass_formula_proposal_followup.log",
            "mass_formula_proposal_corrected_followup.log",
            "mass_formula_generation_followup.log",
            "decoder_formula_response_followup.log",
            "mass_formula_budget_followup.log",
            "overlap_cpu_observation_follow.log",
            "metfrag_jfr_observation_followup.log",
            "metfrag_jfr_corrected_followup.log",
            "metfrag_mass_reuse_followup.log",
            "mass_reuse_cold400_followup.log",
            "mass_reuse_cold400_corrected_followup.log",
            "mass_reuse_full3340_corrected_followup.log",
            "finish_mass_reuse_once236.log",
            "physical_float_observation.log",
            "physical_float_instruction_observation.log",
            "decoder_mixed_formula_followup.log",
            "decoder_mixed_formula_structure_followup.log",
            "decoder_token_loss_attribution_followup.log",
            "mass_reuse_platform_follow.log",
            "decoder_spectrum_condition_followup.log",
            "decoder_latent_only_condition_followup.log",
            "decoder_composition_expectation_followup.log",
            "decoder_heavy_composition_training_followup.log",
            "decoder_heavy_composition_deterministic_followup.log",
            "mass_reuse_float_jit_observation.log",
            "mass_reuse_float_sse_observation.log",
            "full_catalog_availability_followup.log",
            "processor_probe_followup.log",
            "parallel_generation_followup.log",
            "parallel_generation_corrected_followup.log",
            "finish_dualgen_once236.log",
            "finish_dualgen_once236_v2.log",
        ],
        "pending_sequence": "Scientific236/evidence286 retained;319 exact3340 validation cache active. Count337, connected344, aromaticH351, five-slot356, individual-gate361 all rejected by frozen promotion gate.359 scorer trace full3340 exact:116unknown available but absent finally,44outside actual100,72scored,22pass original first critic and physical gates. Streaming platform failed physical timeout under original1200s deadline; no competition POST.362 full-train reference cache built without queries, selected556890 rows and10027422 CSR entries plus74956 observed keys exact; source-bound reference fresh75 exact. Reference cold400409.738s/4276.563MiB/zero budget/all400Top25+265rawdict exact passed; platform timeout1250s, no competition POST. Actual assigned T4x2, metadataP100 unavailable; parallel generation375 CLI defect retained;376 corrected independent-seeded whole-molecule400 CSV/audit/fullhandoff exact; frozenfresh75exact and forced partition cold400460.713s/3397.641MiB/zero budget/all400Top25+265rawdict exact passed. Dualgen platformfailed original1200s physical timeout; actualgeneration217.295s/retrieval63.263s, no competition POST. BalancedCPU385 originalfull3340 exact0budget/newfail passed. Frozen cpubal75 exact0budget44.413s; newcold4004CPU434.995s/3081.590MiB/0budget/full400Top25 and265rawdict exact passed. Cpubal Notebook final APIERROR, original1200 physical timeout; retrieval77.0855s/generation218.1313s; no intent/pending/id/competitionPOST. Same single authorization unused, all original gates retained.386 conditioned120D capacity objective.524647178->.524068418 gain.000578760089, total2.387zero full3340exact0budget/newfail complete;388chemical/389mask fits and390/391 full3340 physical complete0budget/newfail;391mask exact;390unknownMRR+.000166667/+0.379%,knownunchanged, rejected. Only1 truthRR improvement,0 regressions,bootstrap95[0,.0005].394serialGC ABBA fresh75/raw57 exact each but slower; not deploy.395default/ActiveProcessorCount2 ABBA complete75/rank75/raw57 exact each; mean48.548 vs48.256s(-0.60%), no material local gain or deployment. Private original236 platform diagnostic v1 failed assumed inputpath before inference; v2 COMPLETE, bothactual75 Top25exact, physical47 sum330.19 vs324.90s/consumption170.12vs172.86s; no processor-flag deployment, no competition submission.397 all400 emitted generation handoffs/CSV/audit exact.398 prototype second-pool lifecycle risk terminated with partial logs preserved;399 GPU-lock blocked before inference.400 corrected singlepool overlap cold4004CPU363.605s/3187.828MiB/all400Top25+265rawdict exact/0budget, not paired speed proof.402 full3340 sharedscoring/batched originalrequest discovery all2000/1340fullrank exact0budget/newfail complete; Frozen overlap75 fullrankexact0budget52.463s and new4CPU cold400386.253s/3185.078MiB/full400Top25+265rawdict exact0budget passed. OverlapNotebook finalERROR at452.65s individualphysicaltimeout before1200globaldeadline;retrieval58.89746s/generation313.86565s/all400generationCSVexact. Nointent/pending/id/registeredidentity/matchingremotesubmission; platform_verification_failed, authorizationunused. Private unchanged400 overlaprequestdiagnosis v1 COMPLETE but rejected371/400Top25exact and29budgetfallbacks;1204.521s/2723.992MiB;227complete/4failed requests, overlapping227physicalsum1917.279s vs same227local440.593s, not pairedspeed;410CPUgenerator4to1all400CSV/audit/fullhandoffs/events exact;411newcold400allTop25/265rawdict exact0budget;412full3340 sourcebound replay complete original3340 ranks exact0budget/newfail; newCPU frozen75 exact42.815s and4CPU cold400338.824s/3163.031MiB/fullTop25+265rawdict exact0budget complete; private datasetuploaded/CPU Notebookv1 finalERROR physicaltimeout session1291.550s, retrieval107.9205/generation286.1137s/all400generationCSV-audit-fullhandoffexact;229rawdict223exact6floatmismatch14valuesmax1.1369e-13 retained; noPOST/id/registeredidentity/pending;authorizationunused;newprivateCPUobserverCOMPLETErejected372/400Top25exact28budget/228completerequests2exceptions/1204.551s/2719.25MiB/globalremaining8.484-.142s;0420JFRdumpfailedafter10rawexact;0421correctedJFR10rawexact/10138samples;0422massreuseABBAall40rawexactmean36.458to23.639s(-35.16percent)heavyrequestonly;0423inferenceexact400/265butverifierdeletedhandofffilefailed;0424newcold400completeTop25/generatedCSV-audit-fullhandoff/265rawdict exact0budget251.752s/3166.285MiB;0425full3340sourcecontrolalloriginalranks exact0budget/newfailcomplete;newmassreusefrozen75allrank/raw57exact34.704s/newfrozencold400complete248.941s/3145.387MiB/all400Top25-265rawdict exact0budget;identityd3bc8f152e1a22b8e506df3f004a8f20ee404cdd9336b45af9652cfd657c93e4;unusedsingleauthorization/NotebookCOMPLETE807.744s-2763.387MiB/all400Top25exact-zeroBudget/265physicalcomplete257exact-8mismatch18floatvaluesmax1.1369e-13/strictgatefailed-nocompetitionPOST-ID;0428matchedtrainingintegerformula2epochauditcomplete111.382s/softgainovermatchedcontrol.000020851/integergain.000071077/noactivation;0430train-onlytokenattribution12002supported/all3originalsoftNLLarraysexact428/mixedovercontrolatomNLLworse.000024505/branch-ring-EOSslightlybetter/noactivation;0429fixedfivearm128sample200structurepilotcomplete/original9-newarms7each/allnewarmslost2original-noaddedgraph/rejected/noactivation;massreuseprivateNotebookv1COMPLETE/T4x2/strictphysicsgatefailed-noPOST-ID;0431fullconditioninterventionmeta-massconfoundingpreserved;0434latentonly768withquery103meta-2mass-softformulaexact/original12002NLLarrayexact428/9012sameformuladonorNLL.199598to.726356/noaccuracyclaim;0432defaultsorted-reverse-Xbatchall32localrawexact0platformexact;0433SSE2-SSE3-tier1all24localrawexact0platformexact/noLibmIntrinsicchanges125local118platform/noactivation;0426sixfloatmismatchCPUdiagnosticonly;524fullsuitepassed;0435heavyexpectationall12002NLLexact428/mixedworsecountL1.000025133;0436controlweightnonexactfailedbeforetreatment/0437deterministicreplicatedcontrol-heavyauxtrainingactive/syntheticpowonlyCPUdiagnosticpending/strictgatepreferencepending;415fitonly formula top5neural8922/12006 vsfrequency7519/original1292, no structure/system claim;416matched128 structure200pilot complete originalexact; original/neural/frequency graph9/9/9, no gain and no expansion;417trainingonly teacherforced responsecomplete truecounts tinyNLLgain.000274065/zeroandshuffledworse; no graphaccuracyclaim;418all60000heavy-token accounting59972exact/23unencodable/5unsupported0mismatch;419experimental fixed200heavybudgetpilot complete originalexact;massunique311/693/645 butgraph9/9/8, neural8original+1new-1lost,validity/runtimeworse,rejectexpansion;407cold400observer complete342.981s/3170.473MiB/full400Top25+265rawdict exact0budget;265observedrequests complete/max10.096s;physicalsum520.549s overlapping,notsumwall.408full existingtrainlibrary/COCONUT/PubChemLite/ChEMBL declaredmassavailability audit complete3546326structures/1570unknownabsentfullcatalogandcurrent to avoid treating external/currentabsence as fullcatalogabsence. No retries/timeoutextension, full400gates retained. Publisher state below authoritative.401 full3340 unchanged390 physics/ranks exact but numpy-int summary serialization failed;403 corrected audit complete, all3340 ranks390 exact0budget/newfail. Unknown1981 scored queries but truth only131scored/1850absent; criticorder changed1328/top3changed495, truth top3still27/top25still63, truth margin crossedup2. Individually critic+physical qualified53truths,23outsidefusedtop3 and absentfinal; descriptive qualification insufficient for strong-prefix gates. No gate relaxation.404complete original372 summaryarmsexact:2048radius2logcount andradius3logcount heldoutMSE.002862278/.002865183 vsbinary.002855580. Bothworse thanbinary inMSE and heldoutmass pairmargin; deeper representation availability didnot improve diagnostic prediction. Do not run dev/model activation.405initialmissingqueryrowassumptionfailedbeforeaggregate;406completeall3340 funnel retains5unknownemptyrows. Unknown28first/220currentother/105externalscored/44validwindowunscored/1catalogoutwindow/1597absentcatalogandcurrent. Fullcatalogtruth254/validwindow251/current248; raw131scoredtruthisnotoverallrecall. Known758first/425currentother/3external/154absent. No scoring/physical rerun or prediction changes.396 bonus calibration fit47994/holdout12006 complete; conditional120 overbinary60 heldout15 loss reduction.00039051395[.000230061,.000553345], restricted bothheldout massloss.00015478395[.000073878,.000238587]. Frozencritic all60K pretraining, no independent end-to-end acceptance;390 fullranking rejection stands.367/380/390 exact RR audit confirms sameone improved query, no additive gains.392 original384 exact contrast reconstruction, paired molecular bootstrap9758 MSEgain95[.00006004,.00007061] and conditionalpairmargin95[.00002647,.00003248], fixed training-internal cohort only. No auto activation/publication. Current status is in one_time_kaggle below.363 same-mask training capacity gain.000020123 complete, no accuracy claim.364zero/368mask all3340 exact0budget/newfail;367mass-compatible chemistryunknownMRR.044144968 knownunchanged+.379% rejected.372 train-internal fp->diagnostic prediction heldoutMSE8.8% better than20SMARTS, not accuracy.373 direct forward-strength support replacement capacity worse.003135074; do not run dev systems. 369 original2048 exact665 train-pair aliases all distinct actual graph;576mass-compatible, larger4096/8192bits separate0, radius2counts separate108.370 sparse count radius3/4/6/8 separate310/450/526/572 of576, representation availability only. 377zero/381mask full3340exact0budget/newfail;380120Dchemical+.379% unknownMRR rejected, knownunchanged. Parent termination recovered original child without retraining or budget refund.382 measured CE audit complete:nonempty Jaccard .598544 different singletonCE vs .800985 sameCE, observational only.383 fixed train-internal molecular-holdout CE-conditioned fp/context ridge MSE.002800917 vs fp*CE .002735632(-2.33%),9758heldout eligible molecules; no dev accuracy claim.384 complete same-query CE/context bothheldout mass-compatible51014 row-slot pairs:baseline positive22369/negative14684/tie13961,CEinteraction22650/14404/13960. Mean positive-minus-negative .000214308->.000243739; no fitting/dev ranking or activation. Counts from contrast_summary are row slots, not independent molecules. Persistent local_only; no future submissions or GitHub push.",
        "freeze_policy": "unknown absolute+.0005 and relative+2%, known drop<=.001/top1drop<=.005; current best, completed original0206 and historical181 protection; no new physical failed requests or budget fallback promotion",
        "runtime_claims": "Exact packed/object/array all3340 cached ranks and proposal three full2000 dictionaries. Separate scientific236 one-time release cold400 resource/physical/ranking verified; no paired whole-pipeline speed or independent accuracy claim.",
        "one_time_kaggle": {
            key: state.get("one_time_kaggle", {}).get(key)
            for key in (
                "identity",
                "status",
                "remote",
                "submission_id",
                "public_score",
                "automatic_future_submissions",
            )
        },
        "independent_acceptance": False,
        "recent_rounds": [
            {
                "id": r["id"],
                "direction": r["direction"],
                "status": r["status"],
                "decision": r.get("decision"),
                "local_recorded": r.get("local_recorded", False),
            }
            for r in state["rounds"]
            if r["id"] >= "0244"
        ],
    }
    write_json("results/research_loop/local_continuation_manifest.json", value)
    path = Path("results/research_loop/local_diagnostic_cycle_progress.json")
    progress = json.loads(path.read_text()) if path.exists() else {}
    progress.update(
        updated_at=value["updated_at"],
        local_only=True,
        remote_publication=False,
        best_round=state["development_best"]["round"],
        active_rounds=[r["id"] for r in active],
        active_round=active[-1]["id"] if active else None,
        next_sequence=value["pending_sequence"],
        current_query_zero_control=state.get("local_query_zero_control"),
        local_verified_evidence_round=state["local_verified_inference"][
            "evidence_round"
        ],
    )
    progress["array_order_microbenchmark"]["scope"] = (
        "Sorting-segment microbenchmark followed by exact3340 cached inference control; activated, no accuracy or cold acceptance gain"
    )
    for name, filename in [
        ("multispectrum_attribution", "query_multispectrum_attribution.json"),
        ("chemical_only_attribution", "query_chemical_only_attribution.json"),
        ("balanced_attribution", "query_balanced_attribution.json"),
        ("strong_peak_attribution", "query_strong_peak_attribution.json"),
        ("admission_margin_attribution", "query_admission_attribution.json"),
        ("all_spectra_attribution", "query_all_spectra_attribution.json"),
        (
            "candidate_compatibility_attribution",
            "candidate_compatibility_attribution.json",
        ),
        ("extended_candidate_attribution", "extended_candidate_attribution.json"),
        ("intensity_candidate_attribution", "intensity_candidate_attribution.json"),
        ("max_candidate_attribution", "max_candidate_attribution.json"),
        ("full_gap_candidate_attribution", "full_gap_candidate_attribution.json"),
        ("candidate_count_attribution", "candidate_count_attribution.json"),
        ("connected_candidate_attribution", "connected_candidate_attribution.json"),
        ("aromatic_h_candidate_attribution", "aromatic_h_candidate_attribution.json"),
        ("mass_candidate_attribution", "mass_candidate_attribution.json"),
        ("forward_augmented_attribution", "forward_augmented_attribution.json"),
        ("energy_conditioned_attribution", "energy_conditioned_attribution.json"),
        ("conditioned_gate_impact", "conditioned_gate_impact.json"),
    ]:
        report = Path("results/research_loop") / filename
        if report.exists():
            progress[name] = json.loads(report.read_text())
    progress["proposal_runtime_control"] = {
        "complete": True,
        "full_query_matches_per_variant": 2000,
        "variants": ["native", "corrected_only", "corrected_fusion"],
        "accuracy_gain_claim": False,
        "enabled_for_explicit_local_proposal_entry": bool(
            state.get("local_proposal_runtime", {}).get("enabled")
        ),
    }
    write_json(path, progress)
    return value


def main():
    value = refresh()
    print(
        json.dumps(
            {"best": value["best"]["round"], "active": value["active_rounds"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
