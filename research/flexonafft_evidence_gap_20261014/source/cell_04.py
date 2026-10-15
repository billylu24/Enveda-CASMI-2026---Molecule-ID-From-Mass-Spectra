'''3. Submission integrity and current-run evidence report
The charts describe actual processed molecules, not model accuracy. Placeholder rows are excluded. The paired reference is submission_v7c_control.csv; raw per-molecule routes and changes remain inspectable in research_routing_audit.csv/json.'''

class SubmissionBuilder:
    def __init__(self, scope):
        self.scope = scope

    def validate(self, frame, control, active):
        assert list(frame.columns) == ['molecule_id', 'smiles']
        assert list(frame.molecule_id) == list(control.molecule_id) and frame.molecule_id.is_unique
        assert frame.smiles.notna().all()
        original = dict(zip(control.molecule_id, control.smiles))
        for row in frame.itertuples(index=False):
            smiles = row.smiles.split(';')
            assert 1 <= len(smiles) <= 25
            if row.molecule_id in active:
                identities = list(map(self.scope['metric_identity'], smiles))
                assert None not in identities and len(identities) == len(set(identities))
            else:
                assert row.smiles == original[row.molecule_id]

class ExperimentAudit:
    def save(self, audit, stages, fragments, config):
        frame = pd.DataFrame(audit)
        frame.to_csv('research_routing_audit.csv', index=False)
        active = frame[frame.processed]
        summary = dict(experiment='group-gap' if config.group_fragment_fallback else 'initial-faithful',
            active_molecules=len(active), output_rows=len(frame),
            source_control_reconstructed=True, changed_rows=int(frame.changed.sum()),
            changed_top1=int(frame.changed_top1.sum()), fragment_errors=fragments.errors,
            source_control_sha256=hashlib.sha256(Path('submission_v7c_control.csv').read_bytes()).hexdigest(),
            submission_sha256=hashlib.sha256(Path('submission.csv').read_bytes()).hexdigest(),
            group_fallback_count=int(frame.group_fragment_groups.sum()), stages=stages,
            routes=active.branch.value_counts().to_dict(), measured_score=None,
            validation_note='Visible smoke is not independent validation; source public 0.441 is user-reported.')
        Path('research_routing_audit.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        if config.group_fragment_fallback and not summary['changed_rows']:
            print('NO-OP versus v7c: do not spend another submission on identical predictions.')
        self.plot(active)
        return summary

    def plot(self, active):
        try:
            import matplotlib.pyplot as plt
            counts = active.branch.value_counts().sort_values()
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            axes[0].barh(counts.index, counts.values, color='#35618c')
            for i, value in enumerate(counts.values):
                axes[0].text(value, i, f' {value}', va='center')
            axes[0].set(title='Final routing branch — active molecules', xlabel='Molecules')
            changes = [int(active.changed.sum()), int(active.changed_top1.sum())]
            axes[1].bar(['Changed lists', 'Changed top-1'], changes, color='#a17e35')
            for i, value in enumerate(changes):
                axes[1].text(i, value, str(value), ha='center', va='bottom')
            axes[1].set(title='Changes versus same-run v7c', ylabel='Molecules')
            for axis in axes:
                axis.set_axisbelow(True)
                axis.grid(axis='x' if axis is axes[0] else 'y', alpha=.15)
                axis.spines[['top', 'right']].set_visible(False)
            axes[0].set_xlim(0, max(1, int(counts.max()) if len(counts) else 1) * 1.18)
            axes[1].set_ylim(0, max(1, max(changes)) * 1.2)
            fig.suptitle('Current-run audit, not a validation score', fontsize=12)
            fig.tight_layout()
            fig.savefig('research_evidence_report.png', dpi=150, bbox_inches='tight')
            plt.show()
            plt.close(fig)
        except ImportError:
            print('Chart dependency unavailable; inspect research_routing_audit.csv/json.')

class CasmiExperiment:
    def __init__(self, config, stages, fragments, reference_route):
        self.config, self.stages, self.fragments = config, stages, fragments
        self.reference_route = reference_route
        try:
            from IPython.display import display
        except ImportError:
            display = print
        self.scope = {'__name__': '__main__', 'display': display}

    def run(self):
        scope = self.scope
        timing = [stage.run(scope) for stage in self.stages]
        timing.append(self.fragments.run(scope))
        timing.append(self.reference_route.run(scope))
        Path('submission_v7c_control.csv').write_bytes(Path('submission.csv').read_bytes())
        control = pd.read_csv('submission.csv', dtype=str, keep_default_na=False)
        reference = pd.read_csv('submission_reference.csv', dtype=str, keep_default_na=False)
        assert list(control.molecule_id) == list(reference.molecule_id)
        controls = dict(zip(control.molecule_id, control.smiles))
        active = set(scope['te'].molecule_id.astype(str))
        router = EvidenceRouter(scope, self.config, self.fragments)
        rows, audits = [], []
        for row in reference.itertuples(index=False):
            mid = row.molecule_id
            original = row.smiles.split(';')
            faithful, _ = router.rerank_row(mid, original, window=40, ice_fallback=True, processed=mid in active, group=False)
            assert ';'.join(faithful) == controls[mid], f'OOP source parity mismatch: {mid}'
            revised, audit = router.rerank_row(mid, original, window=self.config.window,
                ice_fallback=True, processed=mid in active, group=self.config.group_fragment_fallback)
            old = controls[mid].split(';')
            audit.update(changed=revised != old, changed_top1=revised[:1] != old[:1])
            rows.append((mid, ';'.join(revised)))
            audits.append(audit)
        output = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
        SubmissionBuilder(scope).validate(output, control, active)
        output.to_csv('submission.csv', index=False)
        if not self.config.group_fragment_fallback:
            assert output.equals(control)
        return ExperimentAudit().save(audits, timing, self.fragments, self.config)
