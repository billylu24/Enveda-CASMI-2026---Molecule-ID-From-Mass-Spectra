"""Build a self-contained offline conditional-fingerprint GAN experiment."""
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(__file__).resolve().parent
SOURCES = ('baseline.py','hybrid.py','casmi_ml/__init__.py','casmi_ml/data.py',
           'casmi_ml/chemical_priors.py','casmi_ml/hybrid_chemistry.py',
           'casmi_ml/adversarial.py','casmi_ml/adversarial_release.py',
           'casmi_ml/failure_audit.py','casmi_ml/graph_candidates.py')


def cell(kind,source):
    value={'cell_type':kind,'id':hashlib.sha256(source.encode()).hexdigest()[:12],
           'metadata':{},'source':source.splitlines(keepends=True)}
    if kind=='code':
        value.update(execution_count=None,outputs=[])
    return value


def build():
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            source=(ROOT/name).read_bytes()
            compile(source,name,'exec')
            entry=zipfile.ZipInfo(name,date_time=(2026,10,2,0,0,0))
            entry.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(entry,source)
    payload=buffer.getvalue()
    source_sha=hashlib.sha256(payload).hexdigest()
    previous=json.loads((ROOT/'kaggle_release_chemistry/notebook/chemical_priors_hybrid.ipynb').read_text())
    bootstrap=''.join(previous['cells'][1]['source'])
    bootstrap=bootstrap[bootstrap.index('from pathlib import Path\n'):]
    bootstrap=bootstrap.replace('from casmi_ml.hybrid_chemistry import predict, behavior_check',
                                'from casmi_ml.adversarial_release import run')
    constants=(f'CODE_BASE64 = {base64.b64encode(payload).decode()!r}\n'
               f'CODE_SHA256 = {source_sha!r}\n'
               "SUMS_SHA256 = '1563d3ed2c3a3529926c0507880c4ab39e3477265487978b496cf635ae0553fa'\n"
               "COCONUT_SHA256 = '6d8bd9206fa576fecd2741c020ba64f5f4e60609f767bd8aad8a6628a5b87bbd'\n\n")
    experiment='''import torch
torch.set_num_threads(4)
print("torch",torch.__version__,"cuda available",torch.cuda.is_available())
submission,report=run(DATA_DIR,COCONUT_PATH,CATALOG_PATH,DICTIONARY_PATH,WORKING)
display(submission.head())
print("Actual conditional GAN completed; graph-edit candidates generated:",report['generated_molecular_graphs'])
print("Submit submission.csv; do not confuse visible output with hidden evaluation.")
'''
    explanation='''# Conditional fingerprint GAN: frozen paired experiment

This notebook **actually trains** a conditional generator and discriminator using alternating updates. Its supervised control uses the same generator, training data, seed, and epoch budget with adversarial weight zero. The generator predicts 2048-bit molecular fingerprints from spectra, neutral-loss histograms, metadata and noise; molecular structures come from catalog retrieval plus the limited formula-preserving graph-edit search described below. **There is no trained molecular-graph decoder or claim of unrestricted de novo generation.**

The experiment prepares up to 60,000 training molecules, 1,000 development and 1,000 acceptance molecules, grouping by RDKit 2026.03.3 tautomer-canonical InChIKey14. All natural-product example identities, identifiable visible test identities and identities conservatively matching visible spectrum signatures are excluded from neural fitting. Preprocessing fits training only, epoch selection reads development BCE only; acceptance is evaluated once. Other historical cohort overlap cannot be fully reconstructed. Unknown-spectrum reference libraries and library-origin structures exclude all development/acceptance identities and their raw aliases; independent public database candidates remain eligible.

Routing is fixed before the run: preserve historical rankings at confidence >=0.75, otherwise retain their first hit and blend the shared expanded candidate ranking with the generated-fingerprint ranking at weight 0.35. Chemical weight is 0.1. Low-confidence queries additionally share up to 32 deterministic graph-edit candidates from two mass-compatible historical anchors. Two single-bond endpoint swaps preserve atom inventory and degrees; each accepted graph must sanitize and preserve the exact molecular formula/mass, and must have a new official identity relative to the eligible catalog pool. This limited graph-edit search is **not a trained graph decoder** and does not prove arbitrary de novo capability. Both neural arms use identical generated candidates and routing, separating candidate generation from adversarial training. Results include unprotected and routed MRR@25, candidate recall, scaffold strata, paired bootstrap intervals, training gaps, rule coverage and detailed failures. Official canonical identities are deduplicated before the 25-candidate cutoff.

Attach the competition data, `aidensong123/casmi26-coconut-202609`, and `xiaoyuzhoux120/casmi-2026-chebi-lmsd-chemistry-inputs`. Internet is **off**. The frozen source and public input hashes are checked, and the bundled RDKit wheel is installed offline. CPU is sufficient; each training arm has a 1,800-second cap. The experimental GAN output is `/kaggle/working/submission.csv`; its acceptance performance does not retroactively choose hyperparameters or the submitted arm.
'''
    notebook={'cells':[cell('markdown',explanation),cell('code',constants+bootstrap),cell('code',experiment)],
              'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                          'language_info':{'name':'python','version':'3.12'},
                           'kaggle':{'accelerator':'gpu','isInternetEnabled':False}},
              'nbformat':4,'nbformat_minor':5}
    for index,entry in enumerate(notebook['cells']):
        if entry['cell_type']=='code':
            compile(''.join(entry['source']),f'conditional_fingerprint_gan:cell{index}','exec')
    target=RELEASE/'notebook/conditional_fingerprint_gan.ipynb'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
    metadata=json.loads((ROOT/'kaggle_release_chemistry/notebook/kernel-metadata.json').read_text())
    metadata['code_file']=target.name
    metadata['enable_gpu']=True
    (target.parent/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    write={'notebook':str(target.relative_to(ROOT)),'source_sha256':source_sha,
           'notebook_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}
    (RELEASE/'build_manifest.json').write_text(json.dumps(write,indent=2)+'\n')
    print(json.dumps(write,indent=2))


if __name__=='__main__':
    build()
