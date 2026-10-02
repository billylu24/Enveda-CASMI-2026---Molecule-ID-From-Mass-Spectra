"""Exercise both deployment branches with isomeric chemical evidence."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from baseline import PROTON
from casmi_ml.chemical_priors import dictionary_sha256
from casmi_ml.data import fit_preprocessing
from casmi_ml.models import FingerprintModel
from casmi_ml.secondary_inference import predict


def test_external_isomer_and_chemistry_run_only_below_guard(tmp_path):
    original = 'COCS(=O)(=O)O'
    sulfate = 'CCOS(=O)(=O)O'
    mol = Chem.MolFromSmiles(original)
    mass = Descriptors.ExactMolWt(mol)
    assert mass == Descriptors.ExactMolWt(Chem.MolFromSmiles(sulfate))
    key = lambda smi: Chem.MolToInchiKey(Chem.MolFromSmiles(smi))[:14]
    precursor = mass + PROTON
    signal = precursor - 79.956815
    pd.DataFrame([{
        'inchikey14': key(original), 'normalized_smiles': original,
        'molecular_formula': rdMolDescriptors.CalcMolFormula(mol),
        'ms2_mzs': [signal, 80.], 'ms2_normalized_intensities': [1., .8],
        'precursor_error_ppm': 0.,
    }]).to_parquet(tmp_path / 'train.parquet')
    queries = pd.DataFrame([{
        'molecule_id': name, 'precursor_mz': precursor, 'adduct': '[M+H]+',
        'ms2_mzs': mz, 'ms2_normalized_intensities': intensity,
        'collision_energy_ev': [20.], 'ionization_mode': 'positive',
        'instrument_type': 'timsTOF',
    } for name, mz, intensity in [
        ('protected', [signal, 80.], [1., .8]),
        ('unprotected', [signal], [1.]),
    ]])
    queries.to_parquet(tmp_path / 'test.parquet')
    pd.DataFrame([{'inchikey': key(original), 'canonical_smiles': original,
                   'exact_mass': mass}]).to_parquet(tmp_path / 'coconut.parquet')
    pd.DataFrame([{'inchikey14': key(sulfate), 'normalized_smiles': sulfate,
                   'mass': mass, 'origin': 'lipidmaps', 'formal_charge': 0}])\
        .to_parquet(tmp_path / 'extra.parquet')
    prep = fit_preprocessing(queries)
    dim = 6 + sum(len(v)+1 for v in prep['categories'].values())
    model = FingerprintModel('metadata', dim)
    # An uninformative neural head leaves all discrimination to the tested evidence.
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)
    torch.save({'architecture': 'metadata', 'metadata_dim': dim,
                'preprocessing': prep, 'state_dict': model.state_dict()}, tmp_path / 'model.pt')
    dictionary = Path(__file__).resolve().parents[1] / 'configs/chemical_priors.json'
    digest = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    recipe = {
        'config': {'kind': 'free_top1', 'base': 'coconut15', 'threshold': .9, 'weight': .75},
        'checkpoint': 'model.pt', 'checkpoint_sha256': digest('model.pt'),
        'candidate_expansion': {'path': 'extra.parquet', 'weight': 1., 'sha256': digest('extra.parquet')},
        'chemical_priors': {'path': str(dictionary), 'sha256': dictionary_sha256(dictionary), 'weight': 1.},
    }
    path = tmp_path / 'recipe.json'
    path.write_text(json.dumps(recipe))
    output = tmp_path / 'out.csv'
    result = predict(path, tmp_path, tmp_path / 'coconut.parquet', output).set_index('molecule_id')
    assert result.loc['protected', 'smiles'] == original
    assert result.loc['unprotected', 'smiles'].split(';')[0] == sulfate
    assert set(result.loc['unprotected', 'smiles'].split(';')) == {original, sulfate}
    audit = json.loads(Path(str(output) + '.chemistry.json').read_text())
    assert len(audit) == 1 and audit[0]['molecule_id'] == 'unprotected'
    assert audit[0]['evidence'][0]['rule_id'] == 'sulfate_ester_loss_80'
