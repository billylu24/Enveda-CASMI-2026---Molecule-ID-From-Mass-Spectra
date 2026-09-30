"""Candidate encoders for mass-conditioned listwise spectrum/structure ranking."""

import numpy as np
import torch
from rdkit import Chem
from torch import nn
from torch.nn import functional as F


def graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Invalid candidate molecule")
    atoms = []
    for a in mol.GetAtoms():
        atoms.append(
            [
                min(a.GetAtomicNum(), 118),
                min(a.GetTotalDegree(), 6),
                max(0, min(a.GetFormalCharge() + 3, 6)),
                min(a.GetTotalNumHs(), 4),
                int(a.GetIsAromatic()),
                int(a.IsInRing()),
                min(int(a.GetHybridization()), 7),
            ]
        )
    edges, bonds = [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        kind = {1.0: 0, 2.0: 1, 3.0: 2, 1.5: 3}.get(b.GetBondTypeAsDouble(), 4)
        edges.extend([(i, j), (j, i)])
        bonds.extend([kind, kind])
    return (
        np.asarray(atoms, np.int64),
        np.asarray(edges, np.int64).reshape(-1, 2),
        np.asarray(bonds, np.int64),
    )


def batch_graphs(graphs, device="cpu"):
    atoms, edges, bonds, ids = [], [], [], []
    offset = 0
    for i, (a, e, b) in enumerate(graphs):
        atoms.append(a)
        edges.append(e + offset)
        bonds.append(b)
        ids.append(np.full(len(a), i, np.int64))
        offset += len(a)
    return {
        "atoms": torch.as_tensor(np.concatenate(atoms), device=device),
        "edges": torch.as_tensor(np.concatenate(edges), device=device),
        "bonds": torch.as_tensor(np.concatenate(bonds), device=device),
        "graph_ids": torch.as_tensor(np.concatenate(ids), device=device),
        "n_graphs": len(graphs),
    }


class MolecularGraph(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.atom = nn.ModuleList(
            [nn.Embedding(n, width) for n in [119, 7, 7, 5, 2, 2, 8]]
        )
        self.bond = nn.Embedding(5, width)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width * 2, width),
                    nn.GELU(),
                    nn.Linear(width, width),
                    nn.LayerNorm(width),
                )
                for _ in range(3)
            ]
        )

    def forward(self, data):
        x = sum(emb(data["atoms"][:, i]) for i, emb in enumerate(self.atom))
        source, target = data["edges"].T
        bond = self.bond(data["bonds"])
        degree = (
            torch.bincount(target, minlength=len(x)).clamp_min(1).to(x.dtype)[:, None]
        )
        for layer in self.layers:
            messages = F.gelu(x[source] + bond)
            aggregate = torch.zeros_like(x).index_add_(0, target, messages) / degree
            x = x + layer(torch.cat([x, aggregate], -1))
        pooled = x.new_zeros((data["n_graphs"], x.shape[-1])).index_add_(
            0, data["graph_ids"], x
        )
        count = torch.bincount(data["graph_ids"], minlength=data["n_graphs"]).to(
            x.dtype
        )[:, None]
        return pooled / count.clamp_min(1)


class DirectRanker(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        if architecture not in ["fingerprint", "graph"]:
            raise ValueError(architecture)
        self.architecture = architecture
        self.spectrum = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
        )
        self.molecule_fp = nn.Sequential(
            nn.Linear(2048, 256), nn.GELU(), nn.Linear(256, 128)
        )
        if architecture == "graph":
            self.graph = MolecularGraph()
            self.fusion = nn.Linear(256, 128)
        self.log_scale = nn.Parameter(torch.tensor(2.3))

    def encode_molecules(self, fps, graphs=None):
        fp = self.molecule_fp(fps)
        if self.architecture == "graph":
            fp = self.fusion(torch.cat([fp, self.graph(graphs)], -1))
        return F.normalize(fp, dim=-1)

    def encode_spectra(self, latent):
        return F.normalize(self.spectrum(latent), dim=-1)

    def forward(self, latent, fps, graphs, inverse):
        q = self.encode_spectra(latent)
        candidates = self.encode_molecules(fps, graphs)[inverse]
        return (q[:, None] * candidates).sum(-1) * self.log_scale.exp().clamp(max=50)


@torch.inference_mode()
def score_group(ranker, encoder, group, preprocessing, candidates, fps):
    """CPU deployment score: average normalized per-spectrum query embeddings."""
    from casmi_ml.data import features

    rows = [features(row, preprocessing) for row in group.to_dict("records")]
    pieces = [torch.from_numpy(np.stack([r[i] for r in rows])) for i in [0, 2, 1]]
    latent = encoder.encoder(torch.cat(pieces, -1))
    query = ranker.encode_spectra(latent).mean(0)
    scores = []
    for start in range(0, len(candidates), 256):
        part = candidates.iloc[start : start + 256]
        graphs = (
            batch_graphs([graph(s) for s in part.normalized_smiles])
            if ranker.architecture == "graph"
            else None
        )
        structure = ranker.encode_molecules(
            torch.from_numpy(fps[start : start + 256]), graphs
        )
        scores.append((structure @ query).numpy())
    return np.concatenate(scores) if scores else np.empty(0, np.float32)
