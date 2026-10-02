"""Peak-set self-distillation and spectrum-conditioned autoregressive prototypes."""

import copy
import math
import re
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from casmi_ml.chemistry import RULES, clean_peaks, high_resolution


def peak_tokens(row, maximum=128):
    mz, intensity = clean_peaks(row)
    w = np.sqrt(intensity)
    precursor = float(row.get("precursor_mz") or 0.0)
    precursor = precursor if math.isfinite(precursor) else 0.0
    loss = precursor - mz
    protected = np.zeros(len(mz), bool)
    if high_resolution(row.get("instrument_type")):
        for rule in RULES:
            if (
                row.get("adduct") in rule.adducts
                and row.get("ionization_mode") in rule.modes
            ):
                delta = mz - rule.mass if rule.kind == "ion" else loss - rule.mass
                tol = (
                    0.002 + max(0.002, precursor * 1e-5)
                    if rule.kind == "loss"
                    else max(0.002, rule.mass * 1e-5)
                )
                protected |= np.abs(delta) <= tol
    important = np.flatnonzero(protected)
    important = important[np.argsort(-w[important], kind="stable")][: maximum // 4]
    important_set = set(important)
    remaining = [
        int(i) for i in np.argsort(-w, kind="stable") if i not in important_set
    ]
    ids = np.array(list(important) + remaining[: maximum - len(important)], dtype=int)
    tokens = np.zeros((maximum, 19), np.float32)
    mask, protect = np.zeros(maximum, bool), np.zeros(maximum, bool)
    if len(ids):
        m, weight, d = mz[ids], w[ids], loss[ids]
        freq = np.array([0.01, 0.1, 1.0, 10.0])
        tokens[: len(ids)] = np.concatenate(
            [
                m[:, None] / 1250,
                weight[:, None],
                d[:, None] / 1250,
                np.sin(m[:, None] * freq),
                np.cos(m[:, None] * freq),
                np.sin(d[:, None] * freq),
                np.cos(d[:, None] * freq),
            ],
            axis=1,
        )
        mask[: len(ids)] = True
        protect[: len(ids)] = protected[ids]
    if not mask.any():
        mask[0] = True
    return tokens, mask, protect


class PeakEncoder(nn.Module):
    def __init__(self, metadata_dim, width=256, layers=4):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(19, width), nn.GELU(), nn.Linear(width, width)
        )
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            width,
            8 if width >= 64 else 2,
            width * 4,
            0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )
        self.latent = nn.Sequential(
            nn.Linear(width * 2 + metadata_dim, width), nn.GELU(), nn.LayerNorm(width)
        )
        self.fingerprint = nn.Linear(width, 2048)
        self.reconstruct = nn.Linear(width, 2)
        self.width = width

    def encode(self, batch, return_tokens=False):
        mask = batch["mask"].bool()
        valid = torch.cat(
            [torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device), mask], 1
        )
        x = torch.cat(
            [self.cls.expand(len(mask), -1, -1), self.embedding(batch["peaks"])], 1
        )
        x = self.encoder(x, src_key_padding_mask=~valid)
        mean = (x[:, 1:] * mask[..., None]).sum(1) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        z = self.latent(torch.cat([x[:, 0], mean, batch["meta"]], 1))
        return (z, x[:, 1:]) if return_tokens else z

    def forward(self, batch):
        return self.fingerprint(self.encode(batch))


def augmented(batch, drop=0.1, jitter=0.03):
    result = {k: v.clone() for k, v in batch.items()}
    keep = (torch.rand_like(batch["mask"].float()) >= drop) | (
        batch["peaks"][..., 1] >= 0.5
    )
    keep |= batch["protected"].bool()
    keep &= batch["mask"].bool()
    empty = ~keep.any(1)
    keep[empty, batch["mask"][empty].float().argmax(1)] = True
    result["mask"] = keep
    result["peaks"][..., 1] = (
        result["peaks"][..., 1]
        * (1 + torch.randn_like(result["peaks"][..., 1]) * jitter)
    ).clamp(0, 1)
    return result


class Distillation(nn.Module):
    def __init__(self, encoder, prototypes=1024):
        super().__init__()
        self.student = encoder
        self.project = nn.Sequential(
            nn.Linear(encoder.width, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.LayerNorm(128),
            nn.Linear(128, prototypes, bias=False),
        )
        self.teacher = copy.deepcopy(encoder).requires_grad_(False)
        self.teacher_project = copy.deepcopy(self.project).requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, prototypes))

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_project.eval()
        return self

    def loss(self, batch):
        first, second = augmented(batch), augmented(batch)
        a = self.project(self.student.encode(first)) / 0.1
        b = self.project(self.student.encode(second)) / 0.1
        with torch.no_grad():
            ta = self.teacher_project(self.teacher.encode(first))
            tb = self.teacher_project(self.teacher.encode(second))
            pa, pb = (
                ((ta - self.center) / 0.04).softmax(-1),
                ((tb - self.center) / 0.04).softmax(-1),
            )
            self.center.mul_(0.9).add_(
                torch.cat([ta, tb]).mean(0, keepdim=True), alpha=0.1
            )
        return -0.5 * (
            (pb * a.log_softmax(-1)).sum(-1).mean()
            + (pa * b.log_softmax(-1)).sum(-1).mean()
        )

    @torch.no_grad()
    def update_teacher(self, momentum):
        for student, teacher in [
            (self.student, self.teacher),
            (self.project, self.teacher_project),
        ]:
            for s, t in zip(student.parameters(), teacher.parameters()):
                t.mul_(momentum).add_(s, alpha=1 - momentum)


def masked_loss(encoder, batch):
    selected = (torch.rand_like(batch["mask"].float()) < 0.15) & batch["mask"].bool()
    selected[:, 0] |= ~selected.any(1)
    corrupted = {k: v.clone() for k, v in batch.items()}
    # Remove mass, loss and Fourier channels, not only raw m/z (would leak target).
    corrupted["peaks"][selected] = 0
    _, tokens = encoder.encode(corrupted, return_tokens=True)
    target = batch["peaks"][..., :2]
    return F.smooth_l1_loss(encoder.reconstruct(tokens)[selected], target[selected])


SMILES_TOKEN = re.compile(r"(\[[^\]]+\]|Br|Cl|Si|Na|Li|Ca|Mg|Al|Se|se|@@|%\d\d|.)")


class SmilesVocabulary:
    def __init__(self, tokens):
        self.tokens = ["<pad>", "<bos>", "<eos>", "<unk>"] + sorted(
            set(tokens) - {"<pad>", "<bos>", "<eos>", "<unk>"}
        )
        self.ids = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def fit(cls, smiles):
        return cls(t for s in smiles for t in SMILES_TOKEN.findall(s))

    def encode(self, smiles, limit=256):
        tokens = SMILES_TOKEN.findall(smiles)
        if len(tokens) + 2 > limit or any(t not in self.ids for t in tokens):
            return None
        return [1] + [self.ids[t] for t in tokens] + [2]

    def decode(self, ids):
        tokens = []
        for i in ids:
            if i == 2:
                break
            if i >= 4:
                tokens.append(self.tokens[i])
        return "".join(tokens)


class SmilesDecoder(nn.Module):
    def __init__(self, vocabulary, condition_dim, width=256, layers=6, limit=256):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, width, padding_idx=0)
        self.position = nn.Embedding(limit, width)
        self.condition = nn.Linear(condition_dim, width)
        layer = nn.TransformerDecoderLayer(
            width,
            8 if width >= 64 else 2,
            width * 4,
            0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(width))
        self.head = nn.Linear(width, vocabulary)
        self.limit = limit

    def forward(self, tokens, condition):
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)[None]
        causal = torch.ones(
            (len(positions), len(positions)), dtype=torch.bool, device=tokens.device
        ).triu(1)
        x = self.decoder(
            x,
            self.condition(condition)[:, None],
            tgt_mask=causal,
            tgt_key_padding_mask=tokens == 0,
        )
        return self.head(x)

    @torch.inference_mode()
    def generate(
        self, condition, samples=128, temperature=0.8, generator=None, deadline=None
    ):
        if samples < 1 or temperature <= 0:
            raise ValueError("Positive samples and temperature required")
        self.eval()
        condition = condition.expand(samples, -1)
        tokens = torch.ones((samples, 1), dtype=torch.long, device=condition.device)
        finished = torch.zeros(samples, dtype=torch.bool, device=condition.device)
        logp = torch.zeros(samples, device=condition.device)
        for _ in range(self.limit - 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Generation deadline reached; discard incomplete query"
                )
            logits = self(tokens, condition)[:, -1] / temperature
            logits[:, [0, 1, 3]] = float("-inf")
            probs = logits.softmax(-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            logp += torch.where(
                finished,
                0.0,
                probs.gather(1, nxt[:, None]).squeeze(1).clamp_min(1e-12).log(),
            )
            nxt = torch.where(finished, 2, nxt)
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            finished |= nxt == 2
            if finished.all():
                break
        return tokens, logp, finished


def latent_diagnostics(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        return {"collapsed": True, "mean_std": 0.0, "effective_rank": 0.0}
    centered = values - values.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular**2
    prob = power / max(power.sum(), 1e-12)
    rank = float(np.exp(-sum(p * math.log(p) for p in prob if p > 0)))
    std = float(values.std(0).mean())
    return {
        "collapsed": bool(std < 0.001 or rank < 2),
        "mean_std": std,
        "effective_rank": rank,
    }
