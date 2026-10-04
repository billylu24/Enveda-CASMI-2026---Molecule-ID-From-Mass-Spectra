"""Experimental nonoracle heavy-atom budget sampling with original final validation."""

import re
import time

import numpy as np
import torch

from casmi_ml.generation_experiment import ELEMENTS


def atom_count_tokens(vocabulary):
    """Count explicit heavy atoms; implicit and bracket hydrogen stay unconstrained."""
    known = {e: i for i, e in enumerate(ELEMENTS) if e != "H"}
    matrix = np.zeros((len(vocabulary.tokens), len(ELEMENTS)), np.int64)
    unsupported = np.zeros(len(vocabulary.tokens), bool)
    for index, token in enumerate(vocabulary.tokens[4:], 4):
        match = (
            re.match(r"^\[\d*([A-Z][a-z]?|se|as|[cnospb])", token)
            if token.startswith("[")
            else None
        )
        element = (
            match[1].capitalize()
            if match
            else token.capitalize()
            if re.fullmatch(r"[A-Z][a-z]?|se|as|[cnospb]", token)
            else None
        )
        if element and element != "H":
            if element in known:
                matrix[index, known[element]] = 1
            else:
                unsupported[index] = True
    return matrix, unsupported


def sample(
    decoder,
    condition,
    targets,
    active,
    token_counts,
    unsupported,
    *,
    samples=128,
    temperature=0.8,
    generator=None,
    deadline=None,
):
    """Early element ceilings plus heavy-count EOS feasibility, never a formula oracle.

    Targets must come from observable-mass proposals. Original learned probability
    and SMILES grammar remain unchanged; final RDKit connectedness/exact mass are
    still mandatory. Inactive trajectories have original sampling constraints.
    """
    if samples != 128 or temperature != 0.8:
        raise ValueError("Fixed original128/.8 pilot sampling required")
    decoder.eval()
    condition = condition.expand(samples, -1)
    device = condition.device
    targets = torch.as_tensor(targets, dtype=torch.long, device=device)
    active = torch.as_tensor(active, dtype=torch.bool, device=device)
    weights = torch.as_tensor(token_counts, dtype=torch.long, device=device)
    unsupported = torch.as_tensor(unsupported, dtype=torch.bool, device=device)
    if (
        targets.shape != (samples, len(ELEMENTS))
        or active.shape != (samples,)
        or weights.shape != (decoder.head.out_features, len(ELEMENTS))
        or unsupported.shape != (decoder.head.out_features,)
        or (targets < 0).any()
        or targets[:, 1].any()
        or (weights < 0).any()
        or weights[:, 1].any()
        or weights[2].any()
    ):
        raise ValueError("Aligned nonnegative heavy-atom token budgets required")
    tokens = torch.ones((samples, 1), dtype=torch.long, device=device)
    finished = torch.zeros(samples, dtype=torch.bool, device=device)
    likelihood = torch.zeros(samples, device=device)
    used = torch.zeros_like(targets)
    with torch.inference_mode():
        for _ in range(decoder.limit - 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(
                    "Formula budget generation deadline; discard incomplete query"
                )
            logits = decoder(tokens, condition)[:, -1] / temperature
            logits[:, [0, 1, 3]] = float("-inf")
            exceeded = (used[:, None] + weights[None] > targets[:, None]).any(-1)
            logits.masked_fill_(
                active[:, None] & (exceeded | unsupported[None]), float("-inf")
            )
            logits[active & (used != targets).any(-1), 2] = float("-inf")
            probabilities = logits.softmax(-1)
            if not torch.isfinite(probabilities).all():
                raise ValueError("No finite formula-budget continuation")
            nxt = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            likelihood += torch.where(
                finished,
                0.0,
                probabilities.gather(1, nxt[:, None]).squeeze(1).clamp_min(1e-12).log(),
            )
            nxt = torch.where(finished, 2, nxt)
            used += weights[nxt]
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            finished |= nxt == 2
            if finished.all():
                break
    return tokens, likelihood, finished
