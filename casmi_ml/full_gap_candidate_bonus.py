"""Dimension-general convex candidate bonus with a fixed total cosine budget."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from casmi_ml.full_gap_candidate_compatibility import BONUS_BUDGET


def fit_bonus(cosines, interactions, scale, mask=False, checkpoint=None):
    dimension = np.shape(interactions)[-1]
    if not isinstance(dimension, int) or dimension < 1:
        raise ValueError("Positive feature dimension required")
    cosines = np.asarray(cosines, np.float64)
    interactions = np.asarray(interactions, np.float64)
    if (
        cosines.ndim != 2
        or cosines.shape[1] != 16
        or interactions.shape != (*cosines.shape, dimension)
    ):
        raise ValueError(
            "Positive+15 candidate scores/chemical feature interactions must align"
        )
    if (
        not np.isfinite(cosines).all()
        or not np.isfinite(interactions).all()
        or not (0 < scale <= 50)
    ):
        raise ValueError("Finite frozen critic and chemistry required")
    if ((interactions < 0) | (interactions > 1)).any():
        raise ValueError("Candidate interactions must lie within [0,1]")
    if mask:
        interactions = np.zeros_like(interactions)

    def objective(weight):
        if checkpoint is not None and not checkpoint():
            raise RuntimeError("Candidate chemistry stage budget exhausted")
        bonus = interactions @ weight
        logits = (cosines + bonus) * scale
        normalizer = logsumexp(logits, axis=1)
        probability = np.exp(logits - normalizer[:, None])
        probability[:, 0] -= 1
        loss = (normalizer - logits[:, 0]).mean() + 0.01 * (weight @ weight) / 2
        gradient = (
            np.einsum("bc,bcd->d", probability, interactions) * scale / len(cosines)
            + 0.01 * weight
        )
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(dimension),
        jac=True,
        method="SLSQP",
        bounds=[(0.0, BONUS_BUDGET)] * dimension,
        constraints=[
            {
                "type": "ineq",
                "fun": lambda w: BONUS_BUDGET - w.sum(),
                "jac": lambda w: -np.ones(dimension),
            }
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError(
            "Bounded candidate chemistry optimizer failed: " + result.message
        )
    weight = np.maximum(result.x, 0)
    if weight.sum() > BONUS_BUDGET:
        weight *= BONUS_BUDGET / weight.sum()
    return weight, {
        "initial_ce": float(objective(np.zeros(dimension))[0]),
        "final_objective": float(objective(weight)[0]),
        "optimizer_iterations": int(result.nit),
        "sum_nonnegative_cosine_weights": float(weight.sum()),
        "maximum_possible_bonus": BONUS_BUDGET,
    }
