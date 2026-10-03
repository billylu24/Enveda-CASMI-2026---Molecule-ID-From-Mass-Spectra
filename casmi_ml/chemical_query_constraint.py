"""Training constraint isolating chemical identities in the query correction branch."""

import torch

LATENT_DIM = 768


def constrain_context_weights(residual):
    """Zero latent columns persistently; checkpoint keeps the verified forward graph."""
    weight = residual.correction[0].weight
    if weight.shape != (64, LATENT_DIM + 24):
        raise ValueError("Expected verified64x792 query input layer")
    with torch.no_grad():
        weight[:, :LATENT_DIM].zero_()

    def chemical_gradient(gradient):
        masked = gradient.clone()
        masked[:, :LATENT_DIM] = 0
        return masked

    return weight.register_hook(chemical_gradient)


def verify_context_weights(residual):
    if bool(residual.correction[0].weight[:, :LATENT_DIM].count_nonzero()):
        raise ValueError("Chemical-only correction has nonzero latent weights")
