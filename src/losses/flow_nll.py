"""Flow NLL loss with conditional exponential family prior."""

import math
import torch
import torch.nn as nn


class ConditionalPrior(nn.Module):
    """Exponential family conditional prior p(z|a) = N(μ(a), diag(σ²(a))).

    Parameterizes Gaussian sufficient statistics (μ, log σ) as functions
    of auxiliary variables a, following Khemakhem et al. 2020 (iVAE).

    Args:
        aux_dim: Dimension of auxiliary variables
        latent_dim: Dimension of latent space
    """

    def __init__(self, aux_dim: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(aux_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * latent_dim),
        )
        # Init to identity: μ=0, log_σ=0 → N(0,I) at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) each of shape [batch, latent_dim]."""
        out = self.net(a)
        mu, log_sigma = out.chunk(2, dim=-1)
        return mu, log_sigma.clamp(-4, 4)

    def log_prob(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Log N(z; μ(a), diag(exp(2 log_σ(a)))) summed over latent dim."""
        mu, log_sigma = self(a)
        return (
            -0.5 * ((z - mu) / log_sigma.exp()) ** 2
            - log_sigma
            - 0.5 * math.log(2 * math.pi)
        ).sum(dim=1)


def nll_loss(
    z: torch.Tensor,
    log_det: torch.Tensor,
    prior: ConditionalPrior | None = None,
    a: torch.Tensor | None = None,
) -> torch.Tensor:
    """Negative log-likelihood: -E[log p_a(z) + log|det J|].

    Uses conditional prior N(μ(a), σ²(a)) when prior and a are given,
    otherwise falls back to standard N(0, I).
    """
    if prior is not None and a is not None:
        log_pz = prior.log_prob(z, a)
    else:
        d = z.shape[1]
        log_pz = -0.5 * (z ** 2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)

    return -(log_pz + log_det).mean()
