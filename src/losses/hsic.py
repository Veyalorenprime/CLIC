"""HSIC independence penalty."""

import torch


def _median_bandwidth(dist_sq: torch.Tensor) -> float:
    """Median heuristic: σ = median(pairwise distances), clipped to [0.1, ∞)."""
    positive = dist_sq[dist_sq > 0]
    if positive.numel() == 0:
        return 1.0
    return max(float(positive.median().sqrt()), 0.1)


def compute_hsic(z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Unbiased HSIC independence criterion with RBF kernels.

    Measures statistical dependence between latent codes z and auxiliary
    variables a. Zero means independent; larger values mean more dependent.
    Bandwidths are set automatically via the median heuristic.

    Args:
        z: Latent codes [batch, latent_dim]
        a: Auxiliary variables [batch, aux_dim]

    Returns:
        Scalar HSIC value (≥ 0).
    """
    n = z.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=z.device)

    z_sq = (z ** 2).sum(1, keepdim=True)
    a_sq = (a ** 2).sum(1, keepdim=True)
    dz = (z_sq + z_sq.T - 2 * z @ z.T).clamp(min=0)
    da = (a_sq + a_sq.T - 2 * a @ a.T).clamp(min=0)

    K = torch.exp(-dz / (2 * _median_bandwidth(dz) ** 2))
    L = torch.exp(-da / (2 * _median_bandwidth(da) ** 2))

    H = torch.eye(n, device=z.device) - 1.0 / n
    Kc = H @ K @ H
    Lc = H @ L @ H

    return (Kc * Lc).sum() / (n - 1) ** 2
