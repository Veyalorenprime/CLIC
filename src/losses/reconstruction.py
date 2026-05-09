import torch


def reconstruction_loss(x_recon: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Mean squared reconstruction error.
    """

    return torch.mean((x_recon - x) ** 2)