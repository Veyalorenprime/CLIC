"""OOD detection score computation."""

import math
import torch


def compute_ood_score(
    model,
    x_main: torch.Tensor,
    x_cond: torch.Tensor,
) -> torch.Tensor:
    """OOD score s_OOD = -log p(z'|a).

    Maps z through the flow to get the base-space code z', then evaluates
    the conditional prior log-probability. The Jacobian term is omitted,
    matching the formulation in the paper.

    Higher score means more anomalous.

    Args:
        model:  CLIC model with .encoder, .flow, .prior attributes
        x_main: Main input features [batch, main_dim, seq_len]
        x_cond: Conditional features [batch, cond_dim]

    Returns:
        s_ood: OOD scores [batch]  (higher = more anomalous)
    """
    with torch.no_grad():
        z = model.encoder(x_main, x_cond)
        z_prime, _ = model.flow(z, x_cond)

        if hasattr(model, 'prior') and model.prior is not None:
            log_pz = model.prior.log_prob(z_prime, x_cond)
        else:
            d = z_prime.shape[1]
            log_pz = (
                -0.5 * (z_prime ** 2).sum(dim=1)
                - 0.5 * d * math.log(2 * math.pi)
            )

        return -log_pz
