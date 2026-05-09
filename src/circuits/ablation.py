"""Mean ablation circuit decomposition for additive LSTM decoder."""

import torch


def compute_circuit_contributions(
    model,
    x: torch.Tensor,
    a: torch.Tensor,
    z_mean: torch.Tensor,
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-dimension signed circuit contributions Δᵢ = h(z) - h(z^(i)).

    With the additive decoder x_recon = h∅(a) + Σᵢ fᵢ(zᵢ, a), ablating
    dimension i reduces to:
        Δᵢ = fᵢ(zᵢ, a) - fᵢ(z̄ᵢ, a)
    The baseline h∅(a) cancels exactly, so completeness Σᵢ Δᵢ = h(z) - h(z̄)
    holds by construction (no approximation error).

    Args:
        model:  CLIC model with .encoder and .decoder attributes
        x:      Input observations [batch, C, T]
        a:      Auxiliary variables [batch, aux_dim]
        z_mean: In-distribution mean of latent [latent_dim]
        w:      Linear probe vector [C*T]. Defaults to uniform 1/(C·T).

    Returns:
        deltas: Signed circuit contributions [batch, latent_dim]
    """
    with torch.no_grad():
        z      = model.encoder(x, a)
        x_full = model.decoder(z, a)

        batch, C, T = x_full.shape
        if w is None:
            w = torch.ones(C * T, device=x.device) / (C * T)

        h_full = x_full.reshape(batch, -1) @ w

        deltas = []
        for i in range(z.shape[1]):
            z_abl      = z.clone()
            z_abl[:, i] = z_mean[i]
            x_abl      = model.decoder(z_abl, a)
            deltas.append(h_full - x_abl.reshape(batch, -1) @ w)

    return torch.stack(deltas, dim=1)


def compute_baseline(
    model,
    a: torch.Tensor,
    w: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scalar baseline h∅(a) = w^T decoder.baseline(a).

    Returns the additive decoder's environment-only output — what the model
    predicts when no latent circuit is activated.  Only meaningful when the
    decoder is an AdditiveDecoder (ignored otherwise).

    Args:
        model: CLIC model whose decoder has a .baseline sub-module
        a:     Auxiliary variables [batch, aux_dim]
        w:     Linear probe [C*T]. Defaults to uniform 1/(C·T).

    Returns:
        h_null: Scalar baseline per sample [batch]
    """
    with torch.no_grad():
        if not hasattr(model.decoder, 'baseline'):
            raise AttributeError("decoder has no .baseline — use AdditiveDecoder")

        x_null      = model.decoder.baseline(a)
        batch, C, T = x_null.shape
        if w is None:
            w = torch.ones(C * T, device=a.device) / (C * T)

        return x_null.reshape(batch, -1) @ w
