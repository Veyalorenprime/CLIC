"""CLIC model."""

import torch
import torch.nn as nn
from typing import Tuple

from .encoder import ConditionalLSTMEncoder
from .decoder import AdditiveDecoder
from .flow import ConditionalRealNVP
from src.losses.flow_nll import ConditionalPrior


class CLIC(nn.Module):
    """Causal Latent Interpretable Circuits model.

    Separates main (electrical) features from conditional (environmental)
    features. The encoder produces independent latent factors z; the flow
    maps z to z_prime under a conditional exponential family prior p(z|a);
    the piecewise affine decoder reconstructs main features from z_prime.
    """

    def __init__(
        self,
        main_dim: int = 5,
        cond_dim: int = 8,
        seq_len: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_flows: int = 4,
        num_lstm_layers: int = 2,
        scale_limit: float = 1.5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.main_dim = main_dim
        self.cond_dim = cond_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.encoder = ConditionalLSTMEncoder(
            main_dim=main_dim,
            cond_dim=cond_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )

        self.flow = ConditionalRealNVP(
            latent_dim=latent_dim,
            aux_dim=cond_dim,
            num_flows=num_flows,
            scale_limit=scale_limit,
        )

        # Exponential family conditional prior p(z|a) = N(μ(a), σ²(a))
        self.prior = ConditionalPrior(aux_dim=cond_dim, latent_dim=latent_dim)

        self.decoder = AdditiveDecoder(
            latent_dim=latent_dim,
            aux_dim=cond_dim,
            output_dim=main_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_lstm_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x_main: torch.Tensor,   # [batch, main_dim, seq_len]
        x_cond: torch.Tensor,   # [batch, cond_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x_recon: [batch, main_dim, seq_len]
            z:       pre-flow latent [batch, latent_dim]
            z_prime: post-flow latent [batch, latent_dim]
            log_det: [batch]
        """
        z = self.encoder(x_main, x_cond)
        z_prime, log_det = self.flow(z, x_cond)
        # Decoder uses pre-flow z: reconstruction and NLL have separate gradient paths
        x_recon = self.decoder(z, x_cond)
        return x_recon, z, z_prime, log_det

    def encode(self, x_main: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        return self.encoder(x_main, x_cond)

    def decode(self, z: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, x_cond)
