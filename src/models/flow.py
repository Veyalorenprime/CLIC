"""Stable Conditional RealNVP implementation with proper masking."""

import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    """Affine coupling layer with proper alternating masks."""

    def __init__(
        self,
        dim: int,
        aux_dim: int,
        hidden_dim: int = 64,
        scale_limit: float = 1.0,
        mask_type: str = "even",
    ):
        super().__init__()

        self.dim = dim
        self.scale_limit = scale_limit

        # ---------------------------------------------------
        # Proper binary mask
        # even  -> [1,0,1,0,...]
        # odd   -> [0,1,0,1,...]
        # ---------------------------------------------------
        mask = torch.zeros(dim)

        if mask_type == "even":
            mask[::2] = 1.0
        else:
            mask[1::2] = 1.0

        self.register_buffer("mask", mask)

        # ---------------------------------------------------
        # Networks see ALL masked dimensions
        # ---------------------------------------------------
        input_dim = dim + aux_dim

        self.net_s = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh(),  # bounded output
        )

        self.net_t = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
        )

        # ---------------------------------------------------
        # IMPORTANT: initialize scale output near identity
        # ---------------------------------------------------
        nn.init.zeros_(self.net_s[-2].weight)
        nn.init.zeros_(self.net_s[-2].bias)

        nn.init.zeros_(self.net_t[-1].weight)
        nn.init.zeros_(self.net_t[-1].bias)

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        """
        Forward transform x -> y.

        Args:
            x: [batch, dim]
            a: [batch, aux_dim]

        Returns:
            y: transformed tensor
            log_det: log determinant
        """

        mask = self.mask

        # -----------------------------------------
        # Keep masked part unchanged
        # -----------------------------------------
        x_masked = x * mask

        # Conditioning input
        cond_input = torch.cat([x_masked, a], dim=1)

        # -----------------------------------------
        # Scale + translation
        # -----------------------------------------
        s = self.net_s(cond_input) * self.scale_limit
        t = self.net_t(cond_input)

        # Only apply transform to unmasked dims
        s = s * (1.0 - mask)
        t = t * (1.0 - mask)

        # -----------------------------------------
        # Affine transform
        # -----------------------------------------
        y = x_masked + (1.0 - mask) * (x * torch.exp(s) + t)

        # Log determinant
        log_det = s.sum(dim=1)

        return y, log_det

    def inverse(self, y: torch.Tensor, a: torch.Tensor):
        """
        Inverse transform y -> x.
        """

        mask = self.mask

        y_masked = y * mask

        cond_input = torch.cat([y_masked, a], dim=1)

        s = self.net_s(cond_input) * self.scale_limit
        t = self.net_t(cond_input)

        s = s * (1.0 - mask)
        t = t * (1.0 - mask)

        x = y_masked + (1.0 - mask) * ((y - t) * torch.exp(-s))

        return x


class ConditionalRealNVP(nn.Module):
    """Stable conditional RealNVP."""

    def __init__(
        self,
        latent_dim: int,
        aux_dim: int,
        num_flows: int = 4,
        hidden_dim: int = 64,
        scale_limit: float = 1.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                CouplingLayer(
                    dim=latent_dim,
                    aux_dim=aux_dim,
                    hidden_dim=hidden_dim,
                    scale_limit=scale_limit,
                    mask_type="even" if i % 2 == 0 else "odd",
                )
                for i in range(num_flows)
            ]
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor):
        """
        x -> z
        """

        log_det_total = torch.zeros(x.shape[0], device=x.device)

        z = x

        for layer in self.layers:
            z, log_det = layer(z, a)
            log_det_total += log_det

        return z, log_det_total

    def inverse(self, z: torch.Tensor, a: torch.Tensor):
        """
        z -> x
        """

        x = z

        for layer in reversed(self.layers):
            x = layer.inverse(x, a)

        return x