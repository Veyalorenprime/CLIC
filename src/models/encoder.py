"""Conditional LSTM Encoder for CLIC."""

import torch
import torch.nn as nn


class ConditionalLSTMEncoder(nn.Module):
    """LSTM encoder over electrical features with direct conditional concatenation.

    The LSTM processes main (electrical) features; its final hidden state is
    concatenated with the conditional (environmental) features and projected
    to the latent space.  This gives the encoder awareness of environment
    without FiLM modulation.

    Args:
        main_dim:   Number of main (electrical) input features
        cond_dim:   Number of conditional (environmental) features
        seq_len:    Sequence length
        hidden_dim: LSTM hidden size
        latent_dim: Output latent dimension
        num_layers: Number of LSTM layers
        dropout:    Dropout rate
    """

    def __init__(
        self,
        main_dim: int = 5,
        cond_dim: int = 8,
        seq_len: int = 12,
        hidden_dim: int = 128,
        latent_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.main_dim   = main_dim
        self.cond_dim   = cond_dim
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=main_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )

        # Project [h_last || x_cond] → latent
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        for m in self.projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_main: torch.Tensor, x_cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_main: Electrical features  [batch, main_dim, seq_len]
            x_cond: Environmental features [batch, cond_dim]

        Returns:
            z: Latent codes [batch, latent_dim]
        """
        _, (h_n, _) = self.lstm(x_main.transpose(1, 2))
        h_last = h_n[-1]                              # [batch, hidden_dim]
        return self.projection(torch.cat([h_last, x_cond], dim=-1))


Encoder = ConditionalLSTMEncoder
