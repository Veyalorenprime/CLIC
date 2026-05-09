"""LSTM decoders for CLIC."""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """LSTM decoder that reconstructs sequences from latent codes.

    z initializes the LSTM hidden state; a and timestep embeddings
    are fed as input at each step, giving the decoder temporal structure
    that mirrors the LSTM encoder.

    Args:
        latent_dim: Dimension of latent code z
        aux_dim: Dimension of auxiliary variables a
        output_dim: Number of output features
        seq_len: Sequence length to generate
        hidden_dim: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        aux_dim: int,
        output_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        timestep_dim = 8

        # Map z to initial LSTM hidden state
        self.z_to_h0 = nn.Linear(latent_dim, num_layers * hidden_dim)

        # Learnable timestep embeddings
        self.timestep_embed = nn.Embedding(seq_len, timestep_dim)

        # Each step receives [a, timestep_embedding]
        self.lstm = nn.LSTM(
            input_size=aux_dim + timestep_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.z_to_h0.weight)
        nn.init.zeros_(self.z_to_h0.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes [batch, latent_dim]
            a: Auxiliary variables [batch, aux_dim]

        Returns:
            x_recon: Reconstructed sequences [batch, output_dim, seq_len]
        """
        batch = z.size(0)

        # Build initial hidden state from z
        h0 = self.z_to_h0(z)  # [batch, num_layers * hidden_dim]
        h0 = h0.view(batch, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = torch.zeros_like(h0)

        # Build per-timestep inputs: [a, timestep_embedding]
        t_idx = torch.arange(self.seq_len, device=z.device)
        t_emb = self.timestep_embed(t_idx).unsqueeze(0).expand(batch, -1, -1)  # [batch, seq_len, timestep_dim]
        a_exp = a.unsqueeze(1).expand(-1, self.seq_len, -1)                     # [batch, seq_len, aux_dim]
        lstm_input = torch.cat([a_exp, t_emb], dim=-1)                          # [batch, seq_len, aux_dim+timestep_dim]

        # Decode sequence
        out, _ = self.lstm(lstm_input, (h0, c0))  # [batch, seq_len, hidden_dim]

        x_recon = self.output_proj(out)            # [batch, seq_len, output_dim]
        return x_recon.permute(0, 2, 1)            # [batch, output_dim, seq_len]


class _BaselineDecoder(nn.Module):
    """Baseline h∅ = E_{p*}[h(z)]: output conditioned on a only, independent of z.

    Represents the expected reconstruction under the learned prior — what the
    model predicts from environment alone when no latent circuit is activated.
    The initial hidden state is a learnable parameter rather than a function of z.
    """

    def __init__(
        self,
        aux_dim: int,
        output_dim: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        timestep_dim = 8

        # Learnable initial state — no dependence on z
        self.h0 = nn.Parameter(torch.zeros(num_layers, 1, hidden_dim))

        self.timestep_embed = nn.Embedding(seq_len, timestep_dim)

        self.lstm = nn.LSTM(
            input_size=aux_dim + timestep_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a: Auxiliary variables [batch, aux_dim]

        Returns:
            h_null: [batch, output_dim, seq_len]
        """
        batch = a.size(0)

        h0 = self.h0.expand(-1, batch, -1).contiguous()
        c0 = torch.zeros_like(h0)

        t_idx = torch.arange(self.seq_len, device=a.device)
        t_emb = self.timestep_embed(t_idx).unsqueeze(0).expand(batch, -1, -1)
        a_exp = a.unsqueeze(1).expand(-1, self.seq_len, -1)
        lstm_input = torch.cat([a_exp, t_emb], dim=-1)

        out, _ = self.lstm(lstm_input, (h0, c0))
        return self.output_proj(out).permute(0, 2, 1)   # [batch, output_dim, seq_len]


class _SubDecoder(nn.Module):
    """Single-dimension LSTM decoder: maps one scalar zᵢ + a → sequence."""

    def __init__(
        self,
        aux_dim: int,
        output_dim: int,
        seq_len: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        timestep_dim = 8

        self.z_to_h0       = nn.Linear(1, num_layers * hidden_dim)
        self.timestep_embed = nn.Embedding(seq_len, timestep_dim)

        self.lstm = nn.LSTM(
            input_size=aux_dim + timestep_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.z_to_h0.weight)
        nn.init.zeros_(self.z_to_h0.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_i: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Single latent scalar [batch, 1]
            a:   Auxiliary variables  [batch, aux_dim]

        Returns:
            [batch, output_dim, seq_len]
        """
        batch = z_i.size(0)

        h0 = self.z_to_h0(z_i)
        h0 = h0.view(batch, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
        c0 = torch.zeros_like(h0)

        t_idx = torch.arange(self.seq_len, device=z_i.device)
        t_emb = self.timestep_embed(t_idx).unsqueeze(0).expand(batch, -1, -1)
        a_exp = a.unsqueeze(1).expand(-1, self.seq_len, -1)
        lstm_input = torch.cat([a_exp, t_emb], dim=-1)

        out, _ = self.lstm(lstm_input, (h0, c0))
        return self.output_proj(out).permute(0, 2, 1)   # [batch, output_dim, seq_len]


class AdditiveDecoder(nn.Module):
    """Additive LSTM decoder: x_recon = Σᵢ fᵢ(zᵢ, a).

    Each latent dimension drives its own LSTM sub-decoder independently.
    Summing their outputs guarantees exact Hoeffding additivity — the
    circuit contribution of dimension i is exactly fᵢ(zᵢ, a) - fᵢ(z̄ᵢ, a)
    with zero cross-term leakage.

    Adding or removing latent dimensions only adds/removes one sub-decoder;
    all other sub-decoders are unaffected.

    Args:
        latent_dim: Number of latent dimensions (= number of sub-decoders)
        aux_dim:    Dimension of auxiliary/conditioning variables
        output_dim: Number of reconstructed features
        seq_len:    Output sequence length
        hidden_dim: LSTM hidden size per sub-decoder
        num_layers: LSTM layers per sub-decoder
        dropout:    Dropout rate
    """

    def __init__(
        self,
        latent_dim: int,
        aux_dim: int,
        output_dim: int,
        seq_len: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.baseline = _BaselineDecoder(
            aux_dim=aux_dim,
            output_dim=output_dim,
            seq_len=seq_len,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.sub_decoders = nn.ModuleList([
            _SubDecoder(
                aux_dim=aux_dim,
                output_dim=output_dim,
                seq_len=seq_len,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
            )
            for _ in range(latent_dim)
        ])

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent codes [batch, latent_dim]
            a: Auxiliary variables [batch, aux_dim]

        Returns:
            x_recon: [batch, output_dim, seq_len]
                     = h∅(a) + Σᵢ fᵢ(zᵢ, a)
        """
        out = self.baseline(a)
        for i, sub in enumerate(self.sub_decoders):
            out = out + sub(z[:, i:i + 1], a)
        return out
