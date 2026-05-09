"""Unit tests for CLIC model components."""

import pytest
import torch
from src.models.clic import CLIC


MAIN_DIM  = 5
COND_DIM  = 8
SEQ_LEN   = 12
HIDDEN    = 32
LATENT    = 2
N_FLOWS   = 2
BATCH     = 8


@pytest.fixture
def model():
    return CLIC(
        main_dim=MAIN_DIM, cond_dim=COND_DIM, seq_len=SEQ_LEN,
        hidden_dim=HIDDEN, latent_dim=LATENT, num_flows=N_FLOWS,
        num_lstm_layers=1, scale_limit=0.5, dropout=0.0,
    )


@pytest.fixture
def batch():
    xm = torch.randn(BATCH, MAIN_DIM, SEQ_LEN)
    xc = torch.randn(BATCH, COND_DIM)
    return xm, xc


def test_forward_shapes(model, batch):
    xm, xc = batch
    x_recon, z, z_prime, log_det = model(xm, xc)
    assert x_recon.shape == xm.shape
    assert z.shape       == (BATCH, LATENT)
    assert z_prime.shape == (BATCH, LATENT)
    assert log_det.shape == (BATCH,)


def test_encoder_shape(model, batch):
    xm, xc = batch
    z = model.encoder(xm, xc)
    assert z.shape == (BATCH, LATENT)


def test_decoder_shape(model, batch):
    xm, xc = batch
    z = model.encoder(xm, xc)
    x_recon = model.decoder(z, xc)
    assert x_recon.shape == xm.shape


def test_flow_invertibility(model, batch):
    xm, xc = batch
    z = model.encoder(xm, xc)
    z_prime, _ = model.flow(z, xc)
    z_back = model.flow.inverse(z_prime, xc)
    assert torch.allclose(z, z_back, atol=1e-4)


def test_no_nan(model, batch):
    xm, xc = batch
    x_recon, z, z_prime, log_det = model(xm, xc)
    assert not torch.isnan(x_recon).any()
    assert not torch.isnan(z).any()
    assert not torch.isnan(log_det).any()
