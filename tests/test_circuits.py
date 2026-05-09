"""Unit tests for circuit analysis tools."""

import pytest
import torch
from src.models.clic import CLIC
from src.circuits import compute_circuit_contributions, compute_ood_score


MAIN_DIM = 5
COND_DIM = 8
SEQ_LEN  = 12
LATENT   = 2
BATCH    = 8


@pytest.fixture
def model():
    return CLIC(
        main_dim=MAIN_DIM, cond_dim=COND_DIM, seq_len=SEQ_LEN,
        hidden_dim=32, latent_dim=LATENT, num_flows=2,
        num_lstm_layers=1, scale_limit=0.5, dropout=0.0,
    ).eval()


@pytest.fixture
def batch():
    xm = torch.randn(BATCH, MAIN_DIM, SEQ_LEN)
    xc = torch.randn(BATCH, COND_DIM)
    return xm, xc


def test_circuit_contributions_shape(model, batch):
    xm, xc = batch
    z_mean = torch.zeros(LATENT)
    deltas = compute_circuit_contributions(model, xm, xc, z_mean)
    assert deltas.shape == (BATCH, LATENT)


def test_circuit_contributions_no_nan(model, batch):
    xm, xc = batch
    z_mean = torch.zeros(LATENT)
    deltas = compute_circuit_contributions(model, xm, xc, z_mean)
    assert not torch.isnan(deltas).any()


def test_ood_score_shape(model, batch):
    xm, xc = batch
    scores = compute_ood_score(model, xm, xc)
    assert scores.shape == (BATCH,)


def test_ood_score_no_nan(model, batch):
    xm, xc = batch
    scores = compute_ood_score(model, xm, xc)
    assert not torch.isnan(scores).any()


def test_ood_score_finite(model, batch):
    xm, xc = batch
    scores = compute_ood_score(model, xm, xc)
    assert torch.isfinite(scores).all()
