"""Unit tests for loss functions"""

import pytest
import torch
from src.losses import reconstruction_loss, compute_hsic, nll_loss


def test_reconstruction_loss():
    """Test reconstruction loss."""
    x = torch.randn(16, 3, 24)
    x_recon = x + 0.1 * torch.randn_like(x)
    
    loss = reconstruction_loss(x_recon, x)
    
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"


def test_hsic_computation():
    """Test HSIC computation."""
    z = torch.randn(32, 2)
    a = torch.randn(32, 3)
    
    hsic = compute_hsic(z, a)
    
    assert hsic.shape == torch.Size([]), "HSIC should be scalar"
    assert hsic.item() >= 0, "HSIC should be non-negative"


def test_nll_loss():
    """Test NLL loss."""
    z = torch.randn(16, 2)
    log_det = torch.randn(16)
    
    loss = nll_loss(z, log_det)
    
    assert loss.shape == torch.Size([]), "Loss should be scalar"
    assert loss.item() > 0, "Loss should be positive"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
