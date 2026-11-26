"""Tests for the parametric cometric models."""

import torch

from geodesic_toolbox import cometric


def test_forward_pass_DiagonalCometricModel():
    in_dim = 2
    hidden_dim = 10
    latent_dim = 3
    lbd = 1.0
    model = cometric.DiagonalCometricModel(
        in_dim=in_dim, hidden_dim=hidden_dim, latent_dim=latent_dim, lbd=lbd
    )
    x = torch.randn(4, in_dim)
    out = model.cometric_tensor(x)
    assert out.shape == (4, latent_dim)
    assert torch.all(out > 0)
    inv_out = model.metric_tensor(x)
    assert inv_out.shape == (4, latent_dim)
    assert torch.all(inv_out > 0)
    print(model)


############################################################
# CometricModel cometric
############################################################


def test_forward_pass_CometricModel():
    in_dim = 2
    hidden_dim = 10
    latent_dim = 3
    lbd = 1.0
    model = cometric.CometricModel(
        input_dim=in_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        lbd=lbd,
    )
    x = torch.randn(4, in_dim)
    out = model.cometric_tensor(x)
    assert out.shape == (4, latent_dim, latent_dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)
    inv_out = model.metric_tensor(x)
    assert inv_out.shape == (4, latent_dim, latent_dim)
    for i in range(inv_out.shape[0]):
        eigvals = torch.linalg.eigvalsh(inv_out[i])
        assert torch.all(eigvals > 0)
    print(model)


############################################################
# SmallConvCometricModel cometric
############################################################


def test_forward_pass_SmallConvCometricModel():
    latent_dim = 13
    n_channels = 3
    width = 224
    lbd = 0.01
    model = cometric.SmallConvCometricModel(
        latent_dim=latent_dim,
        n_channels=n_channels,
        width=width,
        lbd=lbd,
    )
    x = torch.randn(4, n_channels, width, width)
    out = model.cometric_tensor(x)
    assert out.shape == (4, latent_dim, latent_dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


############################################################
# Cometric_MLP cometric
############################################################


def test_forward_pass_Cometric_MLP():
    input_dim = (5, 4, 3)
    latent_dim = 13
    lbd = 0.001
    cm = cometric.Cometric_MLP(
        input_dim=input_dim,
        latent_dim=latent_dim,
        lbd=lbd,
    )
    x = torch.randn(2, *input_dim)
    out = cm(x)
    assert out.shape == (2, latent_dim, latent_dim)
    for i in range(out.shape[0]):
        eigvals = torch.linalg.eigvalsh(out[i])
        assert torch.all(eigvals > 0)


############################################################
# SoftAbsCometric cometric
############################################################
