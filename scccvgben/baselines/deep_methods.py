"""deep_methods.py — Deep generative baselines with ELBO variants.

Each method follows the convention:
    run_<method>(X: np.ndarray, n_components: int = 10) -> np.ndarray

Returns z_mean (latent mean) of shape (n_cells, n_components).
All tiny VAEs: encoder MLP → (mean, logvar) → decoder MLP.
Training: 50 epochs, Adam lr=1e-4, batch size = min(256, N).
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

log = logging.getLogger(__name__)

_EPOCHS = 50
_LR = 1e-4
_LATENT_DIM = 10
_HIDDEN = 128


# ──────────────────────────────────────────────────────────────────────────────
# Shared building blocks
# ──────────────────────────────────────────────────────────────────────────────

class _BaseVAE(nn.Module):
    """Minimal VAE backbone shared by all deep baselines."""

    def __init__(self, input_dim: int, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder_fc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.fc_mean = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x: torch.Tensor):
        h = self.encoder_fc(x)
        return self.fc_mean(h), self.fc_logvar(h)

    def reparameterise(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterise(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


def _standard_elbo(x, x_hat, mean, logvar):
    recon = F.mse_loss(x_hat, x, reduction="sum")
    kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    return recon + kl


def _train_vae(
    model: _BaseVAE,
    X: np.ndarray,
    loss_fn,
    epochs: int = _EPOCHS,
    lr: float = _LR,
) -> np.ndarray:
    """Generic train loop; returns z_mean for all cells."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    Xt = torch.tensor(X, dtype=torch.float32)
    bs = min(256, len(Xt))
    loader = DataLoader(TensorDataset(Xt), batch_size=bs, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            batch = batch.to(device)
            x_hat, mean, logvar = model(batch)
            loss = loss_fn(batch, x_hat, mean, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        mean_all, _ = model.encode(Xt.to(device))
    return mean_all.cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# DIPVAE (Disentangled Inferred Prior)
# ELBO + lambda * (off-diagonal covariance terms of E[z] penalised)
# ──────────────────────────────────────────────────────────────────────────────

def run_DIPVAE(X: np.ndarray, n_components: int = _LATENT_DIM) -> np.ndarray:
    """DIPVAE: standard ELBO + DIP-II covariance penalty (lambda=10)."""
    lam = 10.0

    def loss_fn(x, x_hat, mean, logvar):
        base = _standard_elbo(x, x_hat, mean, logvar)
        # DIP-II: penalise off-diagonal entries of Cov[E[z]]
        cov = torch.mm(mean.T, mean) / mean.shape[0]
        off_diag = cov - torch.diag(torch.diag(cov))
        dip = lam * off_diag.pow(2).sum()
        return base + dip

    model = _BaseVAE(X.shape[1], _HIDDEN, n_components)
    return _train_vae(model, X, loss_fn)


# ──────────────────────────────────────────────────────────────────────────────
# InfoVAE (Mutual Information VAE / MMD-VAE)
# ELBO + alpha * MMD(q(z) || p(z))
# ──────────────────────────────────────────────────────────────────────────────

def _rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    dist = torch.cdist(x, y).pow(2)
    return torch.exp(-dist / (2 * sigma ** 2))


def _mmd(z: torch.Tensor) -> torch.Tensor:
    n = z.shape[0]
    z_prior = torch.randn_like(z)
    kzz = _rbf_kernel(z, z).sum() / (n * n)
    kpp = _rbf_kernel(z_prior, z_prior).sum() / (n * n)
    kzp = _rbf_kernel(z, z_prior).sum() / (n * n)
    return kzz + kpp - 2 * kzp


def run_InfoVAE(X: np.ndarray, n_components: int = _LATENT_DIM) -> np.ndarray:
    """InfoVAE (MMD-VAE): ELBO + 100 * MMD(q(z)||N(0,I))."""
    alpha = 100.0

    def loss_fn(x, x_hat, mean, logvar):
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return _standard_elbo(x, x_hat, mean, logvar) + alpha * _mmd(z)

    model = _BaseVAE(X.shape[1], _HIDDEN, n_components)
    return _train_vae(model, X, loss_fn)


# ──────────────────────────────────────────────────────────────────────────────
# TCVAE (Total Correlation VAE)
# ELBO - beta * TC(q(z)) using minibatch-weighted estimator
# ──────────────────────────────────────────────────────────────────────────────

def run_TCVAE(X: np.ndarray, n_components: int = _LATENT_DIM) -> np.ndarray:
    """TCVAE: decomposes ELBO; penalises total-correlation term (beta=6)."""
    beta = 6.0
    N = float(len(X))

    def loss_fn(x, x_hat, mean, logvar):
        recon = F.mse_loss(x_hat, x, reduction="sum")
        # log q(z|x) per sample
        log_q_zx = -0.5 * (logvar + (mean - mean).pow(2) / logvar.exp() + np.log(2 * np.pi)).sum(-1)
        # log p(z)
        log_pz = -0.5 * (mean.pow(2) + logvar.exp() + np.log(2 * np.pi)).sum(-1)
        # minibatch TC approximation (subtract aggregate)
        _z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        log_prod_qz = -0.5 * (
            _z.unsqueeze(0) - mean.unsqueeze(1)
        ).pow(2).div(logvar.exp().unsqueeze(1)).sum(-1).logsumexp(0) + np.log(N)
        tc = (log_q_zx - log_prod_qz - log_pz).mean()
        kl = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp()).sum()
        return recon + kl + (beta - 1) * tc

    model = _BaseVAE(X.shape[1], _HIDDEN, n_components)
    return _train_vae(model, X, loss_fn)


# ──────────────────────────────────────────────────────────────────────────────
# HighBetaVAE (beta-VAE with large beta)
# ELBO with beta=10 KL weight
# ──────────────────────────────────────────────────────────────────────────────

def run_HighBetaVAE(X: np.ndarray, n_components: int = _LATENT_DIM) -> np.ndarray:
    """beta-VAE with beta=10: over-penalises KL to encourage disentanglement."""
    beta = 10.0

    def loss_fn(x, x_hat, mean, logvar):
        recon = F.mse_loss(x_hat, x, reduction="sum")
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return recon + beta * kl

    model = _BaseVAE(X.shape[1], _HIDDEN, n_components)
    return _train_vae(model, X, loss_fn)


# ──────────────────────────────────────────────────────────────────────────────
# scVI wrapper
# ──────────────────────────────────────────────────────────────────────────────

def run_scVI(X: np.ndarray, n_components: int = _LATENT_DIM) -> np.ndarray:
    """scVI latent via scvi-tools (requires scvi-tools installed).

    X should be raw integer count matrix (n_cells × n_genes).
    Returns z_mean of shape (n_cells, n_components).
    """
    try:
        import scvi
        import anndata as ad
    except ImportError as exc:
        raise NotImplementedError(
            "scVI requires scvi-tools and anndata. "
            "Install with: pip install scvi-tools anndata. "
            f"Original error: {exc}"
        ) from exc

    import pandas as pd

    n_cells, n_genes = X.shape
    adata = ad.AnnData(
        X=X.astype(np.float32),
        obs=pd.DataFrame(index=[str(i) for i in range(n_cells)]),
        var=pd.DataFrame(index=[str(j) for j in range(n_genes)]),
    )
    scvi.model.SCVI.setup_anndata(adata)
    vae = scvi.model.SCVI(adata, n_latent=n_components)
    vae.train(max_epochs=_EPOCHS, plan_kwargs={"lr": _LR})
    latent = vae.get_latent_representation()
    return np.array(latent, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Dispatch registry
# ──────────────────────────────────────────────────────────────────────────────

DEEP_REGISTRY: dict[str, callable] = {
    "scVI": run_scVI,
    "DIP": run_DIPVAE,
    "INFO": run_InfoVAE,
    "TC": run_TCVAE,
    "highBeta": run_HighBetaVAE,
}
