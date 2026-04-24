"""Locked hyperparameter configs for scRNA and scATAC modalities."""

from __future__ import annotations

LOCKED_CONFIG: dict[str, dict] = {
    "scrna": {
        "subsample_cells": 3000,
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "hidden_dim": 128,
        "latent_dim": 10,
        "i_dim": 5,
        "n_enc_layers": 2,
        "n_dec_layers": 2,
        "dropout": 0.1,
        "loss_weights": {"recon": 1.0, "irecon": 0.5, "kl": 0.01, "adj": 1.0},
        "kNN_k": 15,
        "pca_dim": 50,
        "hvg_count": 2000,
        "seed": 42,
        "use_mse_likelihood": True,
    },
    "scatac": {
        "subsample_cells": 3000,
        "epochs": 100,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "hidden_dim": 128,
        "latent_dim": 10,
        "i_dim": 5,
        "n_enc_layers": 2,
        "n_dec_layers": 2,
        "dropout": 0.1,
        "loss_weights": {"recon": 1.0, "irecon": 0.5, "kl": 0.01, "adj": 1.0},
        "kNN_k": 15,
        "lsi_dim": 50,
        "hv_peak_count": 2000,
        "seed": 42,
        "use_mse_likelihood": True,
    },
}


def get_config(modality: str) -> dict:
    """Return a copy of the locked config for the given modality."""
    return LOCKED_CONFIG[modality].copy()
