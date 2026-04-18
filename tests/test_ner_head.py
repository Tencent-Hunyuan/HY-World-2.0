"""
test_ner_head.py
══════════════════════════════════════════════════════════════════════════════
NegenWM-JEPA-v2 — Tests unitaires pour NERHead et ner_utils.

Tous les I/O externes sont mockés (VGT tokens, images synthétiques).
Ces tests valident :
  1. Shape forward : ner + mask sont [B, S, H, W]
  2. Valeurs dans [0, 1] pour ner
  3. Soft Collapse R8 : mask in [0, 1] en training, binaire en inference
  4. Priors sky/edge : ner_with_sky < ner_without_sky (pixels ciel)
  5. Hyperparamètres apprenables R6 : log_kappa/logit_lam/logit_tau sont Parameter
  6. ner_utils.compute_ner_filter_mask : sortie booléenne, cohérence avec tau

Run : pytest tests/test_ner_head.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

# ── Helpers de fabrication des token_list ─────────────────────────────────────

def _make_token_list(B: int, S: int, H_patch: int, W_patch: int, dim_in: int, device: str = "cpu"):
    """Simule la sortie des 4 couches VGT intermédiaires.

    _BaseDPTHead.forward reçoit token_list[i] de forme [B, S, N_total, dim_in]
    où N_total = N_register_tokens + H_patch * W_patch.
    On choisit 4 tokens de registre (dinov2-compatible).
    """
    n_reg = 4
    N = n_reg + H_patch * W_patch
    return [
        torch.randn(B, S, N, dim_in, device=device)
        for _ in range(4)
    ]


def _make_images(B: int, S: int, H: int, W: int, device: str = "cpu") -> torch.Tensor:
    return torch.rand(B, S, 3, H, W, device=device)


# ── Fixture NERHead ───────────────────────────────────────────────────────────

@pytest.fixture
def ner_head_cpu():
    """NERHead small (dim_in=768, features=256) sur CPU."""
    from hyworld2.worldrecon.hyworldmirror.models.heads.ner_head import NERHead
    return NERHead(
        dim_in=768,
        patch_size=14,
        features=256,
        kappa=1.0,
        lam=0.5,
        tau=0.35,
        kappa_learnable=True,
    ).eval()


# ── Test 1 : Shapes ──────────────────────────────────────────────────────────

def test_ner_head_output_shapes(ner_head_cpu):
    """ner et mask doivent avoir la forme [B, S, H, W]."""
    B, S, H, W = 1, 3, 56, 56
    patch_size = 14
    H_patch, W_patch = H // patch_size, W // patch_size
    dim_in = 768

    token_list = _make_token_list(B, S, H_patch, W_patch, dim_in)
    images = _make_images(B, S, H, W)

    with torch.no_grad():
        ner, mask = ner_head_cpu(token_list, images, patch_start_idx=4)

    assert ner.shape == (B, S, H, W), f"Unexpected ner shape: {ner.shape}"
    assert mask.shape == (B, S, H, W), f"Unexpected mask shape: {mask.shape}"


# ── Test 2 : Valeurs NER dans [0, 1] ─────────────────────────────────────────

def test_ner_scores_in_range(ner_head_cpu):
    """Tous les scores NER doivent être ∈ [0, 1]."""
    B, S, H, W = 1, 2, 28, 28
    patch_size = 14
    H_patch, W_patch = H // patch_size, W // patch_size

    token_list = _make_token_list(B, S, H_patch, W_patch, 768)
    images = _make_images(B, S, H, W)

    with torch.no_grad():
        ner, _ = ner_head_cpu(token_list, images, patch_start_idx=4)

    assert ner.min().item() >= 0.0 - 1e-5, f"NER min < 0 : {ner.min().item()}"
    assert ner.max().item() <= 1.0 + 1e-5, f"NER max > 1 : {ner.max().item()}"


# ── Test 3 : Soft Collapse R8 ─────────────────────────────────────────────────

def test_soft_mask_range_training(ner_head_cpu):
    """En mode training (hard_mask=False), mask ∈ (0, 1) — différentiable (R8)."""
    ner_head_cpu.train()
    B, S, H, W = 1, 2, 28, 28
    token_list = _make_token_list(B, S, H // 14, W // 14, 768)
    images = _make_images(B, S, H, W)

    _, mask = ner_head_cpu(token_list, images, patch_start_idx=4, hard_mask=False)
    assert mask.min().item() >= 0.0 - 1e-5
    assert mask.max().item() <= 1.0 + 1e-5
    # Au moins certains pixels doivent être intermédiaires (ni 0 ni 1 strict)
    mask_np = mask.detach().numpy()
    n_intermediate = ((mask_np > 0.01) & (mask_np < 0.99)).sum()
    assert n_intermediate > 0, "Soft mask devrait avoir des valeurs intermédiaires"


def test_hard_mask_binary_inference():
    """En mode inference (hard_mask=True), mask ∈ {0.0, 1.0} (R8 — hard path)."""
    from hyworld2.worldrecon.hyworldmirror.models.heads.ner_head import NERHead
    head = NERHead(dim_in=768, patch_size=14, features=256, kappa=1.0, lam=0.5, tau=0.35).eval()

    B, S, H, W = 1, 2, 28, 28
    token_list = _make_token_list(B, S, H // 14, W // 14, 768)
    images = _make_images(B, S, H, W)

    with torch.no_grad():
        _, mask = head(token_list, images, patch_start_idx=4, hard_mask=True)

    unique_vals = mask.unique().tolist()
    for v in unique_vals:
        assert v == 0.0 or v == 1.0, f"Hard mask contient une valeur non binaire : {v}"


# ── Test 4 : Absorption sky prior ─────────────────────────────────────────────

def test_sky_prior_reduces_ner(ner_head_cpu):
    """NER avec prior ciel total (P_sky=1) doit être ≤ NER sans prior."""
    B, S, H, W = 1, 2, 28, 28
    token_list = _make_token_list(B, S, H // 14, W // 14, 768)
    images = _make_images(B, S, H, W)

    # Prior ciel uniforme = 1.0 partout
    sky_prior_full = torch.ones(B, S, H, W)

    with torch.no_grad():
        ner_no_prior, _ = ner_head_cpu(token_list, images, patch_start_idx=4)
        ner_sky, _ = ner_head_cpu(
            token_list, images, patch_start_idx=4, sky_prior=sky_prior_full
        )

    # Avec lambda_sky=0.30 : ner_sky ≤ ner_no_prior (absorption douce)
    diff = (ner_sky - ner_no_prior).max().item()
    assert diff <= 1e-5, f"NER avec prior ciel devrait être ≤ NER sans prior, diff={diff}"


# ── Test 5 : Hyperparamètres apprenables R6 ──────────────────────────────────

def test_learnable_hyperparams():
    """κ/λ/τ doivent être des nn.Parameter (R6)."""
    from hyworld2.worldrecon.hyworldmirror.models.heads.ner_head import NERHead
    head = NERHead(dim_in=768, kappa_learnable=True)

    param_names = {name for name, _ in head.named_parameters()}
    assert "log_kappa" in param_names, "log_kappa n'est pas un Parameter (R6)"
    assert "logit_lam" in param_names, "logit_lam n'est pas un Parameter (R6)"
    assert "logit_tau" in param_names, "logit_tau n'est pas un Parameter (R6)"

    # Vérifier que les propriétés retournent des scalaires ∈ domaine physique
    assert head.kappa_.item() > 0
    assert 0 < head.lam_.item() < 1
    assert 0 < head.tau_.item() < 1


# ── Test 6 : ner_utils.compute_ner_filter_mask ───────────────────────────────

def test_compute_ner_filter_mask_output_type():
    """compute_ner_filter_mask doit retourner un tableau bool [S, H, W]."""
    from hyworld2.worldrecon.hyworldmirror.utils.ner_utils import compute_ner_filter_mask

    S, H, W = 3, 64, 64
    ner_scores = np.random.rand(S, H, W).astype(np.float32)
    mask = compute_ner_filter_mask(ner_scores, tau=0.5)

    assert mask.dtype == bool, f"Masque doit être bool, got {mask.dtype}"
    assert mask.shape == (S, H, W), f"Shape incorrecte: {mask.shape}"


def test_compute_ner_filter_mask_tau_coherence():
    """Pixels avec NER > tau doivent être True, pixels < tau doivent être False."""
    from hyworld2.worldrecon.hyworldmirror.utils.ner_utils import compute_ner_filter_mask

    S, H, W = 1, 4, 4
    ner_scores = np.array([[[0.1, 0.9, 0.4, 0.7],
                             [0.2, 0.8, 0.3, 0.6],
                             [0.0, 1.0, 0.5, 0.35],
                             [0.45, 0.55, 0.34, 0.36]]], dtype=np.float32)  # [1, 4, 4]
    tau = 0.5
    mask = compute_ner_filter_mask(ner_scores, tau=tau)

    expected = ner_scores[0] > tau
    np.testing.assert_array_equal(mask[0], expected, err_msg="Masque NER incohérent avec tau")
