"""
ner_head.py
══════════════════════════════════════════════════════════════════════════════
NegenWM-JEPA-v2 — NERHead : Negentropic Energy Ratio confidence head.

Contribution au dépôt HY-World 2.0 (tencent/HY-World-2.0).
Principe CRISPR-Like : insertion précise, aucune régression sur les benchmarks
7-Scenes / NRGBD / DTU. Les masques heuristiques (sky mask + edge mask) sont
absorbés comme priors souples, non supprimés.

Architecture :
  - Étend _BaseDPTHead (même infrastructure DPT que DPTHead)
  - MHA interne dédié (nn.MultiheadAttention) sur les features DPT fusionnées
  - Proxy de néguentropie J(f) = E[G₁(f̃)] − E[G₁(ν)] via log cosh (Hyvärinen 2000)
  - Score combiné : NER = λ·sigmoid(κ·J) + (1−λ)·att_conf
  - Absorption des priors : NER_final = NER·(1 − λ_sky·P_sky)·(1 − λ_edge·P_edge)
  - Décision :
      Training  (soft, différentiable) : M = sigmoid((NER − τ) / T)  [R8]
      Inference (hard)                 : M = NER > τ

ATLAS Invariants respectés :
  R6  κ, λ, τ sont des nn.Parameters apprenables (jamais hardcodés)
  R8  Soft Collapse différentiel — hard_mask=False en training

Usage dans WorldMirror :
  model = WorldMirror(..., enable_ner=True, ner_kappa=1.0, ner_lambda=0.5, ner_tau=0.35)
  preds = model(views)
  ner_score = preds["ner_score"]  # [B, S, H, W] in [0, 1]
  ner_mask  = preds["ner_mask"]   # [B, S, H, W] bool (hard) ou prob (soft)

Référence :
  NegenWM_JEPA_v2_action_plan.md §1–§3
  aguennoune17/negenWM-jepa-v2 (HuggingFace)
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dense_head import _BaseDPTHead


class NERHead(_BaseDPTHead):
    """Per-pixel Negentropic Energy Ratio (NER) confidence head.

    Remplace les masques heuristiques (sky mask + edge mask) par un critère
    différentiable d'énergie néguentropique.

    Parameters
    ----------
    dim_in : int
        Dimension des tokens d'entrée (= 2 * embed_dim du VGT).
        Exemple : 2048 pour le modèle ``large`` (embed_dim=1024).
    patch_size : int
        Taille de patch du backbone (défaut : 14).
    features : int
        Nombre de canaux dans les représentations intermédiaires DPT (défaut : 256).
    out_channels : List[int]
        Canaux pour les projections multi-échelles.
    kappa : float
        Valeur initiale de κ (échelle néguentropie). Appris via ``log_kappa``.
    lam : float
        Valeur initiale de λ (poids entropie d'attention). Appris via ``logit_lam``.
    tau : float
        Valeur initiale de τ (seuil de masque). Appris via ``logit_tau``.
    lambda_sky : float
        Poids du prior ciel (absorption douce, non apprenant).
    lambda_edge : float
        Poids du prior arête (absorption douce, non apprenant).
    num_attn_heads : int
        Nombre de têtes dans le MHA interne.
    soft_temperature : float
        Température T pour le Soft Collapse (training uniquement).
    kappa_learnable : bool
        Si True, κ/λ/τ sont des nn.Parameters (R6 — activé par défaut).

    Returns
    -------
    ner : torch.Tensor  [B, S, H, W]  in [0, 1]
    mask : torch.Tensor [B, S, H, W]  probabilité soft (training) ou bool (inference)
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14,
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        kappa: float = 1.0,
        lam: float = 0.5,
        tau: float = 0.35,
        lambda_sky: float = 0.30,
        lambda_edge: float = 0.30,
        num_attn_heads: int = 8,
        soft_temperature: float = 0.05,
        kappa_learnable: bool = True,
    ) -> None:
        super().__init__(
            dim_in=dim_in,
            patch_size=patch_size,
            features=features,
            out_channels=out_channels,
            pos_embed=True,
            gradient_checkpoint=False,
        )

        # ── Projection DPT → espace de confiance NER ─────────────────────────
        # output_conv1 (hérité) → [B*S, features//2, H, W]
        # On y ajoute une convolution finale pour réduire à 1 canal NER
        self.ner_conv = nn.Sequential(
            nn.Conv2d(features // 2, features // 4, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(features // 4, 1, kernel_size=1, stride=1, padding=0),
        )

        # ── MHA interne pour entropie d'attention (R6 — auto-attention) ──────
        # Opère sur les features DPT fusionnées projetées en features//2
        attn_dim = features // 2
        self.attn_proj = nn.Conv2d(features // 2, attn_dim, kernel_size=1)
        # MHA sur tokens spatiaux [N, B*S, attn_dim]
        self.mha = nn.MultiheadAttention(
            embed_dim=attn_dim,
            num_heads=num_attn_heads,
            batch_first=False,
            dropout=0.0,
        )

        # ── Hyperparamètres souverains apprenables (R6) ───────────────────────
        if kappa_learnable:
            self.log_kappa = nn.Parameter(torch.tensor(math.log(max(kappa, 1e-6))))
            self.logit_lam = nn.Parameter(torch.tensor(math.log(lam / (1 - lam + 1e-6))))
            self.logit_tau = nn.Parameter(torch.tensor(math.log(tau / (1 - tau + 1e-6))))
        else:
            self.register_buffer("log_kappa", torch.tensor(math.log(max(kappa, 1e-6))))
            self.register_buffer("logit_lam", torch.tensor(math.log(lam / (1 - lam + 1e-6))))
            self.register_buffer("logit_tau", torch.tensor(math.log(tau / (1 - tau + 1e-6))))

        # ── Prior ciel / arête (fixes, non apprenants) ───────────────────────
        self.lambda_sky = float(lambda_sky)
        self.lambda_edge = float(lambda_edge)

        # ── Température Soft Collapse (R8) ───────────────────────────────────
        self.soft_temperature = float(soft_temperature)

        # ── Constante analytique G₁(ν) pour ν~N(0,1) ────────────────────────
        # E[log cosh(ν)] ≈ 0.3746 (Hyvärinen & Oja, 2000)
        self.register_buffer("_g1_ref", torch.tensor(0.3746))

    # ── Properties (R6 : jamais hardcodés) ────────────────────────────────────

    @property
    def kappa_(self) -> torch.Tensor:
        """κ ≥ 0 : échelle de néguentropie."""
        return self.log_kappa.exp()

    @property
    def lam_(self) -> torch.Tensor:
        """λ ∈ (0, 1) : poids entropie d'attention."""
        return torch.sigmoid(self.logit_lam)

    @property
    def tau_(self) -> torch.Tensor:
        """τ ∈ (0, 1) : seuil de masque."""
        return torch.sigmoid(self.logit_tau)

    # ── Proxy de néguentropie ─────────────────────────────────────────────────

    def _negentropy_proxy(self, fused: torch.Tensor) -> torch.Tensor:
        """Calcule J(f) ≈ [E{G₁(f̃)} − E{G₁(ν)}]² puis sigmoid(κ·J).

        Parameters
        ----------
        fused : torch.Tensor  [B*S, C, H, W]
            Features DPT fusionnées (sortie de output_conv1).

        Returns
        -------
        torch.Tensor  [B*S, 1, H, W]  in [0, 1]
            Score de néguentropie normalisé par pixel.
        """
        # Normalisation unitaire sur le canal C
        f_norm = fused / (fused.norm(dim=1, keepdim=True).clamp(min=1e-8))  # [B*S, C, H, W]

        # G₁(u) = log cosh(u)
        g1 = torch.log(torch.cosh(f_norm.clamp(-20.0, 20.0)))  # [B*S, C, H, W]

        # E{G₁(f̃_hw)} — moyenne sur le canal C
        eg1 = g1.mean(dim=1, keepdim=True)  # [B*S, 1, H, W]

        # J(f) = (E{G₁(f̃)} − E{G₁(ν)})²
        j = (eg1 - self._g1_ref.to(fused)) ** 2  # [B*S, 1, H, W]

        return torch.sigmoid(self.kappa_ * j)  # [B*S, 1, H, W]

    # ── Entropie d'attention ───────────────────────────────────────────────────

    def _attention_entropy_confidence(self, fused: torch.Tensor) -> torch.Tensor:
        """Calcule att_conf = 1 − H_att_norm à partir d'un MHA sur les features spatiales.

        Parameters
        ----------
        fused : torch.Tensor  [B*S, C, H, W]
            Features DPT (sortie de output_conv1).

        Returns
        -------
        torch.Tensor  [B*S, 1, H, W]  in [0, 1]
            Confiance d'attention normalisée (faible entropie → haute confiance).
        """
        BS, C, H, W = fused.shape
        num_heads = self.mha.num_heads

        # Projection vers attn_dim
        x = self.attn_proj(fused)          # [BS, attn_dim, H, W]
        attn_dim = x.shape[1]

        # Aplatir spatial → séquence [N, BS, attn_dim]
        x_flat = x.flatten(2).permute(2, 0, 1)  # [H*W, BS, attn_dim]

        # MHA : auto-attention  (on n'a besoin que des poids, pas des valeurs)
        with torch.no_grad() if not self.training else torch.enable_grad():
            _, attn_weights = self.mha(x_flat, x_flat, x_flat, need_weights=True, average_attn_weights=False)
            # attn_weights : [BS, num_heads, H*W, H*W]

        # Entropie d'attention par pixel de requête
        eps = 1e-8
        # Clamping pour stabilité numérique
        attn_weights_safe = attn_weights.clamp(min=eps)
        h_att = -(attn_weights_safe * attn_weights_safe.log()).sum(dim=-1)  # [BS, num_heads, H*W]
        h_att = h_att.mean(dim=1)  # [BS, H*W]

        # Normalisation : H_att / log(num_heads)
        log_heads = math.log(max(num_heads, 2))
        h_att_norm = (h_att / log_heads).clamp(0.0, 1.0)  # [BS, H*W]

        # Confiance = 1 − entropie normalisée
        att_conf = 1.0 - h_att_norm  # [BS, H*W]

        return att_conf.reshape(BS, 1, H, W)  # [BS, 1, H, W]

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        token_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        sky_prior: Optional[torch.Tensor] = None,
        edge_prior: Optional[torch.Tensor] = None,
        hard_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calcule le score NER et le masque de confiance.

        Parameters
        ----------
        token_list : List[torch.Tensor]
            Liste de 4 tenseurs [B, S, N, dim_in] issus des couches VGT intermédiaires.
        images : torch.Tensor  [B, S, 3, H, W]
            Images d'entrée (range [0, 1]).
        patch_start_idx : int
            Index de début des tokens patch (après les tokens de registre).
        sky_prior : torch.Tensor, optional  [B, S, H, W] in [0, 1]
            Probabilité de ciel par pixel (1.0 = ciel certain).
            Si None, aucun prior ciel appliqué.
        edge_prior : torch.Tensor, optional  [B, S, H, W] in [0, 1]
            Probabilité d'arête par pixel (1.0 = arête certaine).
            Si None, aucun prior arête appliqué.
        hard_mask : bool
            False (défaut) → masque soft différentiable (Soft Collapse — R8).
            True → masque binaire (inference).

        Returns
        -------
        ner : torch.Tensor  [B, S, H, W]  in [0, 1]
            Score NER final par pixel.
        mask : torch.Tensor  [B, S, H, W]
            Masque de confiance : probabilité ∈ [0,1] (soft) ou bool (hard).
        """
        B, S, _, H, W = images.shape

        # ── 1. Extraction des features DPT fusionnées ─────────────────────────
        # _extract_fused_features → [B*S, features//2, H, W]  (hérité de _BaseDPTHead)
        # puis output_conv1 est appliqué en interne dans scratch_forward via refinenets
        fused = self._extract_fused_features(
            token_list, B, S, H, W, patch_start_idx,
        )
        # fused : [B*S, features//2, H, W]

        # ── 2. Proxy de néguentropie ──────────────────────────────────────────
        j_score = self._negentropy_proxy(fused)       # [B*S, 1, H, W]

        # ── 3. Confiance d'attention ──────────────────────────────────────────
        att_conf = self._attention_entropy_confidence(fused)  # [B*S, 1, H, W]

        # ── 4. Score NER combiné : λ·J + (1−λ)·att_conf ─────────────────────
        lam = self.lam_
        ner_raw = lam * j_score + (1.0 - lam) * att_conf   # [B*S, 1, H, W]

        # ── 5. Absorption des priors souples (R8 — différentiable) ───────────
        ner_raw = ner_raw.reshape(B, S, H, W)

        if sky_prior is not None:
            # P_sky ∈ [0, 1] : 1.0 = pixel est du ciel
            p_sky = sky_prior.to(ner_raw)
            ner_raw = ner_raw * (1.0 - self.lambda_sky * p_sky)

        if edge_prior is not None:
            p_edge = edge_prior.to(ner_raw)
            ner_raw = ner_raw * (1.0 - self.lambda_edge * p_edge)

        ner = ner_raw.clamp(0.0, 1.0)  # [B, S, H, W]

        # ── 6. Décision de masque (R8 : Soft Collapse en training) ───────────
        tau = self.tau_
        if hard_mask:
            mask = (ner > tau).float()
        else:
            # sigmoid((NER − τ) / T) — différentiable
            mask = torch.sigmoid((ner - tau) / self.soft_temperature)

        return ner, mask
