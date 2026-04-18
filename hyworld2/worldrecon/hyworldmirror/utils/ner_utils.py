"""
ner_utils.py
══════════════════════════════════════════════════════════════════════════════
NegenWM-JEPA-v2 — Utilitaires NER pour le pipeline d'inférence HY-World 2.0.

Remplace compute_filter_mask heuristique par un filtrage basé sur les scores
NER différentiels produits par NERHead.

Référence : NegenWM_JEPA_v2_action_plan.md §4
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def compute_ner_filter_mask(
    ner_scores: np.ndarray,
    tau: float = 0.35,
    sky_mask: Optional[np.ndarray] = None,
    lambda_sky: float = 0.30,
    edge_mask: Optional[np.ndarray] = None,
    lambda_edge: float = 0.30,
) -> np.ndarray:
    """Calcule le masque de filtrage booléen à partir des scores NER.

    Les masques heuristiques (sky, edge) sont absorbés comme priors souples
    avant le seuillage, ce qui les rend non-destructifs (principe CRISPR-Like).

    Parameters
    ----------
    ner_scores : np.ndarray  [S, H, W]  float32 in [0, 1]
        Scores NER produits par NERHead (via ``ner_scores_to_numpy``).
    tau : float
        Seuil de binarisation (défaut : 0.35). Calibrer sur 7-Scenes val.
    sky_mask : np.ndarray, optional  [S, H, W]  bool
        True = pixel non-ciel (même convention que HY-World 2.0 sky_mask).
        Si None, aucun prior ciel n'est appliqué.
    lambda_sky : float
        Poids d'absorption du prior ciel.
    edge_mask : np.ndarray, optional  [S, H, W]  bool
        True = pixel non-arête.
        Si None, aucun prior arête n'est appliqué.
    lambda_edge : float
        Poids d'absorption du prior arête.

    Returns
    -------
    filter_mask : np.ndarray  [S, H, W]  bool
        True = conserver le point 3D, False = supprimer.
    """
    ner = ner_scores.copy().astype(np.float32)

    if sky_mask is not None:
        # P_sky = 1.0 là où c'est du ciel (= NOT sky_mask True)
        p_sky = (~sky_mask).astype(np.float32)
        ner = ner * (1.0 - lambda_sky * p_sky)

    if edge_mask is not None:
        # P_edge = 1.0 là où c'est une arête
        p_edge = (~edge_mask).astype(np.float32)
        ner = ner * (1.0 - lambda_edge * p_edge)

    ner = np.clip(ner, 0.0, 1.0)
    return ner > tau


def ner_scores_to_numpy(ner_tensor: torch.Tensor) -> np.ndarray:
    """Convertit le tenseur NER [B, S, H, W] → numpy [S, H, W] (batch 0).

    Parameters
    ----------
    ner_tensor : torch.Tensor  [B, S, H, W]
        Sortie ``preds["ner_score"]`` du modèle WorldMirror.

    Returns
    -------
    np.ndarray  [S, H, W]  float32
    """
    return ner_tensor[0].detach().cpu().float().numpy()
