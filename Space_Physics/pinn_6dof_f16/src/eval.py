from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

from .aero_true import aero_true

@dataclass
class EvalOutputs:
    coeff_pred: np.ndarray   # [N,6]
    coeff_true: np.ndarray   # [N,6]
    rmse: Dict[str, float]

def evaluate_coefficients(net: torch.nn.Module,
                          X_t: torch.Tensor,
                          extras: Dict[str, np.ndarray]) -> EvalOutputs:
    net.eval()
    with torch.no_grad():
        coeff_pred = net(X_t).detach().cpu().numpy()

    alpha = extras["alpha"]; beta = extras["beta"]
    p_hat = extras["p_hat"]; q_hat = extras["q_hat"]; r_hat = extras["r_hat"]
    de = extras["de"]; da = extras["da"]; dr = extras["dr"]

    coeff_true = np.zeros_like(coeff_pred)
    for i in range(len(alpha)):
        coeff_true[i] = aero_true(alpha[i], beta[i], p_hat[i], q_hat[i], r_hat[i], de[i], da[i], dr[i])

    def rmse(a, b): 
        return float(np.sqrt(np.mean((a-b)**2)))

    names = ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]
    metrics = {n: rmse(coeff_pred[:,k], coeff_true[:,k]) for k, n in enumerate(names)}

    return EvalOutputs(coeff_pred=coeff_pred, coeff_true=coeff_true, rmse=metrics)
