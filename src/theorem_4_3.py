"""
Theorem 4.3 (ICCL paper): prediction error decomposition for task t under masked linear attention.

Definition 3.2: μ_t = E[x y] with element-wise product x y ∈ R^d; for x ~ N(0, Λ), y = w_t^T x,
this gives μ_t = Λ w_t. Theory and eval should use the same {w_t} and Λ when comparing.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch


def build_gamma(Lambda: np.ndarray, N: float) -> np.ndarray:
    """Γ = Λ + (1/N)Λ + (1/N) tr(Λ) I (paper Theorem 4.3)."""
    tr = np.trace(Lambda)
    return Lambda + (1.0 / N) * Lambda + (tr / N) * np.eye(Lambda.shape[0])


def sigma_xy_monte_carlo(
    w: np.ndarray,
    Lambda: np.ndarray,
    n_samples: int = 100_000,
    generator: Optional[torch.Generator] = None,
) -> np.ndarray:
    """
    Σ_t = Var(x ⊙ y) with scalar y = w^T x, x ~ N(0, Λ), z = x * y element-wise (Definition 3.2).
    w: (d,) vector.
    """
    d = w.shape[0]
    g = generator or torch.Generator()
    if generator is None:
        g.manual_seed(0)
    L = np.linalg.cholesky(Lambda + 1e-12 * np.eye(d))
    xs = torch.randn(n_samples, d, generator=g) @ torch.from_numpy(L.T).float()
    w_t = torch.from_numpy(w.astype(np.float32)).view(1, d)
    y = (xs * w_t).sum(dim=1, keepdim=True)
    z = xs * y
    zc = z - z.mean(0, keepdim=True)
    Sigma = (zc.T @ zc) / (n_samples - 1)
    return Sigma.numpy()


def theorem_4_3_decomposition(
    t: int,
    M: int,
    w_list: List[np.ndarray],
    Lambda: np.ndarray,
    N: float,
    n_mc_sigma: int = 100_000,
) -> Dict[str, float]:
    """
    Task index t is 1-based (paper convention), 1 <= t <= T.

    Returns irreducible, variance, bias terms and total for E[(ŷ_{t,q} - y_{t,q})^2]
    under the paper's linear-attention analysis (not the trained softmax Transformer).
    """
    assert 1 <= t <= len(w_list)
    d = Lambda.shape[0]
    eigh = np.linalg.eigh(Lambda)
    lambdas = eigh[0].clip(min=1e-15)
    V = eigh[1]

    mus = [Lambda @ w_list[s - 1].reshape(d, 1) for s in range(1, t + 1)]
    mu_t = mus[t - 1].reshape(d)
    alpha = M / (t * (M + 1))

    Gamma = build_gamma(Lambda, N)
    Gamma_inv = np.linalg.inv(Gamma)
    tr_L = np.trace(Lambda)

    var_term = 0.0
    coeff = M / (t ** 2 * (M + 1) ** 2)
    for s in range(1, t + 1):
        Sigma_s = sigma_xy_monte_carlo(w_list[s - 1], Lambda, n_samples=n_mc_sigma)
        var_term += coeff * np.trace(Sigma_s @ Gamma_inv @ Gamma_inv @ Lambda)

    bias_term = 0.0
    for i in range(d):
        lam_i = lambdas[i]
        v_i = V[:, i]
        s_i = sum(float(np.dot(v_i, mus[k].reshape(d))) for k in range(t))
        m_i = float(np.dot(v_i, mu_t))
        A = (lam_i + tr_L) / N
        num = lam_i * (alpha * s_i - m_i) - m_i * A
        den = lam_i + A
        bias_term += (1.0 / lam_i) * (num / den) ** 2

    irreducible = 0.0
    total = irreducible + var_term + bias_term
    return {
        "irreducible": float(irreducible),
        "variance": float(var_term),
        "bias": float(bias_term),
        "total": float(total),
        "alpha": float(alpha),
        "beta_t": float(1.0 / (t * (M + 1))),
    }


def tensor_weights_to_numpy_list(
    fixed_task_weights: List[torch.Tensor],
) -> List[np.ndarray]:
    """Use first batch row so theory matches the same w as duplicated across batch."""
    out = []
    for w in fixed_task_weights:
        w0 = w[0].detach().cpu().numpy().reshape(-1)
        out.append(w0)
    return out
