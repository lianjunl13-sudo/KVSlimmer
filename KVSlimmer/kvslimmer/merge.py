import torch


def optimal_merge_k_from_alpha_d(
    k1: torch.Tensor,       # (B, kvH, Tpair, D)
    k2: torch.Tensor,       # (B, kvH, Tpair, D)
    alpha1: torch.Tensor,   # (B, kvH, Tpair)
    alpha2: torch.Tensor,   # (B, kvH, Tpair)
    d1: torch.Tensor,       # (B, kvH, Tpair)
    d2: torch.Tensor,       # (B, kvH, Tpair)
    eps: float = 1e-21,
    h11: torch.Tensor = None,
    h22: torch.Tensor = None,
):
    s = d1 + d2
    h12 = (alpha1 * alpha2) * s
    A = h11 - h12
    B = h22 - h12
    D = A + B

    a = A / (D + eps)
    b = B / (D + eps)
    ke = (A[..., None] * k1 + B[..., None] * k2) / (D[..., None] + eps)

    return a, b, ke, h12