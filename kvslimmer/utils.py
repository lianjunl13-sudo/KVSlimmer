import torch


def repeat_kv_3d(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    b, kv_h, t = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    x = hidden_states[:, :, None, :].expand(b, kv_h, n_rep, t)
    return x.reshape(b, kv_h * n_rep, t)