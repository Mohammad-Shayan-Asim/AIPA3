import torch
from quant import TARGETS


def collect_hessians(model, loader, device):
    """H = X^T X averaged over calibration tokens for each quantisable Conv1D layer."""
    H, cnt = {}, {}
    hooks = []

    for i, block in enumerate(model.transformer.h):
        for parent, attr in TARGETS:
            layer = getattr(getattr(block, parent), attr)
            key = f"h.{i}.{parent}.{attr}"

            def _hook(k):
                def hook(mod, inp, _):
                    x = inp[0].detach().float().reshape(-1, inp[0].size(-1))
                    H[k]   = H.get(k, torch.zeros(x.size(1), x.size(1), device=x.device)) + x.T @ x
                    cnt[k] = cnt.get(k, 0) + x.size(0)
                return hook
            hooks.append(layer.register_forward_hook(_hook(key)))

    model.eval()
    with torch.no_grad():
        for (ids,) in loader:
            model(ids.to(device), use_cache=False)

    for h in hooks:
        h.remove()

    # Move to CPU to free GPU memory
    return {k: (H[k] / cnt[k]).cpu() for k in H}


def gptq_layer(W, H, qmax, damp=0.01):
    """
    GPTQ weight optimisation for Conv1D weight W [in, out].
    Sequentially quantises rows (input features) and compensates remaining rows
    via H^{-1} after each step to minimise output reconstruction error.
    """
    W = W.clone().float()
    d_in = W.size(0)
    scale = (W.abs().amax(dim=0, keepdim=True) / qmax).clamp(min=1e-8)

    H = H.clone().float().to(W.device)
    H.diagonal().add_(damp * H.diagonal().abs().mean())
    try:
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    except RuntimeError:
        H.diagonal().add_(0.1 * H.diagonal().abs().mean())
        H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

    for j in range(d_in):
        w     = W[j]
        w_q   = (w / scale[0]).round().clamp(-qmax, qmax) * scale[0]
        delta = w_q - w
        W[j]  = w_q
        if j + 1 < d_in:
            W[j+1:] -= (H_inv[j+1:, j] / H_inv[j, j]).unsqueeze(1) * delta.unsqueeze(0)

    return W


def apply_gptq(model, hessians, qmax):
    """Apply GPTQ to all quantisable layers in-place; returns per-layer recon errors."""
    errors = {}
    for i, block in enumerate(model.transformer.h):
        for parent, attr in TARGETS:
            key = f"h.{i}.{parent}.{attr}"
            if key not in hessians:
                continue
            layer = getattr(getattr(block, parent), attr)
            W = layer.weight.data
            W_q = gptq_layer(W, hessians[key], qmax)
            errors[key] = ((W_q - W.float()).pow(2).sum() / (W.float().pow(2).sum() + 1e-8)).item()
            layer.weight.data.copy_(W_q.to(W.dtype))
    return errors
