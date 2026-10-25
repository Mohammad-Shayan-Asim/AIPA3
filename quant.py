import torch
import torch.nn as nn


class WeightQuantizer:
    def __init__(self, dim: int, bw: int):
        assert 2 <= bw <= 16, "bitwidth must be between 2 and 16"
        self.dim  = dim
        self.bw   = bw
        self.qmax = 2 ** (bw - 1) - 1
        self.s    = None

    def observe(self, w: torch.Tensor):
        self.s = (w.abs().amax(dim=self.dim, keepdim=True) / self.qmax).clamp(min=1e-8)

    def quantize(self, w: torch.Tensor):
        q = torch.clamp(torch.round(w / self.s), -self.qmax, self.qmax)
        return q, self.s


class ActivationQuantizer:
    def __init__(self, bw: int):
        assert 2 <= bw <= 16, "bitwidth must be between 2 and 16"
        self.bw   = bw
        self.qmax = 2 ** bw - 1
        self.reset()

    def reset(self):
        self.xmin = self.xmax = self.s = self.z = None

    def observe(self, x: torch.Tensor):
        xmin = x.min().clamp(max=0)
        xmax = x.max().clamp(min=0)
        if self.xmin is not None:
            xmin = torch.minimum(xmin, self.xmin)
            xmax = torch.maximum(xmax, self.xmax)
        self.xmin, self.xmax = xmin, xmax
        self.s = ((xmax - xmin) / self.qmax).clamp(min=1e-8)
        self.z = torch.round(-xmin / self.s).clamp(0, self.qmax)

    def quantize(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.clamp(torch.round(x / self.s) + self.z, 0, self.qmax)
        return (q - self.z) * self.s


class QuantWrapper(nn.Module):
    def __init__(self, layer, w_bw=8, a_bw=8, mode="off", smooth=None):
        super().__init__()
        assert mode in ("off", "observe", "static", "dynamic")
        self.mode = mode
        self.bias = layer.bias
        self.aq   = ActivationQuantizer(a_bw)

        w = layer.weight.detach().clone()
        if smooth is not None:
            w = w * smooth[:, None]
        self.register_buffer("smooth", smooth)

        scale = None
        if w_bw is not None:
            wq = WeightQuantizer(dim=0, bw=w_bw)
            wq.observe(w)
            q, scale = wq.quantize(w)
            w = q * scale

        self.register_buffer("w_scale", scale)
        self.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        if self.smooth is not None:
            x = x / self.smooth

        if self.mode == "observe":
            self.aq.observe(x)
        elif self.mode == "dynamic":
            self.aq.reset()
            self.aq.observe(x)
        
        if self.mode in ("static", "dynamic"):
            x = self.aq.quantize(x)
        # out = torch.addmm(self.bias, x.reshape(-1, x.size(-1)), self.weight)
        # return out.view(*x.shape[:-1], -1)
        return x @ self.weight + self.bias


TARGETS = [("attn", "c_attn"), ("attn", "c_proj"), ("mlp", "c_fc"), ("mlp", "c_proj")]


def wrap_gpt2(model, w_bw=8, a_bw=8, mode="off", smooth=None):
    for i, block in enumerate(model.transformer.h):
        for parent, name in TARGETS:
            mod = getattr(block, parent)
            s   = smooth[f"h.{i}.{parent}.{name}"] if smooth else None
            setattr(mod, name, QuantWrapper(getattr(mod, name), w_bw, a_bw, mode, s))
    return model


def set_mode(model, mode):
    for m in model.modules():
        if isinstance(m, QuantWrapper):
            m.mode = mode
