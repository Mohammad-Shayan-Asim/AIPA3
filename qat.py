import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class FakeQuantize(torch.autograd.Function):
    """Round-to-nearest fake quantization with saturating straight-through estimator."""

    @staticmethod
    def forward(ctx, x, scale, z, qmin, qmax):
        q    = torch.clamp(torch.round(x / scale) + z, qmin, qmax)
        x_dq = (q - z) * scale
        lo   = (qmin - z) * scale
        hi   = (qmax - z) * scale
        ctx.save_for_backward(((x >= lo) & (x <= hi)))
        return x_dq

    @staticmethod
    def backward(ctx, grad):
        mask, = ctx.saved_tensors
        return grad * mask, None, None, None, None


class QATConv2d(nn.Conv2d):
    def __init__(self, *args, bw_w=8, bw_a=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bw_w    = bw_w
        self.bw_a    = bw_a
        self.qmax_w  = 2 ** (bw_w - 1) - 1
        self.qmax_a  = 2 ** bw_a - 1
        self.observe = True
        self.register_buffer("a_min", torch.tensor(float("inf")))
        self.register_buffer("a_max", torch.tensor(float("-inf")))

    def _weight_fq(self):
        w = self.weight
        with torch.no_grad():
            scale = (
                w.detach().view(w.size(0), -1).abs().amax(dim=1) / self.qmax_w
            ).clamp(min=1e-8).view(-1, 1, 1, 1)
            z = torch.zeros_like(scale)
        return FakeQuantize.apply(w, scale, z, -self.qmax_w, self.qmax_w)

    def _act_fq(self, x):
        if self.observe and self.training:
            with torch.no_grad():
                self.a_min.copy_(
                    torch.minimum(self.a_min, x.detach().min().clamp(max=0.0))
                )
                self.a_max.copy_(
                    torch.maximum(self.a_max, x.detach().max().clamp(min=0.0))
                )
        if self.a_min.isinf().item():
            return x
        with torch.no_grad():
            scale = ((self.a_max - self.a_min) / self.qmax_a).clamp(min=1e-8)
            z     = torch.round((-self.a_min / scale).clamp(0, self.qmax_a))
        return FakeQuantize.apply(x, scale, z, 0, self.qmax_a)

    def forward(self, x):
        return F.conv2d(
            self._act_fq(x), self._weight_fq(), self.bias,
            self.stride, self.padding, self.dilation, self.groups,
        )

    def freeze(self):
        self.observe = False


class QATLinear(nn.Linear):
    def __init__(self, *args, bw_w=8, bw_a=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.bw_w    = bw_w
        self.bw_a    = bw_a
        self.qmax_w  = 2 ** (bw_w - 1) - 1
        self.qmax_a  = 2 ** bw_a - 1
        self.observe = True
        self.register_buffer("a_min", torch.tensor(float("inf")))
        self.register_buffer("a_max", torch.tensor(float("-inf")))

    def _weight_fq(self):
        w = self.weight
        with torch.no_grad():
            scale = (
                w.detach().abs().amax(dim=1, keepdim=True) / self.qmax_w
            ).clamp(min=1e-8)
            z = torch.zeros_like(scale)
        return FakeQuantize.apply(w, scale, z, -self.qmax_w, self.qmax_w)

    def _act_fq(self, x):
        if self.observe and self.training:
            with torch.no_grad():
                self.a_min.copy_(
                    torch.minimum(self.a_min, x.detach().min().clamp(max=0.0))
                )
                self.a_max.copy_(
                    torch.maximum(self.a_max, x.detach().max().clamp(min=0.0))
                )
        if self.a_min.isinf().item():
            return x
        with torch.no_grad():
            scale = ((self.a_max - self.a_min) / self.qmax_a).clamp(min=1e-8)
            z     = torch.round((-self.a_min / scale).clamp(0, self.qmax_a))
        return FakeQuantize.apply(x, scale, z, 0, self.qmax_a)

    def forward(self, x):
        return F.linear(self._act_fq(x), self._weight_fq(), self.bias)

    def freeze(self):
        self.observe = False


def replace_layers_qat(model, bw_w=8, bw_a=8):
    """Recursively replace Conv2d and Linear layers with QAT equivalents in-place."""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.Conv2d) and not isinstance(module, QATConv2d):
            new = QATConv2d(
                module.in_channels, module.out_channels, module.kernel_size,
                stride=module.stride, padding=module.padding,
                dilation=module.dilation, groups=module.groups,
                bias=module.bias is not None, bw_w=bw_w, bw_a=bw_a,
            )
            new.weight = nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                new.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new)
        elif isinstance(module, nn.Linear) and not isinstance(module, QATLinear):
            new = QATLinear(
                module.in_features, module.out_features,
                bias=module.bias is not None, bw_w=bw_w, bw_a=bw_a,
            )
            new.weight = nn.Parameter(module.weight.data.clone())
            if module.bias is not None:
                new.bias = nn.Parameter(module.bias.data.clone())
            setattr(model, name, new)
        else:
            replace_layers_qat(module, bw_w=bw_w, bw_a=bw_a)
    return model


def freeze_observers(model):
    """Freeze activation observers in all QAT layers (use fixed calibration range)."""
    for m in model.modules():
        if isinstance(m, (QATConv2d, QATLinear)):
            m.freeze()


def qat_recon_errors(model, eps=1e-8):
    """Per-layer weight reconstruction error: ||fake_quant(W) - W||^2 / ||W||^2."""
    errors = {}
    for name, m in model.named_modules():
        if not isinstance(m, (QATConv2d, QATLinear)):
            continue
        with torch.no_grad():
            w = m.weight.detach().float()
            if isinstance(m, QATConv2d):
                scale = (
                    w.view(w.size(0), -1).abs().amax(dim=1) / m.qmax_w
                ).clamp(min=1e-8).view(-1, 1, 1, 1)
            else:
                scale = (w.abs().amax(dim=1, keepdim=True) / m.qmax_w).clamp(min=1e-8)
            w_q = torch.clamp(torch.round(w / scale), -m.qmax_w, m.qmax_w) * scale
            errors[name] = ((w - w_q).norm() ** 2 / (w.norm() ** 2 + eps)).item()
    return errors


def train_qat(model, criterion, dataloaders, dataset_sizes, optimizer, scheduler,
              device, epochs=10, freeze_epoch=5):
    """QAT training loop; freezes activation observers at freeze_epoch."""
    model.to(device)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    since   = time.time()

    for epoch in tqdm.tqdm(range(epochs), desc="QAT"):
        if epoch == freeze_epoch:
            freeze_observers(model)
            print(f"\nEpoch {epoch}: activation observers frozen")

        for phase in ("train", "val"):
            model.train() if phase == "train" else model.eval()
            running_loss, running_correct = 0.0, 0

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss    += loss.item() * inputs.size(0)
                running_correct += (outputs.argmax(1) == labels).sum().item()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc  = running_correct / dataset_sizes[phase]
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)
            print(f"  {phase}  loss={epoch_loss:.4f}  acc={100*epoch_acc:.2f}%")

        if scheduler is not None:
            scheduler.step()

    elapsed = time.time() - since
    print(f"\nTotal training time: {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    return model, history
