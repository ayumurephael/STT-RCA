from dataclasses import dataclass
from typing import Mapping
import torch
from torch import Tensor


@dataclass(frozen=True)
class Observation:
    x: Tensor
    valid: Tensor

    @classmethod
    def from_mapping(cls, value: Mapping):
        if set(value) != {"x", "valid"}:
            raise ValueError("Observation permits exactly x and valid")
        return cls(value["x"], value["valid"])


@dataclass(frozen=True)
class NormalizedObservation:
    z: Tensor
    valid: Tensor


@dataclass(frozen=True)
class AffineStats:
    center: Tensor
    scale: Tensor


@dataclass
class Encoded:
    context: Tensor | None
    source: Tensor
    token_valid: Tensor
    logits: Tensor


@torch.no_grad()
def normalize(obs: Observation, epsilon_scale: float = 1e-4):
    x, valid = obs.x, obs.valid
    if x.ndim != 3 or x.dtype != torch.float32 or valid.dtype != torch.bool or valid.shape != x.shape:
        raise ValueError("Expected float32 x and bool valid of shape [B,T,C]")
    if epsilon_scale <= 0 or x.shape[1] < 1 or x.shape[2] < 2:
        raise ValueError("A complete multivariate event and positive scale floor are required")
    if not valid.any(dim=1).all():
        raise ValueError("UNSUPPORTED_UNOBSERVED_CHANNEL")
    if not torch.isfinite(x[valid]).all():
        raise ValueError("Nonfinite valid observation")
    with torch.autocast(x.device.type, enabled=False):
        values = x.double()
        qvalues = torch.tensor([.25, .5, .75], device=x.device, dtype=torch.float64)
        if valid.all():
            q = torch.quantile(values, qvalues, dim=1, interpolation="linear")
            center, scale = q[1, :, None, :], (q[2] - q[0]).clamp_min(epsilon_scale)[:, None, :]
        else:
            center = torch.empty((x.shape[0], 1, x.shape[2]), device=x.device, dtype=torch.float64)
            scale = torch.empty_like(center)
            for b in range(x.shape[0]):
                for c in range(x.shape[2]):
                    q = torch.quantile(values[b, valid[b, :, c], c], qvalues, interpolation="linear")
                    center[b, 0, c] = q[1]
                    scale[b, 0, c] = (q[2] - q[0]).clamp_min(epsilon_scale)
        # Clean invalid values before arithmetic: NaN*0 is still NaN.
        clean = torch.where(valid, values, center)
        z = torch.where(valid, (clean - center) / scale, 0).float()
    return NormalizedObservation(z, valid), AffineStats(center, scale)

