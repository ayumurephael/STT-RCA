from dataclasses import dataclass
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from .records import Observation, normalize
from .model import ModelConfig, Encoder, Decoder, select_source, place_source


@dataclass
class TrainingPair:
    A: Observation
    B: Observation
    F: Observation
    root_A: Tensor
    root_B: Tensor
    slot: Tensor
    normal_in_B: Tensor
    response_in_B: Tensor
    payload: Tensor | None = None
    mode: Tensor | None = None


def masked_mean_per_episode(value, valid):
    if value.shape != valid.shape or not valid.flatten(1).any(1).all():
        raise ValueError("Empty or incompatible loss mask")
    return value.float().masked_fill(~valid, 0).flatten(1).sum(1) / valid.flatten(1).sum(1)


def response_loss(prediction, target, valid, epsilon):
    if epsilon <= 0:
        raise ValueError("epsilon_energy must be positive")
    error = masked_mean_per_episode((prediction.float() - target.float()).square(), valid)
    energy = masked_mean_per_episode(target.float().square(), valid)
    return error / (energy + epsilon), energy


class STTObjective(nn.Module):
    """DDP wraps this complete per-pair graph, including all three root forwards."""
    def __init__(self, config=ModelConfig(), arm="stt", epsilon_energy=1e-3, epsilon_scale=1e-4, ramp=1000):
        super().__init__()
        if arm not in {"stt", "direct", "delete_trans", "oracle", "payload", "reconstruction"}:
            raise ValueError("Unknown training arm")
        self.config, self.arm = config, arm
        self.epsilon_energy, self.epsilon_scale, self.ramp = epsilon_energy, epsilon_scale, ramp
        self.encoder = Encoder(config, context=arm != "direct")
        self.decoder = None if arm == "direct" else Decoder(config)
        self.oracle = nn.Sequential(nn.Linear(config.patch_size + 2, config.d_source), nn.GELU(),
                                    nn.Linear(config.d_source, config.d_source)) if arm == "oracle" else None
        self.payload_head = nn.Linear(config.d_source, config.patch_size + 2) if arm == "payload" else None

    def forward(self, pair: TrainingPair, step: int):
        if pair.A.x.shape[1] != pair.B.x.shape[1] or pair.F.x.shape != pair.B.x.shape:
            raise ValueError("A/B must share real T; F must have B's complete shape")
        A, _ = normalize(pair.A, self.epsilon_scale)
        B, _ = normalize(pair.B, self.epsilon_scale)
        # Independent F coordinates, never B's statistics.
        Fobs, _ = normalize(pair.F, self.epsilon_scale)
        a, b, f = self.encoder(A), self.encoder(B), self.encoder(Fobs)
        with torch.autocast(pair.B.x.device.type, enabled=False):
            root = (.25 * F.cross_entropy(a.logits, pair.root_A, reduction="none") +
                    .25 * F.cross_entropy(b.logits, pair.root_B, reduction="none") +
                    .5 * F.cross_entropy(f.logits, pair.slot, reduction="none"))
            zero = root * 0
            base, trans, aux, energy = zero, zero, zero, zero
            if self.decoder is not None:
                empty = b.source.new_zeros(b.source.shape)
                y0 = self.decoder(b.context, empty, B.valid, B.z.shape[1])
                base = masked_mean_per_episode((y0 - pair.normal_in_B.float()).square(), B.valid)
                if self.arm in {"stt", "oracle"}:
                    if self.arm == "oracle":
                        trajectory = self.oracle(self.payload_features(pair, a.source.shape[1]))
                    else:
                        trajectory = select_source(a.source, pair.root_A)
                    slots = place_source(trajectory, pair.slot, B.z.shape[2])
                    y1 = self.decoder(b.context, slots, B.valid, B.z.shape[1])
                    trans, energy = response_loss(y1 - y0, pair.response_in_B, B.valid, self.epsilon_energy)
                elif self.arm == "payload":
                    target = self.payload_features(pair, a.source.shape[1])
                    predicted = self.payload_head(select_source(a.source, pair.root_A))
                    T = pair.A.x.shape[1]
                    payload_pred = predicted[..., :self.config.patch_size].reshape(predicted.shape[0], -1)[:, :T]
                    root_valid = pair.A.valid.gather(2, pair.root_A[:, None, None].expand(-1, T, 1)).squeeze(-1)
                    payload_loss = ((payload_pred - pair.payload.float()).square().masked_fill(~root_valid, 0).sum(1) / root_valid.sum(1))
                    token_valid = a.token_valid.gather(2, pair.root_A[:, None, None].expand(-1, a.source.shape[1], 1)).squeeze(-1)
                    mode_logits = (predicted[..., -2:] * token_valid[..., None]).sum(1) / token_valid.sum(1)[:, None]
                    aux = .1 * (payload_loss + F.cross_entropy(mode_logits, pair.mode, reduction="none"))
                elif self.arm == "reconstruction":
                    aug = (torch.rand_like(b.token_valid, dtype=torch.float32) < .15) & b.token_valid
                    hidden = self.encoder(B, augmentation_mask=aug)
                    reconstruction = self.decoder(hidden.context, hidden.source, B.valid, B.z.shape[1])
                    mask = aug.repeat_interleave(self.config.patch_size, dim=1)[:, :B.z.shape[1]] & B.valid
                    count = mask.flatten(1).sum(1)
                    aux = .1 * ((reconstruction - B.z).square().masked_fill(~mask, 0).flatten(1).sum(1) / count.clamp_min(1))
            scale = min((step + 1) / max(1, self.ramp), 1.)
            total = root + scale * (.1 * base + trans + aux)
            if not torch.isfinite(total).all():
                raise FloatingPointError("Nonfinite objective; retain original pair cursor")
        return dict(loss=total, root=root.detach(), base=base.detach(), trans=trans.detach(),
                    auxiliary=aux.detach(), energy=energy.detach())

    def payload_features(self, pair, patches):
        if pair.payload is None or pair.mode is None or pair.payload.shape != pair.A.x.shape[:2]:
            raise ValueError("Privileged training payload/mode required for this auxiliary arm")
        payload = F.pad(pair.payload.float(), (0, patches * self.config.patch_size - pair.payload.shape[1]))
        payload = payload.reshape(payload.shape[0], patches, self.config.patch_size)
        return torch.cat((payload, F.one_hot(pair.mode, 2).float()[:, None].expand(-1, patches, -1)), -1)

