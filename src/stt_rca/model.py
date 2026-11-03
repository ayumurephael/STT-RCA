from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .records import Observation, NormalizedObservation, Encoded, normalize


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int = 16
    layers: int = 8
    d_model: int = 512
    heads: int = 8
    d_ff: int = 2048
    d_context: int = 256
    d_source: int = 128
    root_head_hidden: int = 64
    norm_eps: float = 1e-5
    decoder_layers: int = 4
    decoder_heads: int = 4
    decoder_d_ff: int = 1024
    activation_checkpointing: bool = True

    def __post_init__(self):
        if min(self.patch_size, self.layers, self.d_model, self.d_ff, self.d_context, self.d_source) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.d_model % self.heads or self.d_context % self.decoder_heads:
            raise ValueError("Attention width must be divisible by number of heads")


def position(length, width, device, dtype):
    # Shared by every channel. No flattened channel position or channel ID.
    p = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device, dtype=torch.float32) * (-math.log(10000.) / width))
    pe = torch.zeros(length, width, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(p * f)
    pe[:, 1::2] = torch.cos(p * f[:width // 2])
    return pe.to(dtype)[None, :, None, :]


def patchify(obs: NormalizedObservation, width: int):
    z, valid = obs.z, obs.valid
    if z.ndim != 3 or z.shape != valid.shape or valid.dtype != torch.bool:
        raise ValueError("Invalid normalized observation")
    B, T, C = z.shape
    pad = (-T) % width
    values = F.pad(torch.where(valid, z, 0), (0, 0, 0, pad))
    masks = F.pad(valid, (0, 0, 0, pad), value=False)
    values = values.reshape(B, -1, width, C).permute(0, 1, 3, 2)
    masks = masks.reshape(B, -1, width, C).permute(0, 1, 3, 2)
    return torch.cat((values, masks.to(values.dtype)), -1), masks.any(-1), masks


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.out = nn.Linear(width, width)

    def forward(self, x, valid):
        n, length, width = x.shape
        rows = valid.any(-1).nonzero(as_tuple=True)[0]
        # Padded rows never enter softmax, even on backends without safe all-masked behavior.
        if rows.numel() == 0:
            # The zero connection keeps legitimate all-padding microbatches differentiable.
            return x * 0
        selected = x.index_select(0, rows)
        key_valid = valid.index_select(0, rows)
        def project(layer):
            return layer(selected).reshape(-1, length, self.heads, width // self.heads).transpose(1, 2)
        y = F.scaled_dot_product_attention(project(self.q), project(self.k), project(self.v),
            attn_mask=key_valid[:, None, None, :], dropout_p=0., is_causal=False)
        y = self.out(y.transpose(1, 2).reshape(-1, length, width))
        y = y.masked_fill(~key_valid[..., None], 0)
        return y.new_zeros(n, length, width).index_copy(0, rows, y)


class AxialStage(nn.Module):
    def __init__(self, width, heads, d_ff, eps):
        super().__init__()
        self.ln_t = nn.LayerNorm(width, eps=eps)
        self.ln_c = nn.LayerNorm(width, eps=eps)
        self.ln_f = nn.LayerNorm(width, eps=eps)
        self.temporal = Attention(width, heads)
        self.channel = Attention(width, heads)
        self.ffn = nn.Sequential(nn.Linear(width, d_ff), nn.GELU(), nn.Linear(d_ff, width))

    def forward(self, x, valid):
        B, P, C, D = x.shape
        keep = valid[..., None]
        temporal = self.ln_t(x).permute(0, 2, 1, 3).reshape(B * C, P, D)
        tm = valid.permute(0, 2, 1).reshape(B * C, P)
        y = self.temporal(temporal, tm).reshape(B, C, P, D).permute(0, 2, 1, 3)
        x = (x + y).masked_fill(~keep, 0)
        y = self.channel(self.ln_c(x).reshape(B * P, C, D), valid.reshape(B * P, C)).reshape(B, P, C, D)
        x = (x + y).masked_fill(~keep, 0)
        return (x + self.ffn(self.ln_f(x))).masked_fill(~keep, 0)


def stages_forward(stages, x, valid, enabled):
    for stage in stages:
        if enabled and torch.is_grad_enabled():
            x = checkpoint(stage, x, valid, use_reentrant=False)
        else:
            x = stage(x, valid)
    return x


class Encoder(nn.Module):
    def __init__(self, config: ModelConfig, context=True):
        super().__init__()
        self.config = config
        c = config
        self.patch_embed = nn.Linear(2 * c.patch_size, c.d_model)
        self.stages = nn.ModuleList([AxialStage(c.d_model, c.heads, c.d_ff, c.norm_eps) for _ in range(c.layers)])
        self.ln_out = nn.LayerNorm(c.d_model, eps=c.norm_eps)
        self.source_projection = nn.Linear(c.d_model, c.d_source)
        self.context_projection = nn.Linear(c.d_model, c.d_context) if context else None
        self.root_head = nn.Sequential(nn.LayerNorm(c.d_source, eps=c.norm_eps),
            nn.Linear(c.d_source, c.root_head_hidden), nn.GELU(), nn.Linear(c.root_head_hidden, 1))

    def forward(self, obs: NormalizedObservation, augmentation_mask=None):
        features, valid, _ = patchify(obs, self.config.patch_size)
        if augmentation_mask is not None:
            if augmentation_mask.shape != valid.shape or augmentation_mask.dtype != torch.bool:
                raise ValueError("Invalid augmentation mask")
            features = features.masked_fill(augmentation_mask[..., None], 0)
        x = self.patch_embed(features)
        x = (x + position(x.shape[1], x.shape[-1], x.device, x.dtype)).masked_fill(~valid[..., None], 0)
        x = stages_forward(self.stages, x, valid, self.training and self.config.activation_checkpointing)
        x = self.ln_out(x).masked_fill(~valid[..., None], 0)
        with torch.autocast(x.device.type, enabled=False):
            source = self.source_projection(x.float()).masked_fill(~valid[..., None], 0)
            context = None if self.context_projection is None else self.context_projection(x.float()).masked_fill(~valid[..., None], 0)
            counts = valid.sum(1)
            pooled = source.sum(1) / counts.clamp_min(1)[..., None]
            logits = self.root_head(pooled).squeeze(-1).masked_fill(counts == 0, -torch.inf)
        if not valid.any(dim=(1, 2)).all():
            raise ValueError("Empty event")
        return Encoded(context, source, valid, logits)


class Decoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = c = config
        self.source_projection = nn.Linear(c.d_source, c.d_context, bias=False)
        self.stages = nn.ModuleList([AxialStage(c.d_context, c.decoder_heads, c.decoder_d_ff, c.norm_eps) for _ in range(c.decoder_layers)])
        self.ln_out = nn.LayerNorm(c.d_context, eps=c.norm_eps)
        self.patch_output = nn.Linear(c.d_context, c.patch_size)

    def forward(self, context, source_slots, scalar_valid, original_T):
        with torch.autocast(context.device.type, enabled=False):
            if scalar_valid.shape[1] != original_T:
                raise ValueError("Decoder requires the complete original time extent")
            _, valid, _ = patchify(NormalizedObservation(torch.zeros_like(scalar_valid, dtype=torch.float32), scalar_valid), self.config.patch_size)
            if context.shape[:3] != valid.shape or source_slots.shape[:3] != valid.shape:
                raise ValueError("Decoder grid mismatch")
            x = context.float() + self.source_projection(source_slots.float())
            x = (x + position(x.shape[1], x.shape[-1], x.device, x.dtype)).masked_fill(~valid[..., None], 0)
            x = stages_forward(self.stages, x, valid, self.training and self.config.activation_checkpointing)
            x = self.patch_output(self.ln_out(x)).masked_fill(~valid[..., None], 0)
            B, P, C, W = x.shape
            y = x.permute(0, 1, 3, 2).reshape(B, P * W, C)[:, :original_T]
            return y.masked_fill(~scalar_valid, 0)


def select_source(source, root):
    if root.shape != (source.shape[0],) or root.dtype != torch.long or ((root < 0) | (root >= source.shape[2])).any():
        raise ValueError("Invalid donor root")
    return source.gather(2, root[:, None, None, None].expand(-1, source.shape[1], 1, source.shape[3])).squeeze(2)


def place_source(trajectory, slots, channels):
    if slots.shape != (trajectory.shape[0],) or slots.dtype != torch.long or ((slots < 0) | (slots >= channels)).any():
        raise ValueError("Invalid recipient slot")
    shape = (trajectory.shape[0], trajectory.shape[1], channels, trajectory.shape[2])
    return trajectory.new_zeros(shape).scatter(2, slots[:, None, None, None].expand(-1, shape[1], 1, shape[3]), trajectory[:, :, None])


class Ranker(nn.Module):
    """Exported model: no context projection, decoder, or privileged-input path."""
    def __init__(self, config: ModelConfig, epsilon_scale=1e-4):
        super().__init__()
        self.config = config
        self.epsilon_scale = epsilon_scale
        self.encoder = Encoder(config, context=False)

    @torch.no_grad()
    def predict(self, x, valid):
        if self.training:
            raise RuntimeError("Call eval() before deployment inference")
        with torch.autocast(x.device.type, enabled=False):
            obs, _ = normalize(Observation(x, valid), self.epsilon_scale)
            logits = self.encoder(obs).logits
            return dict(logits=logits, q=logits.softmax(-1), ranking=logits.argsort(dim=-1, descending=True, stable=True),
                        status="ok", model_version="stt-rca-0.1.0")


def ieee_precision():
    # Torch 2.5 uses the legacy setting family; do not mix setting families.
    if hasattr(torch.backends.cuda.matmul, "fp32_precision"):
        torch.backends.fp32_precision = "ieee"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
        torch.backends.cudnn.fp32_precision = "ieee"
    else:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

