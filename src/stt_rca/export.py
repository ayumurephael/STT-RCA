from dataclasses import asdict
from pathlib import Path
import os
import torch
from .model import Ranker, ModelConfig, ieee_precision


def export_ranker(objective, path, metadata=None):
    ranker = Ranker(objective.config, objective.epsilon_scale)
    weights = {k: v.detach().cpu() for k, v in objective.encoder.state_dict().items()
               if not k.startswith('context_projection.')}
    ranker.encoder.load_state_dict(weights, strict=True)
    payload = dict(format='stt-rca-ranker-v1', config=asdict(objective.config), epsilon_scale=objective.epsilon_scale,
                   state_dict=ranker.state_dict(), metadata=metadata or {})
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        torch.save(payload, f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def load_ranker(path, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['format'] != 'stt-rca-ranker-v1':
        raise ValueError("Unsupported deployment artifact")
    ieee_precision()
    ranker = Ranker(ModelConfig(**payload['config']), payload['epsilon_scale'])
    ranker.load_state_dict(payload['state_dict'], strict=True)
    return ranker.to(device).eval().requires_grad_(False)

