from copy import deepcopy
from pathlib import Path
import json,hashlib
import yaml
from .model import ModelConfig


def merge(base,override):
    result=deepcopy(base)
    for key,value in override.items():
        result[key]=merge(result.get(key,{}),value) if isinstance(value,dict) else deepcopy(value)
    return result


def read_config(path,overrides=()):
    value=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    for path in overrides:value=merge(value,yaml.safe_load(Path(path).read_text(encoding='utf-8')))
    return value


def architecture(value):
    c=value['model'];decoder=c['decoder']
    if c['backbone']!='axial_transformer' or c['initialization']!='from_scratch':raise ValueError('Unsupported backbone or initialization')
    if c['channel_identity_embedding'] or c['causal_attention'] or c['dropout']!=0 or decoder['dropout']!=0:
        raise ValueError('Frozen v1 requires no channel ID, bidirectional attention, zero dropout')
    if c['activation']!='gelu' or c['normalization']!='pre_layernorm':raise ValueError('Unsupported stage semantics')
    if decoder['d_model']!=c['d_context'] or decoder['output_patch_width']!=c['patch_size'] or decoder['source_projection_bias']:
        raise ValueError('Decoder interface differs from v1')
    names=['patch_size','layers','d_model','heads','d_ff','d_context','d_source','root_head_hidden','norm_eps']
    return ModelConfig(**{k:c[k] for k in names},decoder_layers=decoder['layers'],decoder_heads=decoder['heads'],
                       decoder_d_ff=decoder['d_ff'],activation_checkpointing=value['training']['activation_checkpointing'])


def source_contract(folder,configuration,dataset_contract=None):
    root=Path(folder)
    files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*.py')) if '__pycache__' not in p.parts}
    payload=dict(source_files=files,configuration=configuration,dataset_contract=dataset_contract)
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

