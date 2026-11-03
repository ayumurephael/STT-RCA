"""Own-event evaluation with no test-time fitting or privileged forward inputs."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np
import torch
from .records import Observation,normalize
from .model import ieee_precision
from .metrics import ranking_metrics,summarize,paired_bootstrap


@torch.no_grad()
def evaluate_encoder(encoder,corpus_root,device='cpu',rank=0,world_size=1,epsilon_scale=1e-4):
    from stt_data.storage import Corpus
    corpus=Corpus(corpus_root)
    if corpus.manifest['status']!='complete':raise ValueError('Evaluation requires a complete audited corpus')
    if not 0<=rank<world_size:raise ValueError('Invalid evaluation rank')
    if any(p.dtype!=torch.float32 for p in encoder.parameters()):raise ValueError('Primary evaluation requires FP32 parameters')
    ieee_precision();previous=encoder.training;encoder.eval();rows=[]
    try:
        for index,eid in enumerate(corpus.order):
            if index%world_size!=rank:continue
            meta=corpus.rows[eid]
            if meta['row_role']!='own_event' or meta['split'] not in {'validation','test'}:
                raise ValueError('Primary evaluation accepts only heldout independent own events')
            observation=corpus.observation(eid)
            # Explicit whitelist: metadata/root/normal/graph never enter the encoder.
            x=torch.from_numpy(observation['x'])[None].to(device)
            valid=torch.from_numpy(observation['valid'])[None].to(device)
            with torch.autocast(x.device.type,enabled=False):
                normalized,_=normalize(Observation(x,valid),epsilon_scale)
                logits=encoder(normalized).logits[0].cpu().numpy()
            metrics=ranking_metrics(logits,meta['root_index'])
            row=dict(split=meta['split'],track=meta['track'],episode_id=eid,system_id=meta['mechanism_id'],
                     C=meta['C'],T=meta['T'],mode=meta['mode'],flags=meta['flags'],**metrics)
            # Sink is reporting evidence only; never an input or a filter.
            spec=json.loads((Path(corpus_root)/'worlds'/meta['world_id']/'meta.json').read_text())['spec']
            row['sink']=not any(edge[0]==meta['root_canonical'] for edge in spec['edges'])
            rows.append(row)
    finally:encoder.train(previous)
    return rows


def report(rows,bootstrap_replicates=1000):
    result=summarize(rows);slices=defaultdict(list)
    for row in rows:
        for key,value in [('mode',row['mode']),('C',row['C']),('T',row['T']),('sink',row['sink']),*row['flags'].items()]:
            slices[str(key)+'='+str(value)].append(row)
    result['slices']={key:dict(episodes=len(values),micro_mrr=float(np.mean([r['rr'] for r in values])),
                              micro_top1=float(np.mean([r['top1'] for r in values]))) for key,values in slices.items()}
    result['system_cluster_bootstrap_mrr_95']=paired_bootstrap(rows,replicates=bootstrap_replicates)
    return result

