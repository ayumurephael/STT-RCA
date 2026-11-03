"""Inspection, observation-only inference and explicit evaluation. No auto-training."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .config import read_config,architecture
from .objective import STTObjective
from .export import load_ranker
from .evaluation import evaluate_encoder,report


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('inspect');q.add_argument('--config',required=True);q.add_argument('--override',action='append',default=[])
    q=sub.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--input',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cpu')
    q=sub.add_parser('evaluate');q.add_argument('--model',required=True);q.add_argument('--corpus',required=True);q.add_argument('--output',required=True);q.add_argument('--device',default='cpu')
    a=p.parse_args()
    if a.command=='inspect':
        config=read_config(a.config,a.override);model=STTObjective(architecture(config))
        print(json.dumps(dict(training_parameters=sum(p.numel() for p in model.parameters()),channels=config['data']['channels'],auto_launch=False)))
    elif a.command=='predict':
        with np.load(a.input,allow_pickle=False) as values:
            if set(values.files)!={'x','valid'}:raise ValueError('Input NPZ permits exactly x and valid')
            x,valid=values['x'],values['valid']
        if x.dtype!=np.float32 or valid.dtype!=bool:raise ValueError('x must be float32 and valid bool')
        if x.ndim==2:x=x[None];valid=valid[None]
        model=load_ranker(a.model,a.device)
        result=model.predict(torch.from_numpy(x).to(a.device),torch.from_numpy(valid).to(a.device))
        output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
        np.savez(output,**{k:v.cpu().numpy() for k,v in result.items() if isinstance(v,torch.Tensor)})
    else:
        model=load_ranker(a.model,a.device);rows=evaluate_encoder(model.encoder,a.corpus,a.device,epsilon_scale=model.epsilon_scale)
        output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
        with (output/'episodes.jsonl').open('w',encoding='utf-8') as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        (output/'report.json').write_text(json.dumps(report(rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
