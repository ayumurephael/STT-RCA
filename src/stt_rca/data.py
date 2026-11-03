"""Strict bridge to the frozen normal-first generator and bounded replay cache."""
from bisect import bisect_right
from pathlib import Path
import os, subprocess, json
import numpy as np
import torch
from torch.utils.data import Dataset
from .records import Observation
from .objective import TrainingPair


class FrozenPairDataset(Dataset):
    def __init__(self, corpus, pairs, cache, cache_python=None):
        self.corpus_path, self.pairs_path, self.cache_path = map(str,(corpus,pairs,cache))
        self.cache_python=cache_python
        # Defer arrays and open handles until the DataLoader worker owns them.
        self.runtime=None
        from stt_data.core import read_json
        self.plan=read_json(Path(pairs)/'cache_plan.json')['blocks']
        self.starts=[b['start'] for b in self.plan]
        self.length=read_json(Path(pairs)/'manifest.json')['rows']
        if not self.plan or self.starts[0]!=0 or self.plan[-1]['end']!=self.length:
            raise ValueError('Incomplete frozen cache plan')

    def __len__(self): return self.length

    def __getitem__(self,index):
        if not 0<=index<self.length:raise IndexError(index)
        from stt_data.cache import ReplayCache
        from stt_data.pairs import validate_pair
        if self.runtime is None:
            self.runtime=ReplayCache(self.corpus_path,self.pairs_path,self.cache_path)
        cache=self.runtime;c=cache.corpus;p=cache.pairs[index]
        a,b=c.rows[p['A']],c.rows[p['B']];validate_pair(p,a,b)
        for row in (a,b):
            if sum(row['root_flags'])!=1 or not row['root_flags'][row['root_index']]:raise ValueError('Invalid single-root label')
        start=self.starts[bisect_right(self.starts,index)-1]
        block_id,rows,_=cache._plan(start)
        if rows[-1]['row_id']+1!=self.plan[bisect_right(self.starts,index)-1]['end']:
            raise ValueError('Cache plan does not match original full-pair boundaries')
        # Only a fresh CPU worker may materialize numerical replay targets.
        for attempt in range(8):
            ctx=cache.pin(block_id,logical_step=index//96)
            try:view=ctx.__enter__()
            except KeyError:
                if self.cache_python is None:raise RuntimeError('Missing replay block: explicitly prepare it or enable an isolated cache producer')
                self._produce(start,index//96);continue
            break
        else:raise RuntimeError('Cache block repeatedly evicted before pin acquisition; original row retained')
        try:
            entry=next(r for r in view['rows'] if r['row_id']==index)
            if entry['stats_hash']!=p['B_stats_hash']:raise ValueError('Cache coordinate hash mismatch')
            lo=entry['offset'];hi=lo+entry['length'];shape=(entry['T'],entry['C'])
            target={k:np.array(array[lo:hi].reshape(shape),copy=True) for k,array in view['arrays'].items()}
        finally:ctx.__exit__(None,None,None)
        return dict(A=c.x(p['A']).copy(),B=c.x(p['B']).copy(),F=target['F'],normal_in_B=target['normal_in_B'],response_in_B=target['response_in_B'],
                    root_A=a['root_index'],root_B=b['root_index'],slot=p['s'],payload=c.payload(p['A']).copy().astype(np.float32),mode=0 if p['mode']=='process' else 1,
                    row_id=index,stats_hash=p['B_stats_hash'])

    def _produce(self,start,step):
        import stt_data
        env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
        roots=[str(Path(__file__).resolve().parents[1]),str(Path(stt_data.__file__).resolve().parents[1])]
        env['PYTHONPATH']=os.pathsep.join(roots)
        for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='1'
        command=[self.cache_python,'-m','stt_rca.cache_worker','--corpus',self.corpus_path,'--pairs',self.pairs_path,
                 '--output',self.cache_path,'--start',str(start),'--step',str(step)]
        result=subprocess.run(command,env=env,capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Isolated cache worker failed: '+result.stderr[-3000:])


def training_pair(value,device):
    def tensor(name,dtype=None):return torch.as_tensor(value[name],dtype=dtype).to(device,non_blocking=True)
    def obs(name):
        x=tensor(name,torch.float32)
        if x.ndim==2:x=x[None]
        return Observation(x,torch.ones_like(x,dtype=torch.bool))
    def scalar(name):return tensor(name,torch.long).reshape(-1)
    def grid(name):
        v=tensor(name,torch.float32);return v[None] if v.ndim==2 else v
    payload=tensor('payload',torch.float32)
    if payload.ndim==1:payload=payload[None]
    return TrainingPair(obs('A'),obs('B'),obs('F'),scalar('root_A'),scalar('root_B'),scalar('slot'),
                        grid('normal_in_B'),grid('response_in_B'),payload,scalar('mode'))

