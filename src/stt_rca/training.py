"""Explicit-call optimizer, exact pair assignment, accumulation and restart utilities.

Importing this module never launches a job or initializes a process group.
"""
from contextlib import nullcontext
from pathlib import Path
import math, os, random
import numpy as np
import torch
import torch.distributed as dist


def pair_indices(step,rank,world_size,global_pairs=96):
    if global_pairs%world_size or not 0<=rank<world_size:raise ValueError('Unequal rank pair counts are unsupported')
    per_rank=global_pairs//world_size
    return list(range(step*global_pairs+rank*per_rank,step*global_pairs+(rank+1)*per_rank))


def optimizer_for(model,lr=3e-4,weight_decay=.01):
    decay=[];no_decay=[]
    for name,p in model.named_parameters():
        if p.requires_grad:(no_decay if p.ndim<2 or name.endswith('.bias') else decay).append(p)
    return torch.optim.AdamW([dict(params=decay,weight_decay=weight_decay),dict(params=no_decay,weight_decay=0.)],
                             lr=lr,betas=(.9,.95),eps=1e-8)


def learning_rate(step,max_updates,warmup,peak=3e-4,minimum=3e-5):
    if not 0<=step<max_updates or not 0<warmup<max_updates:raise ValueError('Invalid schedule')
    if step<warmup:return peak*(step+1)/warmup
    progress=(step+1-warmup)/(max_updates-warmup)
    return minimum+.5*(peak-minimum)*(1+math.cos(math.pi*progress))


def calibrate_energy(energies,row_ids):
    values=np.asarray(energies,dtype=np.float64)
    if values.shape!=(4096,) or len(row_ids)!=4096 or len(set(row_ids))!=4096:
        raise ValueError('Energy calibration requires exactly 4096 frozen training row IDs')
    if not np.isfinite(values).all() or (values<0).any():raise ValueError('Invalid response energies')
    positive=values[values>0]
    if not len(positive):raise ValueError('No positive training response: stop and inspect data')
    return dict(epsilon_energy=max(1e-6,.01*float(np.quantile(positive,.5,method='linear'))),
                row_ids=list(row_ids),positive_count=int(len(positive)),zero_count=int((values==0).sum()))


def accumulated_backward(model,pairs,step,amp=False):
    """One equal-count rank contribution. Caller clips/steps after this function.

    A DDP wrapper must own the complete objective, and no_sync encloses forward
    and backward together. No optimizer step is hidden in this utility.
    """
    count=len(pairs)
    if count<1:raise ValueError('No pairs in update')
    totals={}
    for i,pair in enumerate(pairs):
        sync=model.no_sync() if hasattr(model,'no_sync') and i<count-1 else nullcontext()
        with sync:
            with torch.autocast(pair.A.x.device.type,dtype=torch.bfloat16,enabled=amp):result=model(pair,step)
            (result['loss'].float().mean()/count).backward()
        for key,value in result.items():totals[key]=totals.get(key,0.)+float(value.detach().mean())/count
    return totals


def rng_state():
    value=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state())
    if torch.cuda.is_available():value['cuda']=torch.cuda.get_rng_state(torch.cuda.current_device())
    return value


def restore_rng(value):
    random.setstate(value['python']);np.random.set_state(value['numpy']);torch.set_rng_state(value['torch'])
    if 'cuda' in value:torch.cuda.set_rng_state(value['cuda'],torch.cuda.current_device())


def save_checkpoint(path,model,optimizer,next_step,contract,resolved_config):
    """All ranks call at an optimizer boundary. An interrupted micro-step is redone."""
    rank=dist.get_rank() if dist.is_initialized() else 0
    states=[None]*dist.get_world_size() if dist.is_initialized() else [rng_state()]
    if dist.is_initialized():dist.all_gather_object(states,rng_state())
    if rank==0:
        core=model.module if hasattr(model,'module') else model
        payload=dict(format='stt-training-v1',model=core.state_dict(),optimizer=optimizer.state_dict(),next_step=next_step,
                     scheduler=dict(kind='stateless_warmup_cosine',next_step=next_step,config=resolved_config.get('optimizer',{})),
                     sampler_cursor=next_step*96,rank_rng_states=states,contract=contract,resolved_config=resolved_config)
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.tmp')
        with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(temp,path)
    if dist.is_initialized():dist.barrier()


def load_checkpoint(path,model,optimizer,contract):
    # Checkpoints are trusted, locally created training artifacts, not untrusted downloads.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['format']!='stt-training-v1' or state['contract']!=contract:raise ValueError('Checkpoint/data/source contract mismatch')
    world=dist.get_world_size() if dist.is_initialized() else 1;rank=dist.get_rank() if dist.is_initialized() else 0
    if len(state['rank_rng_states'])!=world:raise ValueError('RNG-exact restart requires the same world size')
    if state['sampler_cursor']!=state['next_step']*96:raise ValueError('Checkpoint sampler cursor mismatch')
    if state['scheduler']['next_step']!=state['next_step']:raise ValueError('Checkpoint scheduler cursor mismatch')
    core=model.module if hasattr(model,'module') else model
    core.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer']);restore_rng(state['rank_rng_states'][rank])
    return state['next_step']
