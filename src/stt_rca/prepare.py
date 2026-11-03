"""CPU-only independent pilot pair stream; no torch import or GPU allocation."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse, os, json
import numpy as np
from stt_data.core import rng, dumps, digest, file_hash, save_array, atomic_json, writer_lock, fsync_dir
from stt_data.storage import Corpus
from stt_data.pairs import eligible_pair


def prepare_pilot(corpus, output, count=192000, seed=3407):
    c=Corpus(corpus,lazy=False); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    contract=dict(version='stt-pilot-q-adm-v1',independent_draw_namespace='stt-pilot-independent-v1',
                  corpus_contract=c.manifest['contract'],rows=count,training_permutation_seed=seed)
    with writer_lock(output/'.writer.lock'):
        if (output/'manifest.json').exists():
            from stt_data.pairs import PairManifest
            existing=PairManifest(output)
            if existing.meta['contract']!=contract: raise ValueError('Pilot pair contract changed')
            return existing.meta
        pools=defaultdict(list); totals=Counter()
        for eid in c.order:
            e=c.rows[eid]
            if e['split']!='train': raise ValueError('Heldout pilot ancestor')
            pools[e['T'],e['C']].append(eid);totals[e['C']]+=1
        channels=sorted(totals); ts=sorted({t for t,_ in pools});weights={}
        for t in ts:
            weights[t]=np.array([len(pools[t,k])/totals[k] for k in channels]); weights[t]/=weights[t].sum()
        tp=np.array([sum(len(pools[t,k])/totals[k] for k in channels)/len(channels) for t in ts]);tp/=tp.sum()
        order=np.random.default_rng(seed).permutation(count);offsets=np.empty(count,dtype=np.int64)
        cross=0;hist=Counter();blocks=[];block_start=0;block_bytes=8192;block_cells=0
        with (output/'pair_manifest.jsonl.tmp').open('wb') as f:
            for row,draw_id in enumerate(order):
                r=rng('stt-pilot-independent-v1','row',int(draw_id))
                for attempt in range(1,100001):
                    t=int(r.choice(ts,p=tp));ca,cb=map(int,r.choice(channels,size=2,p=weights[t]))
                    a=c.rows[pools[t,ca][int(r.integers(len(pools[t,ca])))]]
                    b=c.rows[pools[t,cb][int(r.integers(len(pools[t,cb])))]];slot=int(r.integers(cb))
                    if eligible_pair(a,b):break
                else:raise RuntimeError('No admissible pilot pair')
                p=dict(row_id=row,independent_draw_id=int(draw_id),A=a['episode_id'],B=b['episode_id'],s=slot,T=t,C_A=ca,C_B=cb,
                       mode=a['mode'],family=a['family'],d_B=b['d'],split='train',B_stats_hash=b['observed_stats_hash'],payload_ref=a['payload_ref']['path'],attempts=attempt)
                for role,e in [('A',a),('B',b)]:
                    p.update({role+'_'+field:e[key] for field,key in [('world','world_id'),('mechanism','mechanism_id'),('group','split_group_id'),('world_hash','world_hash')]})
                offsets[row]=f.tell();f.write((dumps(p)+'\n').encode())
                cross+=a['mechanism_id']!=b['mechanism_id'];hist[dumps([t,ca,cb,a['mode'],b['d'],a['family']])]+=1
                length=t*cb
                entry=dict(row_id=row,offset=block_cells,length=length,T=t,C=cb,stats_hash=p['B_stats_hash'])
                extra=12*length+len((dumps(entry)+'\n').encode())
                if extra+8192>67108864:raise ValueError('A complete pair exceeds cache block cap')
                if block_bytes+extra>67108864:
                    blocks.append(dict(start=block_start,end=row));block_start=row;block_bytes=8192;block_cells=0
                    entry['offset']=0;extra=12*length+len((dumps(entry)+'\n').encode())
                block_bytes+=extra;block_cells+=length
                if (row+1)%10000==0:print(dumps(dict(event='independent_pilot_pairs',rows=row+1,target=count)),flush=True)
            f.flush();os.fsync(f.fileno())
        blocks.append(dict(start=block_start,end=count))
        os.replace(output/'pair_manifest.jsonl.tmp',output/'pair_manifest.jsonl')
        save_array(output/'row_offsets.npy',offsets)
        atomic_json(output/'cache_plan.json',dict(blocks=blocks,block_bytes=67108864))
        atomic_json(output/'distribution.json',dict(marginals=dict(hist),cross_system_rows=cross,rows=count,channels=channels,
                    draw_namespace='stt-pilot-independent-v1',main_manifest_reused=False))
        paths=['pair_manifest.jsonl','row_offsets.npy','cache_plan.json','distribution.json']
        files=[dict(path=name,sha256=file_hash(output/name),bytes=(output/name).stat().st_size) for name in paths]
        summary=dict(contract=contract,status='complete',rows=count,cross_system_rows=cross,files=files)
        atomic_json(output/'manifest.json',summary);fsync_dir(output)
        return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--output',required=True)
    p.add_argument('--count',type=int,default=192000);p.add_argument('--seed',type=int,default=3407);a=p.parse_args()
    print(dumps(prepare_pilot(a.corpus,a.output,a.count,a.seed)),flush=True)

