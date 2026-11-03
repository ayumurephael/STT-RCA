"""Explicit opt-in CPU integration check. Never discovered as a training job."""
import argparse,json,tempfile
from pathlib import Path
import numpy as np
import torch
from stt_rca.prepare import prepare_pilot
from stt_rca.data import FrozenPairDataset,training_pair
from stt_rca.records import normalize
from stt_data.pairs import PairManifest,materialize_pair,validate_pair
from stt_data.storage import Corpus
from stt_data.core import digest,array_hash


def check(corpus,cache_python):
    with tempfile.TemporaryDirectory(prefix='stt-rca-data-correctness-') as temporary:
        root=Path(temporary);pairs=root/'pairs';cache=root/'cache'
        meta=prepare_pilot(corpus,pairs,count=96,seed=3407)
        assert meta['contract']['independent_draw_namespace']=='stt-pilot-independent-v1'
        assert meta['rows']==96
        manifest=PairManifest(pairs);c=Corpus(corpus)
        for i in range(96):
            p=manifest[i];validate_pair(p,c.rows[p['A']],c.rows[p['B']])
        dataset=FrozenPairDataset(corpus,pairs,cache,cache_python=cache_python)
        indices=sorted({0,95,*[b['start'] for b in dataset.plan],*[b['end']-1 for b in dataset.plan]})
        for i in indices:
            raw=dataset[i];p=manifest[i];target=materialize_pair(c,p)
            for name in ['F','normal_in_B','response_in_B']:np.testing.assert_array_equal(raw[name],target[name])
            pair=training_pair(raw,'cpu');_,stats=normalize(pair.B)
            actual=digest([array_hash(stats.center[0,0].numpy()),array_hash(stats.scale[0,0].numpy())])
            assert actual==raw['stats_hash']
            assert pair.A.x.shape[1]==pair.B.x.shape[1] and pair.F.x.shape==pair.B.x.shape
        # Deterministic reopening cannot silently replace or sample another pool.
        repeated=prepare_pilot(corpus,pairs,count=96,seed=3407)
        assert repeated==meta
        result=dict(status='passed',independent_pairs=96,validated_lineage_rows=96,cache_boundary_rows_checked=indices,
                    blocks=len(dataset.plan),cache_worker=json.loads((cache/'worker_audit.json').read_text()),
                    actual_targets_and_observation_stats_match=True,optimizer_updates=0,gpu_used=False)
        print(json.dumps(result),flush=True)
        return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--cache-python',required=True);a=p.parse_args()
    check(a.corpus,a.cache_python)
