"""Fresh CPU worker; imports no torch and enforces the image's own VmHWM."""
import argparse,json,os
from pathlib import Path
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
from stt_data.cache import ReplayCache
from stt_data.core import atomic_json,read_json,writer_lock


def worker_peak():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):return int(line.split()[1])*1024
    raise RuntimeError('Linux worker image memory measurement unavailable')


def build(corpus,pairs,output,start,step):
    cache=ReplayCache(corpus,pairs,output)
    block=cache.materialize(start,logical_step=step)
    peak=worker_peak()
    if peak>268435456:raise MemoryError('Actual cache worker image exceeds 256 MiB')
    report=Path(output)/'worker_audit.json'
    with writer_lock(Path(output)/'.worker_audit.lock',blocking=True):
        previous=read_json(report) if report.exists() else {}
        atomic_json(report,dict(peak_worker_rss_bytes=max(peak,previous.get('peak_worker_rss_bytes',0)),status='passed',
                              last_start=start,last_end=block['end'],completed_requests=previous.get('completed_requests',0)+1))
    print(json.dumps(dict(status='passed',block=block['id'],end=block['end'],peak_worker_rss_bytes=peak)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--pairs',required=True);p.add_argument('--output',required=True)
    p.add_argument('--start',required=True,type=int);p.add_argument('--step',type=int,default=0);a=p.parse_args()
    build(a.corpus,a.pairs,a.output,a.start,a.step)
