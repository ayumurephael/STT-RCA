"""Exact float32 ties, equal-system averaging, and paired stratified bootstrap."""
from collections import defaultdict
import numpy as np


def ranking_metrics(logits, root):
    v = np.asarray(logits, dtype=np.float32)
    if v.ndim != 1 or not 0 <= root < len(v) or not np.isfinite(v).all():
        raise ValueError("Invalid candidate logits")
    greater, equal = int((v > v[root]).sum()), int((v == v[root]).sum())
    return dict(rr=float(np.reciprocal(np.arange(greater + 1, greater + equal + 1, dtype=np.float64)).mean()),
                top1=1. / equal if greater == 0 else 0.)


def summarize(rows):
    if not rows:
        raise ValueError("Empty evaluation")
    seen = set()
    groups = defaultdict(list)
    for r in rows:
        key = (r['split'], r['track'], r['episode_id'])
        if key in seen:
            raise ValueError("Repeated own-event evaluation row")
        seen.add(key)
        groups[r['track'], r['system_id']].append(r)
    tracks = {}
    for track in sorted({k[0] for k in groups}):
        systems = [dict(rr=np.mean([r['rr'] for r in values]), top1=np.mean([r['top1'] for r in values]))
                   for (t, _), values in groups.items() if t == track]
        tracks[track] = dict(mrr=float(np.mean([s['rr'] for s in systems])),
                             top1=float(np.mean([s['top1'] for s in systems])), systems=len(systems))
    if set(tracks) != {'system_ood', 'composition_ood'}:
        raise ValueError("Both frozen own-event tracks are required for selection")
    return dict(selection_mrr=.5 * sum(v['mrr'] for v in tracks.values()), tracks=tracks,
                micro_mrr=float(np.mean([r['rr'] for r in rows])), micro_top1=float(np.mean([r['top1'] for r in rows])), episodes=len(rows))


def paired_bootstrap(rows, other=None, replicates=1000, seed=93407):
    """Same systems drawn in both tracks and both compared models."""
    keys = lambda r: (r['split'], r['track'], r['episode_id'])
    reference = {keys(r): r for r in other} if other is not None else None
    if reference is not None and set(reference) != {keys(r) for r in rows}:
        raise ValueError("Models must have the same frozen evaluation episodes")
    by = defaultdict(lambda: defaultdict(list)); strata = defaultdict(set)
    for row in rows:
        value = row['rr'] - reference[keys(row)]['rr'] if reference is not None else row['rr']
        by[row['system_id']][row['track']].append(value)
        strata[row['C']].add(row['system_id'])
    rng = np.random.default_rng(seed); samples = defaultdict(list)
    for _ in range(replicates):
        selected = [s for C in sorted(strata) for s in rng.choice(sorted(strata[C]), len(strata[C]), replace=True)]
        for track in ('system_ood', 'composition_ood'):
            if any(track not in by[s] for s in selected):
                raise ValueError("A heldout system is missing one evaluation track")
            samples[track].append(np.mean([np.mean(by[s][track]) for s in selected]))
    return {k: dict(low=float(np.percentile(v, 2.5)), high=float(np.percentile(v, 97.5))) for k, v in samples.items()}

