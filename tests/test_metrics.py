import unittest
from stt_rca.metrics import summarize,paired_bootstrap


def rows():
    return [dict(split='validation',track=t,episode_id=t+'-'+s,system_id=s,C=c,rr=rr,top1=rr)
            for s,c,rr in [('a',8,.5),('b',8,1.),('c',12,.25),('d',12,.75)]
            for t in ['system_ood','composition_ood']]


class MetricTests(unittest.TestCase):
    def test_system_macro_and_duplicate_rejection(self):
        value=rows();self.assertEqual(summarize(value)['selection_mrr'],.625)
        with self.assertRaises(ValueError):summarize(value+[value[0]])

    def test_paired_bootstrap_same_models_zero_interval_and_seed_stability(self):
        value=rows();result=paired_bootstrap(value,value,replicates=100)
        self.assertTrue(all(v==dict(low=0.,high=0.) for v in result.values()))
        self.assertEqual(paired_bootstrap(value,replicates=100),paired_bootstrap(value,replicates=100))


if __name__=='__main__':unittest.main(verbosity=2)
