from pathlib import Path
import unittest
from stt_rca.config import read_config,architecture,merge


class ConfigurationTests(unittest.TestCase):
    def test_frozen_model_architecture_and_explicit_six_group_override(self):
        root=Path(__file__).resolve().parents[1]/'configs'
        cfg=read_config(root/'stt_rca_six_groups.yaml',[root/'specification/pilot.yaml'])
        model=architecture(cfg)
        self.assertEqual((model.layers,model.d_model,model.d_context,model.d_source),(8,512,256,128))
        self.assertEqual(cfg['data']['channels'],[8,12,16,20,32,48])
        self.assertEqual(cfg['training']['max_updates'],2000)
        self.assertFalse(cfg['execution']['launch_training'])
        self.assertEqual(cfg['optimizer']['warmup_updates'],200)

    def test_merge_replaces_lists_and_preserves_unmodified_mapping(self):
        a={'x':{'v':1,'items':[1,2]},'y':2}
        b=merge(a,{'x':{'items':[3]}})
        self.assertEqual(b,{'x':{'v':1,'items':[3]},'y':2})
        self.assertEqual(a['x']['items'],[1,2])


if __name__=='__main__':unittest.main(verbosity=2)
