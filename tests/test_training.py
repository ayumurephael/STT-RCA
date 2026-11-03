from pathlib import Path
import os,tempfile,unittest,random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from stt_rca.training import pair_indices,learning_rate,optimizer_for,accumulated_backward,save_checkpoint,load_checkpoint,calibrate_energy
from stt_rca.records import Observation
from stt_rca.objective import STTObjective,TrainingPair
from stt_rca.model import ModelConfig


def model():
    torch.manual_seed(3407)
    return STTObjective(ModelConfig(layers=1,d_model=16,heads=2,d_ff=64,d_context=8,d_source=4,
        root_head_hidden=4,decoder_layers=1,decoder_heads=2,decoder_d_ff=32,activation_checkpointing=False),arm='direct')


def pairs():
    torch.manual_seed(9876);result=[]
    for i in range(4):
        x=torch.randn(1,17+i,3)
        obs=Observation(x,torch.ones_like(x,dtype=torch.bool))
        result.append(TrainingPair(obs,obs,obs,torch.tensor([0]),torch.tensor([1]),torch.tensor([2]),x.clone(),x.clone()))
    return result


def distributed_worker(rank,world,init_file,output):
    os.environ['GLOO_SOCKET_IFNAME']='lo';torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+init_file,rank=rank,world_size=world)
    local_pairs=pairs()[rank*2:(rank+1)*2];wrapped=torch.nn.parallel.DistributedDataParallel(model())
    accumulated_backward(wrapped,local_pairs,0)
    if rank==0:torch.save({k:p.grad for k,p in wrapped.module.named_parameters()},output)
    dist.destroy_process_group()


class TrainingTests(unittest.TestCase):
    def test_all_world_sizes_keep_exact_order_and_multiplicity(self):
        for world in (1,2,3,4,6):
            ids=[i for rank in range(world) for i in pair_indices(7,rank,world)]
            self.assertEqual(ids,list(range(7*96,8*96)))

    def test_lr_warmup_and_last_update(self):
        self.assertAlmostEqual(learning_rate(0,2000,200),3e-4/200)
        self.assertAlmostEqual(learning_rate(199,2000,200),3e-4)
        self.assertAlmostEqual(learning_rate(1999,2000,200),3e-5)

    def test_energy_calibration_retains_zero_pairs_and_uses_only_train_row_ids(self):
        values=np.r_[0.,np.ones(4095)]
        result=calibrate_energy(values,list(range(4096)))
        self.assertEqual(result['epsilon_energy'],.01);self.assertEqual(result['zero_count'],1)
        with self.assertRaises(ValueError):calibrate_energy(np.zeros(4096),list(range(4096)))

    def test_accumulation_matches_global_mean_without_optimizer_update(self):
        batch=pairs();a=model();b=model()
        accumulated_backward(a,batch,0)
        sum(b(p,0)['loss'].mean() for p in batch).div(4).backward()
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)

    def test_two_rank_gloo_matches_single_rank_global_mean(self):
        if os.name!='posix':self.skipTest('Linux Gloo loopback test')
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);init=str(directory/'init');output=str(directory/'grads.pt')
            mp.spawn(distributed_worker,args=(2,init,output),nprocs=2,join=True)
            reference=model();accumulated_backward(reference,pairs(),0)
            actual=torch.load(output,weights_only=True)
            for name,p in reference.named_parameters():torch.testing.assert_close(actual[name],p.grad,atol=1e-6,rtol=1e-5)

    def test_checkpoint_exact_rng_cursor_and_hash_rejection(self):
        instance=model();optimizer=optimizer_for(instance)
        random.seed(77);np.random.seed(77);torch.manual_seed(77)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt';contract=dict(source='abc',pairs='def',worlds='ghi')
            save_checkpoint(path,instance,optimizer,5,contract,{'training':{'max_updates':2000}})
            expected=(random.random(),float(np.random.random()),float(torch.rand(())))
            random.random();np.random.random();torch.rand(())
            self.assertEqual(load_checkpoint(path,instance,optimizer,contract),5)
            actual=(random.random(),float(np.random.random()),float(torch.rand(())))
            self.assertEqual(actual,expected)
            with self.assertRaises(ValueError):load_checkpoint(path,instance,optimizer,dict(source='changed'))


if __name__=='__main__':unittest.main(verbosity=2)
