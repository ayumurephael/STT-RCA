import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from stt_rca.records import Observation, NormalizedObservation, normalize
from stt_rca.model import ModelConfig, Encoder, Decoder, Ranker, select_source, place_source, ieee_precision
from stt_rca.objective import STTObjective, TrainingPair, response_loss
from stt_rca.metrics import ranking_metrics, summarize, paired_bootstrap
from stt_rca.export import export_ranker, load_ranker

torch.set_num_threads(1)
ieee_precision()


def tiny(**kwargs):
    return replace(ModelConfig(layers=2, d_model=32, heads=4, d_ff=128, d_context=16, d_source=8,
        root_head_hidden=4, decoder_layers=2, decoder_heads=2, decoder_d_ff=64, activation_checkpointing=False), **kwargs)


def observation(T=33, C=4, batch=1):
    x = torch.randn(batch, T, C)
    return Observation(x, torch.ones_like(x, dtype=torch.bool))


class ModelTests(unittest.TestCase):
    def setUp(self): torch.manual_seed(3407)

    def test_whitelist_and_unobserved_channel(self):
        with self.assertRaises(ValueError): Observation.from_mapping(dict(x=1, valid=2, root=0))
        obs = observation(); obs.valid[:, :, 0] = False
        with self.assertRaisesRegex(ValueError, 'UNSUPPORTED'): normalize(obs)

    def test_float64_linear_quantiles_masked_nan(self):
        obs = observation(); obs.valid[:, 0, 0] = False; obs.x[:, 0, 0] = float('nan')
        normalized, stats = normalize(obs)
        for c in range(4):
            values = obs.x[0, obs.valid[0, :, c], c].numpy().astype(np.float64)
            q = np.quantile(values, [.25, .5, .75], method='linear')
            self.assertEqual(stats.center[0, 0, c].item(), q[1])
            self.assertEqual(stats.scale[0, 0, c].item(), max(q[2]-q[0], 1e-4))
        self.assertEqual(normalized.z[0, 0, 0].item(), 0)
        self.assertTrue(torch.isfinite(normalized.z).all())

    def test_exact_full_parameter_count(self):
        model = STTObjective()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 38106897)
        ranker = Ranker(ModelConfig())
        self.assertEqual(sum(p.numel() for p in ranker.parameters()), 33724417)

    def test_permutation_padding_batch_and_partial_patch(self):
        model = Encoder(tiny()).eval(); obs, _ = normalize(observation())
        expected = model(obs)
        perm = torch.tensor([2, 0, 3, 1])
        actual = model(NormalizedObservation(obs.z[:, :, perm], obs.valid[:, :, perm]))
        torch.testing.assert_close(actual.logits, expected.logits[:, perm], atol=1e-5, rtol=1e-5)
        padded = NormalizedObservation(torch.nn.functional.pad(obs.z, (0, 2, 0, 31)),
                                       torch.nn.functional.pad(obs.valid, (0, 2, 0, 31), value=False))
        output = model(padded)
        torch.testing.assert_close(output.logits[:, :4], expected.logits, atol=1e-5, rtol=1e-5)
        self.assertTrue(torch.isneginf(output.logits[:, 4:]).all())
        combined = model(NormalizedObservation(obs.z.expand(2, -1, -1), obs.valid.expand(2, -1, -1)))
        torch.testing.assert_close(combined.logits[0], expected.logits[0], atol=1e-5, rtol=1e-5)

    def test_decoder_equivariance_zero_source_and_context_distinction(self):
        model = STTObjective(tiny()).eval(); obs, _ = normalize(observation()); encoded = model.encoder(obs)
        empty = torch.zeros_like(encoded.source)
        y0 = model.decoder(encoded.context, empty, obs.valid, 33)
        self.assertFalse(torch.equal(y0[:, :, 0], y0[:, :, 1]))
        torch.testing.assert_close(y0 - model.decoder(encoded.context, empty, obs.valid, 33), torch.zeros_like(y0), atol=0, rtol=0)
        p = torch.tensor([2, 0, 3, 1])
        permuted = model.decoder(encoded.context[:, :, p], empty[:, :, p], obs.valid[:, :, p], 33)
        torch.testing.assert_close(permuted, y0[:, :, p], atol=1e-5, rtol=1e-5)

    def test_transplant_gradient_reaches_source_context_and_trunk(self):
        model = STTObjective(tiny()); a, _ = normalize(observation(C=3)); b, _ = normalize(observation(C=4))
        A, B = model.encoder(a), model.encoder(b)
        A.source.retain_grad(); B.context.retain_grad()
        v = place_source(select_source(A.source, torch.tensor([1])), torch.tensor([2]), 4)
        z = model.decoder(B.context, torch.zeros_like(v), b.valid, 33)
        y = model.decoder(B.context, v, b.valid, 33)
        loss, _ = response_loss(y-z, torch.randn_like(y), b.valid, 1e-3)
        loss.mean().backward()
        self.assertGreater(A.source.grad[:, :, 1].abs().sum().item(), 0)
        self.assertEqual(A.source.grad[:, :, [0,2]].abs().sum().item(), 0)
        self.assertGreater(B.context.grad.abs().sum().item(), 0)
        self.assertGreater(model.encoder.patch_embed.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.encoder.root_head[-1].weight.grad)

    def test_fp32_decoder_inside_bf16_autocast_matches_values_and_gradients(self):
        decoder = Decoder(tiny()); clone = copy.deepcopy(decoder)
        h = torch.randn(1, 3, 4, 16, requires_grad=True); v = torch.randn(1, 3, 4, 8, requires_grad=True)
        h2, v2 = h.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()
        valid = torch.ones(1, 33, 4, dtype=torch.bool)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            y0 = decoder(h, torch.zeros_like(v), valid, 33); y1 = decoder(h, v, valid, 33)
            delta = y1 - y0
        reference = clone(h2, v2, valid, 33) - clone(h2, torch.zeros_like(v2), valid, 33)
        self.assertEqual(delta.dtype, torch.float32)
        torch.testing.assert_close(delta, reference, atol=1e-6, rtol=1e-5)
        delta.square().sum().backward(); reference.square().sum().backward()
        torch.testing.assert_close(h.grad, h2.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(v.grad, v2.grad, atol=1e-6, rtol=1e-5)
        for p, q in zip(decoder.parameters(), clone.parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)
        layer = torch.nn.Linear(1,1,bias=False); layer.weight.data.fill_(1.)
        x = torch.tensor([[1.], [1.+2**-10]], requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16): rounded = layer(x).float()
        self.assertEqual((rounded[1]-rounded[0]).item(), 0.)
        full = layer(x); self.assertEqual((full[1]-full[0]).item(), 2**-10)
        (full[1]-full[0]).square().backward(); self.assertTrue(torch.isfinite(x.grad).all())

    def test_reconstruction_keeps_masked_queries(self):
        encoder = Encoder(tiny()).eval(); obs, _ = normalize(observation())
        aug = torch.zeros(1, 3, 4, dtype=torch.bool); aug[:, 0, 1] = True
        features = []
        handle = encoder.patch_embed.register_forward_pre_hook(lambda m, args: features.append(args[0].detach().clone()))
        hidden = encoder(obs, augmentation_mask=aug); handle.remove()
        self.assertEqual(features[0][0,0,1].abs().sum().item(), 0)
        self.assertTrue(hidden.token_valid[0,0,1])
        self.assertGreater(hidden.context[0,0,1].abs().sum().item(), 0)

    def test_objective_all_supported_arms_finite_backward(self):
        A, B, F = observation(C=3), observation(), observation()
        pair = TrainingPair(A,B,F,torch.tensor([1]),torch.tensor([2]),torch.tensor([3]),
                            torch.randn_like(B.x),torch.randn_like(B.x),torch.randn(1,33),torch.tensor([0]))
        for arm in ['stt','direct','delete_trans','oracle','payload','reconstruction']:
            with self.subTest(arm=arm):
                model = STTObjective(tiny(), arm=arm)
                values = model(pair, 0); values['loss'].mean().backward()
                self.assertTrue(torch.isfinite(values['loss']).all())
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_checkpointed_backward_matches_no_checkpoint(self):
        a=Encoder(tiny()).train(); b=Encoder(tiny(activation_checkpointing=True)).train(); b.load_state_dict(a.state_dict())
        obs, _ = normalize(observation()); x=a(obs); y=b(obs)
        x.logits.sum().backward(); y.logits.sum().backward()
        torch.testing.assert_close(x.logits,y.logits)
        for p,q in zip(a.parameters(),b.parameters()):
            if p.grad is not None: torch.testing.assert_close(p.grad,q.grad)

    def test_export_drops_privileged_modules_and_preserves_logits(self):
        model=STTObjective(tiny()).eval(); obs=observation(); normalized,_=normalize(obs)
        logits=model.encoder(normalized).logits
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'ranker.pt'; export_ranker(model,path); ranker=load_ranker(path)
            self.assertFalse(any('context' in k or 'decoder' in k for k in ranker.state_dict()))
            torch.testing.assert_close(ranker.predict(obs.x,obs.valid)['logits'],logits,atol=0,rtol=0)

    def test_pair_mean_of_ratios_and_exact_ties(self):
        target=torch.tensor([[[1.,1.]],[[10.,10.]]]); predicted=torch.zeros_like(target);valid=torch.ones_like(target,dtype=torch.bool)
        loss,_=response_loss(predicted,target,valid,1.)
        self.assertAlmostEqual(loss.mean().item(),(.5+100/101)/2,places=6)
        value=ranking_metrics([2.,2.,1.],0);self.assertEqual(value['rr'],.75);self.assertEqual(value['top1'],.5)
        self.assertEqual(ranking_metrics([1.,1.+2**-20,0.],0)['rr'],.5)

    def test_all_six_channel_groups_and_long_nondivisible_length(self):
        model=Encoder(tiny(layers=1)).eval()
        with torch.no_grad():
            for channels in (8,12,16,20,32,48):
                obs,_=normalize(observation(T=35,C=channels))
                out=model(obs)
                self.assertEqual(tuple(out.logits.shape),(1,channels))
                self.assertTrue(torch.isfinite(out.logits).all())
            obs,_=normalize(observation(T=10001,C=8))
            self.assertEqual(model(obs).source.shape[1],626)

    def test_bf16_trunk_keeps_registered_projections_and_root_fp32(self):
        model=Encoder(tiny()).train();obs,_=normalize(observation())
        with torch.autocast('cpu',dtype=torch.bfloat16):out=model(obs)
        self.assertEqual(out.context.dtype,torch.float32)
        self.assertEqual(out.source.dtype,torch.float32)
        self.assertEqual(out.logits.dtype,torch.float32)
        out.logits.sum().backward()
        self.assertIsNotNone(model.source_projection.weight.grad)

    def test_offset_information_boundary_same_normalized_inputs_same_logits(self):
        base=torch.arange(33,dtype=torch.float32)[None,:,None].expand(1,-1,4).clone()
        a,b=base.clone(),base.clone();a[:,:,0]+=8;b[:,:,3]+=8
        na,_=normalize(Observation(a,torch.ones_like(a,dtype=torch.bool)))
        nb,_=normalize(Observation(b,torch.ones_like(b,dtype=torch.bool)))
        torch.testing.assert_close(na.z,nb.z,atol=0,rtol=0)
        encoder=Encoder(tiny()).eval()
        torch.testing.assert_close(encoder(na).logits,encoder(nb).logits,atol=0,rtol=0)


if __name__=='__main__': unittest.main(verbosity=2)
