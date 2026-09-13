import tempfile
import unittest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask, create_dflash_block_mask

class TestContextOnly(unittest.TestCase):
    def test_masks(self):
        a=torch.tensor([[0,4,7]]); keep=torch.tensor([[True,True,False]])
        for window in (None,2):
            base=create_dflash_sdpa_mask(a,keep,8,3,'cpu',context_window=window)
            mask=create_dflash_sdpa_mask(a,keep,8,3,'cpu',context_window=window,block_attention='context_only')
            torch.testing.assert_close(mask[...,:8],base[...,:8])
            self.assertFalse(mask[...,8:].any())
            self.assertFalse(mask[:,:,:3].any())
            self.assertFalse(mask[:,:,6:].any())
            for mode,dense in [('bidirectional',base),('context_only',mask)]:
                flex=create_dflash_block_mask(a,keep,8,3,'cpu',context_window=window,block_attention=mode)
                actual=flex.mask_mod(torch.tensor(0),torch.tensor(0),torch.arange(9)[:,None],torch.arange(17)[None,:])
                torch.testing.assert_close(actual,dense[0,0])

    def test_model_empty_rows_and_isolation(self):
        c=Qwen3Config(hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,
                      num_key_value_heads=2,head_dim=8,vocab_size=64,layer_types=['full_attention']*2)
        c.block_size=3;c.num_target_layers=4
        c.dflash_config=dict(projector_type='dspark',target_layer_ids=[1],markov_rank=0,enable_confidence_head=False)
        c.dspark_block_attention='context_only';c._attn_implementation='sdpa'
        m=DSparkDraftModel(c).eval()
        mask=create_dflash_sdpa_mask(torch.tensor([[0,3,0]]),torch.tensor([[True,True,False]]),4,3,'cpu',block_attention='context_only')
        noise=torch.randn(1,9,32,requires_grad=True)
        ctx=torch.randn(1,4,32,requires_grad=True)
        kw=dict(position_ids=torch.tensor([[0,1,2,3,0,1,2,3,4,5,0,1,2]]),noise_embedding=noise,target_hidden=ctx,attention_mask=mask)
        results=[]
        for backend in ('sdpa','eager'):
            c._attn_implementation=backend
            out=m(**kw)
            grad=torch.autograd.grad(out[:,3].square().sum(),noise,retain_graph=True)[0]
            self.assertEqual(grad[:,:3].abs().max().item(),0)
            self.assertEqual(grad[:,4:].abs().max().item(),0)
            grads=torch.autograd.grad(out.square().sum(),(noise,ctx,*m.parameters()))
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            results.append((out.detach(),grads))
            changed=noise.detach().clone();changed[:,4:]=torch.randn_like(changed[:,4:])*10
            torch.testing.assert_close(m(**{**kw,'noise_embedding':changed})[:,3],out[:,3])
        torch.testing.assert_close(results[0][0],results[1][0],atol=1e-5,rtol=1e-4)
        for x,y in zip(results[0][1],results[1][1]): torch.testing.assert_close(x,y,atol=1e-4,rtol=1e-3)
        with self.assertRaisesRegex(ValueError,'explicit attention mask'):
            m(**{**kw,'attention_mask':None})
        with tempfile.TemporaryDirectory() as root:
            m.save_pretrained(root)
            restored=DSparkDraftModel.from_pretrained(root,attn_implementation='eager').eval()
            self.assertEqual(restored.config.dspark_block_attention,'context_only')
            torch.testing.assert_close(restored(**kw),results[-1][0])

if __name__=='__main__': unittest.main()
