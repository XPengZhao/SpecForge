"""MLA attention reference math, masking, backends, and serialization."""
import tempfile
import unittest
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config, Qwen3RotaryEmbedding
from transformers import DynamicCache
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.dspark_mla import DSparkMLAAttention
from specforge.modeling.draft.dspark import DSparkDraftModel


def config(kind='mla'):
    c = Qwen3Config(hidden_size=32, intermediate_size=48, num_hidden_layers=1,
                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                    vocab_size=64, attention_dropout=0., layer_types=['full_attention'])
    c.draft_attention_type = kind
    c.mla_kv_lora_rank=12
    c.mla_qk_nope_head_dim=4
    c.mla_qk_rope_head_dim=4
    c.mla_v_head_dim=8
    c.num_target_layers=4
    c.block_size=3
    c.dflash_config=dict(projector_type='dspark',target_layer_ids=[1],markov_rank=0,
                         enable_confidence_head=False,mask_token_id=63)
    c._attn_implementation='eager'
    return c


class TestMLA(unittest.TestCase):
    def test_explicit_reference_and_sdpa_gradients(self):
        torch.manual_seed(4)
        c=config(); m=DSparkMLAAttention(c,0,DEFAULT_DFLASH_KERNELS)
        x=torch.randn(2,3,32,requires_grad=True)
        ctx=torch.randn(2,5,32,requires_grad=True)
        pos=torch.tensor([[0,1,2,3,4,2,3,4]]).expand(2,-1)
        rc=config();rc.head_dim=4
        emb=Qwen3RotaryEmbedding(rc)(x,pos)
        mask=torch.zeros(2,1,3,8);mask[...,4]=-torch.inf
        y,_=m(x,ctx,emb,mask)
        # Independent explicit per-head score: content + separate rotary score.
        src=torch.cat((ctx,x),1)
        a=m.kv_a_proj(src); latent=m.kv_a_norm(a[...,:12])
        kv=m.kv_b_proj(latent).reshape(2,8,4,12)
        q=m.q_proj(x).reshape(2,3,4,8)
        def rotate(v,cos,sin):
            u,w=v.chunk(2,-1)
            return v*cos+torch.cat((-w,u),-1)*sin
        cos,sin=emb
        kr=rotate(a[...,12:],cos,sin)
        heads=[]
        for h in range(4):
            qr=rotate(q[:,:,h,4:],cos[:,-3:],sin[:,-3:])
            score=(q[:,:,h,:4]@kv[:,:,h,:4].transpose(-1,-2)+qr@kr.transpose(-1,-2))*8**-.5
            heads.append((score+mask[:,0]).softmax(-1)@kv[:,:,h,4:])
        ref=m.o_proj(torch.cat(heads,-1))
        torch.testing.assert_close(y,ref)
        params=(x,ctx,*m.parameters())
        gy=torch.autograd.grad(y.square().sum(),params,retain_graph=True)
        gr=torch.autograd.grad(ref.square().sum(),params)
        for actual,expected in zip(gy,gr): torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)
        c._attn_implementation='sdpa'
        sy,_=m(x,ctx,emb,mask)
        torch.testing.assert_close(y,sy,atol=1e-6,rtol=1e-5)
        sg=torch.autograd.grad(sy.square().sum(),params)
        for actual,expected in zip(sg,gy): torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-4)

    def test_model_roundtrip_and_gqa_default(self):
        for kind in ('gqa','mla'):
            c=config(kind)
            if kind=='gqa': del c.draft_attention_type
            m=DSparkDraftModel(c).eval()
            kw=dict(position_ids=torch.arange(7).unsqueeze(0),noise_embedding=torch.randn(1,3,32),
                    target_hidden=torch.randn(1,4,32),attention_mask=torch.zeros(1,1,3,7))
            expected=m(**kw)
            with tempfile.TemporaryDirectory() as root:
                m.save_pretrained(root)
                restored=DSparkDraftModel.from_pretrained(root,attn_implementation='eager').eval()
                torch.testing.assert_close(restored(**kw),expected)
                self.assertEqual(set(m.state_dict()),set(restored.state_dict()))
            self.assertEqual(isinstance(m.layers[0].self_attn,DSparkMLAAttention),kind=='mla')

    def test_invalid_dimensions(self):
        for name,value in [('mla_kv_lora_rank',0),('mla_qk_rope_head_dim',3),('mla_v_head_dim',4)]:
            c=config();setattr(c,name,value)
            with self.assertRaises(ValueError): DSparkMLAAttention(c,0,DEFAULT_DFLASH_KERNELS)

    def test_expanded_cache_matches_full_context(self):
        c=config();m=DSparkMLAAttention(c,0,DEFAULT_DFLASH_KERNELS).eval()
        rc=config();rc.head_dim=4
        rotary=Qwen3RotaryEmbedding(rc)
        prefix=torch.randn(1,4,32); old=torch.randn(1,2,32); new=torch.randn(1,1,32)
        cache=DynamicCache()
        m(old,prefix,rotary(old,torch.arange(6).unsqueeze(0)),None,past_key_values=cache)
        actual,_=m(new,prefix[:,:0],rotary(new,torch.tensor([[6]])),None,past_key_values=cache)
        expected,_=m(new,torch.cat((prefix,old),1),rotary(new,torch.arange(7).unsqueeze(0)),None)
        torch.testing.assert_close(actual,expected)

if __name__=='__main__': unittest.main()
