import tempfile
import unittest
from types import SimpleNamespace
import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

class TestAllMask(unittest.TestCase):
    def test_inputs_positions_and_markov(self):
        c=Qwen3Config(hidden_size=16,intermediate_size=24,num_hidden_layers=1,num_attention_heads=2,
                      num_key_value_heads=1,head_dim=8,vocab_size=32,layer_types=['full_attention'])
        c.block_size=7;c.num_target_layers=4;c._attn_implementation='sdpa'
        c.dflash_config=dict(projector_type='dspark',target_layer_ids=[1],markov_rank=4,
                            enable_confidence_head=True,confidence_head_with_markov=True,mask_token_id=31)
        draft=DSparkDraftModel(c)
        m=OnlineDSparkModel.__new__(OnlineDSparkModel);torch.nn.Module.__init__(m)
        m.draft_model=draft;m.block_size=7;m.mask_token_id=31;m.embed_tokens=torch.nn.Embedding(32,16)
        ids=torch.tensor([[2,3,4,5,6,7,8,9]])
        anchors=torch.tensor([[1,0]]);keep=torch.tensor([[True,False]])
        expected=torch.full((1,14),31);expected[0,0]=3
        baseline=m._create_noise_embed(ids,anchors,keep)
        torch.testing.assert_close(baseline,m.embed_tokens(expected))
        positions=m._create_position_ids(anchors)
        target,mask,_=m._build_dspark_labels_and_mask(ids,torch.ones_like(ids),anchors,keep)
        prev=torch.cat((ids.gather(1,anchors).unsqueeze(-1),target[:,:,:-1]),-1)
        h=torch.randn(1,2,7,16);logits=torch.randn(1,2,7,32)
        bias=draft.apply_logits_head(logits,prev_token_ids=prev,hidden_states=h)
        conf=draft.predict_confidence(h,prev_token_ids=prev)
        c.dspark_anchor_token_input=False
        all_mask=m._create_noise_embed(ids,anchors,keep)
        torch.testing.assert_close(all_mask,m.embed_tokens(torch.full_like(expected,31)))
        torch.testing.assert_close(all_mask[:,1:],baseline[:,1:])
        torch.testing.assert_close(m._create_position_ids(anchors),positions)
        t2,m2,_=m._build_dspark_labels_and_mask(ids,torch.ones_like(ids),anchors,keep)
        torch.testing.assert_close(t2,target);torch.testing.assert_close(m2,mask)
        torch.testing.assert_close(draft.apply_logits_head(logits,prev_token_ids=prev,hidden_states=h),bias)
        torch.testing.assert_close(draft.predict_confidence(h,prev_token_ids=prev),conf)
        with tempfile.TemporaryDirectory() as root:
            draft.save_pretrained(root)
            restored=DSparkDraftModel.from_pretrained(root,attn_implementation='sdpa')
            self.assertIs(restored.config.dspark_anchor_token_input,False)
        c.dspark_anchor_token_input='false'
        with self.assertRaisesRegex(ValueError,'boolean'): DSparkDraftModel(c)

if __name__=='__main__': unittest.main()
