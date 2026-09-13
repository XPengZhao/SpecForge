import unittest
from unittest.mock import patch
import torch
from specforge.algorithms.common.dspark_metrics import tau_loss_terms
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel
from specforge.training.metric_window import MetricWindow


def model(alpha, recompute=False):
    m=OnlineDSparkModel.__new__(OnlineDSparkModel);torch.nn.Module.__init__(m)
    m.block_size=3;m.loss_decay_gamma=4.;m.recompute_loss=recompute
    m.dspark_loss_mode='original';m.dspark_l1_loss_alpha=.9;m.dspark_ce_loss_alpha=.1
    m.dspark_confidence_head_alpha=1.;m.dspark_tau_loss_alpha=alpha
    return m


class TestTauLoss(unittest.TestCase):
    def test_formula_truncation_and_gradient(self):
        a=torch.tensor([[[.8,.7,.6],[.2,.9,.9],[.4,.5,.6]]],requires_grad=True)
        mask=torch.tensor([[[1,1,1],[1,0,0],[0,0,0]]],dtype=torch.bool)
        num,den=tau_loss_terms(a,mask)
        self.assertAlmostEqual(den.item(),2)
        self.assertAlmostEqual(num.item(),(3-.8-.56-.336)/3+.8,places=6)
        g=torch.autograd.grad(num/den,a)[0]
        torch.testing.assert_close(g[0,0],torch.tensor([-(1+.7+.7*.6),-(.8+.8*.6),-.8*.7])/6)
        self.assertEqual(g[0,1,1:].abs().sum().item(),0)
        self.assertEqual(g[0,2].abs().sum().item(),0)
        for fill,expected in [(0.,1.),(1.,0.)]:
            x=torch.full((1,2,3),fill,requires_grad=True)
            n,d=tau_loss_terms(x,torch.ones_like(x,dtype=torch.bool))
            self.assertEqual((n/d).item(),expected)
            self.assertTrue(torch.isfinite(torch.autograd.grad(n,x)[0]).all())

    def test_real_loss_gradient_recompute_and_logging(self):
        torch.manual_seed(9)
        d=torch.randn(1,3,3,8);t=torch.randn_like(d);conf=torch.randn(1,3,3)
        mask=torch.tensor([[[1,1,1],[1,0,0],[0,0,0]]],dtype=torch.bool)
        results=[]
        for alpha,recompute in [(0.,False),(.1,False),(.1,True)]:
            x=d.clone().requires_grad_()
            loss,metrics=model(alpha,recompute)._compute_dspark_loss(draft_logits=x,target_ids=torch.zeros(1,3,3,dtype=torch.long),eval_mask=mask,confidence_pred=conf,aligned_target_logits=t)
            loss.backward();results.append((loss.detach(),x.grad,metrics))
        x=d.clone().requires_grad_()
        rates=1-.5*(x.softmax(-1)-t.softmax(-1)).abs().sum(-1)
        n,den=tau_loss_terms(rates,mask)
        expected_g=torch.autograd.grad(.1*n/den,x)[0]
        torch.testing.assert_close(results[1][0]-results[0][0],.1*n.detach()/den)
        torch.testing.assert_close(results[1][1]-results[0][1],expected_g,atol=1e-7,rtol=1e-4)
        torch.testing.assert_close(results[1][1],results[2][1])
        self.assertNotIn('tau_loss',results[0][2]['eval_metric_sums'])
        w=MetricWindow();metrics=results[1][2]
        w.update(dict(sums=metrics['eval_metric_sums'],denoms=metrics['eval_metric_denoms'],weights={'tau_loss':.1}))
        summary=w.summary()
        self.assertAlmostEqual(summary['tau_loss_weighted'],.1*(n/den).item(),places=6)

    def test_global_block_denominator_and_empty_rank(self):
        torch.manual_seed(2)
        d=torch.randn(1,1,3,8);t=torch.randn_like(d)
        for valid in (True,False):
            mask=torch.full((1,1,3),valid,dtype=torch.bool)
            def reduce(stats):
                # Another rank contributes five valid blocks. Other loss
                # denominators do not matter: coefficients are zero here.
                stats[-1]+=5
            m=model(.1);m.dspark_ce_loss_alpha=0.;m.dspark_l1_loss_alpha=0.;m.dspark_confidence_head_alpha=0.
            x=d.clone().requires_grad_()
            with patch('torch.distributed.is_initialized',return_value=True), patch('torch.distributed.get_world_size',return_value=2), patch('torch.distributed.all_reduce',side_effect=reduce):
                loss,_=m._compute_dspark_loss(draft_logits=x,target_ids=torch.zeros(1,1,3,dtype=torch.long),eval_mask=mask,confidence_pred=None,aligned_target_logits=t)
            rates=1-.5*(x.softmax(-1)-t.softmax(-1)).abs().sum(-1)
            n,den=tau_loss_terms(rates,mask)
            torch.testing.assert_close(loss,.1*n*2/(den+5))
            loss.backward();self.assertTrue(torch.isfinite(x.grad).all())
            if not valid: self.assertEqual(x.grad.abs().sum().item(),0)

if __name__=='__main__': unittest.main()
