"""Regression properties for the follow-up's ambiguity-aware rank objective."""
import pytest
torch=pytest.importorskip('torch')
from scripts.train_condition_listwise import supervised_loss

def test_ambiguous_competitor_gets_lower_repulsive_gradient():
    def gradient(confidence):
        logits=torch.tensor([[0.,1.,0.,-1.,-2.,-3.]],requires_grad=True)
        supervised_loss(logits,torch.tensor([confidence])).sum().backward()
        return float(logits.grad[0,1])
    hard=gradient([1.,1.,1.,1.,1.]);ambiguous=gradient([.25,1.,1.,1.,1.])
    assert 0<ambiguous<hard

def test_rank_loss_is_invariant_to_common_logit_offset():
    logits=torch.tensor([[2.,1.,3.,-1.,-2.,0.]])
    conf=torch.tensor([[1.,.25,1.,1.,1.]])
    assert torch.allclose(supervised_loss(logits,conf),supervised_loss(logits+20,conf))

def test_improving_positive_rank_lowers_loss():
    conf=torch.ones(1,5);low=torch.tensor([[0.,2.,2.,1.,0.,-1.]]);high=low.clone();high[0,0]=3.
    assert supervised_loss(high,conf)<supervised_loss(low,conf)

@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),2.])
def test_invalid_confidence_cannot_silently_remove_candidate(bad):
    with pytest.raises(ValueError):supervised_loss(torch.zeros(1,6),torch.tensor([[bad,1.,1.,1.,1.]]))
