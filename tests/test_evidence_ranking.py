import random
import pytest
from src.insurerag_vlm.evidence_ranking import sample_training_group


def test_sampling_cycles_positives_without_negative_label_collision():
    g={'positive_ids':['p1','p2'],'negative_ids':list('abcdef'),'negative_sources':{a:'current_hard' for a in 'abcdef'}}
    assert sample_training_group(g,random.Random(42),0)[0]=='p1'
    sample=sample_training_group(g,random.Random(42),1)
    assert sample[0]=='p2' and len(set(sample))==6 and 'p1' not in sample


def test_sampling_rejects_known_positive_as_negative():
    g={'positive_ids':['a'],'negative_ids':list('abcde'),'negative_sources':{a:'current_hard' for a in 'abcde'}}
    with pytest.raises(ValueError):sample_training_group(g,random.Random(42),0)


def test_head_loss_preserves_translation_and_frozen_teacher():
    torch=pytest.importorskip('torch')
    from src.insurerag_vlm.evidence_ranking import head_losses
    logits=torch.tensor([[0.,2.,1.,-1.]],requires_grad=True);teacher=torch.tensor([[1.,0.,2.,-1.]],requires_grad=True);confidence=torch.ones(1,3)
    r,k=head_losses(logits,teacher,confidence);a,b=head_losses(logits+8,teacher-5,confidence)
    torch.testing.assert_close(r,a);torch.testing.assert_close(k,b)
    (r+k).sum().backward();assert teacher.grad is None and torch.isfinite(logits.grad).all()


def test_increasing_positive_improves_head_loss_and_inputs_are_checked():
    torch=pytest.importorskip('torch')
    from src.insurerag_vlm.evidence_ranking import head_losses
    low=torch.tensor([[0.,2.,1.,-1.]]);high=low.clone();high[0,0]=3;confidence=torch.ones(1,3)
    assert head_losses(high,low,confidence)[0]<head_losses(low,low,confidence)[0]
    with pytest.raises(ValueError):head_losses(low,low,confidence*0)


def test_finding_one_fact_does_not_count_as_complete_evidence():
    pytest.importorskip('sklearn')
    from src.insurerag_vlm.evidence_ranking import evidence_metrics
    row=evidence_metrics(['a','x'],{'a','b'},['a','x','b'])
    assert row['hit_at_1']==1 and row['all_evidence_at_5']==0 and row['evidence_recall_at_5']==.5
    assert row['candidate_all_evidence']==1
