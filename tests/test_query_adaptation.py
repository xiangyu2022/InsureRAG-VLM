import importlib.util
import numpy as np
import pytest

from src.insurerag_vlm.query_adaptation import blend_queries, multi_positive_losses


def test_query_interpolation_endpoints_and_normalization():
    a = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
    b = np.array([[0., 1.], [1., 0.]], dtype=np.float32)
    assert np.array_equal(blend_queries(a, b, 0), a)
    assert np.array_equal(blend_queries(a, b, 1), b)
    np.testing.assert_allclose(blend_queries(a, b, .5), np.full((2, 2), 1/np.sqrt(2)), atol=1e-6)


def test_invalid_query_interpolation_rejected():
    a = np.array([[1., 0.]])
    for b, alpha in [(a[:, :1], .5), (a, 1.1), (-a, .5), (np.full_like(a, np.nan), .5)]:
        with pytest.raises(ValueError): blend_queries(a, b, alpha)


requires_torch = pytest.mark.skipif(importlib.util.find_spec('torch') is None, reason='Optional training dependency')


@requires_torch
def test_multiple_correct_passages_are_not_treated_as_negatives():
    import torch
    values = torch.tensor([[1., 2., 0.]], requires_grad=True)
    mask = torch.tensor([[True, True, False]])
    ce, kl = multi_positive_losses(values, mask, values.detach())
    expected = -torch.log(torch.softmax(values, dim=1)[0, :2].sum())
    torch.testing.assert_close(ce[0], expected)
    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-7, rtol=0)
    ce.mean().backward()
    assert values.grad[0, 0] < 0 and values.grad[0, 1] < 0 and values.grad[0, 2] > 0


@requires_torch
def test_loss_is_translation_invariant_and_teacher_is_frozen():
    import torch
    student = torch.tensor([[1., 2., 3.], [-1., 2., 0.]], requires_grad=True)
    teacher = torch.tensor([[2., 1., 0.], [0., 2., 1.]], requires_grad=True)
    mask = torch.tensor([[True, False, False], [False, True, True]])
    ce, kl = multi_positive_losses(student, mask, teacher)
    translated = multi_positive_losses(student+7, mask, teacher-3)
    torch.testing.assert_close(ce, translated[0])
    torch.testing.assert_close(kl, translated[1])
    (ce+kl).mean().backward()
    assert teacher.grad is None
    assert torch.isfinite(student.grad).all()


@requires_torch
def test_missing_positive_or_invalid_logits_fail_loudly():
    import torch
    values = torch.tensor([[1., 0.]])
    with pytest.raises(ValueError): multi_positive_losses(values, torch.tensor([[False, False]]), values)
    with pytest.raises(ValueError): multi_positive_losses(values*float('nan'), torch.tensor([[True, False]]), values)
