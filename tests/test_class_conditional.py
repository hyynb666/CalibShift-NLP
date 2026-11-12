import numpy as np
import pytest
from src.calibration import class_conditional_quantiles, class_conditional_sets, lac_quantile


def test_true_label_partition_and_candidate_thresholds():
    rng = np.random.default_rng(6)
    p = rng.dirichlet(np.ones(5), 100)
    y = np.repeat(np.arange(5), 20)
    q = class_conditional_quantiles(p, y)
    for k in range(5):
        assert q[k] == lac_quantile(p[y==k], y[y==k])
    example = np.array([[.1,.2,.3,.2,.2]])
    assert np.array_equal(class_conditional_sets(example, [.9,.1,.7,.1,.1]), [[True,False,True,False,False]])


def test_empty_and_too_small_classes_are_always_included():
    p = np.full((8, 5), .2)
    q = class_conditional_quantiles(p, np.zeros(8, int), .1)
    assert np.isposinf(q).all()  # k=ceil(9*.9)=9 > 8, never a global fallback.
    assert class_conditional_sets(np.full((2,5), .2), q).all()
    assert np.isposinf(class_conditional_quantiles(np.empty((0,5)), np.array([],int))).all()
    q = class_conditional_quantiles(np.full((9,5), .2), np.zeros(9,int), .1)
    assert q[0] == .8 and np.isposinf(q[1:]).all()


def test_label_permutation_and_invalid_inputs():
    rng = np.random.default_rng(4)
    p = rng.dirichlet(np.ones(5), 150)
    y = np.arange(150)%5
    perm = np.array([2,4,0,1,3])
    q = class_conditional_quantiles(p,y)
    assert np.allclose(class_conditional_quantiles(p[:,perm],np.argsort(perm)[y]), q[perm])
    for bad in [np.array([0.]*150), np.array([5]*150)]:
        with pytest.raises(ValueError):
            class_conditional_quantiles(p,bad)
    with pytest.raises(ValueError):
        class_conditional_sets(p,[.1]*4)


def test_exact_within_class_exchangeable_rank_coverage():
    p = np.array([[v]+[(1-v)/4]*4 for v in np.linspace(.05,.95,10)])
    covered=[]
    for i in range(10):
        q=class_conditional_quantiles(np.delete(p,i,axis=0),np.zeros(9,int))
        covered.append(class_conditional_sets(p[i:i+1],q)[0,0])
    assert np.mean(covered)==.9
