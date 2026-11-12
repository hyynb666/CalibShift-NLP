import numpy as np
from matched_size import match_marginal_size
from src.calibration import lac_quantile, prediction_sets


def test_exact_discrete_search_matches_exhaustive_reference():
    rng=np.random.default_rng(55)
    p=rng.dirichlet(np.ones(5),23)
    y=rng.integers(0,5,23)
    val=rng.dirichlet(np.ones(5),11)
    target=29
    result=match_marginal_size(p,y,val,target)
    q=np.r_[np.unique(1-p[np.arange(len(y)),y]),np.inf]
    counts=[int(prediction_sets(val,t).sum()) for t in q]
    best=min(range(len(q)),key=lambda i:(abs(counts[i]-target),counts[i],q[i]))
    assert result['qhat']==q[best]
    assert result['matched_total']==counts[best]
    assert lac_quantile(p,y,result['alpha'])==result['qhat']


def test_unattainable_match_uses_smaller_count_for_tie():
    p=np.array([[.8,.05,.05,.05,.05],[.2,.2,.2,.2,.2]])
    val=np.array([[.6,.4,0,0,0]])
    result=match_marginal_size(p,np.zeros(2,int),val,1)
    assert not result['exact_match']
    assert result['matched_total']==0
    assert result['signed_gap']==-1


def test_tied_scores_use_first_rank_and_deterministic_alpha():
    p=np.full((9,5),.2)
    val=np.full((3,5),.2)
    a=match_marginal_size(p,np.zeros(9,int),val,15)
    b=match_marginal_size(p[::-1],np.zeros(9,int),val[::-1],15)
    assert a==b and a['rank']==1 and a['exact_match']
    assert a['candidate_count']==2
    assert lac_quantile(p,np.zeros(9,int),a['alpha'])==a['qhat']
