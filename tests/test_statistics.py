import numpy as np
import pytest
from src.evaluate import cluster_bootstrap, metrics, class_set_bootstrap


def test_cluster_statistic_matches_direct_metrics():
    rng = np.random.default_rng(8)
    p = rng.dirichlet(np.ones(5), 50)
    y = np.arange(50) % 5
    sets = p >= .12
    ci, _ = cluster_bootstrap(p, y, sets, np.repeat(np.arange(10), 5), replicates=100)
    point = ci.set_index('metric').estimate
    for name, value in metrics(p, y, sets).items():
        if name != 'n':
            assert point[name] == pytest.approx(value)


def test_resampling_unit_is_reviewer_not_review():
    # One reviewer owns 9 covered reviews; the other owns 1 uncovered review.
    # Drawing two reviewers gives only 0, .9 or 1 coverage, not a binomial grid.
    p = np.full((10, 5), .2)
    sets = np.zeros((10, 5), bool)
    sets[:9, 0] = True
    ci, draws = cluster_bootstrap(p, np.zeros(10, int), sets, ['A']*9+['B'], replicates=300)
    assert set(np.unique(draws['coverage'])) <= {0., .9, 1.}
    assert ci.set_index('metric').loc['coverage', 'estimate'] == .9
    assert ci.set_index('metric').loc['reviewer_mean_coverage', 'estimate'] == .5


def test_degenerate_class_and_cluster_handling():
    p = np.full((4, 5), .2)
    y = np.zeros(4, int)
    ci, _ = cluster_bootstrap(p, y, np.ones((4, 5), bool), ['A', 'A', 'B', 'B'], replicates=50)
    row = ci.set_index('metric').loc['class_coverage_4']
    assert np.isnan(row.estimate) and row.bootstrap_valid == 0
    with pytest.raises(ValueError):
        cluster_bootstrap(p, y, np.ones((4, 5), bool), ['A']*4, replicates=50)


def test_class_set_metrics_and_paired_cluster_protocol():
    y=np.tile(np.arange(5),2)
    sets=np.vstack([np.eye(5,dtype=bool),np.ones((5,5),bool)])
    sets[9,4]=False
    users=['A']*5+['B']*5
    ci,draws=class_set_bootstrap(y,sets,users,replicates=100,seed=9)
    point=ci.set_index('metric').estimate
    assert point['coverage']==.9
    assert point['macro_class_coverage']==.9
    assert point['worst_class_coverage']==.5
    assert point['average_set_size']==2.9
    assert point['class_size_4']==2.5
    old,old_draws=cluster_bootstrap(np.full((10,5),.2),y,sets,users,replicates=100,seed=9)
    assert np.array_equal(draws['coverage'],old_draws['coverage'])
    assert np.array_equal(draws['class_coverage_4'],old_draws['class_coverage_4'])


def test_missing_true_class_does_not_inflate_worst_or_macro():
    ci,_=class_set_bootstrap(np.zeros(4,int),np.ones((4,5),bool),['A','A','B','B'],replicates=50)
    indexed=ci.set_index('metric')
    assert np.isnan(indexed.loc['worst_class_coverage','estimate'])
    assert indexed.loc['macro_class_coverage','bootstrap_valid']==0
