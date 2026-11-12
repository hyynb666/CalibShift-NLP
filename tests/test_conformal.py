import numpy as np
import pytest
from src.calibration import lac_quantile, prediction_sets, probabilities, fit_temperature


def test_finite_sample_rank_and_ties():
    p = np.array([[v] + [(1-v)/4]*4 for v in np.linspace(.1, .9, 9)])
    y = np.zeros(9, dtype=int)
    assert lac_quantile(p, y, .1) == pytest.approx(.9)
    assert lac_quantile(p, y, .2) == pytest.approx(.8)
    sets = prediction_sets(np.array([[.1, .2, .3, .2, .2]]), .9)
    assert sets.all()  # <= includes the boundary


def test_small_n_requires_infinity():
    p = np.full((3, 5), .2)
    q = lac_quantile(p, np.array([0, 1, 2]), .1)
    assert np.isinf(q)
    assert prediction_sets(p, q).all()


def test_exact_exchangeable_rank_coverage():
    # Enumerate every possible held-out rank, avoiding a flaky simulation test.
    p = np.array([[v] + [(1-v)/4]*4 for v in np.linspace(.05, .95, 10)])
    covered = []
    for held_out in range(10):
        calibration = np.delete(p, held_out, axis=0)
        q = lac_quantile(calibration, np.zeros(9, dtype=int), .1)
        covered.append(prediction_sets(p[held_out:held_out+1], q)[0, 0])
    assert np.mean(covered) == .9


def test_empty_sets_are_allowed():
    assert not prediction_sets(np.full((2, 5), .2), .1).any()


def test_invalid_inputs():
    for p in [np.zeros((2, 5)), np.empty((0, 5)), np.full((2, 5), np.nan)]:
        with pytest.raises(ValueError):
            lac_quantile(p, np.array([0, 1]))
    with pytest.raises(ValueError):
        lac_quantile(np.full((2, 5), .2), np.array([0, 5]))


def test_monotonicity_and_label_permutation():
    rng = np.random.default_rng(7)
    p = rng.dirichlet(np.ones(5), 200)
    y = rng.integers(0, 5, 200)
    assert lac_quantile(p, y, .05) >= lac_quantile(p, y, .2)
    perm = np.array([2, 0, 4, 1, 3])
    assert lac_quantile(p[:, perm], np.argsort(perm)[y]) == lac_quantile(p, y)


def test_temperature_positive_and_nll_nonincreasing():
    z = np.random.default_rng(1).normal(size=(100, 5)) * 5
    y = np.arange(100) % 5
    t = fit_temperature(z, y)
    assert t > 0
    assert -np.log(probabilities(z, t)[np.arange(100), y]).mean() <= -np.log(probabilities(z)[np.arange(100), y]).mean()
