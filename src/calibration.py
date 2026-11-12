"""Temperature scaling and finite-sample split conformal LAC."""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax


def probabilities(logits, temperature=1.0):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be positive and finite')
    return softmax(np.asarray(logits, dtype=float) / temperature, axis=1)


def fit_temperature(logits, labels):
    logits, labels = np.asarray(logits, float), np.asarray(labels, int)
    def loss(log_t):
        z = logits / np.exp(log_t)
        return float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(labels)), labels]))
    result = minimize_scalar(loss, bounds=(-5, 5), method='bounded')
    if not result.success:
        raise RuntimeError('Temperature optimization failed')
    # Include T=1 so numerical optimization cannot worsen calibration-set NLL.
    return float(np.exp(result.x)) if result.fun < loss(0) else 1.0


def validate_probs(probs):
    p = np.asarray(probs, float)
    if (p.ndim != 2 or not len(p) or p.shape[1] != 5 or
            not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or
            not np.allclose(p.sum(axis=1), 1)):
        raise ValueError('Expected nonempty finite n x 5 probability matrix')
    return p


def lac_quantile(probs, labels, alpha=0.1):
    p = validate_probs(probs)
    y = np.asarray(labels)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer) or ((y < 0) | (y > 4)).any():
        raise ValueError('Labels must be integer 0..4 and aligned with probabilities')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    n = len(y)
    k = math.ceil((n + 1) * (1 - alpha))
    # Order statistic of n scores augmented with +infinity, never clip k to n.
    if k > n:
        return float('inf')
    return float(np.partition(1 - p[np.arange(n), y], k - 1)[k - 1])


def prediction_sets(probs, threshold):
    if np.isnan(threshold):
        raise ValueError('NaN threshold')
    return 1 - validate_probs(probs) <= threshold


def class_conditional_quantiles(probs, labels, alpha=0.1):
    """Label-Mondrian LAC. Missing/insufficient classes get +infinity.

    Each class uses scores from calibration rows whose TRUE label is that class.
    At prediction time all candidate labels are tested; no true label is needed.
    Validity requires within-label exchangeability, including calibration data.
    """
    p, y = np.asarray(probs, float), np.asarray(labels)
    if p.ndim != 2 or p.shape[1] != 5:
        raise ValueError('Expected n x 5 probabilities')
    if len(p):
        validate_probs(p)
    if y.shape != (len(p),) or not np.issubdtype(y.dtype, np.integer) or ((y<0)|(y>4)).any():
        raise ValueError('Labels must be aligned integer 0..4')
    if not 0 < alpha < 1:
        raise ValueError('alpha must lie in (0, 1)')
    thresholds = np.full(5, np.inf)
    for k in range(5):
        selected = y == k
        if selected.any():
            thresholds[k] = lac_quantile(p[selected], y[selected], alpha)
    return thresholds


def class_conditional_sets(probs, thresholds):
    q = np.asarray(thresholds, float)
    if q.shape != (5,) or np.isnan(q).any() or np.isneginf(q).any() or (q < 0).any():
        raise ValueError('Expected five nonnegative thresholds (positive infinity allowed)')
    return 1 - validate_probs(probs) <= q[None, :]
