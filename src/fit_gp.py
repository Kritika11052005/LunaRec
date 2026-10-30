"""
Learn the GP hyperparameters by maximising the marginal likelihood.

fit_variogram.py estimates the same quantities by binning semivariance and
fitting a curve. That is interpretable -- you can look at the binned points and
judge whether the model describes them -- but it is a two-stage procedure: the
binning throws away information, and the least-squares fit to bin means is not
the maximum-likelihood estimate of anything.

This fits the GP directly. The loss is the negative log marginal likelihood of
the observed values under the covariance, optimised over the covariance's own
parameters. It is training in the ordinary sense: a loss, an optimiser, learned
parameters, a convergence curve.

Keeping both methods is deliberate. They share no machinery -- one works on
binned pair differences, the other on the full covariance matrix -- so
agreement between them is real corroboration. Disagreement would be informative
in its own right.

THE COVARIANCE MUST MATCH evaluate.py. It uses

    K = (1 - nugget) * spherical(D, range) + nugget * I

with unit diagonal. Anything learned here against a different form would not be
usable there, which would defeat the purpose of learning it.

Parameters are optimised unconstrained and mapped into range:
    range  = exp(theta_0)          > 0
    nugget = sigmoid(theta_1)      in (0, 1)
    sigma2 = exp(theta_2)          > 0
This keeps L-BFGS-B unconstrained while guaranteeing a valid covariance, and
avoids the optimiser stalling on a boundary.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

JITTER = 1e-8          # guards the Cholesky when the nugget drifts very small


def spherical_corr(D, rng_):
    """Spherical correlation; identical in form to evaluate.spherical_cov."""
    r = np.clip(D / rng_, 0.0, 1.0)
    return 1.0 - 1.5 * r + 0.5 * r ** 3


def _unpack(theta):
    rng_ = float(np.exp(theta[0]))
    nug = float(1.0 / (1.0 + np.exp(-theta[1])))
    s2 = float(np.exp(theta[2]))
    return rng_, nug, s2


def neg_log_marginal(theta, D, y):
    """Negative log marginal likelihood of y under the covariance.

    -log p(y | theta) = 0.5 y' K^-1 y  +  0.5 log|K|  +  (n/2) log 2*pi

    The first term rewards explaining the data, the second penalises a
    covariance that is permissive enough to explain anything. That trade is
    what makes marginal likelihood self-regularising: it will not simply widen
    the range until everything correlates with everything.
    """
    rng_, nug, s2 = _unpack(theta)
    n = len(y)
    K = s2 * ((1.0 - nug) * spherical_corr(D, rng_) + nug * np.eye(n))
    K[np.diag_indices(n)] += JITTER
    try:
        c, low = cho_factor(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e12                      # invalid covariance; steer away
    alpha = cho_solve((c, low), y)
    logdet = 2.0 * np.log(np.diag(c)).sum()
    return float(0.5 * y @ alpha + 0.5 * logdet + 0.5 * n * np.log(2 * np.pi))


def fit_gp(xy, z, init_range=2000.0, init_nugget=0.3, maxiter=200):
    """Learn (range, nugget, sigma2) for a standardised field.

    Returns (range_m, nugget, sigma2, history), where history is the loss at
    each optimiser iteration -- the convergence curve.
    """
    D = np.hypot(xy[:, 0][:, None] - xy[:, 0], xy[:, 1][:, None] - xy[:, 1])
    theta0 = np.array([np.log(init_range),
                       np.log(init_nugget / (1 - init_nugget)),
                       0.0])                      # sigma2 = 1 on standardised z

    history = [neg_log_marginal(theta0, D, z)]
    res = minimize(neg_log_marginal, theta0, args=(D, z), method="L-BFGS-B",
                   callback=lambda th: history.append(
                       neg_log_marginal(th, D, z)),
                   options={"maxiter": maxiter})
    rng_, nug, s2 = _unpack(res.x)
    return rng_, nug, s2, np.array(history)