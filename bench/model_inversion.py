#!/usr/bin/env python3

import numpy as np
import scipy.stats as st
from scipy import optimize

from bench import change_model


def grad(f, p, bounds, dp=1e-6):
    dp = np.maximum(1e-10, abs(p * dp))
    g = []
    for i in range(len(p)):
        pi = np.zeros(len(p))
        pi[i] = dp[i]

        u = p + pi
        if u[i] > bounds[i][1]:
            u[i] = bounds[i][1]

        l = p - pi
        if l[i] < bounds[i][0]:
            l[i] = bounds[i][0]

        g.append((f(u) - f(l)) / (u[i] - l[i]))
    return np.stack(g, axis=0)


def hessian(f, p, bounds, dp=1e-6):
    g = lambda p_: grad(f, p_, bounds, dp)
    return grad(g, p, bounds, dp)


def map_fit(data, noise_cov, model, priors, x0=None):
    """
    Fits a model to a data using maximum a posteriori approach
    Important note: it requires each parameter to have a prior distribution chosen from
    scipy stats continuous rv class.
    :param data:
    :param model:
    :param priors:
    :param noise_cov:
    :param x0:
    :return:
    """
    if x0 is None:
        x0 = np.array([v.mean() for v in priors.values()])

    bounds = np.array([v.interval(1 - 1e-3) for v in priors.values()])

    def neg_log_posterior(params):
        params_dict = {k: v for k, v in zip(priors.keys(), params)}
        lp = np.sum([priors[k].logpdf(params_dict[k]) for k in params_dict.keys()])
        if np.isneginf(lp):
            return -np.inf
        expected = model(**params_dict)
        llh = change_model.log_mvnpdf(
            mean=np.squeeze(expected), cov=np.squeeze(noise_cov), x=np.squeeze(data)
        )
        return -(llh + lp).item()

    p = optimize.minimize(neg_log_posterior, x0=x0, bounds=bounds, method="Nelder-Mead")

    h = hessian(neg_log_posterior, p.x, bounds)
    std = 1 / np.sqrt(np.diag(h))
    return p.x, std


def infer_change(pe1, std_pe1, pe2, std_pe2, alpha=0.05):
    """
        infers the changed parameters given two parameter estimates

    :param pe1: parameter estimates of the baseline
    :param std_pe1: standard deviation of the parameter estimates
    :param pe2: parameter estimates of the second group
    :param std_pe2: standard deviation of the parameter estimates
    :param alpha:
    :return:
    """
    zvals = (pe2 - pe1) / (std_pe1 + std_pe2)
    pvals = st.norm.sf(abs(zvals)) * 2  # twosided

    infered_change = np.argmax(zvals, axis=1)[:, np.newaxis]
    amount = np.take_along_axis(pe1 - pe2, infered_change, axis=1)
    idx = np.take_along_axis(pvals > alpha, infered_change, axis=1)
    infered_change += 1

    infered_change[idx] = 0
    amount[idx] = 0

    return infered_change, amount
