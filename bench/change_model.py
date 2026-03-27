#!/usr/bin/env python3

"""
This module contains classes and functions to train a change model and make inference on new data.
"""
import inspect
import itertools
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Sequence, Union

import dill
import numba
import numpy as np
import scipy
import tqdm
from joblib import Parallel, cpu_count, delayed
from scipy.optimize import curve_fit
from scipy.stats import norm

from bench import acquisition, dti, main, spherical_harmonics

BOUNDS = {'negative': (-np.inf, 0), 'positive': (0, np.inf), 'twosided': (-np.inf, np.inf)}
INTEGRAL_LIMITS = list(BOUNDS.keys())
log2pi = np.log(2 * np.pi)

def default_normaliser(baseline, change=None, noise_cov=None, names=[]):
    return baseline, change, noise_cov


@dataclass
class Trainer:
    """
        A class for training models of change
        Required arguments for init:
        a callable function that maps parameters to measurements. must be callable like this m=f(kwargs, params)
        param priors should define the prior distribution for all parameters.
        change_vecs: list of change vectors.
    """
    forward_model: Callable
    """ The forward model  must be function of the form f(args, **params) that returns a numpy array."""
    priors: Mapping[str, scipy.stats.rv_continuous]
    kwargs: Mapping[str, Any] = None
    change_vecs: List[Mapping] = None
    lims: List = None
    measurement_names: List[str] = None
    amount_priors: Union[float, Sequence] = 1  # model priors (different from param/change priors)
    training_done = False
    normaliser: Callable = default_normaliser

    def __post_init__(self):

        if self.kwargs is None:
            self.kwargs = dict()

        if self.change_vecs is None:
            self.change_vecs = [{p: 1} for p in self.param_names]

        if np.all([[k in self.param_names for k in p] for p in self.change_vecs]):
            self.change_vecs = self.change_vecs
        elif np.all([isinstance(p, str) for p in self.change_vecs]):
            self.change_vecs, self.lims = parse_change_vecs(self.change_vecs)
        else:
            raise ValueError(" Change vectors are not defined properly.")

        if self.lims is None:
            self.lims = [['twosided']] * len(self.change_vecs)

        self.vec_names = [dict_to_string(s) for s in self.change_vecs]
        self.n_vecs = len(self.change_vecs)

        for idx in range(self.n_vecs):
            for pname in self.change_vecs[idx].keys():
                if pname not in self.param_names:
                    raise KeyError(f"parameter {pname} is not defined in the forward model."
                                   f"The parameters are {self.param_names}")
            # normalize change vectors to have a unit L2 norm:
            # scale = np.linalg.norm(list(self.change_vecs[idx].values()))
            # self.change_vecs[idx] = {k: v / scale for k, v in self.change_vecs[idx].items()}

        if np.isscalar(self.amount_priors):
            self.amount_priors = [self.amount_priors] * self.n_vecs
        elif len(self.amount_priors) != self.n_vecs:
            raise ("priors must be either a scalar (same for all change models) "
                   "or a sequence with size of number of change models.")

        params_test = sample_params(self.priors)
        try:
            self.n_dim = self.forward_model(**self.kwargs, **params_test).size
            tril_idx = np.tril_indices(self.n_dim)
            self.diag_idx = np.argwhere(tril_idx[0] == tril_idx[1])

        except Exception as ex:
            print("The provided function does not work with the given parameters and args.")
            raise ex
        if self.n_dim == 1:
            raise ValueError('The forward model must produce at least two dimensional output.')

        if self.measurement_names is None:
            self.measurement_names = [f'meas{i}' for i in range(self.n_dim)]

        self.training_done = False

    @property
    def param_names(self, ):
        return [i for p in self.priors.keys()
                for i in ([p] if isinstance(p, str) else p)]

    def vec_to_dict(self, vec: np.ndarray):
        return {p: v for p, v in zip(self.param_names, vec) if v != 0}

    def dict_to_vec(self, dict_: Mapping):
        return np.array([dict_.get(p, 0) for p in self.param_names])

    def train(self, n_samples=1000, mu_poly_degree=2, sigma_poly_degree=1, alpha=.1, dv0=1e-6,
              verbose=False, parallel=True):
        """
        Train change models (estimates w_mu and w_l) using forward model simulation

        :param n_samples: number simulated samples to estimate forward model
        :param dv0: the amount perturbation in parameter to estimate derivatives
        :param mu_poly_degree: polynomial degree for regressing mu
        :param sigma_poly_degree: polynomial degree for regressing sigma
        :param alpha: ridge regression regularization weight
        :param parallel: train models in parallel (on thread per model of change)

        :param verbose: flag for printing steps.
        """
        print(f'Generating {n_samples} training samples...')
        y_1, y_2 = self.generate_train_samples(n_samples, dv0)
        dy = (y_2 - y_1) / dv0
        y, _, _ = self.normaliser(baseline=y_1, change=None, noise_cov=None,
                            names=self.measurement_names)

        kde = scipy.stats.gaussian_kde(y.T)
        mean_y = y.mean(axis=0, keepdims=True)

        yf_mu = fit_transform(y - mean_y, mu_poly_degree)
        yf_sigma = fit_transform(y - mean_y, sigma_poly_degree)
        n_mu_features = yf_mu.shape[-1]

        print(f'Training models of change for {self.vec_names}. /r'
              f'This may take up to a few hours depending on the number of samples and the number of change vectors.')

        def func(idx):
            mu_weights = scipy.optimize.minimize(neg_log_likelihood,
                                                 x0=np.zeros(self.n_dim * n_mu_features),
                                                 method=None,
                                                 args=(dy[idx], yf_mu, yf_sigma, None, self.diag_idx, 'mu', alpha))
            if verbose:
                print(f' mu optimization for {self.vec_names[idx]}: {mu_weights.message}\n '
                      f'function value:{mu_weights.fun}, {mu_weights.nit}')
            else:
                print(f'optimized regression weights for {self.vec_names[idx]} mean')

            x0 = np.zeros((self.n_dim * (self.n_dim + 1) // 2) * yf_sigma.shape[-1])
            sigma_weights = scipy.optimize.minimize(neg_log_likelihood, method='BFGS', x0=x0,
                                                    args=(
                                                        dy[idx], yf_mu, yf_sigma, mu_weights.x, self.diag_idx, 'sigma',
                                                        alpha))
            if verbose:
                print(f' sigma optimization for {self.vec_names[idx]}: {sigma_weights.message}\n '
                      f'function value:{sigma_weights.fun}, {sigma_weights.nit}')
            else:
                print(f'optimized regression weights for {self.vec_names[idx]} covariance')

            all_weights = scipy.optimize.minimize(neg_log_likelihood,
                                                  method='BFGS',
                                                  x0=np.concatenate([mu_weights.x, sigma_weights.x]),
                                                  args=(dy[idx], yf_mu, yf_sigma, None, self.diag_idx, 'both', alpha))
            if verbose:
                print(f' mu+sigma optimization for {self.vec_names[idx]}: {all_weights.message}\n '
                      f'function value:{all_weights.fun}, {all_weights.nit}')
            else:
                print(f'optimized regression weights for {self.vec_names[idx]} joint mean and covariance')

            w_mu = all_weights.x[:(n_mu_features * self.n_dim)].reshape(n_mu_features, self.n_dim)
            w_sigma = all_weights.x[n_mu_features * self.n_dim:].reshape \
                (yf_sigma.shape[-1], self.n_dim * (self.n_dim + 1) // 2)

            models = []
            for l in self.lims[idx]:
                models.append(
                    MLChangeVector(vec=self.change_vecs[idx],
                                   mu_weight=w_mu,
                                   sig_weight=w_sigma,
                                   mean_y=mean_y,
                                   mu_poly_degree=mu_poly_degree,
                                   sigma_poly_degree=sigma_poly_degree,
                                   prior=self.amount_priors[idx],
                                   lim=l,
                                   name=str(self.vec_names[idx]) + '_' + l)
                )
                if verbose:
                    print(f'Trained models for {models[-1].name}.')
            return models

        models = run_parallel(func, len(self.change_vecs), debug=not parallel, prefer='threads')
        if len(models) == 1 and isinstance(models[0], (list, tuple)):
            models = models[0]

        self.training_done = True
        return ChangeModel(models=models, measurement_names=self.measurement_names,
                           baseline_kde=kde, forward_model=self.forward_model.__name__,
                           normaliser=self.normaliser)

    def generate_train_samples(self, n_samples: int, dv0: float = 1e-6, base_params=None) -> tuple:
        """
        generate samples to estimate derivatives.
        :param n_samples: number of samples
        :param dv0: the amount of change
        :return: M(V),M(V+ Delta V).
        """
        if base_params is None:
            base_params = sample_params(self.priors, n_samples=n_samples)

        for vec in self.change_vecs:
            for (k, v) in base_params.items():
                params_2 = v + vec.get(k, 0) * dv0
                if k in self.priors and hasattr(self.priors[k], 'pdf'):
                    invalid = self.priors[k].pdf(params_2) == 0
                    base_params[k][invalid] -= vec.get(k, 0) * dv0

        y1 = self.forward_model(**self.kwargs, **base_params)

        if y1.shape[0] != n_samples:
            y1 = y1.T

        y2 = []
        for vec in self.change_vecs:
            params_2 = {k: v + vec.get(k, 0) * dv0 for k, v in base_params.items()}
            tmp = self.forward_model(**self.kwargs, **params_2)
            if tmp.shape[0] != n_samples:
                tmp = tmp.T

            y2.append(tmp)
        y2 = np.stack(y2, 0)

        nans = np.isnan(y2).any(axis=(0, 2)) | np.isnan(y1).any(axis=1)
        y1 = y1[~nans]
        y2 = y2[:, ~nans, :]
        if np.sum(nans) > 0:
            warnings.warn(f'{np.sum(nans)} nan samples were generated during training.')

        return y1, y2

    def generate_test_samples(self, n_samples=1000, effect_size=0.1, noise_std=0.0, n_repeats=1,
                              base_params=None, true_change=None, parallel=False):
        """
        Generate signal and covariance matrix for baseline parameters and parameters shifted across change vector

        Important note: for this feature the forward model needs to have an internal noise model that
        accepts the parameter 'noise_std'. Otherwise, gaussian white noise is added to the measurements.

        :param true_change: index of true parameter change for each sample (0 for no change), array or list
        :param base_params: parameter values for the baseline (dict)
        :param n_samples: number of samples
        :param effect_size: the amount of change
        :param noise_std: standard deviation of measurement noise
        :param n_repeats: number of repeats to estimate noise covariance.
        :param parallel: use joblib to make the process parallel.
        :return: for N samples and M observational parameters returns tuple with

            - (N, ) index array with which change vector was applied
            - (N, M) with noisy baseline data
            - (N, M) with noisy data with shifted parameters
            - (N, M, M) or (1, M, M) array with noise matrix for each sample
        """

        models = []
        for vec, name, prior, lims in zip(self.change_vecs, self.vec_names, self.amount_priors, self.lims):
            for l in lims:
                models.append(MLChangeVector(vec=vec, prior=prior, lim=l, mu_weight=None, sig_weight=None,
                                             mean_y=None, mu_poly_degree=None, sigma_poly_degree=None,
                                             name=str(name) + ', ' + l))

        if true_change is None:
            # draw random samples for true change from the models.
            true_change = np.random.randint(0, len(models) + 1, n_samples)
        else:
            n_samples = len(true_change)

        if np.isscalar(effect_size):
            effect_size = np.ones(n_samples) * effect_size
        else:
            assert n_samples == len(effect_size)

        kwargs = self.kwargs.copy()
        has_noise_model = 'noise_std' in inspect.getfullargspec(self.forward_model).args

        if has_noise_model:
            kwargs['noise_std'] = noise_std

        print(f'Generating {n_samples} test samples:')

        def generator_func(s_idx):
            valid = False
            while not valid:
                if base_params is None:
                    params_1 = sample_params(self.priors)
                else:
                    params_1 = base_params

                tc = true_change[s_idx]
                if tc == 0:
                    params_2 = params_1.copy()
                    valid = True
                else:
                    lim = models[tc - 1].lim
                    sign = {'positive': 1, 'negative': -1, 'twosided': 1}[lim]
                    params_2 = {k: v + models[tc - 1].vec.get(k, 0) * effect_size[s_idx] * sign
                                for k, v in params_1.items()}

                    valid = np.all([self.priors[k].pdf(params_2[k]) > 0
                                    if k in self.priors and hasattr(self.priors[k], 'pdf')
                                    else 0 <= params_2[k] <= 1 for k in params_2.keys()])

                    if not valid and base_params is not None:
                        raise ValueError("Cant simulate a change with the baseline "
                                         "parameters and the specified type/amount of change.")

            if has_noise_model:
                y_1_r = np.zeros((n_repeats, self.n_dim))
                y_2_r = np.zeros_like(y_1_r)
                for r in range(n_repeats):
                    y_1_r[r] = self.forward_model(**kwargs, **params_1)
                    y_2_r[r] = self.forward_model(**kwargs, **params_2)
                y_1 = y_1_r[0]  # a random sample.
                y_2 = y_2_r[0]
                sigma_n = np.cov((y_2_r - y_1_r).T)
            else:
                y_1 = self.forward_model(**kwargs, **params_1) + np.random.randn(self.n_dim) * noise_std
                y_2 = self.forward_model(**kwargs, **params_2) + np.random.randn(self.n_dim) * noise_std
                sigma_n = (noise_std ** 2) * np.eye(y_1.shape[-1])

            return y_1, y_2, sigma_n

        y_1, y_2, sigma_n = run_parallel(generator_func, n_samples, debug=not parallel)

        return true_change, np.squeeze(y_1), np.squeeze(y_2), np.squeeze(sigma_n)


@dataclass
class MLChangeVector:
    """
    class for a single model of change trained by maximum likelihood approach
    """
    vec: Mapping
    mu_weight: np.ndarray
    sig_weight: np.ndarray
    mean_y: np.ndarray
    mu_poly_degree: int
    sigma_poly_degree: int
    lim: str
    prior: float = 1
    scale: float = 0.1
    name: str = None

    def __post_init__(self):
        scale = np.round(np.linalg.norm(list(self.vec.values())), 3)
        if scale != 1:
            warnings.warn(f'Change vector {self.name} does not have a unit length')

        if self.name is None:
            self.name = dict_to_string(self.vec)

        if self.mu_weight is not None:
            tril_idx = np.tril_indices(self.mu_weight.shape[-1])
            self.diag_idx = np.argwhere(tril_idx[0] == tril_idx[1])

    def distribution(self, y):
        """
        estimate mu and sigma from y using the trained regression models.
        :param y: normalized baseline measurements
        :return: mu and sigma
        """
        y = np.atleast_2d(y)
        yf_mu = fit_transform(y - self.mean_y, self.mu_poly_degree)
        yf_sigma = fit_transform(y - self.mean_y, self.sigma_poly_degree)
        mu, sigma_inv, _ = regression_model(yf_mu, yf_sigma, self.mu_weight, self.sig_weight, self.diag_idx)
        sigma = np.linalg.pinv(sigma_inv)

        return mu, sigma

    def log_lh(self, dv, y, dy, sigma_n, no_sigmap=False):
        """
        computes the log likelihood function for the amount of change
        P(dy | y, sigma_n, dv) for the vector of change
        :param dv: the amount of change in the parameters (scalar)
        :param y: the baseline measurement.
        :param dy: the amount of change in the measurements
        :param sigma_n: noise covaraince in the measurements
        :param no_sigmap: Dont use degenracy covariance for the estimation.

        :return: log of likelihood function (scalar)
        """
        mu, sigma_p = self.distribution(y)
        mean = np.squeeze(dv * mu)
        if no_sigmap:
            cov = sigma_n
        else:
            cov = (dv ** 2) * np.squeeze(sigma_p) + sigma_n

        return np.squeeze(log_mvnpdf(x=dy, mean=mean, cov=cov))

    def log_lh_precalc(self, y, dy, sigma_n):
        """
        Returns a function that calculates log likelihood without the requirement for inversion.
        :param y:
        :param dy:
        :param sigma_n:
        :return:
        """
        mu, sigma_p = self.distribution(y)
        mu, sigma_p = np.squeeze(mu), np.squeeze(sigma_p)

        sn_inv = np.linalg.inv(sigma_n)
        det_term = -0.5 * np.log(np.linalg.det(sn_inv)) + len(mu) / 2 * np.log(np.pi * 2)

        mat = sn_inv @ sigma_p
        evals, evecs = np.linalg.eigh(mat)

        mt, yt = evecs.T @ mu, evecs.T @ dy
        mtb, ytb = evecs.T @ sn_inv @ mu, evecs.T @ sn_inv @ dy

        def f_noinv(x):
            new_evals = 1 / (1 + x ** 2 * evals)  # should make robust to being zero or negative
            e_term = -0.5 * np.sum(yt * ytb * new_evals +
                                   (- mt * ytb - mtb * yt + mt * mtb * x) * new_evals * x)
            d_term = 0.5 * np.sum(np.log(new_evals)) + det_term

            return e_term - d_term

        return f_noinv

    def log_prior(self, dv):
        if self.lim == 'negative':
            dv = -dv
        elif self.lim == 'twosided':
            dv = np.abs(dv)

        # p = scipy.stats.lognorm(s=np.log(10), scale=self.scale).logpdf(x=dv)  # norm(scale=scale, loc=0).pdf(x=dv)
        # very slow!
        p = lognormal_logpdf(x=dv, shape=np.log(10), scale=self.scale)

        if self.lim == 'twosided':
            p -= np.log(2)

        return p

    def log_posterior(self, dv, y, dy, sigma_n):
        """
                Computes log posterior for the change vector
                :param dv: the amount of change (only scalar)
                :param y: normalized baseline measurement
                :param dy: the vector of change in the measuremnts
                :param sigma_n: noise covariance
                :return: P(dv|y, dy, sigma_n)
                """
        return self.log_prior(dv) + self.log_lh(dv, y, dy, sigma_n)


@dataclass
class NoChangeModel:
    """ no change class. """
    name: str = 'No change'
    prior: float = 1.0
    scale: float = 1.0

    def distribution(self, y):
        """
        returns mu and sigma for the null model given baseline measurements
        :param y:
        :return:
        """
        y = np.atleast_2d(y)
        return np.zeros((y.shape[0], y.shape[1] + 1)), np.zeros((y.shape[0], y.shape[1] + 1, y.shape[1] + 1))

    def log_posterior(self, dv, y, dy, sigma_n):
        """
        Computes log posterior for the null model
        :param dv: not used, to be consistent with the full model signature.
        :param y: not used, to be consistent with the full model signature.
        :param dy: the vector of change in the measuremnts
        :param sigma_n: noise covariance
        :return:
        """
        return log_mvnpdf(x=dy, mean=np.zeros_like(dy), cov=sigma_n)


@dataclass
class ChangeModel:
    """
    Class that contains trained models of change for all of the parameters of a given forward models.
    """
    models: List[MLChangeVector]
    measurement_names: List
    baseline_kde: Any  # scipy or sklearn kde class.
    forward_model: str = 'unnamed'
    normaliser: Callable = default_normaliser

    def __post_init__(self):
        null_model = NoChangeModel(prior=1,
                                   name='nochange')

        self.models.insert(0, null_model)

    @property
    def model_names(self, ):
        return [m.name.replace('_twosided', '') for m in self.models]

    def save(self, file_name=None, path=''):
        """
        Writes the change model to disk

        :param path: directory to write to
        :param file_name: filename (defaults to model name)
        """
        if file_name is None:
            file_name = self.forward_model

        with open(path + file_name, 'wb') as f:
            dill.dump(self, f)

    def estimate_change_vectors(self, data):
        """
            Returns the change vectors given the baseline measurement (mu, sigma)
        :param data: un-normalized baseline measurements.(n, d) n= number of samples, d= number of measurements
        :return: mu (n, m, d) , sigma_p(n, m, d,d) m=number of change vecs.
        """
        data = np.atleast_2d(data)
        n_samples, d = data.shape
        assert d == len(self.measurement_names)
        n_models = len(self.models)
        data_norm, _, _ = self.normaliser(baseline=data,
                                    change=None,
                                    noise_cov=None,
                                    names=self.measurement_names)

        mu = np.zeros((n_samples, n_models, d))
        sigma_p = np.zeros((n_samples, n_models, d, d))

        for i, y in enumerate(data_norm):
            for j, mdl in enumerate(self.models):
                mu[i, j], sigma_p[i, j] = mdl.distribution(y)
        return np.squeeze(mu), np.squeeze(sigma_p)

    def set_prior_scales(self, scale):
        """
         Changes the scale for the prior distribution on the amount of change
         (gamma or log normal distribution)
        :param scale: new scales, either a scalar or list with the size of change models.
        :return: None
        """
        assert np.isscalar(scale) or len(scale) == len(self.models)
        if np.isscalar(scale):
            scale = [scale] * len(self.models)
        for i in range(len(self.models)):
            self.models[i].scale = scale[i]

    def infer(self, data, delta_data, sigma_n, integral_bound=1, parallel=True):
        """
        Computes the posterior probabilities for each model of change
        :param data: numpy array (..., n_dim) containing the first group average.
        :param delta_data: numpy array (..., n_dim) containing the change between groups.
        :param sigma_n: numpy array or list (..., n_dim, n_dim)
        :param integral_bound: the bound to integrate over delta v.
        :param parallel: flag to run inference in parallel across samples. put false for debugging.
        :return:
            - (n_sample, n_changevecs) posterior probabilities for each voxel (..., n_vecs)
            - (n_samples, ) inferred change
            - (n_samples, n_changevecs) estimated amount of change per vec
            - index of samples with nan outputs.
        """

        print(f'running inference for {data.shape[0]} samples ...')
        y, dy, sn = self.normaliser(baseline=data, change=delta_data, noise_cov=sigma_n, names=self.measurement_names)
        lls, amounts = self.compute_log_likelihood(y, dy, sn, parallel=parallel, integral_bound=integral_bound)
        priors = np.array([m.prior for m in self.models])  # the 1 is for empty set
        priors = priors / priors.sum()
        model_log_posteriors = lls + np.log(priors)
        posteriors = np.exp(model_log_posteriors)
        posteriors[posteriors.sum(axis=1) == 0, 0] = 1  # in case all integrals were zeros accept the null model.
        bad_samples = np.argwhere(np.isnan(posteriors).any(axis=1))
        print('number of samples with nan posterior:', len(bad_samples))
        posteriors[np.isnan(posteriors)] = 0
        posteriors[np.isposinf(posteriors)] = np.nanmax(posteriors[np.isfinite(posteriors)]) + 1
        posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
        infered_change = np.argmax(posteriors, axis=1)
        return posteriors, infered_change, amounts, bad_samples

    # warnings.filterwarnings("ignore", category=RuntimeWarning)

    def compute_log_likelihood(self, y, delta_y, sigma_n, integral_bound=1, parallel=True,
                               gaussian_approx=True):
        """
        Computes log_likelihood function for all models of change.

        Compares the observed data with the result from :func:`predict`.

        :param parallel: flag to run inference in parallel
        :param y: (n_samples, n_dim) array of normalized baseline measurements
        :param delta_y: (n_samples, n_dim) array of delta data
        :param sigma_n: (n_samples, n_dim, n_dim) noise covariance per sample
        :param integral_bound
        :return: np array containing log likelihood for each sample per class
        """

        n_samples, n_dim = y.shape
        if sigma_n.ndim == 2:
            sigma_n = sigma_n[np.newaxis, :, :]

        n_models = len(self.models)

        def func(sam_idx):
            #np.seterr(over='ignore')
            log_prob = np.zeros(n_models)
            amount = np.zeros(n_models)

            y_s = y[sam_idx].T
            dy_s = delta_y[sam_idx]
            sigma_n_s = sigma_n[sam_idx]

            if np.isnan(y_s).any() or np.isnan(dy_s).any() or np.isnan(sigma_n_s).any():
                log_prob = np.ones(n_models) / n_models
                warnings.warn(f"Received nan inputs for inference at sample {sam_idx}.")

            else:
                if np.linalg.matrix_rank(sigma_n_s) < sigma_n_s.shape[0]:
                    sigma_n_s += 0.01 * np.diag(np.diag(sigma_n_s))  # regularises the noise covariance.

                log_prob[0] = self.models[0].log_posterior(0, y_s, dy_s, sigma_n_s)

                for vec_idx, ch_mdl in enumerate(self.models[1:], 1):

                    log_post_pdf = lambda dv: ch_mdl.log_posterior(dv, y_s, dy_s, sigma_n_s)
                    # lh = lambda dv: np.exp(ch_mdl.log_lh(dv, y_s, dy_s, sigma_n_s))

                    if ch_mdl.lim == 'positive':
                        neg_int = 0
                    else:  # either negative or two-sided:
                        neg_peak, upper, lower = [-np.exp(x) for x in
                                                  find_range(lambda dv: log_post_pdf(-np.exp(dv)),
                                                             (np.log(1e-6), np.log(integral_bound)))]
                        log_pdf_peak = log_post_pdf(neg_peak)

                        if log_pdf_peak < -200 or neg_peak < 1e-6:
                            neg_int, neg_expected = 0., 0.
                        else:
                            post_pdf = lambda dv: np.exp(log_post_pdf(dv) - log_pdf_peak)
                            neg_int = scipy.integrate.quad(post_pdf, lower, upper, points=[neg_peak],
                                                           epsrel=1e-3, epsabs=1e-6)[0] * np.exp(log_pdf_peak)
                            neg_expected = neg_peak

                    if ch_mdl.lim == 'negative':
                        pos_int = 0
                    else:  # either positive or two-sided
                        pos_peak, lower, upper = [np.exp(x) for x in find_range(lambda dv: log_post_pdf(np.exp(dv)),
                                                                                (np.log(1e-6),
                                                                                 np.log(integral_bound)))]
                        log_pdf_peak = log_post_pdf(pos_peak)

                        if log_pdf_peak < -200 or pos_peak < 1e-6:
                            pos_int, pos_expected = 0., 0.
                        else:
                            post_pdf = lambda dv: np.exp(log_post_pdf(dv) - log_pdf_peak)
                            pos_int = scipy.integrate.quad(post_pdf, lower, upper, points=[pos_peak],
                                                           epsrel=1e-3, epsabs=1e-6)[0] * np.exp(log_pdf_peak)

                            pos_expected = pos_peak

                    integral = pos_int + neg_int
                    if integral > 0:
                        log_prob[vec_idx] = np.log(integral)
                    else:
                        log_prob[vec_idx] = -np.inf

                    if ch_mdl.lim == 'positive':
                        amount[vec_idx] = pos_expected
                    elif ch_mdl.lim == 'negative':
                        amount[vec_idx] = neg_expected
                    else:
                        amount[vec_idx] = pos_expected if pos_int > neg_int else neg_expected

            return log_prob, amount

        def new_func(sam_idx):
            np.seterr(over='ignore')
            log_prob = np.zeros(n_models)
            amount = np.zeros(n_models)

            y_s = y[sam_idx].T
            dy_s = delta_y[sam_idx]
            sigma_n_s = sigma_n[sam_idx]

            if np.isnan(y_s).any() or np.isnan(dy_s).any() or np.isnan(sigma_n_s).any():
                log_prob = np.ones(n_models) / n_models
                warnings.warn(f"Received nan inputs for inference at sample {sam_idx}.")

            else:
                if np.linalg.matrix_rank(sigma_n_s) < sigma_n_s.shape[0]:
                    sigma_n_s += 0.01 * np.diag(np.diag(sigma_n_s))  # regularises the noise covariance.
                    # warnings.warn(f'noise covariance is singular for sample', sam_idx)
                    # log_prob[0] = -np.inf

                log_prob[0] = self.models[0].log_posterior(0, y_s, dy_s, sigma_n_s)

                for vec_idx, ch_mdl in enumerate(self.models[1:], 1):

                    log_post_pdf = lambda dv: ch_mdl.log_posterior(dv, y_s, dy_s, sigma_n_s)
                    log_post_pdf_flipped = lambda dv: ch_mdl.log_posterior(-dv, y_s, dy_s, sigma_n_s)

                    if ch_mdl.lim == 'positive':
                        integral, peak = calc_integral_fast(log_post_pdf, integral_bound)

                    elif ch_mdl.lim == 'negative':
                        integral, peak = calc_integral_fast(log_post_pdf_flipped, integral_bound)
                        peak = -peak
                    else:  # two-sided
                        pos_integral, pos_peak = calc_integral_fast(log_post_pdf, integral_bound)
                        neg_integral, neg_peak = calc_integral_fast(log_post_pdf_flipped, integral_bound)

                        integral = pos_integral + neg_integral
                        peak = pos_peak if pos_integral > neg_integral else -neg_peak

                    if integral > 0:
                        log_prob[vec_idx] = np.log(integral)
                        amount[vec_idx] = peak
                    else:
                        log_prob[vec_idx] = -np.inf
                        amount[vec_idx] = 0

            if np.isnan(log_prob).any():
                print(sam_idx, np.argwhere(np.isnan(log_prob)))
            return log_prob, amount

        if gaussian_approx:
            log_probs1, amounts1 = run_parallel(new_func, n_samples, debug=not parallel, prefer='threads')
        else:
            log_probs1, amounts1 = run_parallel(func, n_samples, debug=not parallel, prefer='threads')

        return log_probs1, amounts1

    def calc_confusion_matrix(self, data, sigma_n, effect_size, n_samples=1000):
        """
        given a baseline measurement, a noise covariance, and an effect size( the amount of change in each parameter)
        computes the expected confusion between change vectors.
        :param n_samples: number of samples
        :param data: (1, d) unnormalized baseline summary measurements
        :param sigma_n: (d, d) noise covariance matrix for the change.
        :param effect_size: (m,) or a scalar on the amount of change.
        :return: (m, m) confusion matrix.

        """
        mu, sigma_p = self.estimate_change_vectors(data)
        n_models = len(self.models)
        if np.isscalar(effect_size):
            effect_size = np.ones(n_models) * effect_size

        conf_mat = np.zeros((n_models, n_models))
        for m1 in range(n_models):
            y = np.random.multivariate_normal(mean=mu[m1] * effect_size[m1],
                                              cov=sigma_n + effect_size[m1] ** 2 * sigma_p[m1],
                                              size=n_samples)
            pdfs = np.zeros((n_models, n_samples))
            for m2 in range(n_models):
                pdfs[m2] = log_mvnpdf(x=y, mean=mu[m2] * effect_size[m2],
                                      cov=sigma_n + effect_size[m2] ** 2 * sigma_p[m2])
            winner = np.argmax(pdfs, axis=0)
            conf_mat[m1] = np.array([(winner == m).mean() for m in range(n_models)])
        return conf_mat

    def estimate_quality_of_fit(self, data, delta_data, sigma_n, predictions, amounts):
        dv = np.array([p[i] for i, p in zip(predictions, amounts)])
        y, dy, sn = self.normaliser(baseline=data, change=delta_data, noise_cov=sigma_n,
                                    names=self.measurement_names)

        dists = [self.models[p].distribution(y) for p, y in zip(predictions, y)]
        mu = np.stack([np.squeeze(d[0]) for d in dists], axis=0)
        sigma = np.stack([np.squeeze(d[1]) for d in dists], axis=0)

        offset = dy - mu * dv[:, np.newaxis]
        sigma_inv = np.linalg.pinv(sn + sigma * dv[:, np.newaxis, np.newaxis] ** 2)
        deviation = np.einsum('ij,ijk,ik->i', offset, sigma_inv, offset)

        return dv, offset, deviation

    @classmethod
    def load(cls, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                mdl = dill.load(f)
                return mdl
        else:
            raise SystemExit('model file not found')


def string_to_dict(str_):
    dict_ = {}
    str_ = str_.replace('+ ', '+')
    str_ = str_.replace('- ', '-')
    str_ = str_.replace('* ', '*')
    str_ = str_.replace(' *', '*')

    for term in str_.split(' '):
        if term[0] in ('+', '-'):
            sign = term[0]
            term = term[1:]
        else:
            sign = '+'

        if '*' in term:
            n = term.split('*')
            coef = n[0]
            pname = n[1]
        else:
            coef = '1'
            pname = term

        dict_[pname] = float(sign + coef)
    return dict_


def dict_to_string(dict_):
    str_ = ' '.join([f'{v:+1.1f}*{p}' for p, v in
                     dict_.items() if v != 0]).replace('1.0*', '')
    if str_[0] == '+':
        str_ = str_[1:]
    str_ = str_.replace(' +', ' + ')
    str_ = str_.replace(' -', ' - ')
    return str_


def parse_change_vecs(vec_texts: List[str]):
    """
    Parses change vectors from strings, the string format must be like this:
      p1: 1, two_sided
      p2: 1, positive

    :param vec_texts:
    :return:
    """
    vecs = list()
    lims = list()
    for txt in vec_texts:
        if '#' in txt:
            txt = txt[:txt.find('#')]

        sub = txt.split(',')
        this_vec = string_to_dict(sub[0])
        if len(sub) > 1:
            this_lims = sub[1].rstrip().lower().split()
            for l in this_lims:
                if l not in INTEGRAL_LIMITS:
                    raise ValueError(f'limits must be one of {INTEGRAL_LIMITS} but got {l}.')
        else:
            this_lims = ['twosided']

        vecs.append(this_vec)
        lims.append(this_lims)

    return vecs, lims


# helper functions:
def knn_estimation(y, dy, k=100, lam=1e-12):
    """
    Computes mean and covariance of change per sample using its K nearest neighbours.

    :param y: (n_samples, n_dim) array of summary measurements
    :param dy: (n_vecs, n_samples, n_dim) array of derivatives per vector of change
    :param k: number of neighbourhood samples
    :param lam: shrinkage value to avoid degenerate covariance matrices relative to minimum change
    :return: mu and l per data point
    """
    # make lambda relative tominimum:
    lam *= np.abs(dy[np.nonzero(dy)]).min()

    n_vecs, n_samples, dim = dy.shape
    mu = np.zeros_like(dy)
    tril = np.zeros((n_vecs, n_samples, dim * (dim + 1) // 2))

    idx = np.tril_indices(dim)
    diag_idx = np.argwhere(idx[0] == idx[1])

    y_whitened = scipy.cluster.vq.whiten(y)
    tree = scipy.spatial.KDTree(y_whitened)
    dists, neigbs = tree.query(y_whitened, k)
    weights = 1 / (dists + 1)
    print('KNN approximation of sample means and covariances:')
    for vec_idx in range(n_vecs):
        for sample_idx in range(n_samples):
            pop = dy[vec_idx, neigbs[sample_idx]]
            mu[vec_idx, sample_idx] = np.average(pop, axis=0, weights=weights[sample_idx])
            try:
                c = np.cov(pop.T, aweights=weights[sample_idx])
                tril[vec_idx, sample_idx] = np.linalg.cholesky(c + lam * np.eye(dim))[np.tril_indices(dim)]
            except Exception as ex:
                print(ex)
                raise ex

            for i in diag_idx:
                tril[vec_idx, sample_idx, i] = np.log(tril[vec_idx, sample_idx, i])

    return mu, tril


def l_to_sigma(l_vec, diag_idx):
    """
         inverse Cholesky decomposition and log transforms diagonals
           :param l_vec: (..., dim(dim+1)/2) vectors
           :param use_numba: use numba to compute the sigma (faster for large n)
           :return: sigma (... , dim, dim) det_sigma (..., dim)
       """

    t = l_vec.shape[-1]
    dim = int((np.sqrt(8 * t + 1) - 1) / 2)  # t = dim*(dim+1)/2

    sigma = np.zeros((l_vec.shape[0], dim, dim))
    _mat_lower_diagonal(l_vec, sigma)

    #  transpose last two dimensions:
    log_dets = 2 * np.squeeze(l_vec[..., diag_idx]).sum(axis=-1)
    return sigma, log_dets


@numba.jit(nopython=True)
def _mat_lower_diagonal(l_vec, l_sigma):
    """Multiplies a lower diagonal matrix with its transpose.

    Numba helper function used in l_to_sigma.
    """
    n, t = l_vec.shape
    n2, dim, dim2 = l_sigma.shape
    assert dim == dim2
    assert n == n2
    assert t == dim * (dim + 1) / 2
    a_i = np.zeros(n)
    a_j = np.zeros(n)
    isum = 0
    for i in range(dim):
        isum += i
        for k in range(i + 1):
            a_i[()] = l_vec[:, isum + k]
            if i == k:
                a_i[()] = np.exp(a_i)
            jsum = 0
            for j in range(dim):
                jsum += j
                if i <= j:
                    if i == j:
                        a_j[()] = a_i
                    else:
                        a_j[()] = l_vec[:, jsum + k]
                        if j == k:
                            a_j[()] = np.exp(a_j)
                    l_sigma[:, i, j] += a_i * a_j
        for j in range(i):
            l_sigma[:, i, j] = l_sigma[:, j, i]


def lognormal_logpdf(x, shape, scale):
    """
    log pdf of lognormal distribution
    :param x:
    :param shape:
    :param scale:
    :return:
    """
    if x == 0:
        return -np.inf
    else:
        return -np.log(shape * x) - ((np.log(x / scale) ** 2) / (2 * shape ** 2)) - log2pi / 2


def log_mvnpdf(x, mean, cov):
    """
    log of multivariate normal distribution. identical output to scipy.stats.multivariate_normal(mean, cov).logpdf(x) but faster.
    :param x: input numpy array (n, d)
    :param mean: mean of the distribution numpy array same size of x (n, d)
    :param cov: covariance of distribution, numpy array or scalar (n, d, d)
    :return scalar
    """
    cov = np.atleast_2d(cov)
    d = mean.shape[-1]
    offset = np.atleast_2d(x - mean)
    if cov.ndim == 2:
        cov = cov[np.newaxis, :, :]

    expo = -0.5 * np.einsum('ij,ijk,ik->i', offset, np.linalg.inv(cov), offset)
    # expo = -0.5 * (x - mean) @ np.linalg.inv(cov) @ (x - mean).T
    nc = -0.5 * (log2pi * d + np.linalg.slogdet(cov)[1])  # we expect the covariance be PD

    # var2 = np.log(multivariate_normal(mean=mean, cov=cov).pdf(x))
    return expo + nc


def find_range(f: Callable, bounds, scale=1e-3):
    """
     find the range for integration

    :param f: function in logarithmic scale, e.g. log_posterior
    :param bounds: range to look for the peak, lower and upper.
    :param scale: the ratio of limits to peak
    :param search_rad: radious to search for limits
    :return: peak, lower limit and higher limit.
    """
    neg_f = lambda dv: -f(dv)
    np.seterr(invalid='raise')
    # this warning caused by minimize scalar + numpy interaction.
    # filtered it because it clutters all other warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    peak = scipy.optimize.minimize_scalar(neg_f, bounds=bounds, method='bounded').x

    f_norm = lambda dv: f(dv) - (f(peak) + np.log(scale))
    lower, upper = bounds
    if f_norm(lower) < 0:  # to check the function is not flat.
        try:
            lower = scipy.optimize.root_scalar(f_norm, bracket=[lower, peak], method='brentq').root
        except Exception as e:
            print(f"Error of type {e.__class__} occurred while finding the lower limit of integral."
                  f"The lower bound is used instead")

    if f_norm(upper) < 0:
        try:
            upper = scipy.optimize.root_scalar(f_norm, bracket=[peak, upper], method='brentq').root
        except Exception as e:
            print(f"Error of type {e.__class__} occurred, while finding higher limit of integral."
                  f"The upper bound is used instead")

    return peak, lower, upper


def estimate_mean(pdf: Callable, bounds):
    """
    Estimates expected value of a probability distribution
    :param pdf: pdf
    :param bounds: the limits
    :return:
    """
    np.seterr(invalid='raise')
    xpx = lambda dv: pdf(dv) * dv
    expected = scipy.integrate.quad(xpx, bounds[0], bounds[1], epsrel=1e-3)[0]
    return expected


def estimate_mode(pdf: Callable, bounds):
    """
    Estimates mode of a probability distribution
    :param pdf: pdf or log_pdf
    :param bounds: the limits
    :return:
    """
    px = lambda dv: -pdf(dv)
    expected = scipy.optimize.minimize_scalar(px, bounds=bounds, method='bounded').x
    return expected


def estimate_median(f: Callable, bounds, n_samples=int(1e3)):
    """
    estimates the median of a probability distribution
    :param f: pdf
    :param bounds: limits
    :param n_samples: number of samples for integration.
    :return:
    """
    x = np.linspace(bounds[0], bounds[1], n_samples)
    p = np.array([f(t).item() for t in x])
    if np.all(p == 0):
        return np.mean(bounds)
    else:
        p_idx = np.argwhere(np.cumsum(p) > 0.5 * np.sum(p))[0]
        return x[p_idx]


def check_exp_underflow(x):
    """
    Checks if underflow happens for calculating exp(x)
    :param x: float
    :return: 1 if underflow error happens for x
    """
    with np.errstate(under='raise'):
        try:
            np.exp(x)
            return False
        except FloatingPointError:
            return True


def performance_measures(posteriors, true_change, set_names):
    """
    Computes accuracy and avg posterior for true changes
    """
    posteriors = posteriors / posteriors.sum(axis=1)[:, np.newaxis]
    predictions = np.argmax(posteriors, axis=1)

    indices = {str(s): i for i, s in enumerate(set_names)}
    true_change_idx = [indices[t] for t in true_change]

    accuracy = (predictions == true_change_idx).mean()
    true_posteriors = np.nanmean(posteriors[np.arange(len(true_change_idx)), true_change_idx])

    return accuracy, true_posteriors


def neg_log_likelihood(weights, dy, yf_mu, yf_sigma, w_mu,
                       diag_idx, opt_param='mu', alpha=0.1):
    """
     Computes the negative log-liklihood for a set of weights given y and dy
    :param weights: vectorized weights (d * n_features), if fixed sigma is False + d(d+1)/2 * n_features
    :param yf_mu: features extracted from summary measurements (n, n_features)
    :param dy: derivatives (n, d)
    :param alpha: regularization weight
    :return: -log(P(dy | yf, weights)) + alpha * |weights|^2
    """
    n_samples, n_dim = dy.shape

    if opt_param == 'mu':
        _, n_features = yf_mu.shape
        w_mu = weights.reshape(n_features, n_dim)
        mu, _, _ = regression_model(yf_mu, yf_sigma, w_mu, None, diag_idx)
        offset = dy - mu
        nll = np.linalg.norm(offset) ** 2 + alpha * np.mean(w_mu[1:, :] ** 2)

    elif opt_param == 'sigma':
        _, n_features = yf_sigma.shape
        w_sigma = weights.reshape(n_features, n_dim * (n_dim + 1) // 2)
        w_mu_ = w_mu.reshape(yf_mu.shape[-1], n_dim)
        mu, sigma_inv, ldet = regression_model(yf_mu, yf_sigma, w_mu_, w_sigma, diag_idx)
        offset = dy - mu
        nll = np.mean(-ldet + np.einsum('ij,ijk,ik->i', offset, sigma_inv, offset)) + \
              alpha * np.mean(w_sigma[1:, :] ** 2)

    elif opt_param == 'both':
        n_mu_features = yf_mu.shape[-1]
        n_sig_features = yf_sigma.shape[-1]
        w_mu = weights[:n_mu_features * n_dim].reshape(n_mu_features, n_dim)
        w_sigma = weights[n_mu_features * n_dim:].reshape(n_sig_features, n_dim * (n_dim + 1) // 2)
        mu, sigma_inv, ldet = regression_model(yf_mu, yf_sigma, w_mu, w_sigma, diag_idx)
        offset = dy - mu
        nll = np.mean(-ldet +
                      np.einsum('ij,ijk,ik->i', offset, sigma_inv, offset)) + \
              alpha * (np.mean(w_mu[1:, :] ** 2) + np.mean(w_sigma[1:, :] ** 2))

    return nll


def regression_model(yf_mu, yf_sigma, w_mu, w_sigma, diag_idx, lam=0):
    """
    Given some measurements and regression weights, computes the hyperparameters of derivatives (mean and covariance)
    :param yf_mu: feature vector (..., n_features)
    :param w_mu: weight matrix for mu (d , n_features)
    :param w_sigma: (d(d+1)/2 , n_features)
    :param lam: shrinkage weight
    :return:
    """
    n_dim = w_mu.shape[1]
    mu = yf_mu @ w_mu
    if w_sigma is None:
        sigma = np.eye(n_dim)
        log_det = 0
    else:
        sigma, log_det = l_to_sigma(yf_sigma @ w_sigma, diag_idx)
        sigma = sigma + lam * np.eye(n_dim)
    return mu, sigma, log_det


def sample_params(priors, n_samples=1):
    """Sample from the prior distributions

    Prior distributions are given by a dictionary from parameter names to a sampling function or scipy distribution.
    Independent parameters can map to a 1D prior distribution or we can map from a list of dependent parameters to a multi-dimensional distribution.
    """
    params = {}
    for p, dist in priors.items():
        func = getattr(dist, 'rvs', dist)
        if isinstance(p, str):
            params[p] = func(n_samples)
        else:
            for single_p, samples in zip(p, func(n_samples)):
                params[single_p] = samples
    return params


def estimate_observation_likelihood(forward_model, kwargs, param_priors, n_samples=1000):
    """
    :param forward_model:
    :param kwargs:
    :param param_priors:
    :param n_samples:
    :return:
    """
    all_params = sample_params(param_priors, n_samples=n_samples)
    y1 = forward_model(kwargs, **all_params)
    return scipy.stats.gaussian_kde(y1)


def run_parallel(func, n_iters, debug=False, print_progress=True, prefer=None):
    """
    runs a for loop in parallel.
    Args:
        func: callable function o = f(i)
        n_iters: number of iterations
        debug: set false to run normally (good for debugging)
        print_progress: prints the progress bar to output
        prefer: backend type. Any of {‘processes’, ‘threads’} or None
    """
    if debug is False:
        iters = range(n_iters)
        n_jobs = -1
        if prefer == 'processes':
            n_jobs = os.environ.get('{NSLOTS:-1}', default=cpu_count())
            n_jobs = np.minimum(n_jobs, n_iters)
            print(f'running jobs with {n_jobs} processes.')

        if print_progress is True:
            iters = tqdm.tqdm(iters, file=sys.stdout, total=n_iters, desc='Processing Jobs')

        res = Parallel(n_jobs=n_jobs, prefer=prefer)(delayed(func)(i) for i in iters)
    else:
        res = []
        for i in range(n_iters):
            res.append(func(i))

    if isinstance(res[0], (np.ndarray, int, float)):  # if output is a number return np.array
        return np.stack(res, axis=0)

    elif hasattr(res[0], '__len__'):  # if output is multiple numbers return arrays for each if possible
        out = []
        for i in range(len(res[0])):
            r = [r[i] for r in res]
            if isinstance(res[0][i], (np.ndarray, int, float)):
                r = np.stack(r, axis=0)
            out.append(r)
        return out
    else:
        return res


def summary_decorator(model, bvals, bvecs, summary_type='sh', sh_degree=2):
    """
    Decorates a diffusion model to add noise and directly calculate a summary measurement. The return function can
    be used like sm = f(microstructure parameters, noise_std= default to zero).

    :param bvecs: array of b-values
    :param bvals: array of b-vectors
    :param model: a diffusion model (function)
    :param summary_type: either 'sh' (spherical harmonics), 'dt' (diffusion tensor), 'sig' (raw signal)
    :param sh_degree: degree for spherical harmonics
    :return:
        function f that maps microstructural parameters to summary measurements, name of the summary measures
    """
    acq = acquisition.Acquisition.from_bval_bvec(bvals, bvecs)
    if summary_type == 'sh':
        def func(noise_std=0.0, **params):
            sig = model(bvals, bvecs, **params)
            noise = np.random.randn(*sig.shape) * noise_std
            sm = spherical_harmonics.fit_shm(sig + noise, acq, sh_degree=sh_degree)
            return sm

        names = spherical_harmonics.summary_names(acq, sh_degree=sh_degree)

    elif summary_type == 'dt':
        def func(noise_std=0.0, **params):
            sig = model(bvals, bvecs, **params)
            noise = np.random.randn(*sig.shape) * noise_std
            sm = dti.fit_dti(sig + noise, acq)
            return sm

        names = dti.summary_names(acq)

    elif summary_type == 'sig':
        def func(noise_std=0.0, **params):
            sig = model(bvals, bvecs, **params)
            noise = np.random.randn(*sig.shape) * noise_std
            return sig + noise

        names = [f'{b:1.3f}-{g[0]:1.3f},{g[1]:1.3f},{g[2]:1.3f}' for b, g in zip(bvals, bvecs)]
    else:
        raise ValueError(f'Summary measurement type {summary_type} is not defined. Must be sh, dt or sig.')
    func.__name__ = model.__name__
    return func, names


def fit_transform(X, degree=1):
    N, K = X.shape

    # Directly handle degree = 0
    if degree == 0:
        return np.ones((N, 1))

    # Directly handle degree = 1
    if degree == 1:
        return np.column_stack([np.ones(N), X])

    # Directly handle degree = 2
    if degree == 2:
        linear_terms = X
        quadratic_terms = np.array([X[:, i] * X[:, j] for i in range(K) for j in range(i, K)]).T
        return np.column_stack([np.ones(N), linear_terms, quadratic_terms])

    output = [np.ones(N), X]
    for d in range(2, degree + 1):
        comb = itertools.combinations_with_replacement(range(K), d)
        for indices in comb:
            output.append(np.prod(X[:, indices], axis=1))

    return np.column_stack(output)


def gaussian_auc(amp, mu, sigma, x_min, x_max):
    return amp * (norm.cdf(x_max, mu, sigma) - norm.cdf(x_min, mu, sigma))


def calc_integral_quad(func, max_x, eps_rel=1e-2):
    """
    Calculate integral of the posterior probability within a range robustly
    :param func: log posterior function
    :param max_x: maximum possible x
    :param eps_rel: ratio of the peak value to the limit of integral
    """

    neg_func_log = lambda x: -func(np.exp(x))
    log_peak = scipy.optimize.minimize_scalar(neg_func_log, bounds=[-10, np.log(max_x)], method='bounded').x
    peak = np.exp(log_peak)
    f_peak = func(peak)

    if peak < 1e-6 or f_peak < -200:
        return 0., 0.

    norm_func = lambda x: neg_func_log(log_peak) - np.log(eps_rel) - neg_func_log(x)

    if norm_func(np.log(max_x)) > 0:
        upper = max_x
    else:
        log_upper = scipy.optimize.root_scalar(norm_func, bracket=[log_peak, np.log(max_x)], method='brentq').root
        upper = np.exp(log_upper)

    exp_func = lambda x: np.exp(func(x) - f_peak)
    integral = \
        scipy.integrate.quad(exp_func, np.maximum(1e-6, peak - upper), upper, points=[peak], epsrel=1e-3, epsabs=1e-6)[
            0] * np.exp(f_peak)

    return integral, peak


def gaussian_auc(amp, mu, sigma, x_min, x_max):
    return amp * (norm.cdf(x_max, mu, sigma) - norm.cdf(x_min, mu, sigma))


def calc_integral_fast(func, max_x, eps_rel=1e-3):
    """
    Calculate integral of a probability function by approximating with a gaussian
    :param func: log probability function
    :param max_x: maximum possible x
    :param eps_rel: ratio of the peak value to the limit of integral
    """

    neg_func_log = lambda x: -func(np.exp(x))
    log_peak = scipy.optimize.minimize_scalar(neg_func_log, bounds=[-10, np.log(max_x)], method='bounded').x
    peak = np.exp(log_peak)
    f_peak = func(peak)
    if peak < 1e-6 or f_peak < -200:  # if the peak is too close to zero, or it's too short don't bother
        return 0., 0.

    norm_func = lambda x: neg_func_log(log_peak) - np.log(eps_rel) - neg_func_log(x)

    if norm_func(np.log(max_x)) > 0:
        upper = max_x
    else:
        log_upper = scipy.optimize.root_scalar(norm_func, bracket=[log_peak, np.log(max_x)], method='brentq').root
        upper = np.exp(log_upper)

    f_upper = func(upper)
    mu = peak

    if f_peak < f_upper:  # numerical issue
        return 0., 0.

    sigma = (upper - peak) / np.sqrt(2 * (f_peak - f_upper))
    amp = np.exp(f_peak) * sigma * np.sqrt(2 * np.pi)
    integral = amp * (norm.cdf(upper, mu, sigma) - norm.cdf(0, mu, sigma))

    return integral, peak


