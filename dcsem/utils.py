#!/usr/bin/env python

# utils.py - useful functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT
import os.path

import numpy as np

constants = {
    'LowerLayerT1' : 800,  # ms
    'UpperLayerT1' : 1200, # ms
}

def create_A_matrix(num_rois, num_layers=1, paired_connections=None, self_connections=None):
    """Make a connectivity matrix
    params:
    -------
    num_rois (int) - number of regions
    num_layers (int) - layers per region
    paired_connections (list or tuple) - e.g. ( [(in_roi,layer),(out_roi,layer),val], ... )
         can also be specified as list of strings: ['Ri, Lj -> Rn, Lm = value', ...]
    self_connections (list or float) - determines the diagonal elements of the matrix

    Returns:
    --------
    2D array (the connectivity matrix A)

    The output A is organised by layers then by ROIs

    """
    A = np.zeros([num_layers*num_rois]*2)
    # Fill diagonal
    if self_connections is not None:
        np.fill_diagonal(A, self_connections)
    # Fill the rest
    if paired_connections is not None:
        for c in paired_connections:
            if type(c) == str:
                import re
                c = [x.replace('R','').replace('L','') for x in re.split(',|->|=',c.replace(' ',''))]
                c = [(int(c[0]), int(c[1])), (int(c[2]), int(c[3])), float(c[-1])]
            (j, lj), (i, li), v = c
            A[i+li*num_rois, j+lj*num_rois] = v

    return A

def create_C_matrix(num_rois, num_layers=1, input_connections=None):
    """Make input connections matrix C
    params:
    -------
    num_rois (int) - number of regions
    num_layers (int) - layers per region
    input_connections (list or tuple) - e.g. ( [(roi,layer,value), ... )
            can also be specified as list of strings: ['Ri, Lj = value', ...]

    Returns:
    --------
    1D array (the connectivity matrix C)
    """
    C = np.zeros(num_layers*num_rois)
    if input_connections is not None:
        for c in input_connections:
            if type(c) == str:
                import re
                c = [x.replace('R','').replace('L','') for x in re.split(',|=',c.replace(' ',''))]
                c = [int(c[0]), int(c[1]), float(c[2])]
            r, l, v = c
            C[r+l*num_rois] = v
    return C

def stim_boxcar(stim):
    """Create boxcar stimulus
    :param stim: three-column array or text file with onset, duration, amplitude (all in seconds)
    :return: function
    """
    # Boxcar input
    import os
    if type(stim) == str:
        if os.path.exists(stim):
            stim = np.loadtxt(stim)
    stim = np.asarray(stim)
    if stim.shape[1] == 3:
        stim = stim.T
    onsets, durations, amplitudes = stim
    @np.vectorize
    def u(t, onsets=onsets, durations=durations, amplitudes=amplitudes):
        for o,d,a in zip(onsets,durations,amplitudes):
            if o<=t<=o+d:
                return a
        return 0.
    return u

def stim_random(tvec, n=1):
    stim = np.random.triangular(-.5,0.,.5,(n,len(tvec)))
    from scipy.interpolate import interp1d
    f = interp1d(tvec, stim)
    def u(t):
        return f(t)
    return u

def stim_random_events(tvec, p=0.5, n=1):
    stim = np.random.uniform(0.,1,(n,len(tvec))) > (1-p)
    from scipy.interpolate import interp1d
    f = interp1d(tvec, stim)
    def u(t):
        return f(t)
    return u


# MCMC fitting class
class MH(object):
    def __init__(self, loglik, logpr, burnin=1000, sampleevery=10, njumps=5000, update=20):
        """Metropolis Hastings class
        Params
        ------
        loglik: function
            Maps parameters to minus log-likelihood
        logpr:  function
            Maps parameters to minus log-prior
        burnin: int
            Number of iterations before actual sampling starts
        sampleevery: int
            Sampling rate
        njumps: int
            Number of sampling iterations
        update: int
            Rate of update of proposal distribution
        """

        self.burnin = burnin
        self.sampleevery = sampleevery
        self.njumps = njumps
        self.update = update
        self.loglik = loglik
        self.logpr = logpr

    def bounds_from_list(self, n, bounds):
        """
        Get bounds from list to two lists
        Args:
            n: num params
            bounds: scipy-optimize-style bounds

        Returns:
        numpy 1D array (Lower bounds)
        numpy 1D array (Upper bounds)
        """
        LB = -np.inf * np.ones(n)
        UB = np.inf * np.ones(n)
        if bounds is None:
            return LB, UB
        if not isinstance(bounds, list):
            raise(Exception('bounds must either be a list or None'))
        for i, b in enumerate(bounds):
            LB[i] = b[0] if b[0] is not None else -np.inf
            UB[i] = b[1] if b[1] is not None else np.inf
        return LB, UB

    def fit(self, p0, mask=None, verbose=False, LB=None, UB=None):
        """
        Run Metropolis Hastings algorithm to fit data

        Parameters
        ----------

        p0 : array-like
            Initial values for the parameters to be fitted
        mask : array-like
            Mask for fixed parameters. Has the same size as p0, contains zero for fixed parameters
        verbose : boolean
        LB: array-like
            Lower bounds on parameters
        UB array-like
            Upper bounds on parameters

        Returns
        -------
        2D array : samples from the posterior distribution (nsamples X nparams)
        """
        # Convert to numpy array
        p0 = np.array(p0, dtype=float)

        if verbose:
            print("Initialisation")

        # Bounds
        LB = np.full(p0.size, -np.inf) if LB is None else LB
        UB = np.full(p0.size, np.inf) if UB is None else UB

        for idx in range(p0.size):
            if not LB[idx] <= p0[idx] <= UB[idx]:
                raise Exception("Initial values outside of range!!!")

        # Initialise p,e,acc,rej,prop
        p = np.array(p0, dtype=float)
        e = self.loglik(p) + self.logpr(p)
        acc = np.zeros(p.size)
        rej = np.zeros(p.size)
        prop = np.abs(p0) / 10  # np.ones(p.size)
        prop[prop == 0] = 1

        samples = np.zeros((self.njumps + self.burnin, p.size))

        # Mask
        if mask is None:
            mask = np.ones(p0.size)

        # Main loop
        maxiter = self.burnin + self.njumps
        if verbose:
            print("Begin MH sampling")
        from tqdm import tqdm
        for iter in tqdm(range(maxiter)):
            if verbose:
                print(".... Iter {}/{}".format(iter, maxiter))
            # Loop through params
            for idx in range(p.size):
                if mask[idx] != 0:
                    oldp = p[idx]
                    p[idx] = p[idx] + np.random.randn() * prop[idx]
                    if not LB[idx] <= p[idx] <= UB[idx]:
                        p[idx] = oldp
                        rej[idx] += 1
                    else:
                        olde = e
                        e = self.loglik(p) + self.logpr(p)
                        if np.exp(olde - e) > np.random.rand():
                            acc[idx] += 1
                        else:
                            p[idx] = oldp
                            rej[idx] += 1
                            e = olde
            # end loop over params
            samples[iter, :] = p
            if iter % self.update == 0:
                if verbose:
                    print(".... >>> Update Proposal ")
                prop *= np.sqrt((1 + acc) / (1 + rej))
                acc *= 0
                rej *= 0

        samples = samples[self.burnin::self.sampleevery]
        return samples

def plot_posterior(means, cov, labels=None, samples=None, actual=None):
    """
    helper function for plotting posterior distribution

    Parameters
    ----------
    means : array like
    cov   : matrix
    labels : list
    samples : 2D (samples x params)
              as ouput by MH
    actual : array like
             true parameter values if known

    Returns
    -------
    matplotlib figure

    """
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    n = means.size
    nbins = 50
    k = 1
    for i in range(n):
        for j in range(n):
            if i == j:
                x = np.linspace(means[i] - 5 * np.sqrt(cov[i, i]),
                                means[i] + 5 * np.sqrt(cov[i, i]), nbins)
                y = norm.pdf(x, means[i], np.sqrt(cov[i, i]))

                plt.subplot(n, n, k)
                plt.plot(x, y)
                if samples is not None:
                    plt.hist(samples[:, i], histtype='step', density=True)
                if labels is not None:
                    plt.title(labels[i])
                if actual is not None:
                    plt.axvline(x=actual[i], c='r')

            else:
                m = np.asarray([means[i], means[j]])
                v = np.asarray([[cov[i, i], cov[i, j]], [cov[j, i], cov[j, j]]])
                xi = np.linspace(means[i] - 5 * np.sqrt(cov[i, i]),
                                 means[i] + 5 * np.sqrt(cov[i, i]), nbins)
                xj = np.linspace(means[j] - 5 * np.sqrt(cov[j, j]),
                                 means[j] + 5 * np.sqrt(cov[j, j]), nbins)
                x = np.asarray([(a, b) for a in xi for b in xj])
                x = x - m
                h = np.sum(-.5 * (x * (x @ np.linalg.inv(v).T)), axis=1)

                h = np.exp(h - h.max())
                h = np.reshape(h, (nbins, nbins))
                plt.subplot(n, n, k)

                plt.contour(xi, xj, h)

                if samples is not None:
                    plt.plot(samples[:, i], samples[:, j], 'k.', alpha=.1)
            k = k + 1

    return fig

