#!/usr/bin/env python

# utils.py - useful functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT
import numpy as np

def create_A_matrix(num_rois, num_layers, paired_connections, self_connections):
    """Make a connectivity matrix
    params:
    -------
    num_rois (int) - number of regions
    num_layers (int) - layers per region
    paired_connections (list or tuple) - e.g. ( [(in_roi,layer),(out_roi,layer),val], ... )
    self_connections (list or float) - determines the diagonal elements of the matrix

    Returns:
    --------
    2D array (the connectivity matrix A)

    The output A is organised by layers then by ROIs

    """
    A = np.zeros([num_layers*num_rois]*2)
    # Fill diagonal
    np.fill_diagonal(A, self_connections)
    # Fill the rest
    for c in paired_connections:
        (j, lj), (i, li), v = c
        A[i+li*num_rois, j+lj*num_rois] = v

    return A

def create_C_matrix(num_rois, num_layers, input_connections):
    """Make input connections matrix C
    params:
    -------
    num_rois (int) - number of regions
    num_layers (int) - layers per region
    input_connections (list or tuple) - e.g. ( [(roi,layer,value), ... )

    Returns:
    --------
    1D array (the connectivity matrix C)
    """
    C = np.zeros(num_layers*num_rois)
    for c in input_connections:
        r, l, v = c
        C[r+l*num_rois] = v
    return C

def stim_boxcar(stim):
    """Create boxcar stimulus
    :param stim: three-column array with onset, duration, amplitude (all in seconds)
    :return: function
    """
    # Boxcar input
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

def stim_random(tvec):
    stim = np.random.triangular(-.5,0.,.5,len(tvec))
    from scipy.interpolate import interp1d
    f = interp1d(tvec, stim)
    @np.vectorize
    def u(t):
        return f(t)
    return u

