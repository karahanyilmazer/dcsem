#!/usr/bin/env python

# tests.py - test functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT

from dcsem import utils, models
import numpy as np

# test utils
def test_create_A_matrix():
    target_A = np.array([[  -3.,    0.,    0.,    0.],[   0.,   -3.,    1.,    0.],[   0., -100.,   -3.,    0.],[   0.,   10.,    0.,   -3.]])
    num_rois = 2
    num_layers = 2
    self_connections = -3.
    paired_connections = ([(0,1),(1,0),1.],
                          [(1,0),(0,1),-100],
                          [(1,0),(1,1),10],
                         )

    A = utils.create_A_matrix(num_rois,num_layers,paired_connections,self_connections)
    assert  np.all(np.isclose(A,target_A))

def test_create_C_matrix():
    target_C = np.array([.1,0.5,0.,.3])
    C = utils.create_C_matrix(2,2,[(0,0,.1),(0,1,0.),(1,0,.5),(1,1,.3)])
    assert  np.all(np.isclose(C,target_C))

def test_stim_boxcar():
    tvec = np.linspace(0,50,300)
    u = utils.stim_boxcar([[0,5,1],[10,10,.5]])
    assert sum(u(tvec)) == 60.

def test_stim_random():
    tvec = np.linspace(0,50,300)
    u = utils.stim_random(tvec)
    assert np.isreal(sum(u(tvec)))

# test models
def test_Parameters():
    p = models.Parameters({'x':2,'y':3})
    assert p.x == 2
    assert p.y == 3
    assert p['x'] == 2
    assert p['y'] == 3

def test_DCM():
    from dcsem.models import DCM
    dcm = models.DCM()
    tvec = np.linspace(0,50,300)
    u = utils.stim_boxcar([[0,5,1]])
    A = np.array([[ -2.0,   0.],
              [  0.2,  -2.0]])
    C = np.array([1,0])
    state_tc = dcm.simulate(tvec,u,A,C,num_roi=2)
    assert np.all(np.isclose(sum(state_tc['bold']),[1.49680682, 0.18591665]))


def test_TwoLayerDCM():
    pass
