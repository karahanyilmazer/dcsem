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

    # try with strings
    paired_connections = ['R0,L1->R1,L0=1.',
                          'R1,L0->R0,L1=-100',
                          'R1,L0->R1,L1->10']
    A = utils.create_A_matrix(num_rois,num_layers,paired_connections,self_connections)
    assert  np.all(np.isclose(A,target_A))

    A = utils.create_A_matrix(num_rois=2)
    assert A.shape == (2,2)
    assert np.all(A == 0)

    A = utils.create_A_matrix(num_rois=2, num_layers=3)
    assert A.shape == (6,6)
    assert np.all(A == 0)

def test_create_C_matrix():
    target_C = np.array([.1,0.5,0.,.3])
    C = utils.create_C_matrix(2,2,[(0,0,.1),(0,1,0.),(1,0,.5),(1,1,.3)])
    assert  np.all(np.isclose(C,target_C))

    input_connections = ['R0,L0=.1', 'R0, L1 = 0.', 'R1, L0 = .5', 'R1,L1=.3']
    C = utils.create_C_matrix(2, 2, input_connections)
    assert  np.all(np.isclose(C,target_C))

    C = utils.create_C_matrix(num_rois=2)
    assert C.shape == (2,)
    assert np.all(C == 0)

    C = utils.create_C_matrix(num_rois=2, num_layers=3)
    assert C.shape == (6,)
    assert np.all(C == 0)

def test_stim_boxcar():
    tvec = np.linspace(0,50,300)
    u = utils.stim_boxcar([[0,5,1],[10,10,.5]])
    assert sum(u(tvec)) == 60.

def test_stim_random():
    tvec = np.linspace(0,50,300)
    u = utils.stim_random(tvec)
    assert np.isreal(np.sum(u(tvec)))

def test_stim_random_events():
    tvec = np.linspace(0,1,10000)
    u = utils.stim_random_events(tvec, p=0.5, n=3)
    np.isclose(np.mean(np.mean(u(tvec),axis=1)),0.5,atol=0.2)

# test models
def test_Parameters():
    p = models.Parameters({'x':2,'y':3})
    assert p.x == 2
    assert p.y == 3
    assert p['x'] == 2
    assert p['y'] == 3

def test_DCM():
    from dcsem.models import DCM
    dcm = models.DCM(2)
    tvec = np.linspace(0,50,300)
    u = utils.stim_boxcar([[0,5,1]])
    A = np.array([[ -2.0,   0.],
              [  0.2,  -2.0]])
    C = np.array([1,0])
    dcm.set_params({'A':A,'C':C})
    state_tc = dcm.simulate(tvec,u)
    assert np.all(np.isclose(sum(state_tc['bold']),[1.49680682, 0.18591665]))


def test_TwoLayerDCM():
    ldcm = models.TwoLayerDCM(num_rois=1)
    TR    = 1  # repetition time
    ntime = 100  # number of time points
    tvec  = np.linspace(0,ntime*TR,ntime)  # seconds

    stim = [[0,30,1]]
    u    = utils.stim_boxcar(stim)

    A = np.array([[-1.,.0],
                  [.0,-1.],
                 ])
    C = np.array([1,1])
    ldcm.set_params({'l_d':.5,'A':A,'C':C})

    state_tc = ldcm.simulate(tvec,u)
    assert np.all(np.isclose(sum(state_tc['bold']),[1.5932011,  2.71178688]))

def test_SEM():
    sem = models.SEM(num_rois=2)
    sem.set_params({'A':[[0,1],[.5,0]],'C':[10,0]})
    TR    = 1  # repetition time
    ntime = 100  # number of time points
    tvec  = np.linspace(0,ntime*TR,ntime)  # seconds

    sem1 = models.SEM(num_rois=2, params={'A':[[0,1],[.5,0]]})
    sem2 = models.SEM(num_rois=2, params={'A':[[0,50],[-50,0]]})
    assert np.linalg.norm(sem1.get_cov()-np.cov(sem1.simulate(tvec).T)) < 5
    assert np.linalg.norm(sem1.get_cov()-np.cov(sem2.simulate(tvec).T)) > 10

    num_rois = 3
    A = np.array([[0,   0, 0], [-5,   0, 0], [1,  -1, 0]])
    sem = models.SEM(num_rois=num_rois, params={'sigma':1,'A':A})
    tvec=np.linspace(0,1,300)
    y = sem.simulate(tvec)
    assert np.isclose(sem.negloglik(y),1270,atol=300)

    # fit
    num_rois = 3
    A = np.array([[0,   0, 0], [-5.,   0, 0], [1.,  -1., 0]])
    sem = models.SEM(num_rois=num_rois, params={'sigma':1,'A':A})
    tvec=np.linspace(0,1,300)
    y = sem.simulate(tvec)
    res = sem.fit(y, method='MH')
    assert np.isclose(res.A[1,0],sem.p.A[1,0],atol=1)
    assert np.isclose(res.sigma,sem.p.sigma,atol=1)

def test_MultiLayerSEM():
    lsem = models.MultiLayerSEM(1,2)
    assert min(lsem.T1s) == utils.constants['LowerLayerT1']
    assert max(lsem.T1s) == utils.constants['UpperLayerT1']

    lsem = models.MultiLayerSEM(2,2)
    TIs  = [400, 600, 800, 1000]
    tvec = np.linspace(0,1,100)
    y    = lsem.simulate_IR(tvec, TIs)
    res = lsem.fit_IR(tvec, y, TIs, method='MH')
    assert np.all(np.isclose(res.x, [1.,.5,1.], atol=2))
