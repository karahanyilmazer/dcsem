#!/usr/bin/env python

# tests.py - test functions
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT

from dcsem import utils, models
import numpy as np
from pathlib import Path
testsPath = Path(__file__).parent

# test utils
def test_parse_matrix_file():
    target_A = np.array([[  -3.,    0.,    0.,    0.],[   0.,   -3.,    1.,    0.],[   0., -100.,   -3.,    0.],[   0.,   10.,    0.,   -3.]])
    A = utils.parse_matrix_file(str(testsPath / 'test_data' / 'Amat.txt'))
    assert np.all(np.isclose(A, target_A))
    A = utils.parse_matrix_file(str(testsPath / 'test_data' / 'Amat_text.txt'),2,2)
    assert np.all(np.isclose(A, target_A))

    A = utils.parse_matrix_file(str(testsPath / 'test_data' / 'Amat_text.txt'),2,2,-3)
    assert np.all(np.isclose(A, target_A))

    target_C = np.array([.1,0.5,0.,.3])
    C = utils.parse_matrix_file(str(testsPath / 'test_data' / 'Cmat.txt'))
    assert np.all(np.isclose(C, target_C))
    C = utils.parse_matrix_file(str(testsPath / 'test_data' / 'Cmat_text.txt'),2,2)
    assert np.all(np.isclose(C, target_C))


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

def test_create_DvE_matrix():
    conn = [
    'R0,L2->R1,L1=1.',
    'R1,L0->R0,L0=1.',
    'R1,L0->R0,L2=1.'
    ]
    num_rois = 2
    num_layers = 3

    A1 = utils.create_DvE_matrix(num_rois,num_layers,self_connections=-1)
    A2 = utils.create_A_matrix(num_rois,num_layers,conn, self_connections=-1)
    assert np.all(np.isclose(A1, A2))


def test_stim_boxcar():
    tvec = np.linspace(0,50,300)
    u = utils.stim_boxcar([[0,5,1],[10,10,.5]])
    assert sum(u(tvec)) == 60.

    stim_file = str(testsPath / 'test_data' / '3col.txt')
    u = utils.stim_boxcar(stim_file)
    assert sum(u(tvec)) == 60.
    #stim_file = str(testsPath / 'test_data' / '3col_1stim.txt')
    #u = utils.stim_boxcar(stim_file)


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
    bold, state_tc = dcm.simulate(tvec,u)
    assert np.all(np.isclose(sum(bold),[1.49680682, 0.18591665]))


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

    bold, state_tc = ldcm.simulate(tvec,u)
    assert np.all(np.isclose(sum(bold),[1.5932011,  2.71178688]))


def test_MultiLayerDCM():

    tvec = np.linspace(0,100,300)
    u = utils.stim_boxcar([[0,2,1]])


    conn = ['R0,L2->R1,L2=3.',
            'R1,L0->R0,L0=.2',
            'R1,L0->R0,L2=.2']

    A = utils.create_A_matrix(2,3,conn,-1)
    C = utils.create_C_matrix(2,3,['R0,L0=1','R0,L1=1','R0,L2=1'])

    ldcm = models.MultiLayerDCM(2,3,params={'A':A,'C':C,'l_d':1,'tau_d':1.})

    bold, state_tc = ldcm.simulate(tvec, u)
    assert np.isclose(np.mean(bold),0.005505226607304188)

    TIs = [300, 600]
    ir_bold = ldcm.simulate_IR(tvec, TIs, u)
    assert np.isclose(np.mean(ir_bold[1]), 0.0017710985449277605)

    # Is MultiLayer(2) like TwoLayer?
    A = utils.create_DvE_matrix(3,2,connections=1,self_connections=-2)
    C = utils.create_C_matrix(3,2,['R0,L0=1','R0,L1=1'])

    ldcm1 = models.MultiLayerDCM(3,2,params={'A':A,'C':C,'l_d':0})
    ldcm2 = models.TwoLayerDCM(3,params={'A':A,'C':C,'l_d':0})

    bold1, _ = ldcm1.simulate(tvec, u)
    bold2, _ = ldcm2.simulate(tvec, u)

    assert np.isclose( np.mean(bold1), np.mean(bold2))


def test_SEM():
    sem = models.SEM(num_rois=2)
    sem.set_params({'A':[[0,1],[.5,0]],'C':[10,0]})
    TR    = 1  # repetition time
    ntime = 100  # number of time points
    tvec  = np.linspace(0,ntime*TR,ntime)  # seconds

    sem1 = models.SEM(num_rois=2, params={'A':[[0,1],[.5,0]]})
    sem2 = models.SEM(num_rois=2, params={'A':[[0,50],[-50,0]]})
    assert np.linalg.norm(sem1.get_cov()-np.cov(sem1.simulate(tvec)[0].T)) < 5
    assert np.linalg.norm(sem1.get_cov()-np.cov(sem2.simulate(tvec)[0].T)) > 10

    num_rois = 3
    A = np.array([[0,   0, 0], [-5,   0, 0], [1,  -1, 0]])
    sem = models.SEM(num_rois=num_rois, params={'sigma':1,'A':A})
    tvec=np.linspace(0,1,300)
    y, _ = sem.simulate(tvec)
    assert np.isclose(sem.negloglik(y),1270,atol=300)

    # fit
    num_rois = 3
    A = np.array([[0,   0, 0], [-5.,   0, 0], [1.,  -1., 0]])
    sem = models.SEM(num_rois=num_rois, params={'sigma':1,'A':A})
    tvec=np.linspace(0,1,300)
    y, _ = sem.simulate(tvec)
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
    y = lsem.simulate_IR(tvec, TIs)
    res = lsem.fit_IR(tvec, y, TIs, method='MH')
    assert np.all(np.isclose(res.x, [1.,.5,1.], atol=2))

def test_state_tc_to_dict():
    # test for dcm
    dcm = models.DCM(num_rois=3)
    tvec = np.linspace(0,50,300)
    u    = utils.stim_boxcar([[0,30,1]])
    _, state_tc = dcm.simulate(tvec, u)
    state_tc = dcm.state_tc_to_dict(state_tc)
    assert len(state_tc['q']) == 3
    assert len(state_tc['q']['R0']) == 300

    ldcm = models.TwoLayerDCM(num_rois=3)
    _, state_tc = ldcm.simulate(tvec, u)
    state_tc = ldcm.state_tc_to_dict(state_tc)
    assert len(state_tc['q']) == 6
    assert len(state_tc['q']['R0L1']) == 300

    # test for sem
    sem = models.SEM(num_rois=3)
    _, state_tc = sem.simulate(tvec)
    state_tc = sem.state_tc_to_dict(state_tc)
    assert len(state_tc['x']) == 3
    assert len(state_tc['x']['R0']) == 300

    lsem = models.MultiLayerSEM(num_rois=3, num_layers=4)
    _, state_tc = lsem.simulate(tvec)
    state_tc = lsem.state_tc_to_dict(state_tc)
    assert len(state_tc['x']) == 12
    assert len(state_tc['x']['R0L3']) == 300
