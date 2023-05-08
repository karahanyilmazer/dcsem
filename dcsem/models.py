#!/usr/bin/env python

# models.py - implementation of DCM and SEM class and their Layered version
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2023 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from scipy.integrate import solve_ivp



# Params class
class Parameters(dict):
    """Dict-like class where elements can be accessed
    either ysing p.x or p['x']
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

# Base class
class BaseModel(object):
    """Base class from which DCM/SEM/etc. inherit
    """
    def __init__(self):
        self.model = self.__class__.__name__
        self.p = Parameters()

    def __str__(self):
        """Print out model parameters
        """
        out = f"{self.model} parameters:\n"
        out += "--------------------------\n"
        for param in self.p:
            out += f" {param} = {self.p[param]}\n"
        out += "--------------------------\n"
        return out

    def simulate(self):
        raise(Exception('This is not implemented in the base class'))

    def fit(self, y):
        raise(Exception('This is not implemented in the base class'))

    def set_params(self, table):
        """Set model parameters
        :param table: dict
        """
        for param in table:
            self.p[param] = table[param]

    def get_params(self):
        return self.p

    def add_noise(self, signal, CNR):
        """
        :param signal: array
        :param CNR: float [CNR defined as std(signal)/std(noise) ]
        :return: signal+noise
        """
        std_signal  = np.std(signal)
        std_noise   = std_signal / CNR
        noise       = np.random.normal(loc=0, scale=std_noise, size=signal.shape)
        return signal + noise



class DCM(BaseModel):
    def __init__(self, num_rois, params=None):
        """Set default values for all parameters of Balloon model
        num_rois (int)
        params (dict) : use this to set up all or a subset of the parameters
        """
        super().__init__()
        # conn params
        self.num_rois = num_rois
        self.p.A = np.zeros((num_rois, num_rois))
        self.p.C = np.zeros(num_rois)
        # rCBF component params
        self.p = Parameters()
        self.p.kappa = 1.92
        self.p.gamma = 0.41
        # Balloon component params
        self.p.alpha = 0.32     # Stiffness param (outflow = volume^(1/alpha))
        self.p.E0 = 0.34        # Resting oxygen extraction fraction
        self.p.tau = 2.66       # Transit time (seconds) (could be V0/F0 where F0=resting flow)
        # BOLD weights
        self.p.k1 = 7.*self.p.E0
        self.p.k2 = 2.
        self.p.k3 = 2.*self.p.E0-0.2
        self.p.V0 = .02         # Resting blood volume fraction
        # Set user-specified parameters
        if params is not None:
            self.set_params(params)
        # State variables
        self.state_vars = ['s', 'f', 'v', 'q', 'x']

    def calc_BOLD(self, q, v):
        """Convert dHb (q) and blood volume (v) to BOLD signal change
        """
        return self.p.V0*(self.p.k1*(1-q)+self.p.k2*(1-q/v)+self.p.k3*(1-v))

    def init_states(self):
        zeros = np.full(self.num_rois, 0.)
        ones  =  np.full(self.num_rois, 1.)
        s0, x0 = zeros, zeros
        f0, v0, q0 = ones, ones, ones
        return np.r_[s0, f0, v0, q0, x0]

    def collect_results(self, ivp):
        BOLD_tc = []
        state_tc = {key:[] for key in self.state_vars}
        num_state = 5
        for idx in range(len(ivp.t)):
            p = ivp.y[:, idx]
            s, f, v, q, x = np.array_split(p, num_state)
            for key in self.state_vars:
                state_tc[key].append(eval(key))
            BOLD_tc.append(self.calc_BOLD(q, v))

        # Turn to numpy arrays and add bold timecourse
        state_tc = {key:np.asarray(state_tc[key]) for key in state_tc}
        state_tc['bold'] = np.asarray(BOLD_tc)
        return state_tc

    def get_func(self):
        # state vector:
        # p = [s,f,v,q,x]
        # dpdt = F(t,p)
        num_state = 5
        def F(t, p, u):
            s,f,v,q,x = np.array_split(p,num_state)
            dsdt = x-self.p.kappa*s-self.p.gamma*(f-1)
            dfdt = s
            dvdt = (1/self.p.tau)*(f-v**(1/self.p.alpha))
            dqdt = (1/self.p.tau)*(f*(1-(1-self.p.E0)**(1/f))/self.p.E0-v**(1/self.p.alpha-1)*q)
            dxdt = np.dot(self.p.A, x)+np.dot(self.p.C, u(t))
            return np.r_[dsdt, dfdt, dvdt, dqdt, dxdt]
        return F

    def simulate(self, tvec, u, CNR=None):
        """Generate BOLD+state time courses using ODE solver
        params:
        tvec (array)  - Times where states are evaluated
        u (function)  - Input function u(t) should be scalar for t scalar
        SNR (float)   - Signal to noise ratio (SNR=mean(abs(signal))/std(noise))
        returns:
        dict with all state time courses + BOLD
        """

        # get main function
        F = self.get_func()

        # intialise
        p0 = self.init_states()

        # run solver
        ivp = solve_ivp(fun=F,
                        t_span=[min(tvec),max(tvec)],
                        y0=p0,
                        args=(u,),
                        t_eval=tvec,
                        method='LSODA')

        # create results dict
        state_tc = self.collect_results(ivp)

        # Add noise to BOLD timecourse
        if CNR is not None:
            state_tc['bold'] = self.add_noise(state_tc['bold'], CNR)

        return state_tc


# TwoLayer-DCM sub-class
class TwoLayerDCM(DCM):
    """Layer DCM class - Can only do TWO layers
    Inherits from the DCM class.
    Must re-implement the following functions:
        - init_params()
        - get_func()
        - collect_results()
    """
    def __init__(self, num_rois, params=None):
        super().__init__(num_rois, params)
        # matrices A and C should be double the size
        self.num_layers = 2
        self.p.A = np.zeros((self.num_rois*self.num_layers,self.num_rois*self.num_layers))
        self.p.C = np.zeros(self.num_rois*2)
        # haemo parmas
        self.p.l_d   = 0.5  # coupling param
        self.p.tau_d = 1.   # delay
        self.p.tau_l = 2.66 # delay
        self.p.tau_d = 2.66 # delay
        # set user-defined params
        if params is not None:
            self.set_params(params)
        self.state_vars.extend(['vs', 'qs'])

    def init_states(self):
        zeros = np.full(self.num_rois*self.num_layers, 0.)
        ones  =  np.full(self.num_rois*self.num_layers, 1.)
        s0,x0,qvs0 = zeros, zeros, zeros
        f0,v0,q0 = ones, ones, ones

        return np.r_[s0, f0, v0, q0, x0, qvs0]

    def get_func(self):
        # state vector:
        # p = [s,f,v,q,x,vs,qs]
        # dpdt = F(t,p)
        # len(p) = n_rois * 5 * 2 + n_rois*2
        num_state = 6
        def F(t, p, u):
            s, f, v, q, x, vqs = np.array_split(p, num_state)
            vs, qs = np.array_split(vqs,self.num_layers)

            # combines lower and upper
            dsdt = x-self.p.kappa*s-self.p.gamma*(f-1)
            dfdt = s
            # drain effect here
            drain_v = np.r_[0*vs,self.p.l_d*vs]
            drain_q = np.r_[0*qs,self.p.l_d*qs]
            dvdt = (1/self.p.tau)*(f-v**(1/self.p.alpha)) + drain_v
            dqdt = (1/self.p.tau)*(f*(1-(1-self.p.E0)**(1/f))/self.p.E0-v**(1/self.p.alpha-1)*q) + drain_q
            # delay eqs
            vl, _ = np.array_split(v,self.num_layers)
            ql, _ = np.array_split(q,self.num_layers)
            dvsdt = 1/self.p.tau_d*(-vs+(vl-1))
            dqsdt = 1/self.p.tau_d*(-qs+(ql-1))
            # combines lower and upper
            dxdt = np.dot(self.p.A, x)+np.dot(self.p.C, u(t))
            return np.r_[dsdt, dfdt, dvdt, dqdt, dxdt, dvsdt, dqsdt]
        return F

    def collect_results(self,ivp):
        BOLD_tc = []
        state_tc = {key:[] for key in self.state_vars}
        num_state = 6
        for idx in range(len(ivp.t)):
            p = ivp.y[:,idx]
            s, f, v, q, x, vqs = np.array_split(p, num_state)
            vs,qs = np.array_split(vqs, self.num_layers)
            BOLD_tc.append(self.calc_BOLD(q, v))
            for key in self.state_vars:
                state_tc[key].append(eval(key))

        # Turn lists into numpy arrays and add BOLD
        state_tc = {key:np.asarray(state_tc[key]) for key in state_tc}
        state_tc['bold'] = np.asarray(BOLD_tc)
        return state_tc

    @staticmethod
    def bold2IR(bold, TI, T1l, T1u):
        TI = np.array(TI)
        return np.dot(bold, np.c_[np.abs(1-2*np.exp(-TI/T1l)), np.abs(1-2*np.exp(-TI/T1u))].T)




# Structural Equation Modelling
# SEM class
class SEM(BaseModel):
    def __init__(self, num_rois, params=None):
        super().__init__()
        self.num_rois = num_rois
        self.p.A = np.zeros((num_rois, num_rois))
        self.p.sigma = 1.
        if params is not None:
            self.set_params(params)

    def simulate(self, tvec):
        """Simulate time courses
        model is : x = Ax + u
        where u ~ N(0, sigma^2)

        model implies: x = (I-A)^(-1)(u)
                       cov(x) = (I-A)^{-1}cov(u)(I-A)^{-T}
        """
        u = np.random.normal(loc=0., scale=self.p.sigma, size=(self.num_rois, len(tvec)))
        I = np.identity(self.num_rois)
        A = self.p.A

        return np.dot(np.linalg.inv(I-A), u).T

    def get_cov(self):
        M = np.linalg.inv(np.identity(self.num_rois)-self.p.A)
        return self.p.sigma**2*np.dot(M,M.T)

    def negloglik(self, y):
        T, N = y.shape
        S = np.dot(y.T,y) / (T-1)
        C = self.get_cov()
        invC = np.linalg.inv(C)
        _, logdetC = np.linalg.slogdet(C)
        _, logdetS = np.linalg.slogdet(S)
        return T/2*(logdetC-logdetS+N*np.log(2*np.pi)+np.trace(S@invC))

    def fit(self, y, method='MH'):
        pass
