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
    either using p.x or p['x']
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
        self.num_rois = None
        self.num_layers = None
        self.state_vars = []
        self.Anz = None
        self.Cnz = None
        # Base class knows about T1s so it can generate IR-BOLD
        self.T1s = None

    def __str__(self):
        """Print out model parameters
        """
        out = f"{self.model} parameters:\n"
        out += "--------------------------\n"
        for param in self.p:
            out += f" {param} = {self.p[param]}\n"
        out += "--------------------------\n"
        return out

    def simulate(self, u=None):
        """Implementation of the simulator
        :return:
        tuple : (BOLD, State_tc)
        """
        raise(Exception('This is not implemented in the base class'))

    def fit(self, y):
        raise(Exception('This is not implemented in the base class'))

    def set_params(self, table):
        """Set model parameters
        :param table: dict
        """
        for param in table:
            if param == 'A':
                self.p.A = np.array(table[param], dtype=np.float64)
                self.Anz = np.nonzero(self.p.A)
            elif param == 'C':
                self.p.C = np.array(table[param], dtype=np.float64)
                self.Cnz = np.nonzero(self.p.C)
            else:
                self.p[param] = table[param]

    def get_params(self):
        return self.p

    # table -> p_names
    def get_p_names(self, param_table=None):
        """ Get list of parameter names (splitting matrix coeffs into own params)
        :param param_table: dict
        :return: list
        """
        if param_table is None:
            param_table = self.p
        names = []
        for param in param_table:
            if isinstance(param_table[param], (int, float)):
                names.append(param)
            else:
                if param == 'A':
                    names.extend([f'a{ij[0]}_{ij[1]}' for ij in np.array(self.Anz).T])
                elif param == 'C':
                    names.extend([f'c{i[0]}' for i in np.array(self.Cnz).T])
        return names

    def get_p(self, param_table=None):
        """ Get parameters as a list
        :param param_table: dict
        :return: list of param values
        """
        if param_table is None:
            param_table = self.p
        values = []
        for param in param_table:
            if isinstance(param_table[param], (int, float)):
                values.append(param_table[param])
            else:
                if param == 'A':
                    A = param_table['A']
                    values.extend([A[ij[0],ij[1]] for ij in np.array(self.Anz).T])
                elif param == 'C':
                    C = param_table['C']
                    values.extend([C[i[0]] for i in np.array(self.Cnz).T])
        return np.array(values)

    # p -> table
    def set_p(self, p):
        """Set model parameters from a list (rather than a table)
        :param p: list
        :return: None
        """
        idx = 0
        for param in self.p:
            if isinstance(self.p[param], (int, float)):
                self.p[param] = p[idx]
                idx += 1
            else:
                if param == 'A':
                    for ij in np.array(self.Anz).T:
                        self.p['A'][ij[0],ij[1]] = p[idx]
                        idx += 1
                elif param == 'C':
                    for i in np.array(self.Cnz).T:
                        self.p['C'][i[0]] = p[idx]
                        idx += 1

    # p -> table
    def p_to_table(self, p):
        """Set model parameters from a list (rather than a table)
        :param p: list
        :return: dict
        """
        import copy
        D = copy.deepcopy(self.p)
        #D = {key: value for key, value in self.p.items()}
        idx = 0
        for param in self.p:
            if isinstance(self.p[param], (int, float)):
                D[param] = float(p[idx])
                idx += 1
            else:
                if param == 'A':
                    for ij in np.array(self.Anz).T:
                        D['A'][ij[0],ij[1]] = float(p[idx])
                        idx += 1
                elif param == 'C':
                    for i in np.array(self.Cnz).T:
                        D['C'][i[0]] = float(p[idx])
                        idx += 1
        return D


    def state_tc_to_dict(self, state_tc):
        """Turn state_tc to dict of dict for saving
        :param state_tc: dict of array
        :return:
        dict of dict
        """
        if type(state_tc) == np.ndarray:
            state_tc = {self.state_vars[0] : state_tc}
        Results = {}
        for v in self.state_vars:
            D = {}
            if state_tc[v].shape[1] == self.num_rois:
                for roi in range(self.num_rois):
                    name = f'R{roi}'
                    D[name] = state_tc[v][:, roi]
            else:
                for layer in range(self.num_layers):
                    for roi in range(self.num_rois):
                        idx = roi+layer*self.num_rois
                        name = f'R{roi}L{layer}'
                        D[name] = state_tc[v][:, idx]
            Results[v] = D
        return Results

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

    def get_Pmat(self, TIs):
        """Make partial volume matrix
        T1s: T1 of each layer
        TIs: TI of each measurement
        """
        E = np.abs(1-2*np.outer(np.array(TIs),1/np.array(self.T1s)))
        E = E / np.sum(E, axis=1, keepdims=True)

        allP = []
        for i, ti in enumerate(TIs):
            P = []
            for j,t1 in enumerate(self.T1s):
                pv = np.identity(self.num_rois)*E[i, j]
                P.append(pv)
            allP.append(np.concatenate(P, axis=1))

        return allP

    def simulate_IR(self, tvec, TIs, u=None):
        """
        x = Ax+u
        for k:
            y[k] = P[k]*x
        """
        if self.T1s is None:
            raise(Exception('Must set the T1s'))
        if self.num_rois is None:
            raise(Exception('Must set num_rois'))

        P = self.get_Pmat(TIs)
        y = []
        for p in P:
            bold, _ = self.simulate(tvec, u=u)
            y.append(np.dot(p, bold.T).T)
        return y

class DCM(BaseModel):
    def __init__(self, num_rois, params=None):
        """Set default values for all parameters of Balloon model
        num_rois (int)
        params (dict) : use this to set up all or a subset of the parameters
        """
        super().__init__()
        # conn params
        self.num_rois = num_rois
        self.num_layers = 1
        self.p.A = np.zeros((num_rois, num_rois))
        self.p.C = np.zeros(num_rois)
        # rCBF component params
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
        # Keep track of non-zero A and C
        self.Anz = np.nonzero(self.p.A)
        self.Cnz = np.nonzero(self.p.C)
        # State variables
        self.state_vars = ['s', 'f', 'v', 'q', 'x']
        # Define default values for T1s
        from dcsem.utils import constants
        self.T1s = (constants['LowerLayerT1']+constants['UpperLayerT1'])/2.

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

        return np.asarray(BOLD_tc), state_tc

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
            dxdt = np.dot(self.p.A, x)+self.p.C * u(t) #np.dot(self.p.C, u(t))
            return np.r_[dsdt, dfdt, dvdt, dqdt, dxdt]
        return F

    def simulate(self, tvec, u=None, CNR=None, solver='LSODA'):
        """Generate BOLD+state time courses using ODE solver
        params:
        tvec (array)  - Times where states are evaluated
        u (function)  - Input function u(t) should be scalar for t scalar
        SNR (float)   - Signal to noise ratio (SNR=mean(abs(signal))/std(noise))
        solver (str) - method of integration (see scipy.integrate.slove_ivp 'method')
        returns:
        dict with all state time courses + BOLD
        """

        # get main function
        F = self.get_func()

        # if no input, set to zero
        if u is None:
            u = lambda x:0

        # intialise
        p0 = self.init_states()

        # run solver
        ivp = solve_ivp(fun=F,
                        t_span=[min(tvec),max(tvec)],
                        y0=p0,
                        args=(u,),
                        t_eval=tvec,
                        method=solver)

        # create results dict
        bold, state_tc = self.collect_results(ivp)

        # Add noise to BOLD timecourse
        if CNR is not None:
            bold = self.add_noise(bold, CNR)

        return bold, state_tc


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
        self.p.A = np.zeros((self.num_rois*self.num_layers, self.num_rois*self.num_layers))
        self.p.C = np.zeros(self.num_rois*self.num_layers)
        # haemo parmas
        self.p.l_d   = 0.5  # coupling param
        self.p.tau_d = 2.66   # delay
        # Define default values for T1s
        from dcsem.utils import constants
        self.T1s = np.linspace(constants['LowerLayerT1'],
                               constants['UpperLayerT1'],
                               self.num_layers)

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
            dxdt = np.dot(self.p.A, x)+ self.p.C * u(t) #np.dot(self.p.C, u(t))
            return np.r_[dsdt, dfdt, dvdt, dqdt, dxdt, dvsdt, dqsdt]
        return F

    def collect_results(self, ivp):
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
        return np.asarray(BOLD_tc), state_tc


class MultiLayerDCM(DCM):
    def __init__(self, num_rois, num_layers, params=None):
        super().__init__(num_rois, params)
        self.num_rois = num_rois
        self.num_layers = num_layers
        self.num_states = 5*num_rois*num_layers + 2*num_rois*(num_layers-1)
        self.p.A = np.zeros((self.num_rois*self.num_layers, self.num_rois*self.num_layers))
        self.p.C = np.zeros(self.num_rois*self.num_layers)
        # haemo parmas
        self.p.l_d   = 0.5  # coupling param
        self.p.tau_d = 2.66 # delay
        # Define default values for T1s
        from dcsem.utils import constants
        self.T1s = np.linspace(constants['LowerLayerT1'],
                               constants['UpperLayerT1'],
                               self.num_layers)

        # set user-defined params
        if params is not None:
            self.set_params(params)
        self.state_vars.extend(['vs', 'qs'])

    def init_states(self):
        zeros = np.full(self.num_rois*self.num_layers, 0.)
        ones  =  np.full(self.num_rois*self.num_layers, 1.)
        s0,x0 = zeros, zeros
        f0,v0,q0 = ones, ones, ones

        zeros_l  =  np.full(self.num_rois*(self.num_layers-1), 0.)
        vs0, qs0 = zeros_l, zeros_l

        return self.merge_p(s0, f0, v0, q0, x0, vs0, qs0)

    def split_p(self, p):
        # do stuff
        n = 5*self.num_rois*self.num_layers
        m = 2*self.num_rois*(self.num_layers-1)
        s, f, v, q, x = np.array_split(p[:n], 5)
        vs, qs = np.array_split(p[n:], 2)
        return s, f, v, q, x, vs, qs

    @staticmethod
    def merge_p(s, f, v, q, x, vs, qs):
        return np.r_[s, f, v, q, x, vs, qs]

    def get_func(self):
        # state vector:
        # p = [s,f,v,q,x,vs,qs]
        # dpdt = F(t,p)

        def F(t, p, u):
            s, f, v, q, x, vs, qs = self.split_p(p)

            # combines all layers
            dsdt = x-self.p.kappa*s-self.p.gamma*(f-1)
            dfdt = s
            # drain effect here
            drain_v = np.r_[np.zeros(self.num_rois),self.p.l_d*vs]
            drain_q = np.r_[np.zeros(self.num_rois),self.p.l_d*qs]
            dvdt = (1/self.p.tau)*(f-v**(1/self.p.alpha)) + drain_v
            dqdt = (1/self.p.tau)*(f*(1-(1-self.p.E0)**(1/f))/self.p.E0-v**(1/self.p.alpha-1)*q) + drain_q
            # delay eqs
            vl = v[:self.num_rois*(self.num_layers-1)]
            ql = q[:self.num_rois*(self.num_layers-1)]

            dvsdt = 1/self.p.tau_d*(-vs+(vl-1))
            dqsdt = 1/self.p.tau_d*(-qs+(ql-1))
            # combines all layers
            dxdt = np.dot(self.p.A, x)+ self.p.C * u(t)
            return self.merge_p(dsdt, dfdt, dvdt, dqdt, dxdt, dvsdt, dqsdt)
        return F

    def collect_results(self, ivp):
        BOLD_tc = []
        state_tc = {key:[] for key in self.state_vars}
        num_state = 6
        for idx in range(len(ivp.t)):
            p = ivp.y[:,idx]
            s, f, v, q, x, vs, qs = self.split_p(p)
            BOLD_tc.append(self.calc_BOLD(q, v))
            for key in self.state_vars:
                state_tc[key].append(eval(key))

        # Turn lists into numpy arrays and add BOLD
        state_tc = {key:np.asarray(state_tc[key]) for key in state_tc}
        return np.asarray(BOLD_tc), state_tc



# Structural Equation Modelling
# SEM class
class SEM(BaseModel):
    def __init__(self, num_rois, params=None):
        super().__init__()
        self.num_rois = num_rois
        self.num_layers = 1
        self.num_states = num_rois*self.num_layers
        self.p.A = np.zeros((self.num_states, self.num_states), dtype=np.float64)
        self.p.sigma = 1.
        if params is not None:
            self.set_params(params)
        # Keep track of non-zero A and C
        self.Anz = np.nonzero(self.p.A)
        self.state_vars = ['x']

    def simulate(self, tvec, u=None):
        """Simulate time courses
        (here u is ignored and instead random noise is used)
        model is : x = Ax + u
        where u ~ N(0, sigma^2)

        model implies: x = (I-A)^(-1)(u)
                       cov(x) = (I-A)^{-1}cov(u)(I-A)^{-T}
        """
        u = np.random.normal(loc=0., scale=self.p.sigma, size=(self.num_states, len(tvec)))
        I = np.identity(self.num_states)
        A = self.p.A
        x = np.dot(np.linalg.inv(I-A), u).T

        return x, x

    def get_cov(self, A=None, sigma=None):
        if A is None:
            A = self.p.A
        if sigma is None:
            sigma = self.p.sigma

        mat = np.linalg.inv(np.identity(self.num_states)-A)
        return sigma**2*np.dot(mat, mat.T)

    def negloglik(self, y, A=None, sigma=None, C=None):
        T, N = y.shape
        S = np.dot(y.T,y) / (T-1)
        if C is None:
            C = self.get_cov(A, sigma)
        invC = np.linalg.inv(C)
        _, logdetC = np.linalg.slogdet(C)
        _, logdetS = np.linalg.slogdet(S)
        return T/2*(logdetC-logdetS+N*np.log(2*np.pi)+np.trace(S@invC))

    def get_num_free_params(self):
        return 1 + np.count_nonzero(self.p.A)

    def init_free_params(self, y=None):
        if y is None:
            return self.p_from_A_sigma(self.p.A, self.p.sigma)
        else:
            return self.p_from_A_sigma(self.p.A*0., np.std(y))

    def A_sigma_from_p(self, p):
        D = self.p_to_table(p)
        A, sigma = D['A'], D['sigma']
        return A, sigma

    def p_from_A_sigma(self, A, sigma):
        return self.get_p({'A':self.p.A*0, 'sigma': sigma})
    #
    def get_bounds(self):
        p = self.get_p()
        n = self.get_p_names()
        UB = np.full(p.shape, np.infty)
        LB = np.full(p.shape, -np.infty)
        LB[ n.index('sigma') ] = 0
        return LB, UB

    def fit_MH(self, p0, fn_negloglik, fn_neglogpr):
        from dcsem.utils import MH
        mh = MH(fn_negloglik, fn_neglogpr)
        LB, UB = self.get_bounds()
        samples = mh.fit(p0, LB=LB, UB=UB)
        p = Parameters()
        p.x = np.mean(samples, axis=0)
        p.cov = np.cov(samples.T)
        p.samples = samples
        p.A, p.sigma = self.A_sigma_from_p(p.x)
        return p

    def fit(self, y, method='MH'):
        p0 = self.init_free_params(y)

        def fn_negloglik(p):
            A, sigma = self.A_sigma_from_p(p)
            return self.negloglik(y, A, sigma)
        def fn_neglogpr(p):
            return 0

        if method == 'MH':
            p = self.fit_MH(p0, fn_negloglik, fn_neglogpr)
        else:
            raise(Exception('Only MH method is implemented'))

        return p

# Layer SEM
class MultiLayerSEM(SEM):
    def __init__(self, num_rois, num_layers, params=None):
        super().__init__(num_rois)
        self.num_layers = num_layers
        self.num_states = num_rois * num_layers
        # Connectivity Matrix should be larger for MultiLayerSEM
        self.p.A = np.zeros((self.num_rois*self.num_layers, self.num_rois*self.num_layers), dtype=np.float64)
        # Define default values for T1s
        from dcsem.utils import constants
        self.T1s = np.linspace(constants['LowerLayerT1'],
                               constants['UpperLayerT1'],
                               self.num_layers)
        if params is not None:
            self.set_params(params)

    def __str__(self):
        ret = super().__str__()
        ret += f"T1s = {self.T1s}"
        return ret

    def get_cov_PV(self, P, A=None, sigma=None):
        return P@self.get_cov(A, sigma)@P.T

    def fit_IR(self, tvec, y, TIs, method='MH'):
        p0 = self.init_free_params(y)
        P  = self.get_Pmat(TIs)

        def fn_negloglik(p):
            A, sigma = self.A_sigma_from_p(p)
            L = 0
            for k, Pmat in enumerate(P):
                Ck = self.get_cov_PV(Pmat, A, sigma)
                L += self.negloglik(y[k], A, sigma, Ck)
            return L
        def fn_neglogpr(p):
            return 0

        if method == 'MH':
            p = self.fit_MH(p0, fn_negloglik, fn_neglogpr)
        else:
            raise(Exception('Only MH method is implemented'))

        return p





