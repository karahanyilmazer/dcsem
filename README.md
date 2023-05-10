# dcsem

This library implements DCM and SEM for FMRI data. In addition, it implements "Layers" versions of both.



## Getting started

Clone the repo and install
```commandline
git clone https://git.fmrib.ox.ac.uk/saad/dcsem.git
cd dcsem
pip install -e .
```

## Usage

### Simulating data using DCM generative model
In the below example, we create a simple two-ROI DCM model. 
```python
import numpy as np
from dcsem import models, utils

# input
tvec  = np.arange(100) # time vector (seconds)
u     = utils.stim_boxcar(np.array([[0,10,1]])) # stimulus function (here onset=0, duration=10s, magnitude=1)

# connectivity params
num_rois = 2
num_layers = 1
# roi0,layer0->roi1,layer0 : magnitude = 0.2
# 
connections = ['R0,L0 -> R1,L0 = .2']
A = utils.create_A_matrix(num_rois,
                          num_layers,
                          connections,
                          self_connections=-1)
# input->roi0,layer0 : c=1
input_connections = ['R0, L0 = 1.']
C = utils.create_C_matrix(num_rois, num_layers,input_connections)

# instantiate object
dcm = models.DCM(num_rois, params={'A':A,'C':C})

# run simulation
state_tc = dcm.simulate(tvec, u)

# plot results
import matplotlib.pyplot as plt
plt.plot(tvec, state_tc['bold'])
```

Below is how you can generate data for layer DCM. We generate a 1 ROI layer DCM, and we change the value of the blood draining parameter $\lambda_d$ (see theory section) and examine its effect on the activity in the two layers (replicating the result from Heinzle et al, figure 2c).

```python
# Let's replicate figure 2c from Heinzle paper

# 1 ROI, 2 Layers. Input feed to both, and some drainage occurs. Change lambda then plot upper and lower layer separately

TR    = 1  # repetition time
ntime = 100  # number of time points 
tvec  = np.linspace(0,ntime*TR,ntime)  # seconds

stim = [[0,30,1]]
u    = utils.stim_boxcar(stim)

# 1 ROI
A = utils.create_A_matrix(num_rois=1, num_layers=2, self_connections=-1.)
C = utils.create_C_matrix(num_rois=1, num_layers=2, input_connections=['R0,L0=1.','R0,L1=1.'])

state_tc = []
lambdas = [0,0.1,0.4,0.6,0.8,0.9] 
for l in lambdas:
    ldcm = models.TwoLayerDCM(num_rois=1, params={'A':A, 'C':C, 'l_d':l})
    state_tc.append(ldcm.simulate(tvec,u))

# Plotting
import matplotlib.pyplot as plt
plt.figure()
for s,l in zip(state_tc,lambdas):
    plt.subplot(1,2,1)
    plt.plot(s['bold'][:,0],c=[l,l,l],alpha=.5)
    plt.title('Lower layer')    
    plt.subplot(1,2,2)
    plt.title('Upper layer')    
    plt.plot(s['bold'][:,1],c=[l,l,l],label=f'$\lambda_d$={l}')
    plt.legend()

plt.grid()
plt.show()
```
<img src="dcsem/static/DCM_layers_lambda.jpg" alt="layers" width="200" height="100">

### Implementation of DCM for layer FMRI

Based on [Heinzle et al. Neuroimage 2016](https://www.sciencedirect.com/science/article/pii/S1053811915009350)

The traditional DCM model is composed of two elements, a state equation describing the evolution of neuronal dynamics $x(t)$ as a function of an input stimulus $u(t)$, and an equation linking neural dynamics to the BOLD signal $y(t)$ via the Buxton Balloon model (see [Buxton and Frank. 1997](https://pubmed.ncbi.nlm.nih.gov/8978388/) and [Friston et al. 2000](https://www.sciencedirect.com/science/article/pii/S105381190090630X)). The state equation is a simple order 1 ODE:

$$ \frac{dx}{dt}=Ax(t)+Cu(t) $$

Where $A$ is a $n\times n$ connectivity matrix between $n$ regions of interest and $C$ is a $n \times 1$ matrix specifying how the input feeds into each region.

The Balloon model relates neural activity to the BOLD signal change via a neuro-vascular coupling mechanism involving 4 state variables: $s(t)$ is a vasodilatory signal which links neural activity to changes in blood flow $f(t)$; this in turn is coupled with changes in blood volume $v(t)$ and in deoxy-hemoglobin (dHb) $q(t)$ via a nonlinear ODE:

$$
\begin{array}{rcl}
\frac{ds}{dt} & = &-\kappa s -\gamma (f-1) + x \\
\frac{df}{dt} & = &s \\
\tau\frac{dv}{dt} & = &-v^{1/\alpha}+f \\
\tau\frac{dq}{dt} & = &-\frac{v^{1/\alpha}}{v}q+f\frac{1-(1-E_0)^{1/f}}{E_0}
\end{array}
$$

Finally, the BOLD signal change is a nonlinear combination of changes in blood flow and dHb concentration:

$$y(t) = V_0\left(k_1(1-q)+k_2(1-\frac{q}{v})+k_3(1-v)\right)$$

Note: in Heinzle2016, the model has been slightly re-parametrised compared to Friston2000, here is the mapping between the two:

$$
\begin{array}{rcl}
\textrm{Heinzle} & - & \textrm{Friston} \\
\kappa & \Longleftrightarrow & 1/\tau_s \\
\gamma & \Longleftrightarrow & 1/\tau_f \\
\tau & \Longleftrightarrow & \tau_0.
\end{array}
$$
