# dcsem

This library implements DCM and SEM for FMRI data. In addition, it implements "Layers" versions of both.



## Getting started

Clone the repo and install
```commandline
git clone https://git.fmrib.ox.ac.uk/saad/dcsem.git
cd dcsem
pip install .
```

## Usage

### Using the command-line wrappers
The easiest way to run the simulations is by using the command line wrapper tools `dcsem_sim` and `dcsem_fit` (BUT: `dcsem_fit` has not yet been implemented).

To get the usage, simply type:

```bash
dcsem_sim --help
```

It is best to run the simulator using a configuration file which defines all the parameters. Here is an example which simulates a DCM model with 3 regions and 2 layers per region:

```bash
# Comments are ignored
outdir = /path/to/output/folder

# Model
model = DCM 

# Network definitions:
num_rois = 3
num_layers = 2
Amat = /path/to/Amat.txt
Cmat = /path/to/Cmat.txt

# Time series params
time_points = 200  
tr = 0.72 
cnr = 20

# Stimulus
stim = /path/to/simulus.txt
```
Once the configuration file has been created, you can run the simulation using:

```bash
dcsem_sim --config my_configuration.txt
```

### Defining the connectivity matrices A and C
The files `Amat.txt` and `Cmat.txt` required to run the simulations can have two different formats. They can either be explicitly given or they can be "described" (see below).

For example, the network shown below has two ROIs with connectivity between the upper and lower layers between the ROIs.

The corresponding connectivity matrix (assuming all connection parameters equal 1 and self-connections equal -1) is:

```bash
echo -1  0  0  0  > Amat.txt
echo  0 -1  0  0 >> Amat.txt
echo  0  1 -1  0 >> Amat.txt
echo  1  0  0 -1 >> Amat.txt
```

The same info could be given with the following text file:

```text
R0, L0 -> R1, L1 = 1
R1, L0 -> R0, L1 = 1
```
The self connections can also be added to the above for each roi and layer, or it can be included in the command-line interface using the `--self_conn` flag.


### Using the python interface

The simulations can also be directly run using the python interface. In the below example, we create a simple two-ROI DCM model. 
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
from dcsem import models, utils
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
Effect of changing $\lambda_d$ on the BOLD activity in the upper layer.

<img src="dcsem/static/DCM_layers_lambda.png" alt="LayerSEM" width="300" >

### Simulating and fitting data with a Layer SEM

The API of the MultiLayerSEM class is quite similar to DCM, except we don't have an input (the input is random noise) and there are no self-connections.

In the below example, we simulate a 2 ROI, 3 layer model and we fit the parameters to the simulated data and plot the posterior distributions.

```python
from dcsem import models, utils
import numpy as np
num_rois   = 2
num_layers = 3
A = utils.create_A_matrix(num_rois,num_layers,
                          ['R0,L2 -> R1,L1 = 1.5',
                           'R1,L0 -> R0,L0 = 0.5',
                           'R1,L2 -> R0,L0 = 0.5'])
sigma = 1.
lsem  = models.MultiLayerSEM(num_rois,num_layers,params={'A':A, 'sigma':sigma})
TIs   = [400, 600, 800, 1000] # inversion times
y     = lsem.simulate_IR(tvec, TIs)
tvec  = np.linspace(0,1,100)
res   = lsem.fit_IR(tvec, y, TIs)
```
Now plot the posterior distributions:

```python
# Compare fitted to simulated params
ground_truth  = lsem.p_from_A_sigma( A, sigma )
estimated     = res.x
estimated_cov = res.cov
fig = utils.plot_posterior(estimated, estimated_cov, samples=res.samples, actual=ground_truth)
```

You should obtain something like the below:

<img src="dcsem/static/LayerSEM_posterior.png" alt="LayerSEM" width="300" >


### Implementation of DCM for layer FMRI

The theory below is based on [Heinzle et al. Neuroimage 2016](https://www.sciencedirect.com/science/article/pii/S1053811915009350)

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

### Multi-Layer extension 
Extending the model to incorporate multiple layers can be straightforwardly done by considering each layer as its own region, but (as in Heinzle et al.) we would like to additionally model the effect of venous blood flowing up towards the pial surface. This is achieved using two additional state variables (per layer) $v^{\star}_k(t), q^{\star}_k(t)$ (volume and dHB concentration) which have the following dynamics:

$$ 
\begin{array}{rcl}
\tau_d\frac{dv^{\star}_{k}}{dt} & = & -v^{\star}_{k} + (1-v_{k-1})  \\
\tau_d\frac{dq^{\star}_{k}}{dt} & = & -q^{\star}_{k} + (1-q_{k-1})  
\end{array}
$$

where $v_{k-1}$ and $q_{k-1}$ are volume/dHb state parameters of the layer below layer $k$ (Sorry, I am labelling layers going from white matter to pial surface, which is obviously the wrong way round!). These equation mean changes in blood volume/dHb in lower layers drive the blood draining signal. 

The state equations for the layer dynamics are slightly different to the standard balloon model, with the addition of the two new state variables added to all but the first layer:

$$ 
\begin{array}{rcl}
\tau\frac{dv_k}{dt} & = & -v_k^{1/\alpha} + f_k + \lambda_d v^{\star}_{k}  \\
\tau\frac{dq_k}{dt} & = & -\frac{v_k^{1/\alpha}}{v_u}q_k+f_k\frac{1-(1-E_0)^{1/f_k}}{E_0} + \lambda_d q^{\star}_{k} 
\end{array}
$$


## SEM and Layer SEM

Structural Equation Modelling is a non-dynamic model where activity in different regions are related through the equation:

$$ x = Ax + u, $$

where the non-zero elements of $A$ determine the structure of the model and where $u$ is $N(0,\sigma^2)$. The matrix $A$ controls how the noise propagates through the rois 

The above equation implies: $x=(I-A)^{-1}(u)$, which is the generative model.

The covariance implied by the model is $C=\sigma^2 (I-A)^{-1}(I-A)^{-T}$, which is to be compared to the empirical covariance $S=cov(x)$ in order to fit the free parameters (the non-zero elements of $A$ as well as the noise variance).

For Layer SEM, we simply consider each layer as its own region in the definition of the $A$ matrix (unlike the DCM model above, there is no BOLD, and there is no blood draining model). In the case where the data that we acquire is an Inversion-Recovery spin echo or gradient echo acquisition, and assuming that we are not measuring the signal from each region but rather from a T1-weighted linear combination of the layers, the observation equation is:

$$ y_k=P_k x, $$

where $\{P_k\}$ are partial volume matrices that combine signals from different layers. Combined with the generative equation for $x$ we have:

$y_k = P_k (I-A)^{-1}(u)$

and therefore $C_k = \sigma^2  P_k(I-A)^{-1}(I-A)^{-T} P_k^T$. These model convariances are compared to the observed covariances in order to fit the free parameters (again, these are the non-zero elements of $A$ and the noise variance).


