# dcsem


## Getting started

Clone the repo and install
```commandline
git clone https://git.fmrib.ox.ac.uk/saad/dcsem.git
cd dcsem
pip install -e .
```

## Usage
```python
import numpy as np
from dcsem import models, utils

# instantiate object
dcm = models.DCM()

# input
tvec  = np.arange(100)
u     = utils.boxcar(np.array([[0,10,1]]))

# connectivity params
# roi0,layer0->roi1,layer0 : a = 1.
A = utils.create_A_matrix(2,1,([(0,0),(1,0),1],),-1)
# input->roi0,layer0 : c=1
C = utils.create_C_matrix(2,1,([0,0,1],))

# run simulation
state_tc = dcm.simulate(tvec,u,A,C,num_roi=2)
```



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
