# DCM Model Integration for Parameter Estimation

Both `pipelines/inversion_generic.py` and `pipelines/mcmc_generic.py` now support DCM 2-ROI BOLD models alongside analytical models.

## Quick Start

### For Optimizer-Based Estimation (`pipelines/inversion_generic.py`)

1. Open `pipelines/inversion_generic.py`
2. Comment out the active model (e.g., quadratic)
3. Uncomment the DCM model section (lines ~119-157):

```python
IS_DCM_MODEL = True
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])
ODE_METHOD = "BDF"  # Stiff solver

def model(theta, x):
    params = dict(zip(param_names, theta))
    bold = simulate_bold(
        params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )
    return bold

model_name = "dcm_2roi"
model_display_name = "DCM 2-ROI BOLD Model"
param_names = ["a01", "a10", "c0", "c1"]
theta_true = np.array([0.4, 0.6, 0.9, 0.2])
theta_zero = np.array([0.3, 0.8, 0.7, 0.3])

param_bounds = [
    (-1.5, 1.5),  # a01 (A matrix)
    (-1.5, 1.5),  # a10 (A matrix)
    (0.0, 1.5),   # c0  (C matrix)
    (0.0, 1.5),   # c1  (C matrix)
]
```

4. Run the script normally

### For MCMC-Based Estimation (`pipelines/mcmc_generic.py`)

1. Open `pipelines/mcmc_generic.py`
2. Comment out the active model (e.g., quadratic)
3. Uncomment the DCM model section (lines ~120-168):

```python
IS_DCM_MODEL = True
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])
ODE_METHOD = "BDF"

def model(theta, x):
    params = dict(zip(param_names, theta))
    bold = simulate_bold(
        params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )
    return bold

model_name = "dcm_2roi"
model_display_name = "DCM 2-ROI BOLD Model"
param_names = ["a01", "a10", "c0", "c1"]
theta_true = np.array([0.4, 0.6, 0.9, 0.2])
theta_zero = np.array([0.3, 0.8, 0.7, 0.3])

# Gaussian priors: [(mu, sigma), ...]
priors = [
    (0.0, 1.0),  # a01: wide prior for excitatory/inhibitory
    (0.0, 1.0),  # a10: wide prior for excitatory/inhibitory
    (0.5, 0.5),  # c0: moderate positive range
    (0.5, 0.5),  # c1: moderate positive range
]
```

4. Run the script normally

## What Gets Adapted Automatically

Both scripts automatically detect `IS_DCM_MODEL = True` and adapt:

### Data Generation
- **Analytical**: 1D x,y arrays with Gaussian noise
- **DCM**: Multi-ROI BOLD signals (T×R) with optional noise

### Optimization
- **Analytical**: BFGS (unconstrained)
- **DCM**: L-BFGS-B (with parameter bounds)

### MSE Calculation
- **Analytical**: Mean over scalar residuals
- **DCM**: Mean over multi-dimensional BOLD residuals

### Plots

#### Data Fit Plot
- **Analytical**: Single scatter + line plot
- **DCM**: Multi-panel time series (one subplot per ROI)

#### Posterior Predictive (MCMC only)
- **Analytical**: Single plot with uncertainty band
- **DCM**: Multi-panel with uncertainty bands per ROI

#### Loss Landscapes (Optimizer only)
- Both work identically, just computed over BOLD MSE instead

#### Corner Plot (MCMC only)
- Works identically for both model types

#### Correlation Matrix
- Works identically for both model types

## DCM Parameters

### A-matrix (Connectivity)
- `a01`: ROI 0 → ROI 1 connection strength
- `a10`: ROI 1 → ROI 0 connection strength
- Can be negative (inhibitory) or positive (excitatory)
- Typical range: [-1.5, 1.5]

### C-matrix (Input Strength)
- `c0`: External input → ROI 0
- `c1`: External input → ROI 1
- Non-negative values
- Typical range: [0.0, 1.5]

## Customization Options

### Stimulus Pattern
```python
u = stim_boxcar([[0, 30, 1]])      # Single boxcar: start=0, duration=30, amplitude=1
u = stim_boxcar([[0, 20, 1], [40, 20, 0.5]])  # Multiple epochs
```

### Time Resolution
```python
time = np.arange(100)        # 100 time points
time = np.linspace(0, 10, 200)  # 200 points from 0 to 10
```

### ODE Solver
```python
ODE_METHOD = "BDF"    # Stiff solver (recommended for DCM)
ODE_METHOD = "RK45"   # Standard Runge-Kutta
ODE_METHOD = None     # Use default
```

### Noise Level
```python
noise_sigma = 0.01    # Low noise (1% BOLD fluctuation)
noise_sigma = 0.05    # Moderate noise
noise_sigma = 0.0     # No noise (perfect observations)
```

### MCMC Settings (`pipelines/mcmc_generic.py` only)
```python
n_walkers = 24        # Number of MCMC walkers
n_burn = 5000         # Burn-in iterations
n_samples_mcmc = 10000  # Sampling iterations
```

## Output Structure

### Optimizer-Based (`pipelines/inversion_generic.py`)
```
img/inversion/L-BFGS-B/dcm_2roi/
├── data_fit.png              # Multi-ROI BOLD time series
├── loss_landscape_1d.png     # 1D parameter slices
├── loss_landscape_2d.png     # 2D parameter contours
└── correlation_matrix.png    # Hessian-based correlations

logs/dcm_2roi_L-BFGS-B.json  # Full results log
```

### MCMC-Based (`pipelines/mcmc_generic.py`)
```
img/inversion/MCMC/dcm_2roi/
├── data_fit.png              # Multi-ROI BOLD with posterior mean
├── posterior_predictive.png  # Multi-ROI with uncertainty bands
├── corner_plot.png           # Posterior distributions
└── correlation_matrix.png    # Posterior correlations

logs/dcm_2roi_MCMC.json      # Full results log
```

## Tips

1. **Use stiff solver**: DCM equations can be stiff, especially with large connection strengths. Use `ODE_METHOD = "BDF"` for stability.

2. **Set appropriate priors (MCMC)**: For A-matrix connections, use zero-centered priors with moderate width (e.g., `(0.0, 1.0)`). For C-matrix inputs, use positive-centered priors (e.g., `(0.5, 0.5)`).

3. **Check Hessian diagnostics**: High condition numbers or negative eigenvalues indicate identifiability issues. Consider:
   - Simplifying the model
   - Adding more informative data
   - Using stronger priors (for MCMC)

4. **Monitor MCMC diagnostics**:
   - Acceptance fraction should be 0.2-0.5
   - Check autocorrelation time is reasonable
   - Ensure effective sample size is sufficient (>1000)

5. **Noise level matters**: For DCM, BOLD noise is typically small (σ ~ 0.01-0.05). Too much noise makes parameters non-identifiable.

## Example: Switching from Analytical to DCM

**Before (Quadratic Model):**
```python
def model(theta, x):
    a, b, c = theta
    return a * x**2 + b * x + c

model_name = "quadratic"
model_display_name = "Quadratic Model"
# ... rest of quadratic setup
```

**After (DCM Model):**
```python
IS_DCM_MODEL = True
NUM_ROIS = 2
time = np.arange(100)
u = stim_boxcar([[0, 30, 1]])
ODE_METHOD = "BDF"

def model(theta, x):
    params = dict(zip(param_names, theta))
    bold = simulate_bold(
        params, time=time, u=u, num_rois=NUM_ROIS, ode_method=ODE_METHOD
    )
    return bold

model_name = "dcm_2roi"
model_display_name = "DCM 2-ROI BOLD Model"
# ... rest of DCM setup
```

Everything else (plotting, diagnostics, logging) adapts automatically! 🎉
