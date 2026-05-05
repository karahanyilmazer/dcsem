"""
spectral.py — Linear Stochastic DCM with Spectral Fitting (LS-spDCM)

Generative model assumptions:
  - Neural model: dx/dt = A·x + ε(t),  ε(t) ~ N(0, σ_e²·I)
  - Innovation noise: white (flat spectrum), spatially independent across ROIs
  - Hemodynamics: nonlinear Balloon model linearized via impulse simulation
  - Stationarity: assumed
  - Observation noise: additive (structured) in CSD space
  - Name: "Linear Stochastic DCM with Spectral Fitting" (LS-spDCM)

All ROIs share the same HRF (standard DCM assumption).
"""

import logging

import numpy as np
from scipy import signal as scipy_signal

from dcsem.models import DCM
from dcsem.utils import stim_boxcar
from dcsem.validation import validate_bold_shape

logger = logging.getLogger(__name__)


def _build_A_matrix(
    theta_A: np.ndarray, R: int, self_connection: float
) -> np.ndarray:
    """Build R×R connectivity matrix from off-diagonal parameter vector.

    Ordering: source-first — for i in 0..R-1, for j in 0..R-1 where j≠i.
    aij (from ROI i to ROI j) is stored at A[j, i].
    """
    A = np.zeros((R, R))
    np.fill_diagonal(A, self_connection)
    k = 0
    for i in range(R):
        for j in range(R):
            if i != j:
                A[j, i] = theta_A[k]
                k += 1
    return A


class SpectralDCM:
    """Forward model and CSD utilities for LS-spDCM.

    Parameters
    ----------
    n_rois : int
        Number of regions of interest (default 2).
    TR : float
        Repetition time in seconds (default 1.0).
    self_connection : float
        Diagonal element of A matrix; must be negative for stability (default -1.0).
    freq_lo : float
        Lower bound of frequency range in Hz (default 0.01).
    freq_hi : float
        Upper bound of frequency range in Hz (default 0.1).
    n_freqs : int
        Number of frequency bins (default 32).
    """

    def __init__(
        self,
        n_rois: int = 2,
        TR: float = 1.0,
        self_connection: float = -1.0,
        freq_lo: float = 0.01,
        freq_hi: float = 0.1,
        n_freqs: int = 32,
    ):
        if not (self_connection < 0):
            raise ValueError(
                f"self_connection must be negative for stability, got {self_connection}."
            )
        self.n_rois = n_rois
        self.TR = TR
        self.self_connection = self_connection
        self.freqs = np.linspace(freq_lo, freq_hi, n_freqs)
        self.hrf_spectrum: np.ndarray  # complex, shape (n_freqs,)
        self._compute_hrf_spectrum()

    def _theta_size(self) -> int:
        """Expected parameter count for [off-diagonal A entries, log_sigma_e]."""
        return self.n_rois * (self.n_rois - 1) + 1

    def get_param_names(self) -> list[str]:
        """Return source-first parameter names followed by log_sigma_e."""
        names = []
        for src in range(self.n_rois):
            for dst in range(self.n_rois):
                if src != dst:
                    names.append(f"a{src}{dst}")
        names.append("log_sigma_e")
        return names

    def get_bounds(self) -> list[tuple[float, float]]:
        """Default bounds for numerical optimisation in LS-spDCM."""
        bounds = [(-4.0, 4.0)] * (self._theta_size() - 1)
        bounds.append((-10.0, 2.0))
        return bounds

    def _validate_theta(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        if theta.ndim != 1:
            raise ValueError(f"theta must be 1D. Got shape {theta.shape}.")
        if theta.size != self._theta_size():
            raise ValueError(
                f"theta has length {theta.size}, expected {self._theta_size()} "
                f"for {self.n_rois} ROIs."
            )
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains non-finite values.")
        return theta

    # ------------------------------------------------------------------
    # HRF spectrum
    # ------------------------------------------------------------------

    def _compute_hrf_spectrum(self) -> None:
        """Compute the pure haemodynamic transfer function (BOLD per unit
        neural-state perturbation), to be composed with the multi-ROI neural
        transfer H_neural = (jωI − A)⁻¹ inside predict_csd.

        The 1-ROI DCM is driven by a 1-second unit boxcar **input** through
        C = 1.  Its output BOLD therefore carries an implicit 1-ROI neural
        filter (jω − self_connection)⁻¹ on top of the haemodynamic kernel:

            H_input→BOLD(ω) = (jω − self_connection)⁻¹ · H_haemo(ω)

        We strip the input-to-state filter analytically so the stored
        ``hrf_spectrum`` represents H_haemo(ω) alone.  Without this correction
        predict_csd would multiply by (jωI − A)⁻¹ a *second* time, double-
        counting the diagonal dynamics (~30 % attenuation by f = 0.1 Hz).

        The DFT is scaled by ``TR`` so it approximates the continuous-time
        Fourier transform ``H(ω) = ∫ h(t) e^{-jωt} dt`` rather than the raw
        ``sum_n h[n] exp(-jωn)``.  Without this factor the stored spectrum is
        too small by ``1/TR`` (and predict_csd is off by ``1/TR²``), which
        would propagate as a TR-dependent units mismatch when fitting against
        ``observed_csd`` (a Welch density in [BOLD]²/Hz).

        Uses T = 300 s to resolve freq_lo ≥ 0.01 Hz with margin, zero-pads to
        4× length to reduce spectral leakage, then interpolates to self.freqs.
        """
        t_hrf = np.arange(0, 300, self.TR)  # 300 s at TR resolution
        u_impulse = stim_boxcar([[0, 1, 1]])  # unit impulse at t=0, width=1 s
        dcm_1roi = DCM(1, params={"A": [[self.self_connection]], "C": [1.0]})
        bold_hrf, _ = dcm_1roi.simulate(t_hrf, u=u_impulse)  # (300, 1)
        h = bold_hrf[:, 0]  # input→BOLD time course

        # Zero-pad to 4× length to reduce spectral leakage; multiply by TR so
        # the DFT approximates the continuous-time Fourier transform.
        n_fft = 4 * len(h)
        H_full = self.TR * np.fft.rfft(h, n=n_fft)
        freqs_full = np.fft.rfftfreq(n_fft, d=self.TR)

        # Interpolate to self.freqs (avoids issues at non-integer cycles)
        h_input_to_bold = np.interp(
            self.freqs, freqs_full, H_full.real
        ) + 1j * np.interp(self.freqs, freqs_full, H_full.imag)

        # Strip the implicit 1-ROI neural filter (jω − self_connection)⁻¹
        # so the stored spectrum is the haemodynamic kernel alone.
        omega = 2 * np.pi * self.freqs
        self.hrf_spectrum = h_input_to_bold * (1j * omega - self.self_connection)
        # shape (n_freqs,) complex

    # ------------------------------------------------------------------
    # Stability
    # ------------------------------------------------------------------

    def _check_stability(self, A: np.ndarray) -> None:
        """Raise ValueError if A has any non-negative real eigenvalue."""
        eigvals = np.linalg.eigvals(A)
        if np.any(eigvals.real >= 0):
            raise ValueError(
                f"A matrix has non-negative eigenvalue: {eigvals}. "
                "System is unstable."
            )

    # ------------------------------------------------------------------
    # CSD vectorization / unvectorization
    # ------------------------------------------------------------------

    def _output_dim(self) -> int:
        """Number of real values in the vectorized CSD.

        Per frequency: R diagonal (real) + 2·R·(R-1)/2 upper-triangle (re+im).
        For R=2: 4 per freq.
        """
        R = self.n_rois
        return len(self.freqs) * (R + 2 * (R * (R - 1) // 2))

    def _vectorize_csd(self, S_stack: np.ndarray) -> np.ndarray:
        """Flatten Hermitian CSD stack to 1D real array.

        Parameters
        ----------
        S_stack : ndarray, shape (n_freqs, R, R), complex, Hermitian

        Returns
        -------
        ndarray, shape (_output_dim(),)
            Frequency-major ordering.  For each ω:
              [S_11(ω), ..., S_RR(ω), Re(S_12(ω)), Im(S_12(ω)), ...]
        """
        R = S_stack.shape[1]
        parts = []
        for k in range(len(self.freqs)):
            S = S_stack[k]
            # Diagonal (real)
            parts.append(S.diagonal().real)
            # Upper triangle (complex → real + imag)
            for i in range(R):
                for j in range(i + 1, R):
                    parts.append([S[i, j].real, S[i, j].imag])
        return np.concatenate(parts)

    def _unvectorize_csd(self, vec: np.ndarray) -> np.ndarray:
        """Reconstruct Hermitian CSD stack from 1D real array.

        Inverse of _vectorize_csd.

        Parameters
        ----------
        vec : ndarray, shape (_output_dim(),)

        Returns
        -------
        S_stack : ndarray, shape (n_freqs, R, R), complex, Hermitian
        """
        R = self.n_rois
        n_diag = R
        n_upper = R * (R - 1) // 2
        vals_per_freq = n_diag + 2 * n_upper

        S_stack = np.zeros((len(self.freqs), R, R), dtype=complex)
        for k in range(len(self.freqs)):
            offset = k * vals_per_freq
            S = np.zeros((R, R), dtype=complex)
            # Diagonal
            for i in range(R):
                S[i, i] = vec[offset + i]
            # Upper triangle
            ptr = offset + n_diag
            for i in range(R):
                for j in range(i + 1, R):
                    S[i, j] = vec[ptr] + 1j * vec[ptr + 1]
                    S[j, i] = np.conj(S[i, j])  # enforce Hermitian
                    ptr += 2
            S_stack[k] = S
        return S_stack

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def predict_csd(self, theta: np.ndarray) -> np.ndarray:
        """Predict cross-spectral density from parameters.

        Parameters
        ----------
        theta : array-like
            For R ROIs: [a01, a02, ..., a0(R-1), a10, a12, ..., log_sigma_e]
            Off-diagonal A entries in source-first order (R*(R-1) values),
            followed by log of neural noise std.
            For R=2: [a01, a10, log_sigma_e] (backward-compatible).

        Returns
        -------
        ndarray, shape (_output_dim(),)
            Vectorized CSD (real-valued).  Returns np.inf array if A is
            unstable.
        """
        theta = self._validate_theta(theta)
        R = self.n_rois
        n_A = R * (R - 1)
        sigma_e = np.exp(theta[n_A])
        A = _build_A_matrix(theta[:n_A], R, self.self_connection)

        try:
            self._check_stability(A)
        except ValueError:
            return np.full(self._output_dim(), np.inf)

        R = self.n_rois
        S_stack = np.zeros((len(self.freqs), R, R), dtype=complex)

        for k, f in enumerate(self.freqs):
            omega = 2 * np.pi * f
            M = 1j * omega * np.eye(R) - A
            # Neural transfer function: (jωI - A)^{-1}
            H_neural = np.linalg.solve(M, np.eye(R))
            # HRF applied per ROI (all ROIs share the same HRF)
            H_hrf = np.diag([self.hrf_spectrum[k]] * R)
            H_tot = H_hrf @ H_neural

            # CSD: S(ω) = σ_e² · H_tot · H_tot^H
            S = sigma_e**2 * (H_tot @ H_tot.conj().T)
            # Enforce Hermitian (numerical symmetrization)
            S_stack[k] = 0.5 * (S + S.conj().T)

            if logger.isEnabledFor(logging.DEBUG):
                cond = np.linalg.cond(M)
                if cond > 1e6:
                    logger.debug(
                        "High condition number %.2e at f=%.4f Hz", cond, f
                    )

        return self._vectorize_csd(S_stack)

    # ------------------------------------------------------------------
    # Data generation
    # ------------------------------------------------------------------

    def generate_noisy_csd(
        self,
        theta_true: np.ndarray,
        snr: float = 10.0,
        rng: np.random.Generator = None,
    ) -> np.ndarray:
        """Generate synthetic observed CSD with structured Hermitian noise.

        Noise is added in complex CSD space per frequency, scaled
        proportionally to ||S(ω)||_F / snr, preserving Hermitian structure
        and positive semi-definiteness (negative eigenvalues are clipped).

        Parameters
        ----------
        theta_true : array-like
            True parameter vector [a01, a10, log_sigma_e].
        snr : float
            Signal-to-noise ratio controlling noise amplitude.
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility.

        Returns
        -------
        ndarray, shape (_output_dim(),)
            Vectorized noisy CSD.
        """
        if snr <= 0:
            raise ValueError("snr must be positive.")
        if rng is None:
            rng = np.random.default_rng()

        R = self.n_rois
        S_true_vec = self.predict_csd(theta_true)
        S_stack = self._unvectorize_csd(S_true_vec)  # (n_freqs, R, R)

        S_noisy = np.zeros_like(S_stack)
        total_clipped = 0

        for k in range(len(self.freqs)):
            S = S_stack[k]
            scale = np.linalg.norm(S, "fro") / snr  # frequency-proportional

            # Add Hermitian noise: ½(N + N^H) to preserve symmetry
            noise = rng.normal(0, scale, (R, R)) + 1j * rng.normal(
                0, scale, (R, R)
            )
            noise = 0.5 * (noise + noise.conj().T)

            S_noisy[k] = S + noise

            # Enforce PSD: clip negative eigenvalues
            eigvals, eigvecs = np.linalg.eigh(S_noisy[k])
            n_clipped = int(np.sum(eigvals < 0))
            total_clipped += n_clipped
            S_noisy[k] = eigvecs @ np.diag(np.maximum(eigvals, 0)) @ eigvecs.conj().T

        if total_clipped > 0:
            logger.debug(
                "PSD enforcement clipped %d negative eigenvalue(s) across "
                "%d frequencies.",
                total_clipped,
                len(self.freqs),
            )

        return self._vectorize_csd(S_noisy)

    # ------------------------------------------------------------------
    # Time-domain simulation
    # ------------------------------------------------------------------

    def simulate_bold(self, theta, T=200, rng=None):
        """Simulate resting-state BOLD using DCM stochastic mode (sdeint).

        Generates a *high-fidelity, fully nonlinear* realisation: the neural
        state evolves under the linear Itô SDE ``dx = A·x dt + sigma_e dW``
        but the haemodynamic mapping is the **full nonlinear Balloon ODE**.
        This is intentionally NOT bit-equivalent to ``predict_csd``, which
        uses the *linearised* haemodynamic kernel around the operating point
        (q = v = f = 1, s = 0). Fitting linearised parameters to data
        produced by this simulator therefore carries a bounded bias from the
        linearisation gap. The bias magnitude is pinned by
        ``test_spectral_simulate_bold_fit_bias_within_documented_bounds``.

        For self-consistent end-to-end tests use ``generate_noisy_csd``,
        which samples directly from the linearised forward model.

        Parameters
        ----------
        theta : [a01, a10, log_sigma_e]
        T     : simulation length in seconds (at TR resolution)
        rng   : numpy.random.Generator for reproducible stochastic simulation

        Returns
        -------
        bold : ndarray, shape (n_steps, R)
        tvec : ndarray, shape (n_steps,)
        """
        theta = self._validate_theta(theta)
        R = self.n_rois
        n_A = R * (R - 1)
        sigma_e = np.exp(theta[n_A])
        A = _build_A_matrix(theta[:n_A], R, self.self_connection)

        n_steps = int(T / self.TR)
        tvec = np.arange(n_steps) * self.TR

        if rng is None:
            rng = np.random.default_rng()
        dcm = DCM(R, params={'A': A, 'C': np.zeros(R)}, stochastic=True)
        dcm.state_noise_std = sigma_e
        bold, _ = dcm.simulate(tvec, u=None, generator=rng)   # u=None → no stimulus
        return bold, tvec   # bold shape (n_steps, R)

    # ------------------------------------------------------------------
    # Observed CSD from BOLD
    # ------------------------------------------------------------------

    def observed_csd(
        self,
        bold: np.ndarray,
        nperseg: int = None,
        noverlap: int = None,
        window: str = "hann",
        detrend: str = "constant",
    ) -> np.ndarray:
        """Estimate CSD from BOLD time series using Welch's method.

        Parameters
        ----------
        bold : ndarray, shape (T, R)
            BOLD time series.
        nperseg : int, optional
            Segment length for Welch.  Defaults to T//4 (4 segments).
        noverlap : int, optional
            Overlap between segments.  Defaults to nperseg//2.
        window : str
            Window function (default 'hann').
        detrend : str
            Detrending mode (default 'constant' = mean subtraction).

        Returns
        -------
        ndarray, shape (_output_dim(),)
            Vectorized CSD interpolated to self.freqs.

        Notes
        -----
        Welch parameters are logged at DEBUG level for reproducibility.
        Hermitian structure is enforced post-estimation.
        """
        validate_bold_shape(bold, expected_rois=self.n_rois, name="BOLD")

        T, R = bold.shape
        nperseg = nperseg or T // 4
        noverlap = noverlap or nperseg // 2
        if nperseg <= 1 or nperseg > T:
            raise ValueError(
                f"nperseg must be in [2, {T}] for observed_csd; got {nperseg}."
            )
        if noverlap < 0 or noverlap >= nperseg:
            raise ValueError(
                f"noverlap must satisfy 0 <= noverlap < nperseg; got {noverlap}."
            )
        fs = 1.0 / self.TR

        logger.debug(
            "observed_csd: T=%d, R=%d, nperseg=%d, noverlap=%d, "
            "window=%s, detrend=%s, fs=%.4f",
            T, R, nperseg, noverlap, window, detrend, fs,
        )

        S_stack = np.zeros((len(self.freqs), R, R), dtype=complex)

        # csd(x, y) and csd(y, x) are conjugates: only compute upper triangle
        # (and diagonals), then reflect.
        for i in range(R):
            for j in range(i, R):
                freqs_welch, Sij = scipy_signal.csd(
                    bold[:, i],
                    bold[:, j],
                    fs=fs,
                    window=window,
                    nperseg=nperseg,
                    noverlap=noverlap,
                    detrend=detrend,
                )
                S_ij_interp = np.interp(
                    self.freqs, freqs_welch, Sij.real
                ) + 1j * np.interp(self.freqs, freqs_welch, Sij.imag)
                S_stack[:, i, j] = S_ij_interp
                if i != j:
                    S_stack[:, j, i] = np.conj(S_ij_interp)

        # Enforce Hermitian structure post-estimation (diagonals real)
        for k in range(len(self.freqs)):
            S = S_stack[k]
            S_stack[k] = 0.5 * (S + S.conj().T)

        return self._vectorize_csd(S_stack)
