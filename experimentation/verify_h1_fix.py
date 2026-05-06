"""Confirm the H1 fix actually changed predict_csd output.

Strategy: rebuild the OLD hrf_spectrum (without the (jω - λ) correction) by
running the same 1-ROI simulation, FFT, and interpolation. Then build a
"naive predict_csd" using that old spectrum and compare to the current
predict_csd. The diagonal ratio of (current / naive_old) should be exactly
|jω - λ|² (not ⁻², because we are dividing the FIXED prediction by the BUGGY
prediction). For self_connection = -1 over [0.01, 0.1] Hz this should range
from ~1.004 at f_lo up to ~1.394 at f_hi.
"""

from __future__ import annotations

import numpy as np

from dcsem.models import DCM
from dcsem.spectral import SpectralDCM, _build_A_matrix
from dcsem.utils import stim_boxcar


def naive_old_hrf_spectrum(spdcm: SpectralDCM) -> np.ndarray:
    """Reproduce the pre-fix hrf_spectrum: input→BOLD without the (jω−λ) strip."""
    t_hrf = np.arange(0, 300, spdcm.TR)
    u_impulse = stim_boxcar([[0, 1, 1]])
    dcm_1roi = DCM(1, params={"A": [[spdcm.self_connection]], "C": [1.0]})
    bold_hrf, _ = dcm_1roi.simulate(t_hrf, u=u_impulse)
    h = bold_hrf[:, 0]

    n_fft = 4 * len(h)
    H_full = np.fft.rfft(h, n=n_fft)
    freqs_full = np.fft.rfftfreq(n_fft, d=spdcm.TR)
    return np.interp(spdcm.freqs, freqs_full, H_full.real) + 1j * np.interp(
        spdcm.freqs, freqs_full, H_full.imag
    )


def naive_predict_csd(spdcm: SpectralDCM, theta: np.ndarray) -> np.ndarray:
    """Predict CSD using the OLD (uncorrected) hrf_spectrum."""
    R = spdcm.n_rois
    n_A = R * (R - 1)
    sigma_e = np.exp(theta[n_A])
    A = _build_A_matrix(theta[:n_A], R, spdcm.self_connection)
    h_old = naive_old_hrf_spectrum(spdcm)

    S_stack = np.zeros((len(spdcm.freqs), R, R), dtype=complex)
    for k, f in enumerate(spdcm.freqs):
        omega = 2 * np.pi * f
        H_neural = np.linalg.solve(1j * omega * np.eye(R) - A, np.eye(R))
        H_hrf = np.diag([h_old[k]] * R)
        H_tot = H_hrf @ H_neural
        S = sigma_e**2 * (H_tot @ H_tot.conj().T)
        S_stack[k] = 0.5 * (S + S.conj().T)
    return S_stack


def main():
    spdcm = SpectralDCM(n_rois=2, TR=1.0, self_connection=-1.0)
    theta = np.array([0.4, 0.6, np.log(0.05)])
    lam = spdcm.self_connection

    actual_vec = spdcm.predict_csd(theta)
    actual = spdcm._unvectorize_csd(actual_vec)
    naive = naive_predict_csd(spdcm, theta)

    print("=" * 72)
    print("H1 fix verification: current predict_csd vs pre-fix predict_csd")
    print("=" * 72)
    print(f"theta = {theta}, lambda = {lam}")
    print()
    print(f"  {'f (Hz)':>8} {'cur[0,0]':>14} {'old[0,0]':>14} "
          f"{'ratio':>10} {'|jw-l|^2':>10}")
    for k in [0, 5, 10, 15, 20, 25, 31]:
        f = spdcm.freqs[k]
        omega = 2 * np.pi * f
        ratio = actual[k, 0, 0].real / naive[k, 0, 0].real
        theory = omega**2 + lam**2
        print(f"  {f:8.4f} {actual[k,0,0].real:14.4e} "
              f"{naive[k,0,0].real:14.4e} {ratio:10.4f} {theory:10.4f}")
    print()

    diag_ratio = np.array([
        actual[k, 0, 0].real / naive[k, 0, 0].real
        for k in range(len(spdcm.freqs))
    ])
    theory = np.array([(2 * np.pi * f) ** 2 + lam**2 for f in spdcm.freqs])
    rel_err = np.max(np.abs(diag_ratio - theory) / np.abs(theory))
    print(f"max relative error vs |jw - lambda|^2: {rel_err:.4e}")
    print()
    if rel_err < 1e-3:
        print("VERDICT: H1 FIX CONFIRMED.")
        print("  Current predict_csd equals (pre-fix predict_csd) * |jw - lambda|^2")
        print("  on the diagonal, exactly as the audit predicted.")
        print(f"  Forward-model output now differs from pre-fix by up to "
              f"{theory.max():.2f}× at f={spdcm.freqs[-1]:.2f} Hz.")
    else:
        print(f"VERDICT: ratio mismatch (rel err {rel_err:.2%}); fix did not land.")


if __name__ == "__main__":
    main()
