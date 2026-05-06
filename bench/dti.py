#!/usr/bin/env python3
"""
Summary metrics based on diffusion tensor fit
"""

import numpy as np
from scipy.linalg import block_diag


def summary_names(acq, b0_threshold=0.05):
    """
    Create the names of summary measurements estimated from diffusion tensor
    :param acq: instance of acquisition class
    :param b0_threshold:
    :return:
    """
    names = []
    for sh in acq.shells:
        if sh.bvals < 0.05:
            names.append("b0.0_mean")
        if sh.bvals >= b0_threshold:
            names.append(f"b{sh.bvals:1.1f}_md")
            names.append(f"b{sh.bvals:1.1f}_fa")

    return names


def fit_dti(signal, acq):
    """
    Fits diffusion tensor model and calculates MD and FA for each shell of diffusion data
    :param signal: diffusion signal (n, m)
    :param acq: an instance of acquisition class
    :return:  A matrix containing MD and FA per shell, b0 deosnt have FA.
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    s0 = signal[:, acq.idx_shells == 0].mean(axis=1)[:, np.newaxis]
    s0[s0 == 0] = 1
    sum_meas = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        bvecs = acq.bvecs[dir_idx]
        shell_signal = signal[..., dir_idx]
        sm = summary_np(shell_signal, bvecs, this_shell.bvals, s0)
        sum_meas.extend(list(sm.values()))

    return np.array(sum_meas).T


def dti_cov(signal, acq, noise_std):
    """
    Estimate covariance matrix of the diffusion tensor metrics
    :param signal:
    :param acq:
    :param noise_std:
    :return:
    """
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]

    variances = list()
    for shell_idx, this_shell in enumerate(acq.shells):
        dir_idx = acq.idx_shells == shell_idx
        bvecs = acq.bvecs[dir_idx]
        shell_signal = signal[..., dir_idx]
        v = noise_variance(shell_signal, bvecs, this_shell.bvals, noise_std, 1)
        variances.append(v)

    cov = block_diag(*variances)
    return cov


def summary_np(signal, gradients, bval, s0):
    """
    Creates summary data based on diffusion MRI data using numpy only for one shell of diffusion

    :param gradients: (M, 3) array with gradient orientations
    :param signal: (..., M) array of diffusion MRI data for the `M` gradients
    :param bval: b-value for this shell, must be scalar
    :param s0:
    :return: Dict mapping 'mean' and 'anisotropy' to (..., )-shaped array
    """

    signal /= s0
    if bval <= 0.05:
        return {"b0.0_mean": s0[:, 0]}

    g = g_mat(gradients)
    d = -np.log(signal) @ np.linalg.pinv(g).T / bval
    dt = np.array(
        [
            [d[..., 0], d[..., 3], d[..., 4]],
            [d[..., 3], d[..., 1], d[..., 5]],
            [d[..., 4], d[..., 5], d[..., 2]],
        ]
    )
    dt = dt.transpose([*np.arange(2, dt.ndim), 0, 1])
    eigs = np.linalg.eigvalsh(dt)

    smm = {
        "MD": np.mean(eigs, axis=-1),
        "FA": 3
        * np.sqrt(1 / 2)
        * np.std(eigs, axis=-1)
        / np.linalg.norm(eigs, axis=-1),
    }

    return smm


def g_mat(gradients):
    x, y, z = gradients.T
    return np.array([x**2, y**2, z**2, 2 * x * y, 2 * x * z, 2 * y * z]).T


def summary(signal, gradients, bval, s0):
    import torch

    """
    Computes  FA and MD from a single shell signal using pytorch
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients
    :param bval: b-value, must be scalar
    :param s0: signal at b=0, might be used for normalization.
    :return: jacobian vectors.
    """

    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        return {"MD": np.mean(signal, axis=-1), "FA": None}

    signal = torch.from_numpy(signal).to(dtype).requires_grad_(False)
    x, y, z = gradients.T
    g = torch.tensor(
        [x**2, 2 * x * y, y**2, 2 * x * z, 2 * y * z, z**2], dtype=dtype
    ).requires_grad_(False)
    d = -torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

    dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
    idx = torch.tril_indices(3, 3)
    dt[..., idx[0], idx[1]] = d

    md = dt.diagonal(dim1=-1, dim2=-2).mean(
        axis=-1
    )  # torch doesnt have trace for nd arrays
    r = dt / (md.view((*md.shape, 1, 1)) * 3)  # normalized tensor matrix
    fa = torch.sqrt(0.5 * (3 - 1 / (r @ r).diagonal(dim1=-1, dim2=-2).sum(axis=-1)))
    smm = {"MD": md.detach().numpy(), "FA": fa.detach().numpy()}

    return smm


def summary_jacobian(signal, gradients, bval, s0):
    import torch

    """
    Compute jacobian matrix for FA and MD w.r.t signal for a single shell of diffusion data
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients tuple (..., 3)
    :param bval: b-value float
    :param s0: signal at b0
    :return: jacobian vectors for md anf fa.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        j_md = None
        j_fa = None
    else:
        signal = torch.from_numpy(signal).to(dtype).requires_grad_(True)
        x, y, z = gradients.T
        g = torch.tensor(
            [x**2, 2 * x * y, y**2, 2 * x * z, 2 * y * z, z**2], dtype=dtype
        ).requires_grad_(False)
        d = -torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

        dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
        idx = torch.tril_indices(3, 3)
        dt[..., idx[0], idx[1]] = d

        md = dt.diagonal(dim1=-1, dim2=-2).mean(
            axis=-1
        )  # torch doesnt have trace for nd arrays

        md.backward(retain_graph=True)
        j_md = signal.grad.detach().numpy()
        signal.grad.data.zero_()

        r = dt / (md.view((*md.shape, 1, 1)) * 3)  # normalized tensor matrix
        fa = torch.sqrt(0.5 * (3 - 1 / (r @ r).diagonal(dim1=-1, dim2=-2).sum(axis=-1)))

        fa.backward()
        j_fa = signal.grad.detach().numpy()
    return j_md, j_fa


def noise_variance(signal, gradients, bval, sigma_n, s0):
    """
    Noise variance for diffusion tensot metrics for a single shell of diffusion MRI data.
    :param signal:
    :param gradients:
    :param bval:
    :param sigma_n:
    :param s0:
    :return:
    """
    j_md = md_jacobian(signal, gradients, bval, s0)
    j_fa = fa_jacobian(signal, gradients, bval, s0)
    if bval > 0:
        j = np.stack([j_md, j_fa])
    else:
        j = j_md
    j = np.squeeze(j)
    m = j @ j.T * (sigma_n**2)
    return m


def md_jacobian(signal, gradients, bval, s0):
    import torch

    """
    Compute jacobian matrix for FA and MD w.r.t signal
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients tuple (..., 3)
    :param bval: b-value float
    :param s0: signal at b0
    :return: jacobian vectors for md anf fa.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        j_md = np.ones_like(signal) / gradients.shape[0]
    else:
        signal = torch.from_numpy(signal).to(dtype).requires_grad_(True)
        x, y, z = gradients.T
        g = torch.tensor(
            [x**2, 2 * x * y, y**2, 2 * x * z, 2 * y * z, z**2], dtype=dtype
        ).requires_grad_(False)
        d = -torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

        dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
        idx = torch.tril_indices(3, 3)
        dt[..., idx[0], idx[1]] = d

        md = dt.diagonal(dim1=-1, dim2=-2).mean(
            axis=-1
        )  # torch doesnt have trace for nd arrays
        md.backward(retain_graph=True)
        j_md = signal.grad.detach().numpy()
    return j_md


def fa_jacobian(signal, gradients, bval, s0):
    import torch

    """
    Compute jacobian matrix for FA and MD w.r.t signal
    :param signal: diffusion signal for a single shell
    :param gradients: direction of gradients tuple (..., 3)
    :param bval: bvalue float
    :param s0: signal at b0
    :return: jacobian vectors for md anf fa.
    """
    dtype = torch.float64
    signal = signal / s0
    if bval == 0:
        j_fa = None
    else:
        signal = torch.from_numpy(signal).to(dtype).requires_grad_(True)
        x, y, z = gradients.T
        g = torch.tensor(
            [x**2, 2 * x * y, y**2, 2 * x * z, 2 * y * z, z**2], dtype=dtype
        ).requires_grad_(False)
        d = -torch.log(torch.abs(signal)) @ torch.pinverse(g) / bval

        dt = torch.zeros((*signal.shape[:-1], 3, 3), dtype=dtype)
        idx = torch.tril_indices(3, 3)
        dt[..., idx[0], idx[1]] = d

        md = dt.diagonal(dim1=-1, dim2=-2).mean(
            axis=-1
        )  # torch doesnt have trace for nd arrays
        r = dt / (md.view((*md.shape, 1, 1)) * 3)  # normalized tensor matrix
        fa = torch.sqrt(0.5 * (3 - 1 / (r @ r).diagonal(dim1=-1, dim2=-2).sum(axis=-1)))

        fa.backward()
        j_fa = signal.grad.detach().numpy()
    return j_fa


def volume_summary(eigs):
    return (32 * np.pi / 945) * (
        2 * (eigs**3).sum(axis=-1)
        - 3
        * (
            eigs[:, 0] ** 2 * (eigs[:, 1] + eigs[:, 2])
            + eigs[:, 1] ** 2 * (eigs[:, 0] + eigs[:, 2])
            + eigs[:, 2] ** 2 * (eigs[:, 0] + eigs[:, 1])
        )
        + 12 * eigs[:, 0] * eigs[:, 1] * eigs[:, 2]
    )


def normalise_summaries(baseline: np.ndarray, change=None, noise_cov=None, names=None):
    """
    Normalises summary measures for all subjects. (divide by average attenuation)
    :param names: name of summaries, is required for knowing how to normalise
    :param baseline: array of summaries for baseline measurements
    :param change: array of summaries for second group, or the change
    :param noise_cov: array or list of covariance matrices
    :param log_l: flag for logarithm l2
    :return: normalised summaries
    """
    assert len(names) == baseline.shape[-1], (
        f"Number of summary measurements doesnt match. "
        f"Expected {len(names)} measures but got {baseline.shape[-1]}."
    )

    b0_idx = names.index("b0.0_mean")
    mean_b0 = np.atleast_1d(baseline[..., b0_idx])
    y1_norm = np.copy(baseline)
    y1_norm = np.delete(y1_norm, b0_idx, axis=-1)

    if change is not None:
        dy_norm = np.copy(change)
        dy_norm[..., b0_idx] = change[..., b0_idx] / mean_b0
    else:
        dy_norm = None

    if noise_cov is not None:
        sigma_n_norm = np.copy(noise_cov)
        sigma_n_norm[..., b0_idx, :] = (
            sigma_n_norm[..., b0_idx, :] / (mean_b0[:, np.newaxis])
        )
        sigma_n_norm[..., :, b0_idx] = (
            sigma_n_norm[..., :, b0_idx] / (mean_b0[:, np.newaxis])
        )
    else:
        sigma_n_norm = None

    return y1_norm, dy_norm, sigma_n_norm
