#!/usr/bin/env python3
"""
This module contains definition of some microstructural diffusion models and a prior distribution for their parameters.
"""

import numba
import numpy as np
from scipy import stats
import warnings

# prior distributions:
dif_coeff = 1.7  # unit: um^2/ms


def sample_signal(n_samples):
    """
    samples signal fraction parameters for 3 compartment models.
    :param n_samples: number of required samples
    :return: samples (n_samples, 3)
    """
    tissue_type = np.random.choice(3, n_samples, p=[0.0, 0.2, 0.8])
    # 0: Pure CSF, 1: CSF partial volume , 2: brain tissue
    s_iso = stats.uniform(loc=0, scale=0.9).rvs(n_samples)
    s_in = stats.truncnorm(loc=0.5, scale=0.2, a=-0.4 / 0.2, b=1 / 0.2).rvs(n_samples)
    s_ex = stats.truncnorm(loc=0.5, scale=0.2, a=-0.4 / 0.2, b=1 / 0.2).rvs(n_samples)

    # CSF:
    s_iso[tissue_type == 0] = 1 - 1e-4

    # inside brain (no CSF partial volume)
    s_iso[tissue_type == 2] = 1e-4

    norm = (1 - s_iso) / (s_in + s_ex)
    return s_iso, s_in * norm, s_ex * norm


prior_distributions = dict(
    ball={"d_iso": stats.truncnorm(loc=3, scale=0.1, a=-3 / 0.1, b=np.inf)},
    stick={
        "d_a": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf)
    },
    cigar={
        "d_a": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "d_r": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
    },
    watson_zeppelin_numerical={
        "d_a": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "d_r": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "odi": stats.beta(a=1, b=4),
    },
    bingham_zeppelin={
        "d_a": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "d_r": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "odi": stats.beta(a=1, b=4),
        "odi2": stats.beta(a=1, b=4),
        "psi": stats.uniform(loc=np.pi / 2, scale=np.pi / 4),
    },
    ball_stick={
        "s_iso": stats.truncnorm(loc=0.5, scale=0.2, a=-0.5 / 0.2, b=np.inf),
        "s_a": stats.truncnorm(loc=0.5, scale=0.2, a=-0.5 / 0.2, b=np.inf),
        "d_iso": stats.truncnorm(loc=3, scale=0.1, a=-3 / 0.1, b=np.inf),
        "d_a": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
    },
    standard_model={
        ("s_iso", "s_in", "s_ex"): sample_signal,
        "odi": stats.beta(a=2, b=3),
        "d_iso": stats.truncnorm(loc=3, scale=0.1, a=-3 / 0.1, b=np.inf),
        "d_in": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "d_ex": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "tau": stats.uniform(loc=0.01, scale=0.98),
    },
    standard_model_bingham={
        ("s_iso", "s_in", "s_ex"): sample_signal,
        "odi": stats.beta(a=2, b=3),
        "odi_ratio": stats.uniform(loc=0.01, scale=0.98),
        "d_iso": stats.truncnorm(loc=3, scale=0.1, a=-3 / 0.1, b=np.inf),
        "d_in": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "d_ex": stats.truncnorm(loc=dif_coeff, scale=0.3, a=-dif_coeff / 0.3, b=np.inf),
        "tau": stats.uniform(loc=0.01, scale=0.98),
    },
    noddi={
        ("s_iso", "s_in", "s_ex"): sample_signal,
        "odi": stats.beta(a=2, b=5),
    },
    noddi_bingham={
        ("s_iso", "s_in", "s_ex"): sample_signal,
        "odi": stats.beta(a=1, b=4),
        "odi_ratio": stats.uniform(loc=0.01, scale=0.98),
    },
)


# compartment definitions:
def ball(bval, bvec, d_iso, s0=1.0):
    """
    Simulates diffusion signal for isotropic diffusion

    This is the equation for this compartment:

    .. math::

        s = e^{-b d_{iso}}

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_iso: diffusion coefficient
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    s0, d_iso = [np.asanyarray(v)[..., np.newaxis] for v in (s0, d_iso)]
    if not np.all(s0 >= 0):
        warnings.warn("s0 cant be negative")
    if not np.all(d_iso >= 0):
        warnings.warn("d_iso cant be negative")
    if not np.isscalar(bval):
        bval = bval * np.ones(bvec.shape[0])

    return s0 * np.exp(-bval * d_iso)


def stick(bval, bvec, d_a, theta, phi, s0=1.0):
    """
    Simulates diffusion signal from single stick model

    The attenuation for the stick is :math:`s=e^{-b d_a (\vec{g} \vec{n})^2}`.

    Simpler equation :math:`a=b`.

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_a: axial diffusion coefficient
    :param theta: angle from z-axis
    :param phi: angle from x axis in xy-plane
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    s0, d_a, theta, phi = [
        np.asarray(v)[..., np.newaxis] for v in (s0, d_a, theta, phi)
    ]
    if not np.all(d_a >= 0):
        warnings.warn("d_a can't be negative")
    if not np.all(s0 >= 0):
        warnings.warn("s0 cant be negative")

    orientation = np.array(spherical2cart(theta, phi)).T
    return s0 * np.exp(-bval * (d_a * orientation.dot(bvec.T) ** 2))


def cigar(bval, bvec, d_a, d_r, theta=0.0, phi=0, s0=1.0):
    """
    Simulates diffusion signal from single stick model

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param theta: angle from z-axis
    :param phi: angle from x axis in xy-plane
    :param d_a: axial diffusion coefficient
    :param d_r: radial diffusion coefficient
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    s0, d_a, d_r = [np.asanyarray(v)[..., np.newaxis] for v in (s0, d_a, d_r)]
    if not np.all(d_r >= 0):
        warnings.warn("d_r can't be negative")
    if not np.all(d_a >= 0):
        warnings.warn("d_a can't be negative")
    if not np.all(s0 >= 0):
        warnings.warn("s0 cant be negative")

    orientation = spherical2cart(theta, phi)
    return s0 * np.exp(-bval * (d_r + (d_a - d_r) * bvec.dot(orientation) ** 2))


def bingham_zeppelin(
    bval, bvec, d_a, d_r, odi, odi2=None, theta=0.0, phi=0.0, psi=0.0, s0=1.0
):
    """
    Simulates diffusion signal for a zeppelin that is dispersed with a bingham distribution

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_a: axial diffusion coefficient
    :param d_r: radial diffusion coefficient
    :param odi: first dispersion coefficient
    :param odi2: second dispersion coefficient
    :param theta: theta for main diffusion direction
    :param phi: phi for main diffusion direction
    :param psi: first dispersion orientation
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    if odi2 is None:
        odi2 = odi  # make it watson distribution.

    s0, d_a, d_r, odi, odi2, theta, phi, psi = [
        np.atleast_1d(v) for v in (s0, d_a, d_r, odi, odi2, theta, phi, psi)
    ]
    n_samples = s0.shape[0]

    if not np.all(s0 >= 0):
        warnings.warn("s0 cannot be negative")

    if not np.all((odi >= odi2) & (odi2 > 0)):
        warnings.warn("odis must be positive and in order")

    if bvec.ndim == 1:
        bvec = bvec[np.newaxis, :]

    r_psi = np.array(
        [
            [np.cos(psi), np.sin(psi), np.zeros_like(psi)],
            [-np.sin(psi), np.cos(psi), np.zeros_like(psi)],
            [np.zeros_like(psi), np.zeros_like(psi), np.ones_like(psi)],
        ]
    ).transpose()

    r_theta = np.array(
        [
            [np.cos(theta), np.zeros_like(theta), -np.sin(theta)],
            [np.zeros_like(theta), np.ones_like(theta), np.zeros_like(theta)],
            [np.sin(theta), np.zeros_like(theta), np.cos(theta)],
        ]
    ).transpose()

    r_phi = np.array(
        [
            [np.cos(phi), np.sin(phi), np.zeros_like(phi)],
            [-np.sin(phi), np.cos(phi), np.zeros_like(phi)],
            [np.zeros_like(phi), np.zeros_like(phi), np.ones_like(phi)],
        ]
    ).transpose()

    r = r_psi @ r_theta @ r_phi

    k1 = 1 / np.tan(odi * np.pi / 2)
    k2 = 1 / np.tan(odi2 * np.pi / 2)
    b_diag = np.zeros(k1.shape + (3, 3))
    b_diag[..., 0, 0] = -k1
    b_diag[..., 1, 1] = -k2

    if r.shape[0] == 1:
        bing_mat = np.array([r[0].T @ b_diag[i] @ r[0] for i in range(b_diag.shape[0])])
    elif r.shape[0] == n_samples:
        bing_mat = np.array([r[i].T @ b_diag[i] @ r[i] for i in range(b_diag.shape[0])])

    denom = hyp_sapprox(np.stack([np.zeros_like(k1), -k2, -k1], -1))
    q = (
        bing_mat[:, np.newaxis, :, :]
        - (bval * (d_a - d_r)[..., np.newaxis])[..., np.newaxis, np.newaxis]
        * ((bvec[:, np.newaxis, :] * bvec[:, :, np.newaxis])[np.newaxis, ...])
    )
    num = hyp_sapprox(np.linalg.eigvalsh(q)[..., ::-1]) * np.exp(
        -d_r[..., np.newaxis] * bval
    )

    return s0[:, np.newaxis] * num / denom[:, np.newaxis]


def watson_zeppelin_numerical(
    bval, bvec, d_a, d_r, odi, theta=0.0, phi=0.0, s0=1.0, n_samples=10000
):
    """
    Simulates diffusion signal for a zeppelin that is dispersed watson distribution using numerical integration

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param d_a: axial diffusion coefficient
    :param d_r: radial diffusion coefficient
    :param odi: first dispersion coefficient
    :param theta: theta for main diffusion direction
    :param phi: phi for main diffusion direction
    :param s0: attenuation for b=0
    :param n_samples: resolution of the surface integral
    :return: simulated signal (M,)
    """
    if not odi > 0:
        warnings.warn("odis must be positive")

    if bvec.ndim == 1:
        bvec = bvec[np.newaxis, :]

    if np.isscalar(bval):
        bval = bval * np.ones(bvec.shape[0])

    k = 1 / np.tan(odi * np.pi / 2)
    mu = np.array(spherical2cart(theta, phi))

    theta_samples, phi_samples = uniform_sampling_sphere(n_samples=n_samples)
    normal_samples = np.array(spherical2cart(theta_samples, phi_samples)).T
    wat_pdf_samples = np.exp(k * normal_samples.dot(mu) ** 2)
    wat_pdf_samples = wat_pdf_samples / wat_pdf_samples.sum()

    s = np.zeros_like(bval)
    for g_i, (b, g) in enumerate(zip(bval, bvec)):
        resp = cigar(
            b, g, d_a=d_a, d_r=d_r, theta=theta_samples, phi=phi_samples, s0=s0
        )
        s[g_i] = (resp * wat_pdf_samples).sum()
    return np.array(s)


# multi-compartment models:


def ball_stick(bval, bvec, d_a, d_iso, s_a, s_iso, theta=0.0, phi=0.0, s0=1.0):
    """
    Simulates diffusion signal from ball and stick model

    :param bval: acquisition b-value
    :param bvec: acquisition b-vec (M,3)
    :param theta: angle from z-axis
    :param phi: angle from x axis in xy-plane
    :param d_a: axial diffusion coefficient
    :param d_iso: radial diffusion coefficient
    :param s_iso: signal fraction of isotropic diffusion
    :param s_a:  signal fraction of anisotropic diffusion
    :param s0: attenuation for b=0
    :return: simulated signal (M,)
    """
    if not np.all(s_iso >= 0):
        warnings.warn("volume fraction cant be negative.")
    if not np.all(s_a >= 0):
        warnings.warn("volume fraction cant be negative")

    s_a = np.atleast_1d(s_a)
    s_iso = np.atleast_1d(s_iso)
    return (
        stick(bval, bvec, d_a, theta, phi, s0) * s_a[:, np.newaxis]
        + ball(bval, bvec, d_iso, s0) * s_iso[:, np.newaxis]
    )


def standard_model(
    bval,
    bvec,
    s_iso,
    s_in,
    s_ex,
    d_iso,
    d_in,
    d_ex,
    tau,
    odi,
    theta=0.0,
    phi=0.0,
    s0=1.0,
):
    """
    Simulates diffusion signal with Watson dispersed standard model

    :param bval: b-values
    :param bvec: (,3) gradient directions(x, y, z)
    :param s_iso: signal fraction of isotropic diffusion
    :param s_in: signal fraction of intra-axonal diffusion
    :param s_ex: signal fraction of extra-axonal water
    :param d_iso: isotropic diffusion coefficient
    :param d_in: axial diffusion coefficient
    :param d_ex: axial diffusion coefficient for extra-axonal compartment
    :param tau: ratio of radial to axial diffusivity
    :param odi: dispersion parameter of watson distribution
    :param theta: orientation of stick from z axis
    :param phi: orientation of stick from x axis
    :param s0: attenuation for b=0
    :return: (M,) diffusion signal
    """
    if not s0 >= 0:
        warnings.warn("s0 cant be negative")
    a_iso = ball(bval=bval, bvec=bvec, d_iso=d_iso, s0=s_iso)
    a_int = bingham_zeppelin(
        bval=bval,
        bvec=bvec,
        d_a=d_in,
        d_r=0,
        odi=odi,
        odi2=odi,
        psi=0,
        theta=theta,
        phi=phi,
        s0=s_in,
    )

    a_ext = bingham_zeppelin(
        bval=bval,
        bvec=bvec,
        d_a=d_ex,
        d_r=d_ex * tau,
        odi=odi,
        odi2=odi,
        psi=0,
        theta=theta,
        phi=phi,
        s0=s_ex,
    )

    return (a_iso + a_int + a_ext) * s0


def standard_model_bingham(
    bval,
    bvec,
    s_iso,
    s_in,
    s_ex,
    d_iso,
    d_in,
    d_ex,
    tau,
    odi,
    odi_ratio,
    psi=0.0,
    theta=0.0,
    phi=0.0,
    s0=1.0,
):
    """
    Simulates diffusion signal with Bingham dispressed NODDI model

    :param bval: b-values
    :param bvec: (,3) gradient directions(x, y, z)
    :param s_iso: signal fraction of isotropic diffusion
    :param s_in: signal fraction of intra-axonal diffusion
    :param s_ex: signal fraction of extra-axonal water
    :param d_iso: isotropic diffusion coefficient
    :param d_in: axial diffusion coefficient
    :param d_ex: axial diffusion coefficient for extra-axonal compartment
    :param tau: ratio for radial diffusion coefficient for intra-axonal compartment
    :param odi: dispersion parameter of bingham distribution
    :param odi_ratio: ratio for dispersion parameter of bingham distribution
    :param psi: fanning orientation
    :param theta: orientation angle of stick from z axis
    :param phi: orientation angle of stick from x axis
    :param s0: attenuation for b=0
    :return: (M,) diffusion signal
    """

    a_iso = ball(bval=bval, bvec=bvec, d_iso=d_iso, s0=s_iso)

    a_int = bingham_zeppelin(
        bval=bval,
        bvec=bvec,
        d_a=d_in,
        d_r=0,
        odi=odi,
        odi2=odi * odi_ratio,
        psi=psi,
        theta=theta,
        phi=phi,
        s0=s_in,
    )
    a_ext = bingham_zeppelin(
        bval=bval,
        bvec=bvec,
        d_a=d_ex,
        d_r=d_ex * tau,
        odi=odi,
        odi2=odi * odi_ratio,
        psi=psi,
        theta=theta,
        phi=phi,
        s0=s_ex,
    )
    return (a_iso + a_int + a_ext) * s0


def noddi(bval, bvec, s_iso, s_in, s_ex, odi, theta=0.0, phi=0.0, s0=1.0):
    # fixed parameters:
    d_iso = 3
    dax_int = 1.7
    dax_ext = 1.7
    tau = s_ex / (s_in + s_ex + np.finfo(s_in.dtype).eps)

    signal = standard_model(
        bval=bval,
        bvec=bvec,
        s_iso=s_iso,
        s_in=s_in,
        s_ex=s_ex,
        d_iso=d_iso,
        d_in=dax_int,
        d_ex=dax_ext,
        tau=tau,
        odi=odi,
        s0=s0,
        theta=theta,
        phi=phi,
    )
    return signal


def bingham_noddi(
    bval, bvec, s_iso, s_in, s_ex, odi, odi_ratio, theta=0.0, phi=0.0, psi=0.0, s0=1.0
):
    # fixed parameters:
    d_iso = 3
    d_in = 1.7
    d_ex = 1.7
    tau = s_in / (s_in + s_ex)

    signal = standard_model_bingham(
        bval=bval,
        bvec=bvec,
        s_iso=s_iso,
        s_in=s_in,
        s_ex=s_ex,
        d_iso=d_iso,
        d_in=d_in,
        d_ex=d_ex,
        tau=tau,
        odi=odi,
        odi_ratio=odi_ratio,
        s0=s0,
        theta=theta,
        phi=phi,
        psi=psi,
    )
    return signal


# helper functions:
def spherical2cart(theta, phi, r=1):
    """
    Converts spherical to cartesian coordinates
    :param theta: angel from z axis
    :param phi: angle from x axis
    :param r: radius
    :return: tuple [x, y, z]-coordinates
    """
    z = r * np.cos(theta)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    return x, y, z


def cart2spherical(n):
    """
    Converts to spherical coordinates
    :param n: (:, 3) array containing vectors in (x,y,z) coordinates
    :return: tuple with (phi, theta, r)-coordinates
    """
    r = np.sqrt(np.sum(n**2, axis=1))
    theta = np.arccos(n[:, 2] / r)
    phi = np.arctan2(n[:, 1], n[:, 0])
    phi[r == 0] = 0
    theta[r == 0] = 0
    return r, phi, theta


def uniform_grid_sphere(n_theta, n_phi=None):
    """
    Generates uniformly distributed grid over the surface of sphere:

    :param n_theta: number of theta grids
    :param n_phi: number of phi_grids
    :return: grid of theta and phi
    """
    if n_phi is None:
        n_phi = n_theta

    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    cos_theta = np.linspace(-1, 1, n_theta)
    theta = np.arccos(cos_theta)

    pairs = np.array([(t, p) for t in theta for p in phi])
    theta, phi = pairs.T
    return theta, phi


def uniform_sampling_sphere(n_samples):
    """
    Generates uniformly distributed samples over the surface of sphere:

    :param n_samples: number of theta grids
    :return: samples of theta and phi
    """

    phi = np.random.uniform(0, 2 * np.pi, n_samples)

    cos_theta = np.random.uniform(-1, 1, n_samples)
    theta = np.arccos(cos_theta)
    return theta, phi


def plot_response_function(response, shells, idx_shells, bvecs, res=40, maxs=5):
    import matplotlib.pyplot as plt
    from scipy.interpolate import griddata
    from matplotlib import cm

    fig = plt.figure(figsize=(12, 8))

    for shell_idx in np.arange(1, 1):
        sig = -np.log(response[idx_shells == shell_idx]) / shells[shell_idx].bval
        dirs = bvecs[idx_shells == shell_idx]
        dirs = np.vstack((dirs, -dirs))
        sig = np.append(sig, sig)
        _, phi, theta = cart2spherical(dirs)

        p, t = np.meshgrid(np.linspace(-np.pi, np.pi, res), np.linspace(0, np.pi, res))
        s = griddata((phi, theta), sig, (p, t), method="nearest")
        x = np.sin(p) * np.cos(t)
        y = np.sin(p) * np.sin(t)
        z = np.cos(p)

        ax = fig.add_subplot(2, 2, shell_idx, projection="3d")
        plt.set_cmap("jet")
        plot = ax.plot_surface(
            x,
            y,
            z,
            rstride=1,
            cstride=1,
            cmap=cm.jet,
            linewidth=0,
            antialiased=False,
            alpha=0.8,
            facecolors=cm.jet(s / maxs),
        )
        plot.set_clim(vmin=0, vmax=maxs)
        fig.colorbar(plot, shrink=0.5, aspect=2)
        plt.title(f"bvals={shells[shell_idx].bval}")
        (
            ax.set_xlim3d(-maxs, maxs),
            ax.set_ylim3d(-maxs, maxs),
            ax.set_zlim3d(-maxs, maxs),
        )
        ax.view_init(azim=0, elev=0)
    plt.show()


@numba.jit(nopython=True)
def find_t(l1, l2, l3):
    """
    Helper function for hyp_Sapprox

    Args:
        l1: float
            negative first eigenvalue
        l2: float
            negative second eigenvalue
        l3: float
            negative third eigenvalue

    Returns: float
        I guess the return value is t

    """
    a3 = l1 * l2 + l2 * l3 + l1 * l3
    a2 = 1.5 - l1 - l2 - l3
    a1 = a3 - l1 - l2 - l3
    a0 = 0.5 * (a3 - 2 * l1 * l2 * l3)

    inv3 = 1.0 / 3.0
    p = (a1 - a2 * a2 * inv3) * inv3
    q = (-9 * a2 * a1 + 27 * a0 + 2 * a2 * a2 * a2) / 54
    d = q * q + p * p * p
    offset = a2 * inv3

    if d > 0:
        ee = np.sqrt(d)
        z1 = (-q + ee) ** inv3 + (-q - ee) ** inv3 - offset
        z2 = z1
        z3 = z1
    elif d < 0:
        ee = np.sqrt(-d)
        angle = 2 * inv3 * np.arctan(ee / (np.sqrt(q * q + ee * ee) - q))
        sqrt3 = np.sqrt(3.0)
        c = np.cos(angle)
        s = np.sin(angle)
        ee = np.sqrt(-p)
        z1 = 2 * ee * c - offset
        z2 = -ee * (c + sqrt3 * s) - offset
        z3 = -ee * (c - sqrt3 * s) - offset
    else:
        tmp = -(q**inv3)
        z1 = 2 * tmp - offset
        if p != 0 or q != 0:
            z2 = tmp - offset
        else:
            z2 = z1
        z3 = z2
    if z1 < z2 and z1 < z3:
        return z1
    elif z2 < z3:
        return z2
    else:
        return z3


@numba.guvectorize([(numba.float64[:], numba.float64[:])], "(n)->()")
def hyp_sapprox(x, res):
    """
    Computes 1F1(1/2; 3/2; M) where ``x`` are the eigenvalues from M

    see ``der_hyp_Sapprox`` to only numerically estimate the derivative

    Args:
        x: (3, ) float np.ndarray
            eigenvalues in descending order

    Returns: float
        Result of the hypergeometric function

    """
    if x[0] == 0 and x[1] == 0 and x[2] == 0:
        res[0] = 1.0
    else:
        t = find_t(-x[0], -x[1], -x[2])
        r = 1.0
        k2 = 0.0
        k3 = 0.0
        k4 = 0.0

        for idx in range(3):
            r /= np.sqrt(-x[idx] - t)
            k2 += 0.5 * (x[idx] + t) ** -2
            k3 -= (x[idx] + t) ** -3
            k4 += 3 * (x[idx] + t) ** -4

        tau = k4 / (8 * k2 * k2) - 5 * k3 * k3 / (24 * k2**3)
        c1 = (np.sqrt(2 / k2) * np.pi * r * np.exp(-t)) * np.exp(tau) / (4 * np.pi)
        res[0] = c1
