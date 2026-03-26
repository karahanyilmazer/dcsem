import numpy as np
import scipy.stats as st

from bench import model_inversion as mi


def toy_model(x, a, b, c):
    return a * x ** 2 + b * x + c


param_priors = {'a': st.norm(loc=1, scale=1),
                'b': st.norm(loc=1, scale=1),
                'c': st.norm(loc=1, scale=1)}


def test_parameter_estimation():
    x = np.array([1, 2, 3])[:, np.newaxis]
    func = lambda params: toy_model(x, **params)
    actual = np.array([1, 2, 2])
    noise_level = 1e-3
    data = toy_model(x, *actual)
    noisy_data = data + noise_level * np.random.randn(*data.shape)

    fits, _ = mi.map_fit_sig(func, param_priors, noisy_data, noise_level)
    np.testing.assert_allclose(actual, fits, rtol=1e-2)


def test_std_estimation():
    x = np.array([1, 2, 3])[:, np.newaxis]
    func = lambda params: toy_model(x, **params)
    actual_params = np.array([1, 2, 2])
    noise_level = 1e-3

    data = toy_model(x, *actual_params)
    noisy_data = data + noise_level * np.random.randn(*data.shape)

    _, stde = mi.map_fit_sig(func, param_priors, noisy_data, noise_level)

    a = np.array([[p ** 2, p, 1] for p in x[:, 0]])
    a_inv = np.linalg.inv(a)
    pe_a = np.squeeze(a_inv @ data)

    cov = a_inv.dot(a_inv.T) * (noise_level ** 2)
    stda = np.sqrt(np.diagonal(cov))

    np.testing.assert_allclose(pe_a, actual_params, rtol=1e-3)
    np.testing.assert_allclose(stda, stde, rtol=1e-7)


def test_confmats():
    n_samples = 100
    noise_level = 1e-3
    effect_size = 0.1
    x = np.array([1, 2, 3])[:, np.newaxis]
    func = lambda params: toy_model(x, **params)
    p_names = list(param_priors.keys())

    true_changes = np.random.randint(0, len(param_priors), n_samples)
    actual_p1 = {k: v.rvs() for k, v in param_priors.items()}
    data_1 = func(actual_p1) + noise_level * np.random.rand(*x.shape)
    pe1, std1 = mi.map_fit_sig(func, param_priors, data_1, noise_level)
    z_vals = []
    for i in range(n_samples):
        actual_p2 = actual_p1.copy()
        if true_changes[i] > 0:
            actual_p2[p_names[true_changes[i]]] += effect_size

        data_2 = func(actual_p2) + noise_level * np.random.rand(*x.shape)
        pe2, std2 = mi.map_fit_sig(func, param_priors, data_2, noise_level)
        z_vals.append(abs(pe2 - pe1) / np.sqrt(std1 ** 2 + std2 ** 2))

    p_vals = st.norm.sf(np.array(z_vals))
    predicts = np.argmin(p_vals, axis=-1) + 1
    predicts[p_vals.mean(axis=-1) > 0.05] = 0
    np.testing.assert_equal(predicts, true_changes)


def test_hessian():
    func = lambda x: x[0] ** 3 + x[1] ** 3
    h = mi.hessian(func, np.array([1, 1]))

    np.testing.assert_array_almost_equal(h, np.array([[6, 0], [0, 6]]), decimal=3)