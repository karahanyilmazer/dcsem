import numpy as np
import pytest
from dipy.data import default_sphere
from numpy import testing
from scipy.stats import distributions

import change_model
import spherical_harmonics
from bench import acquisition, diffusion_models
from bench.change_model import Trainer


def forward_model(x, a, b, c) -> np.ndarray:
    """produces (n_samples, n_summary)
    """
    a, b, c = [np.asarray(v)[..., None] for v in (a, b, c)]
    return a * x ** 2 + b * x + c


@pytest.fixture
def stupid_trainer():
    return Trainer(
        forward_model, 
        kwargs={'x': np.array([-1, 1])},
        priors={name: distributions.norm(loc=idx, scale=0.) for idx, name in enumerate('abc')},
        )


@pytest.fixture
def multi_shell():
    single_bvecs = default_sphere.vertices
    assert single_bvecs.shape[1] == 3
    non_zero_bvals = np.concatenate([np.full(single_bvecs.shape[0], bval) for bval in (1, 2, 3)])
    bvals = np.append(np.zeros(10), non_zero_bvals)
    bvecs = np.concatenate([np.zeros((10, 3))] + [single_bvecs] * 3, 0)
    idx_shell, shells = acquisition.ShellParameters.create_shells(bval=bvals)
    return acquisition.Acquisition(shells, idx_shell, bvecs)


@pytest.fixture
def trainer(multi_shell):
    model = change_model.summary_decorator(diffusion_models.standard_model, summary_type='sh')
    return Trainer(
        model,
        dict(acq=multi_shell, shm_degree=4, noise_level=0),
        priors=diffusion_models.prior_distributions[model.__name__]
    )


def test_multi_shell(trainer: Trainer):
    np.random.seed(123)
    d1, d2 = trainer.generate_train_samples(1000, 0.01, old=True, parallel=False)
    np.random.seed(123)
    p1, p2 = trainer.generate_train_samples(1000, 0.01)

    assert np.all([(x == x).all() for x in (d1, d2, p1, p2)])
    testing.assert_array_almost_equal(d1, p1)
    testing.assert_array_almost_equal(d2, p2)


def test_generate_data(stupid_trainer):
    y_1, y_2 = stupid_trainer.generate_train_samples(100, 1)
    assert y_1.shape == (100, 2)
    assert y_2.shape == (3, 100, 2)
    testing.assert_equal(y_1[:, 0], 1)
    testing.assert_equal(y_1[:, 1], 3)
    testing.assert_equal(y_2[0, :, 0], 2)
    testing.assert_equal(y_2[0, :, 1], 4)
    testing.assert_equal(y_2[1, :, 0], 0)
    testing.assert_equal(y_2[1, :, 1], 4)
    testing.assert_equal(y_2[2, :, 0], 2)
    testing.assert_equal(y_2[2, :, 1], 4)


def test_nan():
    a = np.array([2, np.nan])
    b = np.array([2, np.nan])
    #assert np.all([(x == x).all() for x in (a, b)])
    testing.assert_almost_equal(a, b)


def test_derivatives():
    """

    :return:
    """