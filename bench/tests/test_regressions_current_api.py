import os
import tempfile

import numpy as np
import pytest
import scipy.stats as st

from bench import dti, glm, model_inversion
from bench.change_model import NoChangeModel, Trainer


def test_no_change_model_distribution_uses_configured_dimension():
    model = NoChangeModel(n_dim=3)
    mu, sigma = model.distribution(np.array([[1.0, 2.0]]))

    assert mu.shape == (1, 3)
    assert sigma.shape == (1, 3, 3)


def test_trainer_amount_priors_length_error_is_value_error():
    trainer_kwargs = dict(
        forward_model=lambda a, b: np.stack([a + b, a - b], axis=-1),
        priors={"a": st.uniform(0, 1), "b": st.uniform(0, 1)},
        amount_priors=[1, 2, 3],
    )

    with pytest.raises(ValueError):
        Trainer(**trainer_kwargs)


def test_dti_normalise_summaries_uses_change_input():
    baseline = np.array([[100.0, 1.5, 0.4]])
    change = np.array([[10.0, 0.1, 0.05]])
    noise = np.eye(3)[None, ...]

    y, dy, sigma = dti.normalise_summaries(
        baseline, change, noise, names=["b0.0_mean", "b1.0_md", "b1.0_fa"]
    )

    np.testing.assert_allclose(y, np.array([[1.5, 0.4]]))
    np.testing.assert_allclose(dy, np.array([[0.1, 0.1, 0.05]]))
    np.testing.assert_allclose(sigma[0, 0], np.array([0.0001, 0.0, 0.0]))


def test_infer_change_uses_combined_standard_error_and_absolute_z():
    pe1 = np.array([[0.0, 0.0, 0.0]])
    std1 = np.array([[0.1, 0.1, 0.5]])
    pe2 = np.array([[0.2, 0.0, -2.0]])
    std2 = np.array([[0.1, 0.1, 0.5]])

    inferred_change, amount = model_inversion.infer_change(pe1, std1, pe2, std2)

    np.testing.assert_array_equal(inferred_change, np.array([[3]]))
    np.testing.assert_allclose(amount, np.array([[2.0]]))


def test_map_fit_masks_negative_hessian_diagonal(monkeypatch):
    class DummyResult:
        x = np.array([0.0, 0.0])

    monkeypatch.setattr(model_inversion.optimize, "minimize", lambda *a, **k: DummyResult())
    monkeypatch.setattr(
        model_inversion, "hessian", lambda *a, **k: np.diag(np.array([-1.0, 4.0]))
    )

    pe, std = model_inversion.map_fit(
        data=np.array([0.0, 0.0]),
        noise_cov=np.eye(2),
        model=lambda a, b: np.array([a + b, a - b]),
        priors={"a": st.norm(), "b": st.norm()},
    )

    np.testing.assert_allclose(pe, np.array([0.0, 0.0]))
    assert np.isnan(std[0])
    np.testing.assert_allclose(std[1], 0.5)


def test_group_glm_uses_residual_degrees_of_freedom():
    data = np.array(
        [
            [[1.0, 2.0]],
            [[1.2, 2.1]],
            [[0.9, 1.9]],
            [[2.0, 3.0]],
            [[2.2, 3.1]],
        ]
    )
    x = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ]
    )
    c = np.array([[1.0, 0.0], [-1.0, 1.0]])

    with tempfile.TemporaryDirectory() as tmpdir:
        design_mat = os.path.join(tmpdir, "design.mat")
        design_con = os.path.join(tmpdir, "design.con")

        with open(design_mat, "w") as f:
            f.write("/NumWaves 2\n/NumPoints 5\n/PPheights 1 1\n/Matrix\n")
            for row in x:
                f.write(f"{row[0]} {row[1]}\n")

        with open(design_con, "w") as f:
            f.write("/ContrastName1 baseline\n/ContrastName2 diff\n")
            f.write("/NumWaves 2\n/NumContrasts 2\n/Matrix\n")
            for row in c:
                f.write(f"{row[0]} {row[1]}\n")

        _, _, sigma_n = glm.group_glm(data, design_mat, design_con)

    y = np.transpose(data, [1, 2, 0])
    beta = y @ np.linalg.pinv(x).T
    residuals = y - beta @ x.T
    sigma_sq = np.cov(residuals[0], ddof=0) * (x.shape[0] / (x.shape[0] - x.shape[1]))
    expected = sigma_sq * np.diagonal(c @ np.linalg.inv(x.T @ x) @ c.T)[1]

    np.testing.assert_allclose(sigma_n[0], expected)
