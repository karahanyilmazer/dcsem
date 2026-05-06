"""Regression tests for ``dcsem.plotting.plot_dcm_graph``.

The function renders DCM connectivity diagrams for presentations and
thesis figures. These tests pin behaviour we don't want to silently
break — supported ROI counts, output formats, and the kwargs that
toggle major elements (self-connections, inputs).
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from dcsem import plot_dcm_graph
from dcsem.models import DCM, TwoLayerDCM
from dcsem.utils import create_A_matrix, create_C_matrix


def _make_two_roi_dcm() -> DCM:
    A = create_A_matrix(
        2,
        1,
        ["R0,L0->R1,L0=0.5", "R1,L0->R0,L0=0.4"],
        self_connections=-1,
    )
    C = create_C_matrix(2, 1, ["R0,L0=1.0", "R1,L0=0.5"])
    return DCM(2, params={"A": A, "C": C})


def test_returns_fig_and_ax():
    dcm = _make_two_roi_dcm()
    fig, ax = plot_dcm_graph(dcm)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_two_roi_renders_expected_text_labels():
    """Node labels x_0, x_1 and at least one a_ij appear in the axes."""
    dcm = _make_two_roi_dcm()
    fig, ax = plot_dcm_graph(dcm)
    texts = [t.get_text() for t in ax.texts]
    assert any("x_{0}" in t for t in texts)
    assert any("x_{1}" in t for t in texts)
    assert any(("a_{01}" in t) or ("a_{10}" in t) for t in texts)
    assert any("c_{0}" in t for t in texts)
    assert any("u(t)" in t for t in texts)
    plt.close(fig)


def test_three_roi_layout_runs_without_error():
    """3 ROIs should layout on a triangle without exception."""
    A = create_A_matrix(
        3,
        1,
        ["R0,L0->R1,L0=0.4", "R1,L0->R2,L0=0.3", "R2,L0->R0,L0=-0.2"],
        self_connections=-1,
    )
    C = create_C_matrix(3, 1, ["R0,L0=1.0"])
    dcm = DCM(3, params={"A": A, "C": C})
    fig, _ = plot_dcm_graph(dcm)
    plt.close(fig)


def test_disabling_self_connections_removes_a_ii_labels():
    dcm = _make_two_roi_dcm()  # self_connections=-1, so a_ii are non-zero
    fig, ax = plot_dcm_graph(dcm, show_self_connections=False)
    texts = [t.get_text() for t in ax.texts]
    assert not any(("a_{00}" in t) or ("a_{11}" in t) for t in texts)
    plt.close(fig)


def test_disabling_inputs_removes_c_i_labels():
    dcm = _make_two_roi_dcm()
    fig, ax = plot_dcm_graph(dcm, show_inputs=False)
    texts = [t.get_text() for t in ax.texts]
    assert not any("c_{0}" in t for t in texts)
    assert not any("u(t)" in t for t in texts)
    plt.close(fig)


def test_zero_C_entries_skip_input_arrows():
    """A ROI with C[i] == 0 must not get a c_i arrow."""
    A = create_A_matrix(2, 1, ["R0,L0->R1,L0=0.5"], self_connections=-1)
    C = create_C_matrix(2, 1, ["R0,L0=1.0", "R1,L0=0.0"])
    dcm = DCM(2, params={"A": A, "C": C})
    fig, ax = plot_dcm_graph(dcm)
    texts = [t.get_text() for t in ax.texts]
    assert any("c_{0}" in t for t in texts)
    assert not any("c_{1}" in t for t in texts)
    plt.close(fig)


def test_save_path_writes_svg(tmp_path):
    """SVG output is the headline use case (presentations, thesis)."""
    dcm = _make_two_roi_dcm()
    target = tmp_path / "dcm.svg"
    fig, _ = plot_dcm_graph(dcm, save_path=target)
    assert target.exists()
    assert target.stat().st_size > 0
    # Crude SVG sanity check
    assert target.read_text(errors="ignore").startswith("<?xml")
    plt.close(fig)


def test_save_path_writes_png(tmp_path):
    """PNG path also works (raster output for slide previews)."""
    dcm = _make_two_roi_dcm()
    target = tmp_path / "dcm.png"
    fig, _ = plot_dcm_graph(dcm, save_path=target)
    assert target.exists()
    assert target.stat().st_size > 0
    plt.close(fig)


def test_works_with_two_layer_dcm():
    """TwoLayerDCM exposes the same .p.A / .p.C attributes; should render."""
    A = create_A_matrix(1, 2, self_connections=-1)
    C = create_C_matrix(1, 2, ["R0,L0=1.0", "R0,L1=1.0"])
    ldcm = TwoLayerDCM(1, params={"A": A, "C": C, "l_d": 0.5})
    fig, _ = plot_dcm_graph(ldcm)
    plt.close(fig)


def test_threshold_skips_small_connections():
    """Connections below threshold should not produce a_ij labels."""
    A = create_A_matrix(2, 1, ["R0,L0->R1,L0=1e-15"], self_connections=-1)
    C = create_C_matrix(2, 1, ["R0,L0=1.0"])
    dcm = DCM(2, params={"A": A, "C": C})
    fig, ax = plot_dcm_graph(dcm)
    texts = [t.get_text() for t in ax.texts]
    # 1e-15 is below default threshold 1e-12
    assert not any("a_{01}" in t for t in texts)
    plt.close(fig)


def test_non_square_A_raises():
    """Bad A shape should fail loudly, not produce a garbled diagram."""

    class FakeDCM:
        class P:
            A = np.ones((2, 3))
            C = np.zeros(2)

        p = P()

    with pytest.raises(ValueError, match="square"):
        plot_dcm_graph(FakeDCM())
