import importlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest


@pytest.fixture
def view_utils(monkeypatch):
    mod = importlib.import_module("cohere_core.utilities.view_utils")

    class DummyDevLib:
        def to_numpy(self, x):
            return np.asarray(x)

        def absolute(self, x):
            return np.abs(x)

        def center_of_mass(self, x):
            x = np.asarray(x, dtype=float)
            total = x.sum()
            inds = np.indices(x.shape)
            return tuple((inds[i] * x).sum() / total for i in range(x.ndim))

    monkeypatch.setattr(mod, "devlib", DummyDevLib(), raising=False)
    return mod


@pytest.fixture
def no_gui(monkeypatch, view_utils):
    monkeypatch.setattr(view_utils.plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(view_utils.plt, "pause", lambda *a, **k: None)
    monkeypatch.setattr(view_utils.plt, "draw", lambda *a, **k: None)


def test_set_lib_from_pkg_calls_get_lib(monkeypatch, view_utils):
    sentinel = object()

    monkeypatch.setattr(view_utils.ut, "get_lib", lambda pkg: sentinel)
    view_utils.set_lib_from_pkg("np")

    assert view_utils.devlib is sentinel


def test_show_3d_slices_runs(no_gui, monkeypatch, view_utils):
    arr = np.random.rand(4, 5, 6)

    called = {"subplots": False}

    original_subplots = view_utils.plt.subplots

    def wrapped_subplots(*args, **kwargs):
        called["subplots"] = True
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(view_utils.plt, "subplots", wrapped_subplots)

    view_utils.show_3d_slices(arr)

    assert called["subplots"] is True


def test_liveviewbackend_select_singlepeak_data_returns_center_slice(view_utils):
    class DummyLib:
        def absolute(self, x):
            return np.abs(x)

        def center_of_mass(self, x):
            return (1.0, 1.0, 2.0)

    backend = view_utils.MatplotlibBackend
    base = backend.__mro__[1]() if False else None  # unused, just to avoid ABC direct instantiation

    class ConcreteBackend(view_utils.LiveViewBackend):
        def update_singlepeak(self, ds_image, errors, support, title=""):
            pass

        def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
            pass

        def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
            pass

        def save(self, save_as):
            pass

        def block(self):
            pass

    b = ConcreteBackend()
    ds_image = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    support = np.ones_like(ds_image)

    ds_out, support_out = b.select_singlepeak_data(ds_image, support, DummyLib())

    np.testing.assert_array_equal(ds_out, ds_image[:, :, 2])
    np.testing.assert_array_equal(support_out, support[:, :, 2])


def test_liveviewbackend_select_singlepeak_data_handles_none_support(view_utils):
    class DummyLib:
        def absolute(self, x):
            return np.abs(x)

        def center_of_mass(self, x):
            return (0.0, 0.0, 1.0)

    class ConcreteBackend(view_utils.LiveViewBackend):
        def update_singlepeak(self, ds_image, errors, support, title=""):
            pass

        def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
            pass

        def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
            pass

        def save(self, save_as):
            pass

        def block(self):
            pass

    b = ConcreteBackend()
    ds_image = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    ds_out, support_out = b.select_singlepeak_data(ds_image, None, DummyLib())

    np.testing.assert_array_equal(ds_out, ds_image[:, :, 1])
    assert support_out is None


def test_matplotlibbackend_update_singlepeak(no_gui, view_utils):
    backend = view_utils.MatplotlibBackend()
    ds_image = np.ones((5, 5), dtype=np.complex128) * (1 + 1j)
    errors = [1.0, 0.5, 0.25]
    support = np.ones((5, 5))

    backend.update_singlepeak(ds_image, errors, support, title="singlepeak")

    assert len(backend.axs) == 4
    assert backend.axs[0].get_title() == "Amplitude"
    assert backend.axs[1].get_title() == "Phase"
    assert backend.axs[2].get_title() == "Error"
    assert backend.axs[3].get_title() == "Support"


def test_matplotlibbackend_update_multipeak_fourier(no_gui, view_utils):
    backend = view_utils.MatplotlibBackend()
    proj = np.ones((4, 4))
    mask = np.zeros((4, 4))
    meas = np.full((4, 4), 2.0)
    data = np.full((4, 4), 3.0)

    backend.update_multipeak_fourier(proj, mask, meas, data, title="fourier")

    titles = [ax.get_title() for ax in backend.axs]
    assert titles == ["Measurement", "Projection", "Mask", "Fourier Constraint"]


def test_matplotlibbackend_update_multipeak_direct(no_gui, view_utils):
    backend = view_utils.MatplotlibBackend()
    rho = np.ones((5, 5))
    u0 = np.full((5, 5), 0.01)
    u1 = np.full((5, 5), -0.02)
    u2 = np.zeros((5, 5))

    backend.update_multipeak_direct(rho, u0, u1, u2, title="direct")

    assert len(backend.axs) == 4


def test_matplotlibbackend_save_creates_file_and_reinitializes(no_gui, tmp_path, view_utils):
    backend = view_utils.MatplotlibBackend()
    old_fig = backend.fig

    out = tmp_path / "subdir" / "figure.png"
    backend.save(out)

    assert out.exists()
    assert backend.fig is not old_fig
    assert len(backend.axs) == 4


def test_matplotlibbackend_reinitialize_replaces_figure(no_gui, view_utils):
    backend = view_utils.MatplotlibBackend()
    old_fig = backend.fig

    backend._reinitialize()

    assert backend.fig is not old_fig
    assert len(backend.axs) == 4


def test_set_and_get_default_live_backend(view_utils):
    class DummyBackend:
        pass

    backend = DummyBackend()
    view_utils.set_default_live_backend(backend)

    assert view_utils.get_default_live_backend() is backend

    view_utils.set_default_live_backend(None)
    assert view_utils.get_default_live_backend() is None


def test_liveviewer_uses_explicit_backend(view_utils):
    class DummyBackend:
        def __init__(self):
            self.calls = []

        def update_singlepeak(self, ds_image, errors, support, title=""):
            self.calls.append(("single", ds_image, errors, support, title))

        def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
            self.calls.append(("fourier", proj, mask, meas, data, title))

        def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
            self.calls.append(("direct", rho, u0, u1, u2, title))

        def save(self, save_as):
            self.calls.append(("save", save_as))

        def block(self):
            self.calls.append(("block",))

    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)

    arr = np.ones((2, 2))
    viewer.update_singlepeak(arr, [1.0], arr, title="t1")
    viewer.update_multipeak_fourier(arr, arr, arr, arr, title="t2")
    viewer.update_multipeak_direct(arr, arr, arr, arr, title="t3")
    viewer.save("x.png")
    viewer.block()

    assert backend.calls[0][0] == "single"
    assert backend.calls[1][0] == "fourier"
    assert backend.calls[2][0] == "direct"
    assert backend.calls[3] == ("save", "x.png")
    assert backend.calls[4] == ("block",)


def test_liveviewer_uses_global_default_backend(view_utils):
    class DummyBackend:
        def __init__(self):
            self.called = False

        def update_singlepeak(self, ds_image, errors, support, title=""):
            self.called = True

        def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
            pass

        def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
            pass

        def save(self, save_as):
            pass

        def block(self):
            pass

    backend = DummyBackend()
    view_utils.set_default_live_backend(backend)

    viewer = view_utils.LiveViewer()
    viewer.update_singlepeak(np.ones((2, 2)), [1.0], np.ones((2, 2)))

    assert viewer._backend is backend
    assert backend.called is True

    view_utils.set_default_live_backend(None)


def test_liveviewer_falls_back_to_matplotlibbackend(no_gui, view_utils):
    view_utils.set_default_live_backend(None)
    viewer = view_utils.LiveViewer()

    assert isinstance(viewer._backend, view_utils.MatplotlibBackend)


def test_liveviewer_getattr_forwards_to_backend(view_utils):
    class DummyBackend:
        def __init__(self):
            self.custom_attr = 123

        def update_singlepeak(self, ds_image, errors, support, title=""):
            pass

        def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
            pass

        def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
            pass

        def save(self, save_as):
            pass

        def block(self):
            pass

    viewer = view_utils.LiveViewer(backend=DummyBackend())

    assert viewer.custom_attr == 123


# What this covers
# set_lib_from_pkg
# show_3d_slices
# LiveViewBackend.select_singlepeak_data
# MatplotlibBackend
# update_singlepeak
# update_multipeak_fourier
# update_multipeak_direct
# save
# _reinitialize
# default backend registration helpers
# LiveViewer
# explicit backend
# global default backend
# fallback backend
# delegated attribute access
# Notes
# 1. Use a non-interactive backend
# The line:
#
#
# matplotlib.use("Agg")
# is important for CI/headless environments.
#
# 2. dvut.use_numpy
# Since view_utils uses @dvut.use_numpy, I patched view_utils.devlib with a small stub so decorated functions can safely call devlib.to_numpy(...).
#
# 3. GUI behavior
# The tests don’t verify pixels rendered on screen; they verify:
#
# methods run successfully
# expected titles/state are set
# files are created
# If you want, I can also give you:
#
# a more minimal version,
# a parametrized version,
# or a version split into TestMatplotlibBackend / TestLiveViewer classes.