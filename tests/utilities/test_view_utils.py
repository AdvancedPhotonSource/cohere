import numpy as np
import pytest
from pathlib import Path
import cohere_core.utilities.view_utils as view_utils
import cohere_core.utilities.dvc_utils as dvut


class FakeDevLib:
    @staticmethod
    def to_numpy(x):
        return x

@pytest.fixture(autouse=True)
def fake_devlib(monkeypatch):
    monkeypatch.setattr(dvut, "devlib", FakeDevLib(), raising=False)

class DummyBackend(view_utils.LiveViewBackend):
    def __init__(self):
        self.calls = []

    def update_singlepeak(self, ds_image, errors, support, title=""):
        self.calls.append(
            ("singlepeak", ds_image, errors, support, title)
        )

    def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
        self.calls.append(
            ("fourier", proj, mask, meas, data, title)
        )

    def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
        self.calls.append(
            ("direct", rho, u0, u1, u2, title)
        )

    def save(self, save_as):
        self.calls.append(("save", save_as))

    def block(self):
        self.calls.append(("block",))


@pytest.fixture(autouse=True)
def reset_default_backend():
    old = view_utils.get_default_live_backend()
    view_utils.set_default_live_backend(None)
    yield
    view_utils.set_default_live_backend(old)


def test_set_and_get_default_live_backend():
    backend = DummyBackend()
    view_utils.set_default_live_backend(backend)
    assert view_utils.get_default_live_backend() is backend


def test_liveviewer_uses_explicit_backend():
    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)
    assert viewer._backend is backend


def test_liveviewer_uses_default_backend():
    backend = DummyBackend()
    view_utils.set_default_live_backend(backend)

    viewer = view_utils.LiveViewer()
    assert viewer._backend is backend


def test_liveviewer_falls_back_to_matplotlib_backend(monkeypatch):
    class FakeMatplotlibBackend:
        def __init__(self, shape=(2, 2), figsize=(12, 13)):
            self.shape = shape
            self.figsize = figsize

        def save(self, save_as):
            pass

        def block(self):
            pass

    monkeypatch.setattr(view_utils, "MatplotlibBackend", FakeMatplotlibBackend)

    viewer = view_utils.LiveViewer(shape=(1, 1), figsize=(4, 4))
    assert isinstance(viewer._backend, FakeMatplotlibBackend)
    assert viewer._backend.shape == (1, 1)
    assert viewer._backend.figsize == (4, 4)


def test_liveviewer_update_singlepeak_dispatches_to_backend():
    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)

    ds_image = np.ones((4, 4), dtype=np.complex64)
    errors = np.array([0.0, 0.1, 0.05])
    support = np.ones((4, 4))
    title = "test title"

    viewer.update_singlepeak(ds_image, errors, support, title)

    assert len(backend.calls) == 1
    kind, got_ds, got_err, got_support, got_title = backend.calls[0]
    assert kind == "singlepeak"
    np.testing.assert_array_equal(got_ds, ds_image)
    np.testing.assert_array_equal(got_err, errors)
    np.testing.assert_array_equal(got_support, support)
    assert got_title == title


def test_liveviewer_update_multipeak_fourier_dispatches_to_backend():
    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)

    proj = np.ones((3, 3))
    mask = np.zeros((3, 3))
    meas = np.full((3, 3), 2.0)
    data = np.full((3, 3), 3.0)

    viewer.update_multipeak_fourier(proj, mask, meas, data, "fourier title")

    assert len(backend.calls) == 1
    kind, got_proj, got_mask, got_meas, got_data, got_title = backend.calls[0]
    assert kind == "fourier"
    np.testing.assert_array_equal(got_proj, proj)
    np.testing.assert_array_equal(got_mask, mask)
    np.testing.assert_array_equal(got_meas, meas)
    np.testing.assert_array_equal(got_data, data)
    assert got_title == "fourier title"


def test_liveviewer_update_multipeak_direct_dispatches_to_backend():
    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)

    rho = np.ones((3, 3))
    u0 = np.full((3, 3), 0.1)
    u1 = np.full((3, 3), 0.2)
    u2 = np.full((3, 3), 0.3)

    viewer.update_multipeak_direct(rho, u0, u1, u2, "direct title")

    assert len(backend.calls) == 1
    kind, got_rho, got_u0, got_u1, got_u2, got_title = backend.calls[0]
    assert kind == "direct"
    np.testing.assert_array_equal(got_rho, rho)
    np.testing.assert_array_equal(got_u0, u0)
    np.testing.assert_array_equal(got_u1, u1)
    np.testing.assert_array_equal(got_u2, u2)
    assert got_title == "direct title"


def test_liveviewer_save_dispatches_to_backend(tmp_path):
    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)

    out = tmp_path / "image.png"
    viewer.save(out)

    assert backend.calls == [("save", out)]


def test_liveviewer_block_dispatches_to_backend():
    backend = DummyBackend()
    viewer = view_utils.LiveViewer(backend=backend)

    viewer.block()

    assert backend.calls == [("block",)]


def test_liveviewer_getattr_forwards_to_backend():
    backend = DummyBackend()
    backend.custom_attr = "forwarded"

    viewer = view_utils.LiveViewer(backend=backend)

    assert viewer.custom_attr == "forwarded"


def test_select_singlepeak_data_returns_center_slice():
    backend = DummyBackend()

    ds_image = np.zeros((4, 5, 6), dtype=np.complex64)
    support = np.ones((4, 5, 6), dtype=np.float32)

    target_idx = 3
    ds_image[:, :, target_idx] = 7 + 2j
    support[:, :, target_idx] = 9

    class FakeDevLib:
        @staticmethod
        def absolute(arr):
            return np.abs(arr)

        @staticmethod
        def center_of_mass(arr):
            return (1.0, 2.0, float(target_idx))

    out_img, out_support = view_utils.LiveViewBackend.select_singlepeak_data(
        backend, ds_image, support, FakeDevLib
    )

    assert out_img.shape == (4, 5)
    assert out_support.shape == (4, 5)
    np.testing.assert_array_equal(out_img, ds_image[:, :, target_idx])
    np.testing.assert_array_equal(out_support, support[:, :, target_idx])


def test_select_singlepeak_data_handles_none_support():
    backend = DummyBackend()
    ds_image = np.zeros((2, 3, 4), dtype=np.complex64)

    class FakeDevLib:
        @staticmethod
        def absolute(arr):
            return np.abs(arr)

        @staticmethod
        def center_of_mass(arr):
            return (0.0, 0.0, 1.0)

    out_img, out_support = view_utils.LiveViewBackend.select_singlepeak_data(
        backend, ds_image, None, FakeDevLib
    )

    assert out_img.shape == (2, 3)
    assert out_support is None


def test_matplotlibbackend_save_creates_parent_and_reinitializes(monkeypatch, tmp_path):
    backend = view_utils.MatplotlibBackend.__new__(view_utils.MatplotlibBackend)

    class FakeFig:
        def __init__(self):
            self.saved = None

        def savefig(self, path, dpi=300):
            self.saved = (path, dpi)

    fake_fig = FakeFig()
    backend.fig = fake_fig

    called = {"reinit": False}

    def fake_reinitialize():
        called["reinit"] = True

    backend._reinitialize = fake_reinitialize

    out = tmp_path / "nested" / "plot.png"
    backend.save(out)

    assert out.parent.exists()
    assert fake_fig.saved == (Path(out), 300)
    assert called["reinit"] is True


def test_matplotlibbackend_draw_calls_pyplot(monkeypatch):
    calls = []

    monkeypatch.setattr(view_utils.plt, "draw", lambda: calls.append("draw"))
    monkeypatch.setattr(view_utils.plt, "pause", lambda x: calls.append(("pause", x)))

    view_utils.MatplotlibBackend.draw()

    assert calls == ["draw", ("pause", 0.15)]


def test_matplotlibbackend_block_calls_show(monkeypatch):
    calls = []
    monkeypatch.setattr(view_utils.plt, "show", lambda: calls.append("show"))

    view_utils.MatplotlibBackend.block()

    assert calls == ["show"]

# What this tests
# This test suite covers:
#
# set_default_live_backend / get_default_live_backend
# LiveViewer backend selection:
# explicit backend
# default backend
# fallback to MatplotlibBackend
# dispatching from LiveViewer methods to backend methods
# __getattr__ forwarding
# LiveViewBackend.select_singlepeak_data
# MatplotlibBackend.save
# MatplotlibBackend.draw
# MatplotlibBackend.block
# Notes
# 1. GUI-safe testing
# These tests avoid real plotting windows except for one place:
#
# test_matplotlibbackend_save_creates_parent_and_reinitializes uses __new__ so it does not run the real MatplotlibBackend.__init__.
# That keeps tests headless-friendly.
#
# 2. If your CI has matplotlib GUI issues
# You may also want to force a non-interactive backend in your test environment, for example in conftest.py:
#
#
# import matplotlib
# matplotlib.use("Agg")
# 3. If you want stricter tests for decorators
# Because @dvut.use_numpy is applied, if you want to explicitly test conversion behavior, I can also write tests that monkeypatch dvut.use_numpy or simulate non-numpy inputs depending on how that decorator behaves in your codebase.
#
# If you want, I can also provide:
#
# a minimal version of this test file, or
# a version adapted to your project’s existing test style.