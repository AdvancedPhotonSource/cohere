"""Live-view machinery for cohere reconstructions.

This module provides:

- ``LiveViewer``: backend-agnostic facade. Existing callers (Qt cohere_gui,
  CLI scripts, tests) construct it the same way as before; default behavior
  is unchanged matplotlib live view.
- ``LiveViewBackend``: abstract base for pluggable rendering backends.
  Subclass and register via :func:`set_default_live_backend` to redirect all
  live-view output (e.g. a Jupyter consumer ships PNG bytes over a queue;
  a PyVista backend renders 3D volumes).
- ``MatplotlibBackend``: the existing matplotlib-based viewer logic, lifted
  out of ``LiveViewer`` and made into a backend.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
# import matplotlib
import numpy as np

import cohere_core.utilities.utils as ut
import cohere_core.utilities.dvc_utils as dvut

# matplotlib.use("TkAgg")

def set_lib_from_pkg(pkg):
    global devlib

    # get the lib object
    devlib = ut.get_lib(pkg)


@dvut.use_numpy
def show_3d_slices(arr):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    # Display initial slice
    slice_index = 0
    img = ax.imshow(arr[slice_index], cmap='gray')

    # Create slider axis and widget
    ax_slider = plt.axes((0.2, 0.05, 0.6, 0.03))  # Positioning the slider
    slider = Slider(ax_slider, 'Slice', 0, arr.shape[0] - 1, valinit=slice_index, valstep=1)

    # Update function for slider
    def update(val):
        img.set_data(arr[int(val)])
        fig.canvas.draw_idle()

    # Connect slider to update function
    slider.on_changed(update)

    plt.show()


class LiveViewBackend(ABC):
    """Pluggable renderer for :class:`LiveViewer`.

    Subclass and register via :func:`set_default_live_backend` to redirect
    live reconstruction output. ``update_*`` methods receive numpy arrays
    (the :class:`LiveViewer` facade applies ``@dvut.use_numpy`` upstream).
    """

    def select_singlepeak_data(self, ds_image, support, devlib):
        """Choose the on-device data to ship to :meth:`update_singlepeak`.

        ``Rec.live_operation`` invokes this with the still-on-device 3D
        ``ds_image`` and ``support`` and the active backend lib (numpy /
        cupy / torch). Return any ``(ds_out, support_out)`` pair: a 2D
        slice, a strided downsample, the full volume, or a custom view.
        Returned arrays cross the GPU to the host boundary via ``@use_numpy``
        before reaching :meth:`update_singlepeak`, so smaller returns
        mean cheaper transfers.

        Default: a 2D center-of-mass slice along axis 2.
        """
        com = devlib.center_of_mass(devlib.absolute(ds_image))
        idx = int(com[2])
        sl = (slice(None), slice(None), idx)
        return ds_image[sl], (support[sl] if support is not None else None)

    @abstractmethod
    def update_singlepeak(self, ds_image, errors, support, title=""):
        """Render the current single-peak reconstruction.

        Shape of ``ds_image`` and ``support`` is whatever
        :meth:`select_singlepeak_data` returned (2D or 3D).
        """

    @abstractmethod
    def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
        """Render multipeak Fourier-space state (2D arrays)."""

    @abstractmethod
    def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
        """Render multipeak direct-space state (3D arrays)."""

    @abstractmethod
    def save(self, save_as):
        """Persist the current frame to disk."""

    @abstractmethod
    def block(self):
        """Block until the user dismisses the view, or no-op."""


class MatplotlibBackend(LiveViewBackend):
    """Interactive matplotlib renderer using ``plt.show(block=False)``.

    Suitable for Qt applications and native CLI runs with an interactive
    matplotlib backend. Not suitable for Jupyter inline (figure is consumed
    once and subsequent draws don't refresh).
    """

    def __init__(self, shape=(2, 2), figsize=(12, 13)):
        self.shape = shape
        self.figsize = figsize
        self.fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            *shape, figsize=figsize, layout='constrained'
        )
        self.axs = [ax0, ax1, ax2, ax3]
        plt.show(block=False)
        self.view = None

    def update_singlepeak(self, ds_image, errors, support, title=""):
        [ax.clear() for ax in self.axs]
        plt.suptitle(title)
        self.axs[0].set(title="Amplitude", xticks=[], yticks=[])
        self.axs[0].imshow(np.absolute(ds_image), cmap="gray")
        self.axs[1].set(title="Phase", xticks=[], yticks=[])
        self.axs[1].imshow(np.angle(ds_image), cmap="hsv", interpolation_stage="rgba")
        self.axs[2].set(title="Error", xlabel="Iteration", yscale="log")
        self.axs[2].plot(errors[1:])
        self.axs[3].set(title="Support", xticks=[], yticks=[])
        if support is not None:
            self.axs[3].imshow(support, cmap="gray")
        self.draw()

    def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
        [ax.clear() for ax in self.axs]
        plt.suptitle(title)
        vmax = np.log(max([np.amax(arr+1) for arr in (proj, meas, data)]))

        self.axs[0].set_title("Measurement")
        self.axs[0].imshow(np.log(meas + 1), cmap="magma", clim=(0, vmax))
        self.axs[1].set_title("Projection")
        self.axs[1].imshow(np.log(proj + 1), cmap="magma", clim=(0, vmax))
        self.axs[2].set_title("Mask")
        self.axs[2].imshow(mask, cmap="magma")
        self.axs[3].set_title("Fourier Constraint")
        self.axs[3].imshow(np.log(data + 1), cmap="magma", clim=(0, vmax))
        plt.setp(self.fig.get_axes(), xticks=[], yticks=[])
        self.draw()

    def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
        [ax.clear() for ax in self.axs]
        plt.suptitle(title)

        rho = rho / np.max(rho)
        data = [rho, rho*u0, rho*u1, rho*u2]

        # abcd = ("(a)", "(b)", "(c)", "(d)")
        abcd = ("rho", "u[0]", "u[1]", "u[2]")
        locations = ("left", "right", "left", "right")
        cmaps = ("gray_r", "seismic", "seismic", "seismic")

        for ax, u, txt, cmap, loc in zip(self.axs, data, abcd, cmaps, locations):
            ax.clear()
            if cmap == "gray_r":
                clim = (0, 1)
                contour_color = 'r'
            else:
                u = u - np.mean([np.quantile(u, 0.02), np.quantile(u, 0.98)])
                u = (data[0] > 0) * u
                clim = [-0.05, 0.05]
                contour_color = 'k'
            img = ax.imshow(u, cmap=cmap, vmin=clim[0], vmax=clim[1])
            ax.contour(data[0], [0.65], colors=contour_color)
            ax.text(0.06, 0.92, txt, transform=ax.transAxes, fontsize=16,
                    horizontalalignment="center", verticalalignment="center")
            plt.setp(ax, xticks=[], yticks=[])
            plt.colorbar(img, ax=ax, location=loc)
        self.draw()

    def save(self, save_as):
        save_as = Path(save_as)
        save_as.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(save_as, dpi=300)
        self._reinitialize()

    def _reinitialize(self):
        plt.close(self.fig)
        self.fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
            *self.shape, figsize=self.figsize, layout='constrained')
        self.axs = [ax0, ax1, ax2, ax3]

    @staticmethod
    def draw():
        plt.draw()
        plt.pause(0.15)

    @staticmethod
    def block():
        plt.show()


_default_backend: Optional[LiveViewBackend] = None


def set_default_live_backend(backend: Optional[LiveViewBackend]):
    """Register the backend used by future :class:`LiveViewer` instances
    constructed without an explicit ``backend=`` kwarg. Pass ``None`` to
    fall back to a fresh :class:`MatplotlibBackend` per construction.
    """
    global _default_backend
    _default_backend = backend


def get_default_live_backend() -> Optional[LiveViewBackend]:
    return _default_backend


class LiveViewer:
    """Backend-agnostic dispatcher for the live reconstruction view.

    Constructed by :class:`Rec` when ``live_trigger`` is configured. With no
    ``backend=`` argument and no global backend registered, falls back to
    :class:`MatplotlibBackend`. The ``@dvut.use_numpy`` decorator on the
    update methods converts cupy/torch arrays to numpy at the facade so
    backend implementations always receive numpy.

    Attribute access not defined on the facade (e.g. ``viewer.fig``) is
    forwarded to the underlying backend.
    """

    def __init__(self, shape=(2, 2), figsize=(12, 13), backend=None):
        if backend is None:
            backend = _default_backend
        if backend is None:
            backend = MatplotlibBackend(shape=shape, figsize=figsize)
        self._backend = backend

    @dvut.use_numpy
    def update_singlepeak(self, ds_image, errors, support, title=""):
        self._backend.update_singlepeak(ds_image, errors, support, title)

    @dvut.use_numpy
    def update_multipeak_fourier(self, proj, mask, meas, data, title=""):
        self._backend.update_multipeak_fourier(proj, mask, meas, data, title)

    @dvut.use_numpy
    def update_multipeak_direct(self, rho, u0, u1, u2, title=""):
        self._backend.update_multipeak_direct(rho, u0, u1, u2, title)

    def save(self, save_as):
        self._backend.save(save_as)

    def block(self):
        self._backend.block()

    def __getattr__(self, name):
        return getattr(self._backend, name)
