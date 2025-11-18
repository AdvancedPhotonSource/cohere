from pathlib import Path

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


class LiveViewer:
    def __init__(self, shape=(2, 2), figsize=(12,13)):
        self.shape = shape
        self.figsize = figsize
        self.fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(*shape, figsize=figsize, layout='constrained')
        self.axs = [ax0, ax1, ax2, ax3]
        plt.show(block=False)
        self.view = None

    @dvut.use_numpy
    def update_singlepeak(self, ds_image, errors, support, title=""):
        [ax.clear() for ax in self.axs]
        plt.suptitle(title)
        qtr = ds_image.shape[0] // 4
        img = ds_image[qtr:-qtr, qtr:-qtr]
        self.axs[0].set(title="Amplitude", xticks=[], yticks=[])
        self.axs[0].imshow(np.absolute(img), cmap="gray")
        self.axs[1].set(title="Phase", xticks=[], yticks=[])
        self.axs[1].imshow(np.angle(img), cmap="hsv", interpolation_stage="rgba")
        self.axs[2].set(title="Error", xlabel="Iteration", yscale="log")
        self.axs[2].plot(errors[1:])
        self.axs[3].set(title="Support", xticks=[], yticks=[])
        self.axs[3].imshow(support[qtr:-qtr, qtr:-qtr], cmap="gray")
        self.draw()

    @dvut.use_numpy
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

    @dvut.use_numpy
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
            ax.text(0.06, 0.92, txt, transform=ax.transAxes, fontsize=16, horizontalalignment="center",
                    verticalalignment="center")
            plt.setp(ax, xticks=[], yticks=[])
            plt.colorbar(img, ax=ax, location=loc)
        self.draw()

    def save(self, save_as):
        save_as = Path(save_as)
        save_as.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(save_as, dpi=300)
        self.reinitialize()

    def reinitialize(self):
        plt.close(self.fig)
        self.fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(*self.shape, figsize=self.figsize, layout='constrained')
        self.axs = [ax0, ax1, ax2, ax3]

    @staticmethod
    def draw():
        plt.draw()
        plt.pause(0.15)

    @staticmethod
    def block():
        plt.show()
