import numpy as np
import pytest

import cohere_core.controller.phasing as phasing


class FakeDevLib:
    def __init__(self):
        self.device_set = None

    def set_device(self, device_id):
        self.device_set = device_id
        return f"gpu:{device_id}"

    def load(self, path, device=None):
        return np.ones((4, 4, 4), dtype=np.complex64)

    def from_numpy(self, arr, device=None):
        return np.array(arr)

    def fftshift(self, arr):
        return arr

    def ifftshift(self, arr):
        return arr

    def dims(self, arr):
        return arr.shape

    def copy(self, arr):
        return np.copy(arr)

    def random(self, dims, dtype=None, device=None):
        dtype = np.dtype(dtype) if dtype is not None else np.complex64
        if np.issubdtype(dtype, np.complexfloating):
            return np.ones(dims, dtype=dtype) + 1j * np.ones(dims, dtype=dtype)
        return np.ones(dims, dtype=np.float32)

    def full(self, shape, fill_value, device=None):
        return np.full(shape, fill_value)

    def amax(self, arr):
        return np.max(arr)

    def absolute(self, arr):
        return np.abs(arr)

    def where(self, cond, x, y):
        return np.where(cond, x, y)

    def ifft(self, arr):
        return arr

    def fft(self, arr):
        return arr

    def hasnan(self, arr):
        return np.isnan(arr).any()

    def to_numpy(self, arr):
        return np.array(arr)

    def save(self, path, arr):
        pass

    def gaussian_filter(self, arr, sigma):
        return arr

    def array(self, arr):
        return np.array(arr)

    def zeros(self, shape):
        return np.zeros(shape)

    def sum(self, arr):
        return np.sum(arr)

    def mean(self, arr):
        return np.mean(arr)

    def median(self, arr, axis=0):
        return np.median(arr, axis=axis)

    def exp(self, arr):
        return np.exp(arr)

    def dot(self, a, b):
        return np.dot(a, b)

    def angle(self, arr):
        return np.angle(arr)

    def moveaxis(self, arr, src, dst):
        return np.moveaxis(arr, src, dst)

    def stack(self, seq):
        return np.stack(seq)

    def flip(self, arr, axis=None):
        return np.flip(arr, axis=axis)

    def roll(self, arr, shift, axis=None):
        return np.roll(arr, shift, axis=axis)

    def round(self, arr):
        return np.round(arr)

    def center_of_mass(self, arr):
        inds = np.argwhere(arr == arr.max())
        return inds[0]

    def binary_erosion(self, arr):
        return arr

    def median_filter(self, arr, size):
        return arr

    def sqrt(self, arr):
        return np.sqrt(arr)

    def histogram2d(self, a, b, log=False):
        return np.ones((4, 4))

    def argmax(self, arr):
        return np.argmax(arr)

    def take_along_axis(self, arr, indices, axis):
        return np.take_along_axis(arr, indices, axis=axis)

    def gradient(self, arr, voxel_size):
        grads = np.gradient(arr)
        return grads[0], grads[1], grads[2]

    def indices(self, shape):
        return np.indices(shape)

    def lstsq(self, a, b):
        return np.linalg.lstsq(a, b, rcond=None)

    def astype(self, arr, *args):
        return arr.astype(np.float32) if hasattr(arr, "astype") else np.float32(arr)

    def coordinate_dev(self, *args):
        return args

    def argmin(self, arr, axis=0):
        return np.argmin(arr, axis=axis)

    def xlogy(self, x, y):
        y_safe = np.where(y == 0, 1, y)
        return np.where(x == 0, 0, x * np.log(y_safe))

    def cross(self, a, b):
        return np.cross(a, b)


@pytest.fixture
def fake_devlib(monkeypatch):
    fake = FakeDevLib()
    monkeypatch.setattr(phasing, "devlib", fake, raising=False)
    monkeypatch.setattr(phasing, "set_lib_from_pkg", lambda pkg: None)
    return fake


@pytest.fixture
def basic_params():
    return {"algorithm_sequence": "ER"}