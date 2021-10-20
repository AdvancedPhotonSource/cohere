from cohere.lib.cohlib import cohlib
import numpy as np
import scipy
from scipy import ndimage


class nplib(cohlib):
    def array(obj):
        return np.array(obj)

    def dot(arr1, arr2):
        return np.dot(arr1, arr2)

    def set_device(dev_id):
        pass

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return arr

    def load(filename):
        return np.load(filename)

    def from_numpy(arr):
        return arr

    def save(filename, arr):
        np.save(filename, arr)

    def dtype(arr):
        return arr.dtype

    def size(arr):
        return arr.size

    def hasnan(arr):
        return np.any(np.isnan(arr))

    def copy(arr):
        return np.copy(arr)

    def random(shape, **kwargs):
        import time
        import os

        rng = np.random.default_rng(time.time()* 10000 * os.getpid() + os.getpid())
        return rng.random.rand(*shape)

    def fftshift(arr):
        return np.fft.fftshift(arr)

    def ifftshift(arr):
        return np.fft.fftshift(arr)

    def shift(arr, sft):
        sft = [int(s) for s in sft]
        return np.roll(arr, sft)

    def fft(arr):
        return np.fft.fftn(arr)

    def ifft(arr):
        return np.fft.ifftn(arr)

    def fftconvolve(arr1, arr2):
        return scipy.ndimage.convolve(arr1, arr2)

    def where(cond, x, y):
        return np.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.shape

    def absolute(arr):
        return np.absolute(arr)

    def sqrt(arr):
        return np.sqrt(arr)

    def square(arr):
        return np.square(arr)

    def sum(arr, axis=None):
        return np.sum(arr, axis)

    def real(arr):
        return np.real(arr)

    def imag(arr):
        return np.imag(arr)

    def amax(arr):
        return np.amax(arr)

    def unravel_index(indices, shape):
        return np.unravel_index(indices, shape)

    def maximum(arr1, arr2):
        return np.maximum(arr1, arr2)

    def argmax(arr, axis=None):
        return np.argmax(arr, axis)

    def ceil(arr):
        return np.ceil(arr)

    def fix(arr):
        return np.fix(arr)

    def round(val):
        return np.round(val)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return np.angle(arr)

    def flip(arr, axis=None):
        return np.flip(arr, axis)

    def tile(arr, rep):
        return np.tile(arr, rep)

    def full(shape, fill_value, **kwargs):
        return np.full(shape, fill_value)

    def gaussian(shape, sigmas, **kwargs):
        grid = np.full(shape, 1.0)
        for i in range(len(shape)):
            # prepare indexes for tile and transpose
            tile_shape = list(shape)
            tile_shape.pop(i)
            tile_shape.append(1)
            trans_shape = list(range(len(shape) - 1))
            trans_shape.insert(i, len(shape) - 1)

            multiplier = - 0.5 / pow(sigmas[i], 2)
            line = np.linspace(-(shape[i] - 1) / 2.0, (shape[i] - 1) / 2.0, shape[i])
            gi = np.tile(line, tile_shape)
            gi = np.transpose(gi, tuple(trans_shape))
            exponent = np.power(gi, 2) * multiplier
            gi = np.exp(exponent)
            grid = grid * gi

        grid_total = np.sum(grid)
        return grid / grid_total

    def gaussian_filter(arr, sigma, **kwargs):
        return ndimage.gaussian_filter(arr, sigma)

    def center_of_mass(inarr):
        return ndimage.center_of_mass(np.absolute(inarr))

    def meshgrid(*xi):
        return np.meshgrid(*xi)

    def exp(arr):
        return np.exp(arr)

    def conj(arr):
        return np.conj(arr)

    def save(file, arr):
        np.save(file, arr)

        # print(nplib.real(nplib.fft(nplib.gaussian((3,4,5),(3.0,4.0,5.0)))))