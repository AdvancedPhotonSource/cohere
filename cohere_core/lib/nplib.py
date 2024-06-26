from cohere_core.lib.cohlib import cohlib
import numpy as np
import scipy.ndimage as ndi
import scipy.signal as sig


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

    def astype(arr, dtype):
        return arr.astype(dtype=dtype)

    def size(arr):
        return arr.size

    def hasnan(arr):
        return np.any(np.isnan(arr))

    def copy(arr):
        return np.copy(arr)

    def random(shape, **kwargs):
        import time
        import os

        # return np.random.rand(*shape)
        #rng = np.random.default_rng((int(time.time()* 10000000) + os.getpid()))
        np.random.seed((int(time.time()  * os.getpid()) + os.getpid()) % 2**31)
        r = np.random.rand(*shape)
        return r
        #return rng.random(*shape).astype(float)

    def roll(arr, sft, axis=None):
        if type(sft) != list:
            sft = [sft]
        if axis is None:
            axis = list(range(len(sft)))
        sft = [int(s) for s in sft]
        return np.roll(arr, sft, axis=axis)

    def shift(arr, sft):
        return ndi.shift(arr, sft)

    def fftshift(arr):
        return np.fft.fftshift(arr)

    def ifftshift(arr):
        return np.fft.ifftshift(arr)

    def fft(arr):
        return np.fft.fftn(arr, norm='forward')

    def ifft(arr):
        return np.fft.ifftn(arr, norm='forward')

    def fftconvolve(arr1, kernel):
        return ndi.convolve(arr1, kernel)

    def correlate(arr1, arr2, mode='same', method='auto'):
        return sig.correlate(arr1, arr2, mode, method)

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

    def expand_dims(arr, axis):
        return np.expand_dims(arr, axis)

    def squeeze(arr):
        return np.squeeze(arr)

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
        return ndi.gaussian_filter(arr, sigma)

    def center_of_mass(inarr):
        return ndi.center_of_mass(np.absolute(inarr))

    def meshgrid(*xi):
        return np.meshgrid(*xi)

    def exp(arr):
        return np.exp(arr)

    def conj(arr):
        return np.conj(arr)

    def array_equal(arr1, arr2):
        return np.array_equal(arr1, arr2)
        # print(nplib.real(nplib.fft(nplib.gaussian((3,4,5),(3.0,4.0,5.0)))))

    def cos(arr):
        return np.cos(arr)

    def linspace(start, stop, num):
        return np.linspace(start, stop, num)

    def clip(arr, min, max=None):
        return np.clip(arr, min, max)

    def diff(arr, axis=None, prepend=0):
        return np.diff(arr, axis=axis, prepend=prepend)

    def gradient(arr, dx=1):
        return np.gradient(arr, dx)

    def argmin(arr, axis=None):
        return np.argmin(arr, axis)

    def take_along_axis(a, indices, axis):
        return np.take_along_axis(a, indices, axis)

    def moveaxis(arr, source, dest):
        return np.moveaxis(arr, source, dest)

    def lstsq(A, B):
        return np.linalg.lstsq(A, B, rcond=None)

    def zeros(shape):
        return np.zeros(shape)

    def indices(dims):
        return np.indices(dims)

    def concatenate(tup, axis=0):
        return np.concatenate(tup, axis)

    def amin(arr):
        return np.amin(arr)

    def affine_transform(arr, matrix, order=3, offset=0):
        raise NotImplementedError

    def pad(arr, padding):
        raise NotImplementedError

    def histogram2d(meas, rec, n_bins=100, log=False):
        raise NotImplementedError

    def calc_nmi(hgram):
        raise NotImplementedError

    def calc_ehd(hgram):
        raise NotImplementedError

    def clean_default_mem():
        pass
