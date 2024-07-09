from cohere_core.lib.cohlib import cohlib
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage
import cupyx.scipy.stats as stats
import cupyx.scipy.ndimage as sc


class cplib(cohlib):
    @staticmethod
    def array(obj):
        return cp.array(obj)

    @staticmethod
    def dot(arr1, arr2):
        return cp.dot(arr1, arr2)

    @staticmethod
    def set_device(dev_id):
        if dev_id != -1:
            cp.cuda.Device(dev_id).use()

    @staticmethod
    def set_backend(proc):
        pass

    @staticmethod
    def to_numpy(arr):
        return cp.asnumpy(arr)

    @staticmethod
    def from_numpy(arr):
        return cp.array(arr)

    @staticmethod
    def save(filename, arr):
        cp.save(filename, arr)

    @staticmethod
    def load(filename):
        return cp.load(filename, allow_pickle=True)

    @staticmethod
    def dtype(arr):
        return arr.dtype

    @staticmethod
    def astype(arr, dtype):
        return arr.astype(dtype=dtype)

    @staticmethod
    def size(arr):
        return arr.size

    @staticmethod
    def hasnan(arr):
        return cp.any(cp.isnan(arr))

    @staticmethod
    def copy(arr):
        return cp.copy(arr)

    @staticmethod
    def random(shape, **kwargs):
        import time
        import os

        seed = np.array([time.time() * 10000 * os.getpid(), os.getpid()])
        rs = cp.random.RandomState(seed=seed)
        return cp.random.random(shape, dtype=cp.float32) + 1j * cp.random.random(shape, dtype=cp.float32)

    @staticmethod
    def roll(arr, sft, axis=None):
        if axis is None:
            axis = list(range(len(sft)))
        if type(sft) != list:
            sft = [sft]
        sft = [int(s) for s in sft]
        return cp.roll(arr, sft, axis=axis)

    @staticmethod
    def shift(arr, sft):
        return sc.fourier_shift(arr, sft)

    @staticmethod
    def fftshift(arr):
        return cp.fft.fftshift(arr)

    @staticmethod
    def ifftshift(arr):
        return cp.fft.ifftshift(arr)

    @staticmethod
    def fft(arr, norm='forward'):
        return cp.fft.fftn(arr, norm=norm)

    @staticmethod
    def ifft(arr, norm='forward'):
        return cp.fft.ifftn(arr, norm=norm)

    @staticmethod
    def fftconvolve(arr1, kernel):
        from cupyx.scipy import signal
        return signal.fftconvolve(arr1, kernel, mode='same')

    @staticmethod
    def correlate(arr1, arr2, mode='same', method='fft'):
        from cupyx.scipy.signal import correlate
        return correlate(arr1, arr2, mode, method)

    @staticmethod
    def where(cond, x, y):
        return cp.where(cond, x, y)

    @staticmethod
    def dims(arr):
        # get array dimensions
        return arr.shape

    @staticmethod
    def absolute(arr):
        return cp.absolute(arr)

    @staticmethod
    def sqrt(arr):
        return cp.sqrt(arr)

    @staticmethod
    def square(arr):
        return cp.square(arr)

    @staticmethod
    def sum(arr, axis=None):
        sm = cp.sum(arr, axis)
        # if axis is None:
        #     return sm.tolist()
        return sm

    @staticmethod
    def real(arr):
        return cp.real(arr)

    @staticmethod
    def imag(arr):
        return cp.imag(arr)

    @staticmethod
    def amax(arr):
        return cp.amax(arr)

    @staticmethod
    def argmax(arr, axis=None):
        return cp.argmax(arr, axis)

    @staticmethod
    def unravel_index(indices, shape):
        return cp.unravel_index(indices, shape)

    @staticmethod
    def maximum(arr1, arr2):
        return cp.maximum(arr1, arr2)

    @staticmethod
    def ceil(arr):
        return cp.ceil(arr)

    @staticmethod
    def fix(arr):
        return cp.fix(arr)

    @staticmethod
    def round(val):
        return cp.round(val)

    @staticmethod
    def print(arr, **kwargs):
        print(arr)

    @staticmethod
    def angle(arr):
        return cp.angle(arr)

    @staticmethod
    def flip(arr, axis=None):
        return cp.flip(arr, axis)

    @staticmethod
    def tile(arr, rep):
        return cp.tile(arr, rep)

    @staticmethod
    def full(shape, fill_value, **kwargs):
        return cp.full(shape, fill_value)

    @staticmethod
    def expand_dims(arr, axis):
        return cp.expand_dims(arr, axis)

    @staticmethod
    def squeeze(arr):
        return cp.squeeze(arr)

    @staticmethod
    def gaussian(shape, sigma, **kwargs):
        from functools import reduce
        import operator

        n_el = reduce(operator.mul, shape)
        inarr = cp.zeros((n_el))
        inarr[int(n_el / 2)] = 1.0
        inarr = cp.reshape(inarr, shape)
        gaussian = sc.gaussian_filter(inarr, sigma)
        return gaussian / cp.sum(gaussian)

    @staticmethod
    def gaussian_filter(arr, sigma, **kwargs):
        return sc.gaussian_filter(arr, sigma)

    @staticmethod
    def center_of_mass(inarr):
        t = sc.center_of_mass(cp.absolute(inarr))
        return t

    @staticmethod
    def meshgrid(*xi):
        return cp.meshgrid(*xi)

    @staticmethod
    def exp(arr):
        return cp.exp(arr)

    @staticmethod
    def conj(arr):
        return cp.conj(arr)

    @staticmethod
    def array_equal(arr1, arr2):
        return cp.array_equal(arr1, arr2)

    @staticmethod
    def cos(arr):
        return cp.cos(arr)

    @staticmethod
    def linspace(start, stop, num):
        return cp.linspace(start, stop, num)

    @staticmethod
    def clip(arr, min, max=None):
        return cp.clip(arr, min, max)

    @staticmethod
    def diff(arr, axis=None, prepend=0):
        return cp.diff(arr, axis=axis, prepend=prepend)

    @staticmethod
    def gradient(arr, dx=1):
        return cp.gradient(arr, dx)

    @staticmethod
    def argmin(arr, axis=None):
        return cp.argmin(arr, axis)

    @staticmethod
    def take_along_axis(a, indices, axis):
        return cp.take_along_axis(a, indices, axis)

    @staticmethod
    def moveaxis(arr, source, dest):
        return cp.moveaxis(arr, source, dest)

    @staticmethod
    def lstsq(A, B):
        return cp.linalg.lstsq(A, B, rcond=None)

    @staticmethod
    def zeros(shape):
        return cp.zeros(shape)

    @staticmethod
    def indices(dims):
        return cp.indices(dims)

    @staticmethod
    def concatenate(tup, axis=0):
        return cp.concatenate(tup, axis)

    @staticmethod
    def amin(arr):
        return cp.amin(arr)

    @staticmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        return cupyx.scipy.ndimage.affine_transform(arr, matrix, order=order, offset=offset, prefilter=True)

    def pad(arr, padding):
        return cp.pad(arr, padding)

    @staticmethod
    def histogram2d(meas, rec, n_bins=100, log=False):
        norm = cp.max(meas) / cp.max(rec)
        if log:
            bins = cp.logspace(cp.log10(1), cp.log10(cp.max(meas)), n_bins+1)
        else:
            bins = n_bins
        return cp.histogram2d(cp.ravel(meas), cp.ravel(norm*rec), bins)[0]

    @staticmethod
    def calc_nmi(hgram):
        h0 = stats.entropy(np.sum(hgram, axis=0))
        h1 = stats.entropy(np.sum(hgram, axis=1))
        h01 = stats.entropy(np.reshape(hgram, -1))
        return (h0 + h1) / h01

    @staticmethod
    def calc_ehd(hgram):
        n = hgram.shape[0] * 1j
        x, y = cp.mgrid[0:1:n, 0:1:n]
        return cp.sum(hgram * cp.abs(x - y)) / cp.sum(hgram)

    @staticmethod
    def clean_default_mem():
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()
