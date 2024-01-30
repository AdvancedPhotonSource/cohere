from cohere_core.lib.cohlib import cohlib
import cupy as cp
import numpy as np
import cupyx.scipy.ndimage
import cupyx.scipy.stats as stats
import cupyx.scipy.ndimage as sc


class cplib(cohlib):
    def array(obj):
        return cp.array(obj)

    def dot(arr1, arr2):
        return cp.dot(arr1, arr2)

    def set_device(dev_id):
        if dev_id != -1:
            cp.cuda.Device(dev_id).use()

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return cp.asnumpy(arr)

    def from_numpy(arr):
        return cp.array(arr)

    def save(filename, arr):
        cp.save(filename, arr)

    def load(filename):
        return cp.load(filename, allow_pickle=True)

    def dtype(arr):
        return arr.dtype

    def astype(arr, dtype):
        return arr.astype(dtype=dtype)

    def size(arr):
        return arr.size

    def hasnan(arr):
        return cp.any(cp.isnan(arr))

    def copy(arr):
        return cp.copy(arr)

    def random(shape, **kwargs):
        import time
        import os

        seed = np.array([time.time() * 10000 * os.getpid(), os.getpid()])
        rs = cp.random.RandomState(seed=seed)
        return cp.random.random(shape, dtype=cp.float32) + 1j * cp.random.random(shape, dtype=cp.float32)

    def roll(arr, sft, axis=None):
        if axis is None:
            axis = list(range(len(sft)))
        if type(sft) != list:
            sft = [sft]
        sft = [int(s) for s in sft]
        return cp.roll(arr, sft, axis=axis)

    def shift(arr, sft):
        return sc.fourier_shift(arr, sft)

    def fftshift(arr):
        return cp.fft.fftshift(arr)

    def ifftshift(arr):
        return cp.fft.ifftshift(arr)

    def fft(arr, norm='forward'):
        return cp.fft.fftn(arr, norm=norm)

    def ifft(arr, norm='forward'):
        return cp.fft.ifftn(arr, norm=norm)

    def fftconvolve(arr1, arr2):
        return sc.convolve(arr1, arr2)
        # return cupyx.scipy.signal.convolve(arr1, arr2, mode='same')

    def correlate(arr1, arr2, mode='same', method='fft'):
        from cupyx.scipy.signal import correlate
        return correlate(arr1, arr2, mode, method)

    def where(cond, x, y):
        return cp.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.shape

    def absolute(arr):
        return cp.absolute(arr)

    def sqrt(arr):
        return cp.sqrt(arr)

    def square(arr):
        return cp.square(arr)

    def sum(arr, axis=None):
        sm = cp.sum(arr, axis)
        # if axis is None:
        #     return sm.tolist()
        return sm

    def real(arr):
        return cp.real(arr)

    def imag(arr):
        return cp.imag(arr)

    def amax(arr):
        return cp.amax(arr)

    def argmax(arr, axis=None):
        return cp.argmax(arr, axis)

    def unravel_index(indices, shape):
        return cp.unravel_index(indices, shape)

    def maximum(arr1, arr2):
        return cp.maximum(arr1, arr2)

    def ceil(arr):
        return cp.ceil(arr)

    def fix(arr):
        return cp.fix(arr)

    def round(val):
        return cp.round(val)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return cp.angle(arr)

    def flip(arr, axis=None):
        return cp.flip(arr, axis)

    def tile(arr, rep):
        return cp.tile(arr, rep)

    def full(shape, fill_value, **kwargs):
        return cp.full(shape, fill_value)

    def expand_dims(arr, axis):
        return cp.expand_dims(arr, axis)

    def squeeze(arr):
        return cp.squeeze(arr)

    def gaussian(shape, sigma, **kwargs):
        from functools import reduce
        import operator

        n_el = reduce(operator.mul, shape)
        inarr = cp.zeros((n_el))
        inarr[int(n_el / 2)] = 1.0
        inarr = cp.reshape(inarr, shape)
        gaussian = sc.gaussian_filter(inarr, sigma)
        return gaussian / cp.sum(gaussian)

    def gaussian_filter(arr, sigma, **kwargs):
        return sc.gaussian_filter(arr, sigma)

    def center_of_mass(inarr):
        t = sc.center_of_mass(cp.absolute(inarr))
        return t

    def meshgrid(*xi):
        return cp.meshgrid(*xi)

    def exp(arr):
        return cp.exp(arr)

    def conj(arr):
        return cp.conj(arr)

    def array_equal(arr1, arr2):
        return cp.array_equal(arr1, arr2)

    def cos(arr):
        return cp.cos(arr)

    def linspace(start, stop, num):
        return cp.linspace(start, stop, num)

    def clip(arr, min, max=None):
        return cp.clip(arr, min, max)

    def diff(arr, axis=None, prepend=0):
        return cp.diff(arr, axis=axis, prepend=prepend)

    def gradient(arr, dx=1):
        return cp.gradient(arr, dx)

    def argmin(arr, axis=None):
        return cp.argmin(arr, axis)

    def take_along_axis(a, indices, axis):
        return cp.take_along_axis(a, indices, axis)

    def moveaxis(arr, source, dest):
        return cp.moveaxis(arr, source, dest)

    def lstsq(A, B):
        return cp.linalg.lstsq(A, B, rcond=None)

    def zeros(shape):
        return cp.zeros(shape)

    def indices(dims):
        return cp.indices(dims)

    def concatenate(tup, axis=0):
        return cp.concatenate(tup, axis)

    def amin(arr):
        return cp.amin(arr)

    def affine_transform(arr, matrix, order=3, offset=0):
        return cupyx.scipy.ndimage.affine_transform(arr, matrix, order=order, offset=offset, prefilter=True)

    def pad(arr, padding):
        return cp.pad(arr, padding)

    def histogram2d(meas, rec, n_bins=100, log=False):
        norm = cp.max(meas) / cp.max(rec)
        if log:
            bins = cp.logspace(cp.log10(1), cp.log10(cp.max(meas)), n_bins+1)
        else:
            bins = n_bins
        return cp.histogram2d(cp.ravel(meas), cp.ravel(norm*rec), bins)[0]

    def calc_nmi(hgram):
        h0 = stats.entropy(np.sum(hgram, axis=0))
        h1 = stats.entropy(np.sum(hgram, axis=1))
        h01 = stats.entropy(np.reshape(hgram, -1))
        return (h0 + h1) / h01

    def calc_ehd(hgram):
        n = hgram.shape[0] * 1j
        x, y = cp.mgrid[0:1:n, 0:1:n]
        return cp.sum(hgram * cp.abs(x - y)) / cp.sum(hgram)

    def clean_default_mem():
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()
