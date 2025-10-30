# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

from cohere_core.lib.cohlib import cohlib
import cupy as cp
import numpy as np
import cupyx.scipy.stats as stats
import cupyx.scipy.ndimage as sc
import cupyx.scipy.special as sp
import cupyx.scipy.signal as sig
import cupyx.scipy.fft as fft


class cplib(cohlib):
    @staticmethod
    def array(obj):
        return cp.array(obj)

    @staticmethod
    def dot(arr1, arr2):
        return cp.dot(arr1, arr2)

    @staticmethod
    def cross(arr1, arr2):
        return cp.cross(arr1, arr2)

    @staticmethod
    def set_device(dev_id):
        if dev_id != -1:
            cp.cuda.Device(dev_id).use()

    @staticmethod
    def to_numpy(arr):
        return cp.asnumpy(arr)

    @staticmethod
    def from_numpy(arr, **kwargs):
        return cp.array(arr)

    @staticmethod
    def save(filename, arr):
        cp.save(filename, arr)

    @staticmethod
    def load(filename, **kwargs):
        return cp.load(filename, allow_pickle=True)

    @staticmethod
    def dtype(arr):
        return arr.dtype

    @staticmethod
    def astype(arr, dtype):
        return arr.astype(dtype=dtype)

    @staticmethod
    def reshape(arr, shape):
        return cp.reshape(arr, shape)

    @staticmethod
    def size(arr):
        return arr.size

    @staticmethod
    def next_fast_len(target):
        return fft.next_fast_len(target)

    @staticmethod
    def hasnan(arr):
        return cp.any(cp.isnan(arr))

    @staticmethod
    def nan_to_num(arr, **kwargs):
        return cp.nan_to_num(arr, **kwargs)

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
        return fft.fftshift(arr)

    @staticmethod
    def ifftshift(arr):
        return fft.ifftshift(arr)

    @staticmethod
    def fft(arr, norm='forward'):
        return fft.fftn(arr, norm=norm)

    @staticmethod
    def ifft(arr, norm='forward'):
        return fft.ifftn(arr, norm=norm)

    @staticmethod
    def fftconvolve(arr1, kernel):
        return sig.fftconvolve(arr1, kernel, mode='same')

    @staticmethod
    def correlate(arr1, arr2, mode='same', method='fft'):
        return sig.correlate(arr1, arr2, mode, method)

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
    def amin(arr):
        return cp.amin(arr)

    @staticmethod
    def argmax(arr, axis=None):
        return cp.argmax(arr, axis)

    @staticmethod
    def argmin(arr, axis=None):
        return cp.argmin(arr, axis)

    @staticmethod
    def unravel_index(indices, shape):
        return cp.unravel_index(indices, shape)

    @staticmethod
    def ravel(arr):
        return cp.ravel(arr)

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
    def entropy(arr):
        return stats.entropy(arr)

    @staticmethod
    def gaussian_filter(arr, sigma, **kwargs):
        return sc.gaussian_filter(arr, sigma, **kwargs)

    @staticmethod
    def median_filter(arr, size, **kwargs):
        return sc.median_filter(arr, size)

    @staticmethod
    def uniform_filter(arr, size, **kwargs):
        return sc.uniform_filter(arr, size)

    @staticmethod
    def binary_erosion(arr, **kwargs):
        return sc.binary_erosion(arr, iterations=1)

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
    def geomspace(start, stop, num):
        return cp.geomspace(start, stop, num)

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
    def stack(tup):
        return cp.stack(tup)

    @staticmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        return sc.affine_transform(arr, matrix, order=order, offset=offset, prefilter=True)

    @staticmethod
    def pad(arr, padding):
        return cp.pad(arr, padding)

    @staticmethod
    def histogram2d(arr1, arr2, bins):
        return cp.histogram2d(cp.ravel(arr1), cp.ravel(arr2), bins)[0]

    # @staticmethod
    # def calc_nmi(hgram):
    #     h0 = stats.entropy(cp.sum(hgram, axis=0))
    #     h1 = stats.entropy(cp.sum(hgram, axis=1))
    #     h01 = stats.entropy(cp.reshape(hgram, -1))
    #     return (h0 + h1) / h01
    #
    @staticmethod
    def log(arr):
        return cp.log(arr)

    @staticmethod
    def log10(arr):
        return cp.log10(arr)

    @staticmethod
    def xlogy(x, y=None):
        if y is None:
            y = x
        return sp.xlogy(x, y)

    @staticmethod
    def mean(arr):
        return cp.mean(arr)

    @staticmethod
    def median(arr):
        return cp.median(arr)

    # @staticmethod
    # def calc_ehd(hgram):
    #     n = hgram.shape[0] * 1j
    #     x, y = cp.mgrid[0:1:n, 0:1:n]
    #     return cp.sum(hgram * cp.abs(x - y)) / cp.sum(hgram)
    #
    # @staticmethod
    # def integrate_jacobian(jacobian, dx=1):
    #     nx, ny, nz, _, _ = jacobian.shape
    #     u = cp.zeros((nx, ny, nz, 3))
    #     for ax in range(3):
    #         u = u + dx * cp.cumsum(jacobian[:, :, :, ax, :], axis=ax)
    #     return u
    #
    @staticmethod
    def clean_default_mem():
        cp._default_memory_pool.free_all_blocks()
        cp._default_pinned_memory_pool.free_all_blocks()
