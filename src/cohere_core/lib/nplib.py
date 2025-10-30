# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

from cohere_core.lib.cohlib import cohlib
import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
import scipy.signal as sig
import scipy.stats as stats

class nplib(cohlib):
    @staticmethod
    def array(obj):
        return np.array(obj)

    @staticmethod
    def dot(arr1, arr2):
        return np.dot(arr1, arr2)

    @staticmethod
    def cross(arr1, arr2):
        return np.cross(arr1, arr2)

    @staticmethod
    def set_device(dev_id):
        pass

    @staticmethod
    def to_numpy(arr):
        return arr

    @staticmethod
    def load(filename, **kwargs):
        return np.load(filename)

    @staticmethod
    def from_numpy(arr, **kwargs):
        return arr

    @staticmethod
    def save(filename, arr):
        np.save(filename, arr)

    @staticmethod
    def dtype(arr):
        return arr.dtype

    @staticmethod
    def astype(arr, dtype):
        return arr.astype(dtype=dtype)

    @staticmethod
    def reshape(arr, shape):
        return np.reshape(arr, shape)

    @staticmethod
    def size(arr):
        return arr.size

    @staticmethod
    def next_fast_len(target):
        return fft.next_fast_len(target)

    @staticmethod
    def nan_to_num(arr, **kwargs):
        return np.nan_to_num(arr, **kwargs)

    @staticmethod
    def hasnan(arr):
        return np.any(np.isnan(arr))

    @staticmethod
    def copy(arr):
        return np.copy(arr)

    @staticmethod
    def random(shape, **kwargs):
        import time
        import os

        # return np.random.rand(*shape)
        #rng = np.random.default_rng((int(time.time()* 10000000) + os.getpid()))
        np.random.seed((int(time.time()  * os.getpid()) + os.getpid()) % 2**31)
        r = np.random.rand(*shape)
        return r
        #return rng.random(*shape).astype(float)

    @staticmethod
    def roll(arr, sft, axis=None):
        if type(sft) != list:
            sft = [sft]
        if axis is None:
            axis = list(range(len(sft)))
        sft = [int(s) for s in sft]
        return np.roll(arr, sft, axis=axis)

    @staticmethod
    def shift(arr, sft):
        return ndi.shift(arr, sft)

    @staticmethod
    def fftshift(arr):
        return np.fft.fftshift(arr)

    @staticmethod
    def ifftshift(arr):
        return np.fft.ifftshift(arr)

    @staticmethod
    def fft(arr):
        return fft.fftn(arr, norm='forward')

    @staticmethod
    def ifft(arr):
        return fft.ifftn(arr, norm='forward')

    @staticmethod
    def fftconvolve(arr1, kernel):
        return sig.fftconvolve(arr1, kernel, mode='same')

    @staticmethod
    def correlate(arr1, arr2, mode='same', method='auto'):
        return sig.correlate(arr1, arr2, mode, method)

    @staticmethod
    def where(cond, x, y):
        return np.where(cond, x, y)

    @staticmethod
    def dims(arr):
        # get array dimensions
        return arr.shape

    @staticmethod
    def absolute(arr):
        return np.absolute(arr)

    @staticmethod
    def sqrt(arr):
        return np.sqrt(arr)

    @staticmethod
    def square(arr):
        return np.square(arr)

    @staticmethod
    def sum(arr, axis=None):
        return np.sum(arr, axis)

    @staticmethod
    def real(arr):
        return np.real(arr)

    @staticmethod
    def imag(arr):
        return np.imag(arr)

    @staticmethod
    def amax(arr):
        return np.amax(arr)

    @staticmethod
    def unravel_index(indices, shape):
        return np.unravel_index(indices, shape)

    @staticmethod
    def ravel(arr):
        return np.ravel(arr)

    @staticmethod
    def maximum(arr1, arr2):
        return np.maximum(arr1, arr2)

    @staticmethod
    def argmax(arr, axis=None):
        return np.argmax(arr, axis)

    @staticmethod
    def ceil(arr):
        return np.ceil(arr)

    @staticmethod
    def fix(arr):
        return np.fix(arr)

    @staticmethod
    def round(val):
        return np.round(val)

    @staticmethod
    def print(arr, **kwargs):
        print(arr)

    @staticmethod
    def angle(arr):
        return np.angle(arr)

    @staticmethod
    def flip(arr, axis=None):
        return np.flip(arr, axis)

    @staticmethod
    def tile(arr, rep):
        return np.tile(arr, rep)

    @staticmethod
    def full(shape, fill_value, **kwargs):
        return np.full(shape, fill_value)

    @staticmethod
    def expand_dims(arr, axis):
        return np.expand_dims(arr, axis)

    @staticmethod
    def squeeze(arr):
        return np.squeeze(arr)

    @staticmethod
    def entropy(arr):
        return stats.entropy(arr)

    @staticmethod
    def gaussian_filter(arr, sigma, **kwargs):
        return ndi.gaussian_filter(arr, sigma, **kwargs)

    @staticmethod
    def median_filter(arr, size, **kwargs):
        return ndi.median_filter(arr, size)

    @staticmethod
    def uniform_filter(arr, size, **kwargs):
        return ndi.uniform_filter(arr, size)

    @staticmethod
    def binary_erosion(arr, **kwargs):
        return ndi.binary_erosion(arr, iterations=1)

    @staticmethod
    def center_of_mass(inarr):
        return ndi.center_of_mass(np.absolute(inarr))

    @staticmethod
    def meshgrid(*xi):
        return np.meshgrid(*xi)

    @staticmethod
    def exp(arr):
        return np.exp(arr)

    @staticmethod
    def conj(arr):
        return np.conj(arr)

    @staticmethod
    def array_equal(arr1, arr2):
        return np.array_equal(arr1, arr2)

    @staticmethod
    def cos(arr):
        return np.cos(arr)

    @staticmethod
    def linspace(start, stop, num):
        return np.linspace(start, stop, num)

    @staticmethod
    def geomspace(start, stop, num):
        return np.geomspace(start, stop, num)

    @staticmethod
    def clip(arr, min, max=None):
        return np.clip(arr, min, max)

    @staticmethod
    def diff(arr, axis=None, prepend=0):
        return np.diff(arr, axis=axis, prepend=prepend)

    @staticmethod
    def gradient(arr, dx=1):
        return np.gradient(arr, dx)

    @staticmethod
    def argmin(arr, axis=None):
        return np.argmin(arr, axis)

    @staticmethod
    def take_along_axis(a, indices, axis):
        return np.take_along_axis(a, indices, axis)

    @staticmethod
    def moveaxis(arr, source, dest):
        return np.moveaxis(arr, source, dest)

    @staticmethod
    def lstsq(A, B):
        return np.linalg.lstsq(A, B, rcond=None)

    @staticmethod
    def zeros(shape):
        return np.zeros(shape)

    @staticmethod
    def indices(dims):
        return np.indices(dims)

    @staticmethod
    def concatenate(tup, axis=0):
        return np.concatenate(tup, axis)

    @staticmethod
    def stack(tup):
        return np.stack(tup)

    @staticmethod
    def amin(arr):
        return np.amin(arr)

    @staticmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        return ndi.affine_transform(arr, matrix, order=order, offset=offset, prefilter=True)

    @staticmethod
    def pad(arr, padding):
        return np.pad(arr, padding)

    @staticmethod
    def histogram2d(arr1, arr2, bins):
        return np.histogram2d(np.ravel(arr1), np.ravel(arr2), bins)[0]

    @staticmethod
    def log(arr):
        return np.log(arr)

    @staticmethod
    def log10(arr):
        return np.log10(arr)

    @staticmethod
    def xlogy(x, y=None):
        if y is None:
            y = x
        return np.xlogy(x, y)

    @staticmethod
    def mean(arr):
        return np.mean(arr)

    @staticmethod
    def median(arr):
        return np.median(arr)

    @staticmethod
    def clean_default_mem():
        pass
