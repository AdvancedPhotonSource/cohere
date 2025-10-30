# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import abc
class cohlib(metaclass=abc.ABCMeta):
    # Interface
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'array') and
                callable(subclass.array) and
                hasattr(subclass, 'dot') and
                callable(subclass.dot) and
                hasattr(subclass, 'cross') and
                callable(subclass.cross) and
                hasattr(subclass, 'set_device') and
                callable(subclass.set_device) and
                hasattr(subclass, 'to_numpy') and
                callable(subclass.to_numpy) and
                hasattr(subclass, 'save') and
                callable(subclass.save) and
                hasattr(subclass, 'load') and
                callable(subclass.load) and
                hasattr(subclass, 'from_numpy') and
                callable(subclass.from_numpy) and
                hasattr(subclass, 'dtype') and
                callable(subclass.dtype) and
                hasattr(subclass, 'astype') and
                callable(subclass.astype) and
                hasattr(subclass, 'reshape') and
                callable(subclass.reshape) and
                hasattr(subclass, 'size') and
                callable(subclass.size) and
                hasattr(subclass, 'next_fast_len') and
                callable(subclass.next_fast_len) and
                hasattr(subclass, 'hasnan') and
                callable(subclass.hasnan) and
                hasattr(subclass, 'copy') and
                callable(subclass.copy) and
                hasattr(subclass, 'roll') and
                callable(subclass.roll) and
                hasattr(subclass, 'shift') and
                callable(subclass.shift) and
                hasattr(subclass, 'fftshift') and
                callable(subclass.fftshift) and
                hasattr(subclass, 'ifftshift') and
                callable(subclass.ifftshift) and
                hasattr(subclass, 'fft') and
                callable(subclass.fft) and
                hasattr(subclass, 'ifft') and
                callable(subclass.ifft) and
                hasattr(subclass, 'fftconvolve') and
                callable(subclass.fftconvolve) and
                hasattr(subclass, 'correlate') and
                callable(subclass.correlate) and
                hasattr(subclass, 'where') and
                callable(subclass.where) and
                hasattr(subclass, 'dims') and
                callable(subclass.dims) and
                hasattr(subclass, 'absolute') and
                callable(subclass.absolute) and
                hasattr(subclass, 'square') and
                callable(subclass.square) and
                hasattr(subclass, 'sqrt') and
                callable(subclass.sqrt) and
                hasattr(subclass, 'sum') and
                callable(subclass.sum) and
                hasattr(subclass, 'real') and
                callable(subclass.real) and
                hasattr(subclass, 'imag') and
                callable(subclass.imag) and
                hasattr(subclass, 'amax') and
                callable(subclass.amax) and
                hasattr(subclass, 'amin') and
                callable(subclass.amin) and
                hasattr(subclass, 'argmax') and
                callable(subclass.argmax) and
                hasattr(subclass, 'argmin') and
                callable(subclass.argmin) and
                hasattr(subclass, 'unravel_index') and
                callable(subclass.unravel_index) and
                hasattr(subclass, 'ravel') and
                callable(subclass.ravel) and
                hasattr(subclass, 'maximum') and
                callable(subclass.maximum) and
                hasattr(subclass, 'ceil') and
                callable(subclass.ceil) and
                hasattr(subclass, 'fix') and
                callable(subclass.fix) and
                hasattr(subclass, 'round') and
                callable(subclass.round) and
                hasattr(subclass, 'random') and
                callable(subclass.random) and
                hasattr(subclass, 'full') and
                callable(subclass.full) and
                hasattr(subclass, 'print') and
                callable(subclass.print) and
                hasattr(subclass, 'angle') and
                callable(subclass.angle) and
                hasattr(subclass, 'flip') and
                callable(subclass.flip) and
                hasattr(subclass, 'tile') and
                callable(subclass.tile) and
                hasattr(subclass, 'expand_dims') and
                callable(subclass.expand_dims) and
                hasattr(subclass, 'squeeze') and
                callable(subclass.squeeze) and
                hasattr(subclass, 'gaussian_filter') and
                callable(subclass.gaussian_filter) and
                hasattr(subclass, 'center_of_mass') and
                callable(subclass.center_of_mass) and
                hasattr(subclass, 'meshgrid') and
                callable(subclass.meshgrid) and
                hasattr(subclass, 'exp') and
                callable(subclass.exp) and
                hasattr(subclass, 'conj') and
                callable(subclass.conj) and
                hasattr(subclass, 'array_equal') and
                callable(subclass.array_equal) and
                hasattr(subclass, 'cos') and
                callable(subclass.cos) and
                hasattr(subclass, 'linspace') and
                callable(subclass.linspace) and
                hasattr(subclass, 'geomspace') and
                callable(subclass.geomspace) and
                hasattr(subclass, 'clip') and
                callable(subclass.clip) and
                hasattr(subclass, 'gradient') and
                callable(subclass.gradient) and
                hasattr(subclass, 'take_along_axis') and
                callable(subclass.take_along_axis) and
                hasattr(subclass, 'moveaxis') and
                callable(subclass.moveaxis) and
                hasattr(subclass, 'lstsq') and
                callable(subclass.lstsq) and
                hasattr(subclass, 'zeros') and
                callable(subclass.zeros) and
                hasattr(subclass, 'indices') and
                callable(subclass.indices) and
                hasattr(subclass, 'diff') and
                callable(subclass.diff) and
                hasattr(subclass, 'concatenate') and
                callable(subclass.concatenate) and
                hasattr(subclass, 'nan_to_num') and
                callable(subclass.nan_to_num) and
                hasattr(subclass, 'entropy') and
                callable(subclass.entropy) and
                hasattr(subclass, 'median_filter') and
                callable(subclass.median_filter) and
                hasattr(subclass, 'uniform_filter') and
                callable(subclass.uniform_filter) and
                hasattr(subclass, 'binary_erosion') and
                callable(subclass.binary_erosion) and
                hasattr(subclass, 'stack') and
                callable(subclass.stack) and
                hasattr(subclass, 'affine_transform') and
                callable(subclass.affine_transform) and
                hasattr(subclass, 'pad') and
                callable(subclass.pad) and
                hasattr(subclass, 'histogram2d') and
                callable(subclass.histogram2d) and
                hasattr(subclass, 'log') and
                callable(subclass.log) and
                hasattr(subclass, 'log10') and
                callable(subclass.log10) and
                hasattr(subclass, 'xlogy') and
                callable(subclass.xlogy) and
                hasattr(subclass, 'mean') and
                callable(subclass.mean) and
                hasattr(subclass, 'median') and
                callable(subclass.median) and
                hasattr(subclass, 'clean_default_mem') and
                callable(subclass.clean_default_mem) or
                NotImplemented)

    @staticmethod
    @abc.abstractmethod
    def array(obj):
        pass

    @staticmethod
    @abc.abstractmethod
    def dot(arr1, arr2):
        pass

    @staticmethod
    @abc.abstractmethod
    def cross(arr1, arr2):
        pass

    @staticmethod
    @abc.abstractmethod
    def set_device(dev_id):
        pass

    @staticmethod
    @abc.abstractmethod
    def to_numpy(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def save(filename, arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def load(filename, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def from_numpy(arr, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def dtype(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def astype(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def reshape(arr, shape):
        pass

    @staticmethod
    @abc.abstractmethod
    def size(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def next_fast_len(target):
        pass

    @staticmethod
    @abc.abstractmethod
    def hasnan(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def nan_to_num(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def copy(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def roll(arr, sft, axis):
        pass

    @staticmethod
    @abc.abstractmethod
    def shift(arr, sft):
        pass

    @staticmethod
    @abc.abstractmethod
    def fftshift(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def ifftshift(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def fft(arr, norm='forward'):
        pass

    @staticmethod
    @abc.abstractmethod
    def ifft(arr, norm='forward'):
        pass

    @staticmethod
    @abc.abstractmethod
    def fftconvolve(arr1, arr2):
        pass

    @staticmethod
    @abc.abstractmethod
    def correlate(arr1, arr2, mode='same', method='fft'):
        pass

    @staticmethod
    @abc.abstractmethod
    def where(cond, x, y):
        pass

    @staticmethod
    @abc.abstractmethod
    def dims(arr):
        # get array dimensions
        pass

    @staticmethod
    @abc.abstractmethod
    def absolute(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def square(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def sqrt(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def sum(arr, axis=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def real(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def imag(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def amax(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def argmax(arr, axis=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def unravel_index(indices, shape):
        pass

    @staticmethod
    @abc.abstractmethod
    def ravel(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def maximum(arr1, arr2):
        pass

    @staticmethod
    @abc.abstractmethod
    def ceil(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def fix(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def round(val):
        pass

    @staticmethod
    @abc.abstractmethod
    def random(shape, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def full(shape, fill_value, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def print(arr, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def angle(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def flip(arr, axis):
        pass

    @staticmethod
    @abc.abstractmethod
    def tile(arr, rep):
        pass

    @staticmethod
    @abc.abstractmethod
    def expand_dims(arr, axis):
        pass

    @staticmethod
    @abc.abstractmethod
    def squeeze(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def entropy(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def gaussian_filter(arr, sigma, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def median_filter(arr, size, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def uniform_filter(arr, size, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def binary_erosion(arr, **kwargs):
        pass

    @staticmethod
    @abc.abstractmethod
    def center_of_mass(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def meshgrid(*xi):
        pass

    @staticmethod
    @abc.abstractmethod
    def exp(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def conj(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def array_equal(arr1, arr2):
        pass

    @staticmethod
    @abc.abstractmethod
    def cos(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def linspace(start, stop, num):
        pass

    @staticmethod
    @abc.abstractmethod
    def geomspace(start, stop, num):
        pass

    @staticmethod
    @abc.abstractmethod
    def diff(arr, axis=None, prepend=0):
        pass

    @staticmethod
    @abc.abstractmethod
    def clip(arr, min, max=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def gradient(arr, dx=1):
        pass

    @staticmethod
    @abc.abstractmethod
    def argmin(arr, axis=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def take_along_axis(a, indices, axis):
        pass

    @staticmethod
    @abc.abstractmethod
    def moveaxis(arr, source, dest):
        pass

    @staticmethod
    @abc.abstractmethod
    def lstsq(A, B):
        pass

    @staticmethod
    @abc.abstractmethod
    def zeros(shape):
        pass

    @staticmethod
    @abc.abstractmethod
    def indices(dims):
        pass

    @staticmethod
    @abc.abstractmethod
    def concatenate(tup, axis=0):
        pass

    @staticmethod
    @abc.abstractmethod
    def stack(tup):
        pass

    @staticmethod
    @abc.abstractmethod
    def amin(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        pass

    @staticmethod
    @abc.abstractmethod
    def pad(arr, padding):
        pass

    @staticmethod
    @abc.abstractmethod
    def histogram2d(arr1, arr2, bins):
        pass

    @staticmethod
    @abc.abstractmethod
    def log(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def log10(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def xlogy(arr, y=None):
        pass

    @staticmethod
    @abc.abstractmethod
    def mean(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def median(arr):
        pass

    @staticmethod
    @abc.abstractmethod
    def clean_default_mem():
        pass
