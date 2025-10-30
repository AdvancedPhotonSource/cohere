# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

from cohere_core.lib.cohlib import cohlib
import numpy as np
import torch

import sys
import math


class torchlib(cohlib):
    device = "cpu"
    # Interface
    @staticmethod
    def array(obj):
        return torch.Tensor(obj, device=torchlib.device)

    @staticmethod
    def dot(arr1, arr2):
        return torch.dot(arr1, arr2)

    @staticmethod
    def cross(arr1, arr2):
        raise NotImplementedError

    @staticmethod
    def set_device(dev_id):
        if sys.platform == 'darwin':
            # Check that MPS is available
            if not torch.backends.mps.is_available():
                print("MPS not available because the current PyTorch install was not built with MPS enabled. Using cpu.")
                torch.device("cpu")
            else:
                torch.device("mps")
        elif torch.backends.cuda.is_built() and dev_id != -1:
            torch_dev = f"cuda:{str(dev_id)}"
            torchlib.device = torch_dev
        else:
            torch.device("cpu")

    @staticmethod
    def to_numpy(arr):
        try:
            out = arr.detach().cpu().numpy()
        except:
            out = arr
        return out

    @staticmethod
    def save(filename, arr):
        try:
            arr = arr.detach().cpu().numpy()
        except:
            pass
        np.save(filename, arr)

    @staticmethod
    def load(filename, **kwargs):
        arr = np.load(filename)
        return torch.as_tensor(arr, device=torchlib.device)

    @staticmethod
    def from_numpy(arr, **kwargs):
        return torch.as_tensor(arr, device=torchlib.device)

    @staticmethod
    def dtype(arr):
        return arr.dtype

    @staticmethod
    def astype(arr, dtype):
        # this is kind of nasty, it does not understand 'int', so need to convert for the cases
        # that will be used
        if dtype == 'int32':
            dtype = torch.int32
        return arr.type(dtype=dtype)

    @staticmethod
    def reshape(arr, shape):
        raise NotImplementedError

    @staticmethod
    def size(arr):
        return torch.numel(arr)

    @staticmethod
    def next_fast_len(target):
        import scipy
        return scipy.fft.next_fast_len(target)

    @staticmethod
    def hasnan(arr):
        return torch.any(torch.isnan(arr))

    @staticmethod
    def nan_to_num(arr):
        raise NotImplementedError

    @staticmethod
    def copy(arr):
        return arr.clone()

    @staticmethod
    def random(shape, **kwargs):
        arr = torch.rand(shape, device=torchlib.device)
        # return torch.rand(shape, device=torchlib.device)
        return arr

    @staticmethod
    def roll(arr, sft, axis):
        sft = [int(s) for s in sft]
        dims = tuple([i for i in range(len(sft))])
        try:
            return torch.roll(arr, sft, dims)
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def shift(arr, sft):
        sft = [int(s) for s in sft]
        dims = tuple([i for i in range(len(sft))])
        try:
            return torch.roll(arr, sft, dims)
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def fftshift(arr):
        try:
            return torch.fft.fftshift(arr)
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def ifftshift(arr):
        try:
            return torch.fft.ifftshift(arr)
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def fft(arr):
        try:
            return torch.fft.fftn(arr, norm='forward')
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def ifft(arr):
        try:
            return torch.fft.ifftn(arr, norm='forward')
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def fftconvolve(arr1, kernel):
        print('not supported yet in torch, use different library')
        raise
        # # kernel shape can be smaller than arr1 shape in each dim
        # sh1 = list(arr1.size())
        # sh2 = list(kernel.size())
        # if sh1 != sh2:
        #     # the pad is added from last dim to first
        #     sh1.reverse()
        #     sh2.reverse()
        #     pad = [((sh1[i]-sh2[i])//2, sh1[i] - sh2[i] - (sh1[i]-sh2[i])//2) for i in range(len(sh1))]
        #     pad = tuple(sum(pad, ()))
        #     kernel = torch.nn.functional.pad(kernel, pad)
        # conv = torch.fft.ifftn(torch.fft.fftn(arr1) * torch.fft.fftn(kernel))
        # return conv

    @staticmethod
    def correlate(arr1, arr2, mode='same', method='fft'):
        raise NotImplementedError

    @staticmethod
    def where(cond, x, y):
        return torch.where(cond, x, y)

    @staticmethod
    def dims(arr):
        # get array dimensions
        return arr.size()

    @staticmethod
    def absolute(arr):
        return torch.abs(arr)

    @staticmethod
    def square(arr):
        return torch.square(arr)

    @staticmethod
    def sqrt(arr):
        return torch.sqrt(arr)

    @staticmethod
    def sum(arr, axis=None):
        return torch.sum(arr, dim=axis)

    @staticmethod
    def real(arr):
        return arr.real

    @staticmethod
    def imag(arr):
        return arr.imag

    @staticmethod
    def amax(arr):
        return torch.amax(arr)

    @staticmethod
    def argmax(arr, axis=None):
        return torch.argmax(arr, axis)

    @staticmethod
    def unravel_index(indices, shape):
        return np.unravel_index(indices.detach().cpu().numpy(), shape)

    @staticmethod
    def ravel(arr):
        raise NotImplementedError

    @staticmethod
    def maximum(arr1, arr2):
        return torch.maximum(arr1, arr2)

    @staticmethod
    def ceil(arr):
        return torch.ceil(arr)

    @staticmethod
    def fix(arr):
        return torch.fix(arr)

    @staticmethod
    def round(val):
        return torch.round(val)

    @staticmethod
    def full(shape, fill_value, **kwargs):
        return torch.full(shape, fill_value, device=torchlib.device)

    @staticmethod
    def print(arr, **kwargs):
        print(arr)

    @staticmethod
    def angle(arr):
        return torch.angle(arr)

    @staticmethod
    def flip(arr, axis=None):
        if axis is None:
            axis = [i for i in range(len(arr.size()))]
        return torch.flip(arr, axis)

    @staticmethod
    def tile(arr, rep):
        return torch.tile(arr, rep)

    @staticmethod
    def expand_dims(arr, axis):
        return arr.unsqeeze(axis)

    @staticmethod
    def squeeze(arr):
        return torch.squeeze(arr)

    @staticmethod
    # this method is only in torchlib, not part of cohlib
    def gaussian(sigma, size=5, **kwargs): #sigma, size=5, n=None):
        """
        Creates a nD Gaussian kernel with the specified shape and sigma.
        It is assumed that all dimensions in shape are equal.
        """
        dims = kwargs.get('dims', None)
        ranges = [torch.arange(size)] * dims
        grid = torch.meshgrid(*ranges)
        grid = torch.stack(grid, dim=-1).float()
        center = (size - 1) / 2
        kernel = torch.exp(-torch.sum((grid - center) ** 2, dim=-1) / (2 * sigma ** 2))
        kernel /= kernel.sum()
        return kernel

    @staticmethod
    def entropy(arr):
        raise NotImplementedError

    @staticmethod
    def gaussian_filter(arr, sigma, **kwargs):
        """
        Convolves a nD input tensor with a nD Gaussian kernel.
        """
        if arr.ndim == 3:
            from torch.nn.functional import conv3d as convn
        elif arr.ndim == 2:
            from torch.nn.functional import conv2d as convn
        elif arr.ndim == 1:
            from torch.nn.functional import conv1d as convn

        if 'kernel' in kwargs:
            # kernel should be on device the arr is
            kernel = kwargs.get('kernel')
            gauss_kernel_size = kernel.size()[0]
        else:
            gauss_kernel_size = 5
            kernel = torchlib.gaussian(sigma, gauss_kernel_size, dims=arr.ndim)
            kernel = kernel.to(arr.device)

        padding = gauss_kernel_size // 2

        blurred = convn(arr.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding)
        return blurred.squeeze()

    @staticmethod
    def median_filter(arr, size, **kwargs):
        raise NotImplementedError

    @staticmethod
    def uniform_filter(arr, size, **kwargs):
        raise NotImplementedError

    @staticmethod
    def binary_erosion(arr, **kwargs):
        raise NotImplementedError

    @staticmethod
    def center_of_mass(arr):
        normalizer = torch.sum(arr)
        shape = arr.shape
        ranges = [torch.arange(shape[i], device=torchlib.device) for i in range(arr.ndim)]
        grids = torch.meshgrid(*ranges)
        com = [(torch.sum(arr * grids[i]) / normalizer).tolist() for i in range(arr.ndim)]
        return com

    @staticmethod
    def meshgrid(*xi):
        # check if need to move to device
        return torch.meshgrid(*xi)

    @staticmethod
    def exp(arr):
        return torch.exp(arr)

    @staticmethod
    def conj(arr):
        return torch.conj(arr)

    @staticmethod
    def cos(arr):
        return torch.cos(arr)

    @staticmethod
    def array_equal(arr1, arr2):
        return torch.equal(arr1, arr2)

    @staticmethod
    def linspace(start, stop, num):
        return torch.linspace(start, stop, num)

    @staticmethod
    def geomspace(start, stop, num):
        raise NotImplementedError

    @staticmethod
    def clip(arr, min, max=None):
        return torch.clip(arr, min, max)

    @staticmethod
    def diff(arr, axis=None, prepend=0):
        raise NotImplementedError

    @staticmethod
    def gradient(arr, dx=1):
        raise NotImplementedError

    @staticmethod
    def argmin(arr, axis=None):
        raise NotImplementedError

    @staticmethod
    def take_along_axis(a, indices, axis):
        raise NotImplementedError

    @staticmethod
    def moveaxis(arr, source, dest):
        raise NotImplementedError

    @staticmethod
    def lstsq(A, B):
        raise NotImplementedError

    @staticmethod
    def zeros(shape):
        raise NotImplementedError

    @staticmethod
    def indices(dims):
        raise NotImplementedError

    @staticmethod
    def concatenate(tup, axis=0):
        raise NotImplementedError

    @staticmethod
    def stack(tup):
        raise NotImplementedError

    @staticmethod
    def amin(arr):
        raise NotImplementedError

    @staticmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        raise NotImplementedError

    @staticmethod
    def pad(arr, padding):
        raise NotImplementedError

    @staticmethod
    def histogram2d(meas, rec, n_bins=100, log=False):
        raise NotImplementedError

    @staticmethod
    def log(arr):
        raise NotImplementedError

    @staticmethod
    def log10(arr):
        raise NotImplementedError

    @staticmethod
    def xlogy(arr, y=None):
        raise NotImplementedError

    @staticmethod
    def mean(arr):
        raise NotImplementedError

    @staticmethod
    def median(arr):
        raise NotImplementedError

    @staticmethod
    def clean_default_mem():
        pass


# a1 = torch.Tensor([0.1, 0.2, 0.3, 1.0, 1.2, 1.3])
# a2 = torch.Tensor([10.1, 10.2, 10.3, 11.0])
# conv = torchlib.fftconvolve(a1,a2)
# print('torch conv', conv)
# print(conv.real)
# print(torch.abs(conv))
# print(torch.nn.functional.conv1d(a1,a2))
