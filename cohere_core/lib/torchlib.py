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
    def set_device(dev_id):
        if dev_id == -1 or sys.platform == 'darwin':
            torch.device("cpu")
            torchlib.device = "cpu"
        else:
            torch.device(dev_id)
            torchlib.device = dev_id

        # the pytorch does not support very needed functions such as fft, ifft for
        # the GPU M1 (backendMPS), so for this platform it will be set to cpu until the
        # functions are supported
        # torchlib.device = torch.device("mps")

    @staticmethod
    def set_backend(proc):
        # this function is not used currently
        if sys.platform == 'darwin':
            # Check that MPS is available
            if not torch.backends.mps.is_available():
                print("MPS not available because the current PyTorch install was not "
                      "built with MPS enabled. Using cpu.")
                torch.device("cpu")
            else:
                torch.device("mps")
        elif torch.backends.cuda.is_built():
                torch.device("cuda")
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
    def load(filename):
        arr = np.load(filename)
        return torch.as_tensor(arr, device=torchlib.device)

    @staticmethod
    def from_numpy(arr):
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
    def size(arr):
        return torch.numel(arr)

    @staticmethod
    def hasnan(arr):
        return torch.any(torch.isnan(arr))

    @staticmethod
    def copy(arr):
        return arr.clone()

    @staticmethod
    def random(shape, **kwargs):
        arr = torch.rand(shape, device=torchlib.device)
        print(arr.dtype)
        # return torch.rand(shape, device=torchlib.device)
        return arr

    @staticmethod
    def roll(arr, sft):
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
        # if roll is implemented before fftshift
        # shifts = tuple([d // 2 for d in arr.size()])
        # dims = tuple([d for d in range(len(arr.size()))])
        # return torch.roll(arr, shifts, dims=dims)
        try:
            return torch.fft.fftshift(arr)
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def ifftshift(arr):
        # if roll is implemented before ifftshift
        # shifts = tuple([(d + 1) // 2 for d in arr.size()])
        # dims = tuple([d for d in range(len(arr.size()))])
        # return torch.roll(arr, shifts, dims=dims)
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
        return torch.sum(arr)

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
    def gaussian(dims, sigma, **kwargs):
        def gaussian1(dim, sig):
            x = torch.arange(dim, device=torchlib.device) - dim // 2
            if dim % 2 == 0:
                x = x + 0.5
            gauss = torch.exp(-(x ** 2.0) / (2 * sig ** 2))
            return gauss / gauss.sum()

        if isinstance(sigma, int) or isinstance(sigma, float):
            sigmas = [sigma] * len(dims)
        else: # assuming a list of values
            sigmas = sigma

        sigmas = [dims[i] / (2.0 * math.pi * sigmas[i]) for i in range(len(dims))]

        last_axis = len(dims) - 1
        last_dim = dims[-1]
        gaussian = gaussian1(dims[last_axis], sigmas[last_axis]).repeat(*dims[:-1], 1)
        for i in range(int(len(dims)) - 1):
            tdims = dims[:-1]
            tdims[i] = last_dim
            temp = gaussian1(dims[i], sigmas[i])
            temp = temp.repeat(*tdims, 1)
            temp = torch.swapaxes(temp, i, last_axis)
            gaussian *= temp
        return gaussian / gaussian.sum()

    @staticmethod
    def gaussian_filter(arr, sigma, **kwargs):
        # will not work on M1 until the fft fuctions are supported
        normalizer = torch.sum(arr)
        dims = list(arr.shape)
        dist = torchlib.gaussian(dims, 1 / sigma)
        arr_f = torch.fft.ifftshift(torch.fft.fftn(torch.fft.ifftshift(arr)))
        filter = arr_f * dist
        filter = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(filter)))
        filter = torch.real(filter)
        filter = torch.where(filter >= 0, filter, 0.0)
        corr = normalizer/torch.sum(filter)
        return filter * corr

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
    def linspace(start, stop, num):
        return torch.linspace(start, stop, num)

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
    def calc_nmi(hgram):
        raise NotImplementedError

    @staticmethod
    def calc_ehd(hgram):
        raise NotImplementedError

    @staticmethod
    def clean_default_mem():
        pass
