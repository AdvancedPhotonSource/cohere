from cohere_core.lib.cohlib import cohlib
import numpy as np
import torch
from torch.nn.functional import conv1d

import sys

class torchlib(cohlib):
    # Interface
    def array(obj):
        return torch.Tensor(obj, device=torchlib.device)

    def dot(arr1, arr2):
        return torch.dot(arr1, arr2)

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

    def to_numpy(arr):
        try:
            out = arr.detach().cpu().numpy()
        except:
            out = arr
        return out

    def save(filename, arr):
        try:
            arr = arr.detach().cpu().numpy()
        except:
            pass
        np.save(filename, arr)

    def load(filename):
        arr = np.load(filename)
        return torch.as_tensor(arr, device=torchlib.device)

    def from_numpy(arr):
        return torch.as_tensor(arr, device=torchlib.device)

    def dtype(arr):
        return arr.dtype

    def size(arr):
        return torch.numel(arr)

    def hasnan(arr):
        return torch.any(torch.isnan(arr))

    def copy(arr):
        return arr.clone()

    def random(shape, **kwargs):
        return torch.rand(shape, device=torchlib.device)

    def fftshift(arr):
        # if roll is implemented before fftshift
        # shifts = tuple([d // 2 for d in arr.size()])
        # dims = tuple([d for d in range(len(arr.size()))])
        # return torch.roll(arr, shifts, dims=dims)
        try:
            return torch.fft.fftshift(arr)
        except Exception as e:
            print('not supported error: ' + repr(e))

    def ifftshift(arr):
        # if roll is implemented before ifftshift
        # shifts = tuple([(d + 1) // 2 for d in arr.size()])
        # dims = tuple([d for d in range(len(arr.size()))])
        # return torch.roll(arr, shifts, dims=dims)
        try:
            return torch.fft.ifftshift(arr)
        except Exception as e:
            print('not supported error: ' + repr(e))

    def shift(arr, sft):
        sft = [int(s) for s in sft]
        dims = tuple([i for i in range(len(sft))])
        try:
            return torch.roll(arr, sft, dims)
        except Exception as e:
            print('not supported error: ' + repr(e))

    def fft(arr):
        try:
            return torch.fft.fftn(arr, norm='forward')
        except Exception as e:
            print('not supported error: ' + repr(e))

    def ifft(arr):
        try:
            return torch.fft.ifftn(arr, norm='forward')
        except Exception as e:
            print('not supported error: ' + repr(e))

    def fftconvolve(arr1, arr2):
        shape1 = arr1.size()
        shape2 = arr2.size()
        pad1 = tuple([d//2 for d in shape2 for _ in (0, 1)])
        pad2 = tuple([d//2 for d in shape1 for _ in (0, 1)])
        parr1 = torch.nn.functional.pad(arr1, pad1)
        parr2 = torch.nn.functional.pad(arr2, pad2)
        conv = torch.fft.ifftn(torch.fft.fftn(parr1) * torch.fft.fftn(parr2))

        for i in range(len(shape2)):
            splitted = torch.split(conv, [shape2[i] //2, shape1[i], shape2[i] //2], dim=i)
            conv = splitted[1]
        return conv

    def where(cond, x, y):
        return torch.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.size()

    def absolute(arr):
        return torch.abs(arr)

    def square(arr):
        return torch.square(arr)

    def sqrt(arr):
        return torch.sqrt(arr)

    def sum(arr, axis=None):
        return torch.sum(arr)

    def real(arr):
        return arr.real

    def imag(arr):
        return arr.imag

    def amax(arr):
        return torch.amax(arr)

    def argmax(arr, axis=None):
        return torch.argmax(arr, axis)

    def unravel_index(indices, shape):
        return np.unravel_index(indices.detach().cpu().numpy(), shape)

    def maximum(arr1, arr2):
        return torch.maximum(arr1, arr2)

    def ceil(arr):
        return torch.ceil(arr)

    def fix(arr):
        return torch.fix(arr)

    def round(val):
        return torch.round(val)

    def full(shape, fill_value, **kwargs):
        return torch.full(shape, fill_value, device=torchlib.device)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return torch.angle(arr)

    def flip(arr, axis=None):
        if axis is None:
            axis = [i for i in range(len(arr.size()))]
        return torch.flip(arr, axis)

    def tile(arr, rep):
        return torch.tile(arr, rep)

    def expand_dims(arr, axis):
        return arr.unsqeeze(axis)

    def squeeze(arr):
        return torch.squeeze(arr)

    def gaussian(dims, sigma, **kwargs):
        def gaussian1(dim, sig):
            x = torch.arange(dim, device=torchlib.device) - dim // 2
            if dim % 2 == 0:
                x = x + 0.5
            gauss = torch.exp(-(x ** 2.0) / (2 * sig ** 2))
            return gauss / gauss.sum()

        if isinstance(sigma, int) or isinstance(sigma, float):
            sigma = [sigma] * len(dims)

        last_axis = len(dims) - 1
        last_dim = dims[-1]
        gaussian = gaussian1(dims[last_axis], sigma[last_axis]).repeat(*dims[:-1], 1)
        for i in range(int(len(dims)) - 1):
            tdims = dims[:-1]
            tdims[i] = last_dim
            temp = gaussian1(dims[i], sigma[i])
            temp = temp.repeat(*tdims, 1)
            temp = torch.swapaxes(temp, i, last_axis)
            gaussian *= temp
        return gaussian / gaussian.sum()

    def gaussian_filter(arr, sigma, **kwargs):
        # will not work on M1 until the fft fuctions are supported
        dims = list(arr.shape)
        dist = torchlib.gaussian(dims, 1/sigma)
        arr_f = torch.fft.ifftshift(torch.fft.fftn(torch.fft.ifftshift(arr)))
        filter = arr_f * dist
        filter = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(filter)))
        filter = torch.real(filter)
        filter = torch.where(filter >= 0, filter, 0.0)
        return filter

    def center_of_mass(arr):
        normalizer = torch.sum(arr)
        shape = arr.shape
        ranges = [torch.arange(shape[i], device=torchlib.device) for i in range(arr.ndim)]
        grids = torch.meshgrid(*ranges)
        com = [(torch.sum(arr * grids[i]) / normalizer).tolist() for i in range(arr.ndim)]
        return com

    def meshgrid(*xi):
        # check if need to move to device
        return torch.meshgrid(*xi)

    def exp(arr):
        return torch.exp(arr)

    def conj(arr):
        return torch.conj(arr)

    def cos(arr):
        return torch.cos(arr)

    def linspace(start, stop, num):
        return torch.linspace(start, stop, num)

    def clip(arr, min, max=None):
        return torch.clip(arr, min, max)

# a1 = torch.Tensor([0.1, 0.2, 0.3, 1.0, 1.2, 1.3])
# a2 = torch.Tensor([10.1, 10.2, 10.3, 11.0])
# conv = torchlib.fftconvolve(a1,a2)
# print('torch conv', conv)
# print(conv.real)
# print(torch.abs(conv))
# print(torch.nn.functional.conv1d(a1,a2))
