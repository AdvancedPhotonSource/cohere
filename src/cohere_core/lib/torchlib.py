# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

from cohere_core.lib.cohlib import cohlib
import numpy as np
import sys
import torch
import scipy
from cohere_core.utilities import pad_to_cube
import os

os.environ['SCIPY_ARRAY_API']='1'

class torchlib(cohlib):
    #device = "cpu"
    # Interface
    @staticmethod
    def array(obj):
        # return torch.tensor(obj, device=torchlib.device, dtype=torch.float64)
        if isinstance(obj, list) and torch.is_tensor(obj[0]):
            obj = torch.stack(obj)
        return torch.tensor(obj, device=torchlib.device, dtype=torch.float32)

    @staticmethod
    def dot(arr1, arr2):
        return torch.matmul(arr1, arr2)

    @staticmethod
    def cross(arr1, arr2):
        return torch.cross(arr1, arr2)

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
        return torch.as_tensor(arr, device=torchlib.device).double()

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
        if shape == -1:
            return torch.flatten(arr)
        else:
            return torch.reshape(arr, shape)

    @staticmethod
    def size(arr):
        return torch.numel(arr)

    @staticmethod
    def next_fast_len(target):
        return scipy.fft.next_fast_len(target)

    @staticmethod
    def hasnan(arr):
        return torch.any(torch.isnan(arr))

    @staticmethod
    def nan_to_num(arr, **kwargs):
        return torch.nan_to_num(arr, **kwargs)

    @staticmethod
    def copy(arr):
        return arr.clone()

    @staticmethod
    def random(shape, **kwargs):
        return torch.rand(shape, device=torchlib.device)

    @staticmethod
    def moveaxis(arr, src, dst):
        return torch.moveaxis(arr, src, dst)

    @staticmethod
    def roll(arr, sft, axis):
        sft = [int(s) for s in sft]
        dims = tuple([i for i in range(len(sft))])
        try:
            return torch.roll(arr, sft, dims)
        except Exception as e:
            print('not supported error: ' + repr(e))

    @staticmethod
    def fftshift(arr):
        return torch.fft.fftshift(arr)

    @staticmethod
    def ifftshift(arr):
        return torch.fft.ifftshift(arr)

    @staticmethod
    def fft(arr, norm='forward'):
        return torch.fft.fftn(arr, norm=norm)

    @staticmethod
    def ifft(arr, norm='forward'):
        return torch.fft.ifftn(arr, norm=norm)

    @staticmethod
    def fftconvolve(arr1, kernel, mode='same'):
        try:
            import torchaudio.functional as tf
        except Exception as e:
            print('torchaudio not installed')
            raise
        return tf.fftconvolve(arr1, kernel, mode=mode)

    @staticmethod
    def correlate(arr1, arr2, mode='same', method='fft'):
        A = torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(torch.conj(arr1)), norm='forward'))
        B = torch.fft.ifftshift(torch.fft.fftn(torch.fft.fftshift(arr2), norm='forward'))
        CC = A * B
        return torch.fft.ifftshift(torch.fft.ifftn(torch.fft.fftshift(CC), norm='forward'))

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
    def amin(arr):
        return torch.amin(arr)

    @staticmethod
    def argmax(arr, axis=None):
        return torch.argmax(arr, axis)

    @staticmethod
    def unravel_index(indices, shape):
        return np.unravel_index(indices.detach().cpu().numpy(), shape)

    @staticmethod
    def ravel(arr):
        return torch.ravel(arr)

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
        if isinstance(axis, int):
            axis = [axis]
        return torch.flip(arr, axis)

    @staticmethod
    def coordinate_dev(*args):
        # check if any tensor and if so check if any on cuda device
        device = torch.device('cpu')
        for arg in args:
            if isinstance(arg, torch.Tensor):
                if arg.device != device:
                    device = arg.device
                    break
        # load all args on the chosen device
        # if on different device
        out = []
        for arg in args:
            if not isinstance(arg, torch.Tensor) or arg.device != device:
                arg = arg.to(device)
            out.append(arg)
        return out

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
    def entropy(arr):
        arr = arr / torch.sum(arr, axis=0, keepdim=True)
        epsilon = 1e-12
        entropy = -torch.sum(arr * torch.log(arr + epsilon), axis=0)
        return entropy

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
        arr = arr.to(torch.float32)
        padding = gauss_kernel_size // 2

        blurred = convn(arr.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=padding)
        return blurred.squeeze().to(torch.float32)

    @staticmethod
    def median_filter(arr, kernel_size, **kwargs):
        raise NotImplementedError("Median filtering not implemented.")
        #
        # # input_tensor shape: (B, C, H, W)
        # padding = kernel_size // 2
        #
        # # Extract patches: (B, C * kernel_size * kernel_size, L)
        # patches = torch.nn.functional.unfold(arr.unsqueeze(0), kernel_size=kernel_size, padding=padding)
        #
        # # Reshape to separate channels and patch pixels
        # B, C_kh_kw, L = patches.shape
        # patches = patches.view(B, arr.size(1), -1, L)
        #
        # # Compute median across the patch dimension (dim=2)
        # # .values is used because torch.median returns (values, indices)
        # median_values = patches.median(dim=2).values
        #
        # # Reshape back to original image spatial dimensions
        # return median_values.view_as(arr)

    @staticmethod
    def uniform_filter(arr, size, **kwargs):
        raise NotImplementedError("Uniform filter not implemented.")
        # if arr.ndim == 3:
        #     return torch.nn.functional.avg_pool3d(arr, kernel_size=size, stride=1, padding=size // 2, count_include_pad=False)
        # else:
        #     # currently not used for 1D or 2D
        #     print('not implemented for 1D or 2D')
        #     raise NotImplementedError

    @staticmethod
    def binary_erosion(arr, **kwargs):
        raise NotImplementedError("Binary erosion not implemented.")
        # if arr.ndim == 3:
        #     if 'structure' in kwargs:
        #         # Ensure kernel is Float
        #         structure = kwargs['structure'].float()
        #     else:
        #         # Default 3x3x3 cube structuring element
        #         structure = torch.ones((1, 1, 3, 3, 3), device=arr.device)
        #
        #     padding = (structure.shape[2] // 2, structure.shape[3] // 2, structure.shape[4] // 2)
        #     num_elements = structure.sum()
        #
        #     # Perform convolution
        #     convolved = torch.nn.functional.conv3d(arr.float().unsqueeze(0).unsqueeze(0), structure, padding=padding).squeeze()
        #
        #     # Erosion: only True if ALL elements of the structure matched
        #     return (convolved >= num_elements).float()
        # else:
        #     # currently not used for 1D or 2D
        #     print('not implemented for 1D or 2D')
        #     raise NotImplementedError

    def binary_dilation(arr, **kwargs):
        raise NotImplementedError("Binary dilation not implemented.")
        # if arr.ndim == 3:
        #     if 'structure' in kwargs:
        #         # Ensure kernel is Float
        #         structure = kwargs['structure'].float()
        #     else:
        #         # Default 3x3x3 cube structuring element
        #         structure = torch.ones((1, 1, 3, 3, 3), device=arr.device)
        #
        #     # Conv3d with weight=1 behaves like dilation when combined with >0 thresholding
        #     dilated = torch.nn.functional.conv3d(arr.unsqueeze(0).unsqueeze(0), structure, padding=1).squeeze()
        #     return (dilated > 0).float()
        # else:
        #     # currently not used for 1D or 2D
        #     print('not implemented for 1D or 2D')
        #     raise NotImplementedError

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
        raise NotImplementedError("Geometry space not implemented.")
        # return np.logspace(start, stop, num)

    @staticmethod
    def clip(arr, min, max=None):
        raise NotImplementedError("Clipping not implemented.")
        # return torch.clip(arr, min=min, max=max)

    @staticmethod
    def diff(arr, axis=-1, prepend=None, append=None):
        raise NotImplementedError("Difference not implemented.")
        # return torch.diff(arr, dim=axis, prepend=prepend, append=append)

    @staticmethod
    def gradient(arr, dx=1):
        raise NotImplementedError("Gradient not implemented.")
        # return torch.gradient(arr, spacing=torch.tensor([float(dx)]))

    @staticmethod
    def argmin(arr, axis=None):
        raise NotImplementedError("Argmin not implemented.")
        # return torch.argmin(arr, dim=axis)

    @staticmethod
    def take_along_axis(arr, indices, axis=None):
        return torch.take_along_dim(arr, indices, dim=axis)

    @staticmethod
    def lstsq(A, B):
        raise NotImplementedError("LSTSQ not implemented.")
        # return torch.linalg.lstsq(A, B, rcond=None)

    @staticmethod
    def zeros(shape):
        raise NotImplementedError("Zeros not implemented.")
        # return torch.zeros(shape)

    @staticmethod
    def indices(dims):
        raise NotImplementedError("Indices not implemented.")
        # grids = torch.meshgrid([torch.arange(dim) for dim in dims], indexing='ij')
        #
        # # Stack the coordinate tensors along a new first dimension
        # # This results in a single tensor of shape (N, r0, ..., rN-1)
        # return torch.stack(grids, dim=0)

    @staticmethod
    def concatenate(tup, axis=0):
        raise NotImplementedError("Concatenate not implemented.")
        # return torch.cat(tup, dim=axis)

    @staticmethod
    def stack(tup):
        raise NotImplementedError("Stack not implemented.")
        # same_dev_tup = torchlib.coordinate_dev(*tup)
        # return torch.stack(same_dev_tup)

    @staticmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        raise NotImplementedError("Affine transform not implemented.")
        # # Create grid
        # matrix = torch.nn.functional.pad(matrix, (0,1))
        # matrix = matrix.unsqueeze(0)
        # grid = torch.nn.functional.affine_grid(matrix, (1,1)+arr.shape, align_corners=False)
        # # if on different device
        # [arr, grid] = torchlib.coordinate_dev(*[arr, grid])
        #
        # arr = arr.unsqueeze(0).unsqueeze(0)
        # out = torch.nn.functional.grid_sample(arr.to(torch.float32), grid.to(torch.float32), padding_mode="zeros", align_corners=False)
        # out = out.squeeze().to(torch.float64)
        # return out

    @staticmethod
    def pad(arr, padding):
        raise NotImplementedError("Pad not implemented.")
        # if isinstance(padding, tuple) or isinstance(padding, list):
        #     # convert the padding in numpy format ((px0,px1),(py0,py1),(pz0,pz1)) to torch format
        #     # (px0,px1,py0,py1,pz0,pz1)
        #     padding = list(sum(padding, ()))
        #     padding.reverse() # in torch it starts from the last dim
        # elif isinstance(padding, int):
        #     padding = (padding,) * arr.ndim * 2
        # return torch.nn.functional.pad(arr, padding)

    @staticmethod
    def histogram2d(meas, rec, n_bins=100, log=False):
        raise NotImplementedError("Histogram2d not implemented.")
        # device = torch.device('cpu')
        # if meas.device != torch.device('cpu'):
        #     device = meas.device
        # if rec.device != torch.device('cpu'):
        #     device = rec.device
        #
        # data = torch.stack((meas.to('cpu'), rec.to('cpu'))).T
        # # torch.histogram2 is implemented only on cpu device, need to load it to cpu and then back to cuda if on cuda
        # hist, bin_edges = torch.histogramdd(data, bins=n_bins)
        # if device != torch.device('cpu'):
        #     hist = hist.to(device)
        # return hist

    @staticmethod
    def log(arr):
        raise NotImplementedError("Log not implemented.")
        # return torch.log(arr)

    @staticmethod
    def log10(arr):
        raise NotImplementedError("Log10 not implemented.")
        # return torch.log10(arr)

    @staticmethod
    def xlogy(x, y=None):
        raise NotImplementedError("Xlogy not implemented.")
        # if y is None:
        #     y = x
        # return torch.xlogy(x, y)

    @staticmethod
    def mean(arr, axis=None):
        raise NotImplementedError("Mean not implemented.")
        # return torch.mean(arr, axis)

    @staticmethod
    def median(arr, axis=-1):
        raise NotImplementedError("Median not implemented.")
        # (values, indices) = torch.median(arr, dim=axis)
        # return values

    @staticmethod
    def norm(arr, ord=None, axis=None, keepdim=True):
        raise NotImplementedError("Norm not implemented.")
        # return torch.linalg.norm(arr, ord=ord, dim=axis, keepdim=keepdim)

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
