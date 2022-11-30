from cohere_core.lib.cohlib import cohlib
import numpy as np
import torch

class torchlib(cohlib):
    # Interface
    def array(obj):
        return torch.Tensor(obj)

    def dot(arr1, arr2):
        return torch.dot(arr1, arr2)

    def set_device(dev_id):
        torch.cuda.set_device(dev_id)

    def set_backend(proc):
        pass

    def to_numpy(arr):
        return arr.detach().cpu().numpy()

    def save(filename, arr):
        arr = arr.detach().cpu().numpy()
        np.save(arr)

    def load(filename):
        arr = np.load(filename)
        return torch.from_numpy(arr)

    def from_numpy(arr):
        return torch.from_numpy(arr)

    def dtype(arr):
        return arr.dtype

    def size(arr):
        return torch.numel(arr)

    def hasnan(arr):
        return torch.isnan(arr)

    def copy(arr):
        return arr.clone()

    def random(shape, **kwargs):
        return torch.rand(shape)

    def fftshift(arr):
        return torch.fft.fftshift(arr)

    def ifftshift(arr):
        return torch.fft.ifftshift(arr)

    def shift(arr, sft):
        torch.roll(arr, sft)

    def fft(arr):
        return torch.fft.fftn(arr)

    def ifft(arr):
        return torch.fft.ifftn(arr)

    def fftconvolve(arr1, arr2):
        pass

    def where(cond, x, y):
        return torch.where(cond, x, y)

    def dims(arr):
        # get array dimensions
        return arr.size()

    def absolute(arr):
        return torch.absolute(arr)

    def square(arr):
        return torch.square(arr)

    def sqrt(arr):
        return torch.sqrt(arr)

    def sum(arr, axis=None):
        return torch.sum(arr)

    def real(arr):
        torch.real(arr)

    def imag(arr):
        torch.imag(arr)

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
        return torch.full(shape, fill_value)

    def print(arr, **kwargs):
        print(arr)

    def angle(arr):
        return torch.angle(arr)

    def flip(arr, axis):
        torch.flip(arr, axis)

    def tile(arr, rep):
        return torch.tile(arr, rep)

    def expand_dims(arr, axis):
        return arr.unsqeeze(axis)

    def squeeze(arr):
        return torch.squeeze(arr)

    def gaussian_filter(arr, sigma, **kwargs):
        pass

    def gaussian(dims, sigma, **kwargs):
        pass

    def center_of_mass(arr):
        normalizer = torch.sum(arr)
        grids = torch.meshgrid(arr.shape)
        results = [torch.sum(arr * grids[dir].astype(float)) / normalizer
                   for dir in range(arr.ndim)]
        print(results)
        return results
    t=torch.tensor([[[1.0, 2.5,3.0,7.1],[2.0, 7.5,3.0,7.1],[1.5, 8.5,4.0,7.1]],[[1.0, 2.5,3.0,7.1],[2.0, 7.5,3.0,7.1],[1.5, 8.5,4.0,7.1]],[[1.0, 6.5,3.6,7.1],[2.0, 1.5,3.5,7.9],[1.9, 8.5,47.0,2.1]]])
    com = center_of_mass(t)

    def meshgrid(*xi):
        return torch.meshgrid(*xi)

    def exp(arr):
        pass

    def conj(arr):
        pass

