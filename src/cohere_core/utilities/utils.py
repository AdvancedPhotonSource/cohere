# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.utils
=================

This module is a suite of utility functions.
"""

import tifffile as tf
import numpy as np
import os
import logging
import stat
import scipy.ndimage as ndi
import ast
import importlib


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [
           'adjust_dimensions',
           'binning',
           'center_max',
           'crop_center',
           'get_central_object_extent',
           'get_good_dim',
           'get_lib',
           'get_logger',
           'get_oversample_ratio',
           'join',
           'normalize',
           'pad_center',
           'read_config',
           'read_tif',
           'resample',
           'select_central_object',
           'save_tif',
           'save_metrics',
           'threshold_by_edge',
           'write_config',
           'write_plot_errors',
           ]


def adjust_dimensions(arr, pads):
    """
    This function adds to or subtracts from each dimension of the array elements defined by pad. If the pad is positive, the array is padded in this dimension. If the pad is negative, the array is cropped.

    :param arr: ndarray, the array to pad/crop
    :param pad: list of pad values, a tuple of two int for each dimension. The values in each tuple will be added/subtracted to the sides of array in corresponding dimension. 
    :return: the padded/cropped and adjusted to opencl compatible format array
    """
   # up the dimensions to 3D
    for _ in range(len(arr.shape), 3):
        arr = np.expand_dims(arr,axis=0)
        pads = [(0,0)] + pads

    old_dims = arr.shape
    start = [max(0, -pad[0]) for pad in pads]
    stop = [arr.shape[i] - max(0, -pads[i][1]) for i in range(3)]
    cropped = arr[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]  #for 1D and 2D it was upped to 3D

    dims = cropped.shape
    c_vals = []
    new_pad = []
    for i in range(len(dims)):
        pad = pads[i]
        new_dim = old_dims[i] + pad[0] + pad[1]
        # if the pad is positive
        pad_front = max(0, pad[0])
        pad_end = new_dim - dims[i] - pad_front
        new_pad.append((pad_front, pad_end))
        c_vals.append((0.0, 0.0))
    adjusted = np.pad(cropped, new_pad, 'constant', constant_values=c_vals)

    return np.squeeze(adjusted)


def array_to_good_dims(arr, pkg):
    """
    Returns array with each dimension found to be good for fast Fourier transform and not smaller than given dimension.
    The original array is in the center.

    :param arr: array to be resized
    :param pck: python package that will be used for reconstruction: 'np' for numpy, 'cp' for cupy, 'torch' for torch.
    :return: the array padded to the 'good' dimensions
    """
    new_shape = [get_good_dim(d, pkg) for d in arr.shape]
    return pad_center(arr, tuple(new_shape))


def binning(array, binsizes):
    """
    This function does the binning of the array. The array is binned in each dimension by the corresponding binsizes elements.
    If binsizes list is shorter than the array dimensions, the remaining dimensions are not binned.

    :param array: ndarray the original array to be binned
    :param binsizes: a list defining binning factors for corresponding dimensions
    :return: binned array
    """
    data_dims = array.shape
    # trim array
    for ax in range(len(binsizes)):
        cut_slices = range(data_dims[ax] - data_dims[ax] % binsizes[ax], data_dims[ax])
        array = np.delete(array, cut_slices, ax)

    binned_array = array
    new_shape = list(array.shape)

    for ax in range(len(binsizes)):
        if binsizes[ax] > 1:
            new_shape[ax] = binsizes[ax]
            new_shape.insert(ax, int(array.shape[ax] / binsizes[ax]))
            binned_array = np.reshape(binned_array, tuple(new_shape))
            binned_array = np.sum(binned_array, axis=ax + 1)
            new_shape = list(binned_array.shape)
    return binned_array


def center_max(arr):
    """
    Finds maximum value in the array, and shifts in each dimension to put the max in a center.

    :param arr: ndarray, array to be centered
    :return: centered array
    """
    shift = (np.array(arr.shape) // 2) - np.unravel_index(np.argmax(arr), arr.shape)
    return np.roll(arr, shift.astype(int), tuple(range(arr.ndim))), shift.astype(int)


def crop_center(arr, crop_shape):
    """
    Crop the center of a 1D, 2D, or 3D NumPy array.

    Parameters
    ----------
    arr : np.ndarray
        Input array (1D, 2D, or 3D).
    crop_shape : int or tuple
        Desired output shape.
        - For 1D: int or tuple of length 1
        - For 2D: tuple of length 2
        - For 3D: tuple of length 3

    Returns
    -------
    np.ndarray
        Center-cropped array.
    """
    arr = np.asarray(arr)

    # Normalize crop_shape to a tuple
    if isinstance(crop_shape, int):
        crop_shape = (crop_shape,)
    elif len(crop_shape) != arr.ndim:
        raise ValueError(f"crop_shape must have {arr.ndim} dimensions.")

    if any(c > s for c, s in zip(crop_shape, arr.shape)):
        raise ValueError("crop_shape cannot be larger than the input array shape.")

    slices = []
    for dim_size, crop_size in zip(arr.shape, crop_shape):
        start = (dim_size - crop_size) // 2
        end = start + crop_size
        slices.append(slice(start, end))

    return arr[tuple(slices)]


def get_central_object_extent(fp: np.ndarray) -> list:
    """
    Calculates extent of central cluster.

    :param fp: ndarray tofind extend for
    :return: list, an extend
    """
    fp_cut = threshold_by_edge(np.abs(fp))
    need = select_central_object(fp_cut)

    # get extend of cluster
    extent = [np.max(s) + 1 - np.min(s) for s in np.nonzero(need)]
    return extent


def get_good_dim(dim, pkg):
    """
    Returns the even dimension that the given package found to be good for fast Fourier transform and not smaller than given dimension. .

    :param dim: int, initial dimension
    :param pkg: python package that will be used for reconstruction: 'np' for numpy, 'cp' for cupy, 'torch' for torch.
    :return: a new dimension
    """
    devlib = get_lib(pkg)
    new_dim = devlib.next_fast_len(dim)
    while new_dim % 2 == 1:
        new_dim += 1
        new_dim = devlib.next_fast_len(new_dim)
    return new_dim


def get_lib(pkg):
    """
    Dynamically imports library module specified in input.

    :param pkg: package acronym
    :return: object, library module
    """
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    elif pkg == 'torch':
        devlib = importlib.import_module('cohere_core.lib.torchlib').torchlib
    else:
        devlib = None

    return devlib


def get_logger(name, ldir=''):
    """
    Creates looger instance that will write to default.log file in a given directory.

    :param name: str, logger name
    :param ldir: str, directory where to create log file
    :return: logger object from logging module
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_file = join(ldir, 'default.log')
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_oversample_ratio(fp: np.ndarray) -> np.ndarray:
    """
    Author: Yudong Yao

    Calculates oversampling ratio.

    :param fp: ndarray to calculate oversampling ratio
    :return: oversample ratio in each dimension
    """
     # autocorrelation
    acp = np.fft.fftshift(np.fft.ifftn(np.abs(fp)**2.))
    aacp = np.abs(acp)

    # get extent
    blob = get_central_object_extent(aacp)

    # correct for underestimation due to thresholding
    correction = [0.025, 0.025, 0.0729][:fp.ndim]

    extent = [
        min(m, s + int(round(f * aacp.shape[i], 1)))
        for i, (s, f, m) in enumerate(zip(blob, correction, aacp.shape))
    ]

    # oversample ratio
    oversample = [
        2. * s / (e + (1 - s % 2)) for s, e in zip(aacp.shape, extent)
    ]
    return np.round(oversample, 3)


def join(*args):
    """
    Operation on path. Joins arguments in Path string and replaces OS separators with Linux type separators.

    :param args: variable number of arguments
    :return: path
    """
    return os.path.join(*args).replace(os.sep, '/')


def normalize(vec):
    """
    Normalizes vector.

    :param vec: vector
    :return: normalized vector
    """
    return vec / np.linalg.norm(vec)


def pad_center(arr, target_shape):
    """
    Pad a 1D, 2D, or 3D NumPy array with zeros so that it is centered
    in a new array of shape `target_shape`.

    Parameters
    ----------
    arr : np.ndarray
        Input 1D, 2D, or 3D array.
    target_shape : int or tuple
        Desired output shape. Must have the same number of dimensions as arr.
        For 1D, this can be an int or a 1-tuple.

    Returns
    -------
    np.ndarray
        Zero-padded array with the original array centered.

    Raises
    ------
    ValueError
        If target_shape has different dimensionality from arr
        or if any target dimension is smaller than the input dimension.
    """
    arr = np.asarray(arr)

    if arr.ndim not in (1, 2, 3):
        raise ValueError("Only 1D, 2D, and 3D arrays are supported.")

    if isinstance(target_shape, int):
        target_shape = (target_shape,)
    else:
        target_shape = tuple(target_shape)

    if len(target_shape) != arr.ndim:
        raise ValueError("target_shape must have the same number of dimensions as arr.")

    if any(t < s for t, s in zip(target_shape, arr.shape)):
        raise ValueError("Each target dimension must be >= the corresponding input dimension.")

    pad_width = []
    for s, t in zip(arr.shape, target_shape):
        total_pad = t - s
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))

    return np.pad(arr, pad_width, mode='constant', constant_values=0)


def read_config(config):
    """
    Checks if the file exists and parses it into a dictionary.

    :param config: str, configuration file name
    :return: dictionary containing parsed configuration, None if the given file does not exist
    """
    config = config.replace(os.sep, '/')
    if not os.path.isfile(config):
        print(config, 'is not a file')
        return None

    param_dict = {}
    input = open(config, 'r')
    line = input.readline()
    while line:
        # Ignore comment lines and move along
        line = line.strip()
        if line.startswith('/') or line.startswith('#'):
            line = input.readline()
            continue
        elif "=" in line:
            param, value = line.split('=')
            # do not replace in strings
            value = value.strip()
            if value.startswith('('):
                value = value.strip().replace('(','[').replace(')',']')
            try:
                param_dict[param.strip()] = ast.literal_eval(value)
            except:
                msg = f'{param}: {value}\n  string value in configuration file should be surrounded by "" '
                raise ValueError(msg)

        line = input.readline()
    input.close()
    return param_dict


def read_tif(filename):
    """
    This method reads tif type file and returns the data as ndarray.

    :param filename: file name
    :return: ndarray with the tif file data
    """
    return tf.imread(filename.replace(os.sep, '/')).transpose()


def resample(IN, dim):
    """
    Author: Yudong Yao

    Resamples to new dimensions.

    :param IN: ndarray
    :param dim: new dim
    :return: resampled array
    """
    ft = np.fft.fftshift(np.fft.fftn(IN)) / np.prod(IN.shape)

    pad_value = np.array(dim) // 2 - np.array(ft.shape) // 2
    pad = [[pad_value[0], pad_value[0]], [pad_value[1], pad_value[1]],
           [pad_value[2], pad_value[2]]]
    ft_resize = adjust_dimensions(ft, pad)
    output = np.fft.ifftn(np.fft.ifftshift(ft_resize)) * np.prod(dim)
    return output


def select_central_object(fp: np.ndarray) -> np.ndarray:
    """
    Author: Yudong Yao

    Returns array with central object from input array.

    :param fp: array
    :return: central object array
    """
    # import scipy.ndimage as ndimage
    zero = 1e-6
    binary = np.abs(fp)
    binary[binary > zero] = 1
    binary[binary <= zero] = 0

    # cluster by connectivity
    struct = ndi.morphology.generate_binary_structure(fp.ndim, 1).astype("uint8")
    label, nlabel = ndi.label(binary, structure=struct)

    # select largest cluster
    select = np.argmax(np.bincount(np.ravel(label))[1:]) + 1

    binary[label != select] = 0

    fp[binary == 0] = 0
    return fp


def save_tif(arr, filename):
    """
    Saves array in tif format file.

    :param arr: ndarray, array to save
    :param filename: file name
    """
    tf.imwrite(filename.replace(os.sep, '/'), np.abs(arr).transpose().astype(np.float32))


def save_metrics(errs, dir, metrics=None):
    """
    Saves arrays metrics and errors by iterations in text file.

    :param errs: list of "chi" error by iteration
    :param dir: directory to write the file containing array metrics
    :param metrics: dictionary with metric type keys, and metric values
    """
    metric_file = join(dir, 'summary')
    linesep = os.linesep
    with open(metric_file, 'w+') as mf:
        if metrics is not None:
            mf.write(f'metric     result{linesep}')
            for key in metrics:
                value = metrics[key]
                mf.write(f'{key} = {str(value)}{linesep}')
        mf.write(f'{linesep}errors by iteration{linesep}')
        for er in errs:
            mf.write(f'{er} ')
    mf.close()


def threshold_by_edge(fp: np.ndarray) -> np.ndarray:
    """
    Author: Yudong Yao

    :param fp:
    :return:
    """
    # threshold by left edge value
    mask = np.ones_like(fp, dtype=bool)
    mask[tuple([slice(1, None)] * fp.ndim)] = 0
    zero = 1e-6
    cut = np.max(fp[mask])
    binary = np.zeros_like(fp)
    binary[(np.abs(fp) > zero) & (fp > cut)] = 1
    return binary


def write_config(param_dict, config):
    """
    Writes configuration to a file.

    :param param_dict: dictionary containing configuration parameters
    :param config: configuration file name the parameters will be written into
    """
    with open(config.replace(os.sep, '/'), 'w+') as cf:
        cf.truncate(0)
        linesep = os.linesep
        for key, value in param_dict.items():
            if type(value) == str:
                value = f'"{value}"'
            cf.write(f'{key} = {str(value)}{linesep}')


def write_plot_errors(save_dir):
    """
    Creates python executable that draw plot of error by iteration. It assumes that the given directory
    contains "errors.npy" file

    :param save_dir: directory containing errors.npy file
    """
    plot_file = join(save_dir, 'plot_errors.py')
    f = open(plot_file, 'w+')
    f.write("#! /usr/bin/env python\n")
    f.write("import matplotlib.pyplot as plt\n")
    f.write("import numpy as np\n")
    f.write("import sys\n")
    f.write("import os\n")
    f.write("current_dir = sys.path[0]\n")
    f.write("errs = np.load(current_dir + '/errors.npy').tolist()\n")
    f.write("errs.pop(0)\n")
    f.write("plt.plot(errs)\n")
    f.write("plt.ylabel('errors')\n")
    f.write("plt.show()")
    f.close()
    st = os.stat(plot_file)
    os.chmod(plot_file, st.st_mode | stat.S_IEXEC)


# https://stackoverflow.com/questions/51503672/decorator-for-timeit-timeit-method/51503837#51503837
from functools import wraps
from time import time

def measure(func):
    @wraps(func)
    def _time_it(*args, **kwargs):
        start = int(round(time() * 1000))
        try:
            return func(*args, **kwargs)
        finally:
            end_ = int(round(time() * 1000)) - start
            print("Total execution time: {end_ if end_ > 0 else 0} ms")

    return _time_it
