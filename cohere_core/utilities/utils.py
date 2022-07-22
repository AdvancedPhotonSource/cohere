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


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_logger',
           'read_tif',
           'save_tif',
           'read_config',
           'get_good_dim',
           'binning',
           'get_centered',
           'get_zero_padded_centered',
           'adjust_dimensions',
           'gaussian',
           'gauss_conv_fft',
           'read_results',
           'save_metrics',
           'write_plot_errors',
           'save_results',
           'arr_property',
           ]


def get_logger(name, ldir=''):
    """
    Creates looger instance that will write to default.log file in a given directory.
    Parameters
    ----------
    name : str
        logger name
    ldir : str
        directory where to create log file
    Returns
    -------
    logger
        logger object from logging module
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    log_file = ldir + '/default.log'
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def read_tif(filename):
    """
    This method reads tif type file and returns the data as ndarray.

    Parameters
    ----------
    filename : str
        tif format file name

    Returns
    -------
    ndarray
        an array containing the data parsed from the file
    """
    ar = tf.imread(filename.replace(os.sep, '/')).transpose()
    return ar


def save_tif(arr, filename):
    """
    This method saves array in tif format file.

    Parameters
    ----------
    arr : ndarray
        array to save
    filename : str
        tif format file name
    """
    tf.imsave(filename.replace(os.sep, '/'), arr.transpose().astype(np.float32))


def read_config(config):
    """
    This function gets configuration file. It checks if the file exists and parses it into a dictionary.

    Parameters
    ----------
    config : str
        configuration file name

    Returns
    -------
    dict
        dictionary containing parsed configuration, None if the given file does not exist
    """
    import ast

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
        if line.startswith('//') or line.startswith('#'):
            line = input.readline()
            continue
        elif "=" in line:
            param, value = line.split('=')
            # do not replace in strings
            value = value.strip()
            if value.startswith('('):
                value = value.strip().replace('(','[').replace(')',']')
            param_dict[param.strip()] = ast.literal_eval(value)
        line = input.readline()
    input.close()
    return param_dict


def get_good_dim(dim):
    """
    This function calculates the dimension supported by opencl library (i.e. is multiplier of 2, 3, or 5) and is closest to the given starting dimension.
    If the dimension is not supported the function adds 1 and verifies the new dimension. It iterates until it finds supported value.

    Parameters
    ----------
    dim : int
        initial dimension
    Returns
    -------
    int
        a dimension that is supported by the opencl library, and closest to the original dimension
    """

    def is_correct(x):
        sub = x
        if sub % 3 == 0:
            sub = sub / 3
        if sub % 3 == 0:
            sub = sub / 3
        if sub % 5 == 0:
            sub = sub / 5
        while sub % 2 == 0:
            sub = sub / 2
        return sub == 1

    new_dim = dim
    if new_dim % 2 == 1:
        new_dim += 1
    while not is_correct(new_dim):
        new_dim += 2
    return new_dim


def binning(array, binsizes):
    """
    This function does the binning of the array. The array is binned in each dimension by the corresponding binsizes elements.
    If binsizes list is shorter than the array dimensions, the remaining dimensions are not binned.

    Parameters
    ----------
    array : ndarray
        the original array to be binned
    binsizes : list
        a list defining binning factors for corresponding dimensions

    Returns
    -------
    ndarray
        binned array
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


def get_centered(arr, center_shift):
    """
    This function finds maximum value in the array, and puts it in a center of a new array. If center_shift is not zeros,
     the array will be shifted accordingly. The shifted elements are rolled into the other end of array.

    Parameters
    ----------
    arr : ndarray
        the original array to be centered

    center_shift : list
        a list defining shift of the center

    Returns
    -------
    ndarray
        centered array
    """
    max_coordinates = list(np.unravel_index(np.argmax(arr), arr.shape))
    max_coordinates = np.add(max_coordinates, center_shift)
    shape = arr.shape
    centered = arr
    for i in range(len(max_coordinates)):
        centered = np.roll(centered, int(shape[i] / 2) - max_coordinates[i], i)

    return centered


def get_zero_padded_centered(arr, new_shape):
    """
    This function pads the array with zeros to the new shape with the array in the center.
    Parameters
    ----------
    arr : ndarray
        the original array to be padded
    new_shape : tuple
        new dimensions
    Returns
    -------
    ndarray
        the zero padded centered array
    """
    shape = arr.shape
    pad = []
    c_vals = []
    for i in range(len(new_shape)):
        pad.append((0, new_shape[i] - shape[i]))
        c_vals.append((0.0, 0.0))
    arr = np.lib.pad(arr, (pad), 'constant', constant_values=c_vals)

    centered = arr
    for i in range(len(new_shape)):
        centered = np.roll(centered, int((new_shape[i] - shape[i] + 1) / 2), i)

    return centered


def adjust_dimensions(arr, pads):
    """
    This function adds to or subtracts from each dimension of the array elements defined by pad. If the pad is positive, the array is padded in this dimension. If the pad is negative, the array is cropped.
    The dimensions of the new array is then adjusted to be supported by the opencl library.

    Parameters
    ----------
    arr : ndarray
        the array to pad/crop

    pad : list
        list of three pad values, for each dimension

    Returns
    -------
    ndarray
        the padded/cropped array
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
        # find a good dimension and find padding
        temp_dim = old_dims[i] + pad[0] + pad[1]
        if temp_dim > 1:
            new_dim = get_good_dim(temp_dim)
        else:
            new_dim = temp_dim
        added = new_dim - temp_dim
        # if the pad is positive
        pad_front = max(0, pad[0]) + int(added / 2)
        pad_end = new_dim - dims[i] - pad_front
        new_pad.append((pad_front, pad_end))
        c_vals.append((0.0, 0.0))
    adjusted = np.pad(cropped, new_pad, 'constant', constant_values=c_vals)

    return np.squeeze(adjusted)


def gaussian(shape, sigmas, alpha=1):
    """
    Calculates Gaussian distribution grid in given dimensions.

    Parameters
    ----------
    shape : tuple
        shape of the grid
    sigmas : list
        sigmas in all dimensions
    alpha : float
        a multiplier
    Returns
    -------
    ndarray
        Gaussian distribution grid
    """
    grid = np.full(shape, 1.0)
    for i in range(len(shape)):
        # prepare indexes for tile and transpose
        tile_shape = list(shape)
        tile_shape.pop(i)
        tile_shape.append(1)
        trans_shape = list(range(len(shape) - 1))
        trans_shape.insert(i, len(shape) - 1)

        multiplier = - 0.5 * alpha / pow(sigmas[i], 2)
        line = np.linspace(-(shape[i] - 1) / 2.0, (shape[i] - 1) / 2.0, shape[i])
        gi = np.tile(line, tile_shape)
        gi = np.transpose(gi, tuple(trans_shape))
        exponent = np.power(gi, 2) * multiplier
        gi = np.exp(exponent)
        grid = grid * gi

    grid_total = np.sum(grid)
    return grid / grid_total


def gauss_conv_fft(arr, sigmas):
    """
    Calculates convolution of array with the Gaussian.

    A Guassian distribution grid is calculated with the array dimensions. Fourier transform of the array is
    multiplied by the grid and inverse transformed. A correction is calculated and applied.

    Parameters
    ----------
    arr : ndarray
        subject array
    sigmas : list
        sigmas in all dimensions
    Returns
    -------
    ndarray
        convolution array
    """
    arr_sum = np.sum(abs(arr))
    arr_f = np.fft.ifftshift(np.fft.fftn(np.fft.ifftshift(arr)))
    shape = list(arr.shape)
    for i in range(len(sigmas)):
        sigmas[i] = shape[i] / 2.0 / np.pi / sigmas[i]
    convag = arr_f * gaussian(shape, sigmas)
    convag = np.fft.ifftshift(np.fft.ifftn(np.fft.ifftshift(convag)))
    convag = convag.real
    convag = np.clip(convag, 0, None)
    correction = arr_sum / np.sum(convag)
    convag *= correction
    return convag


def read_results(read_dir):
    """
    Reads results and returns array representation.

    Parameters
    ----------
    read_dir : str
        directory to read the results from
    Returns
    -------
    ndarray, ndarray, ndarray (or None)
        image, support, and coherence arrays
    """
    read_dir = read_dir.replace(os.sep, '/')
    try:
        imagefile = read_dir + '/image.npy'
        image = np.load(imagefile)
    except:
        image = None

    try:
        supportfile = read_dir + '/support.npy'
        support = np.load(supportfile)
    except:
        support = None

    try:
        cohfile = read_dir + '/coherence.npy'
        coh = np.load(cohfile)
    except:
        coh = None

    return image, support, coh


def save_metrics(errs, dir, metrics=None):
    """
    Saves arrays metrics and errors by iterations in text file.

    Parameters
    ----------
    errs : list
        list of "chi" error by iteration

    dir : str
        directory to write the file containing array metrics

    metrics : dict
        dictionary with metric type keys, and metric values
    """
    dir = dir.replace(os.sep, '/')
    metric_file = dir + '/summary'
    with open(metric_file, 'w+') as f:
        if metrics is not None:
            f.write('metric     result\n')
            for key in metrics:
                value = metrics[key]
                f.write(key + ' = ' + str(value) + '\n')
        f.write('\nerrors by iteration\n')
        for er in errs:
            f.write(str(er) + ' ')
    f.close()


def write_plot_errors(save_dir):
    """
    Creates python executable that draw plot of error by iteration. It assumes that the given directory
    contains "errors.npy" file

    Parameters
    ----------
    save_dir : str
        directory containing errors.npy file
    """
    save_dir = save_dir.replace(os.sep, '/')
    plot_file = save_dir + '/plot_errors.py'
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


def save_results(image, support, coh, errs, save_dir, metric=None):
    """
    Saves results of reconstruction. Saves the following files: image.np, support.npy, errors.npy,
    optionally coherence.npy, plot_errors.py, graph.npy, flow.npy, iter_array.npy


    Parameters
    ----------
    image : ndarray
        reconstructed image array

    support : ndarray
        support array related to the image

    coh : ndarray
        coherence array when pcdi feature is active, None otherwise

    errs : ndarray
        errors "chi" by iterations

    save_dir : str
        directory to write the files

    metrics : dict
        dictionary with metric type keys, and metric values
    """
    save_dir = save_dir.replace(os.sep, '/')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_file = save_dir + '/image'
    np.save(image_file, image)
    support_file = save_dir + '/support'
    np.save(support_file, support)

    errs_file = save_dir + '/errors'
    np.save(errs_file, errs)
    if not coh is None:
        coh_file = save_dir + '/coherence'
        np.save(coh_file, coh)

    write_plot_errors(save_dir)

    save_metrics(errs, save_dir, metric)


def arr_property(arr):
    """
    Used only in development. Prints max value of the array and max coordinates.

    Parameters
    ----------
    arr : ndarray
        array to find max
    """
    arr1 = abs(arr)
    print('norm', np.sum(pow(abs(arr), 2)))
    max_coordinates = list(np.unravel_index(np.argmax(arr1), arr.shape))
    print('max coords, value', max_coordinates, arr[max_coordinates[0], max_coordinates[1], max_coordinates[2]])


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