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


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_logger',
           'read_tif',
           'save_tif',
           'read_config',
           'write_config',
           'read_results',
           'save_results',
           'save_metrics',
           'write_plot_errors',
           'get_good_dim',
           'threshold_by_edge',
           'select_central_object',
           'get_central_object_extent',
           'get_oversample_ratio',
           'Resize',
           'binning',
           'crop_center',
           'pad_center',
           'center_max',
           'adjust_dimensions',
           'normalize',
           'get_gpu_load',
           'get_gpu_distribution',
           'estimate_no_proc',
           'arr_property'
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
    return tf.imread(filename.replace(os.sep, '/')).transpose()


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
    tf.imsave(filename.replace(os.sep, '/'), np.abs(arr).transpose().astype(np.float32))


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


def write_config(param_dict, config):
    """
    Writes configuration to a file.
    Parameters
    ----------
    param_dict : dict
        dictionary containing configuration parameters
    config : str
        configuration name theparameters will be written into
    """
    with open(config.replace(os.sep, '/'), 'w+') as f:
        f.truncate(0)
        for key, value in param_dict.items():
            if type(value) == str:
                value = '"' + value + '"'
            f.write(key + ' = ' + str(value) + os.linesep)


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


def select_central_object(fp: np.ndarray) -> np.ndarray:
    """
    Author: Yudong Yao

    :param fp:
    :return:
    """
    # import scipy.ndimage as ndimage
    zero = 1e-6
    binary = np.abs(fp)
    binary[binary > zero] = 1
    binary[binary <= zero] = 0

    # cluster by connectivity
    struct = ndi.morphology.generate_binary_structure(fp.ndim,
                                                          1).astype("uint8")
    label, nlabel = ndi.label(binary, structure=struct)

    # select largest cluster
    select = np.argmax(np.bincount(np.ravel(label))[1:]) + 1

    binary[label != select] = 0

    fp[binary == 0] = 0
    return fp


def get_central_object_extent(fp: np.ndarray) -> list:
    """
    Author: Yudong Yao

    :param fp:
    :return:
    """
    fp_cut = threshold_by_edge(np.abs(fp))
    need = select_central_object(fp_cut)

    # get extend of cluster
    extent = [np.max(s) + 1 - np.min(s) for s in np.nonzero(need)]
    return extent


def get_oversample_ratio(fp: np.ndarray) -> np.ndarray:
    """
    Author: Yudong Yao

    :param fp: diffraction pattern
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


def Resize(IN, dim):
    """
    Author: Yudong Yao

    :param IN:
    :param dim:
    :return:
    """
    ft = np.fft.fftshift(np.fft.fftn(IN)) / np.prod(IN.shape)

    pad_value = np.array(dim) // 2 - np.array(ft.shape) // 2
    pad = [[pad_value[0], pad_value[0]], [pad_value[1], pad_value[1]],
           [pad_value[2], pad_value[2]]]
    ft_resize = adjust_dimensions(ft, pad)
    output = np.fft.ifftn(np.fft.ifftshift(ft_resize)) * np.prod(dim)
    return output


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


def crop_center(arr, new_shape):
    """
    This function crops the array to the new size, leaving the array in the center.
    The new_size must be smaller or equal to the original size in each dimension.
    Parameters
    ----------
    arr : ndarray
        the array to crop
    new_shape : tuple
        new size
    Returns
    -------
    cropped : ndarray
        the cropped array
    """
    shape = arr.shape
    principio = []
    finem = []
    for i in range(3):
        principio.append(int((shape[i] - new_shape[i]) / 2))
        finem.append(principio[i] + new_shape[i])
    if len(shape) == 1:
        cropped = arr[principio[0]: finem[0]]
    elif len(shape) == 2:
        cropped = arr[principio[0]: finem[0], principio[1]: finem[1]]
    elif len(shape) == 3:
        cropped = arr[principio[0]: finem[0], principio[1]: finem[1], principio[2]: finem[2]]
    else:
        raise NotImplementedError
    return cropped


def pad_center(arr, new_shape):
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
    centered = np.zeros(new_shape, arr.dtype)
    if len(shape) == 1:
        centered[: shape[0]] = arr
    elif len(shape) == 2:
        centered[: shape[0], : shape[1]] = arr
    elif len(shape) == 3:
        centered[: shape[0], : shape[1], : shape[2]] = arr

    for i in range(len(new_shape)):
        centered = np.roll(centered, (new_shape[i] - shape[i] + 1) // 2, i)

    return centered


def center_max(arr, center_shift):
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
    shift = (np.array(arr.shape)/2) - np.unravel_index(np.argmax(arr), arr.shape) + np.array(center_shift)
    return np.roll(arr, shift.astype(int), tuple(range(arr.ndim))), shift.astype(int)


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


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_gpu_load(mem_size, ids):
    """
    This function is only used when running on Linux OS. The GPUtil module is not supported on Mac.
    This function finds available GPU memory in each GPU that id is included in ids list. It calculates
    how many reconstruction can fit in each GPU available memory.
    Parameters
    ----------
    mem_size : int
        array size
    ids : list
        list of GPU ids user configured for use
    Returns
    -------
    list
        list of available runs aligned with the GPU id list
    """
    import GPUtil

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getGPUs()
    total_avail = 0
    available_dir = {}
    for gpu in gpus:
        if gpu.id in ids:
            free_mem = gpu.memoryFree
            avail_runs = int(free_mem / mem_size)
            if avail_runs > 0:
                total_avail += avail_runs
                available_dir[gpu.id] = avail_runs
    available = []
    for id in ids:
        try:
            avail_runs = available_dir[id]
        except:
            avail_runs = 0
        available.append(avail_runs)
    return available


def get_gpu_distribution(runs, available):
    """
    Finds how to distribute the available runs to perform the given number of runs.
    Parameters
    ----------
    runs : int
        number of reconstruction requested
    available : list
        list of available runs aligned with the GPU id list
    Returns
    -------
    list
        list of runs aligned with the GPU id list, the runs are equally distributed across the GPUs
    """
    from functools import reduce

    all_avail = reduce((lambda x, y: x + y), available)
    distributed = [0] * len(available)
    sum_distr = 0
    while runs > sum_distr and all_avail > 0:
        # balance distribution
        for i in range(len(available)):
            if available[i] > 0:
                available[i] -= 1
                all_avail -= 1
                distributed[i] += 1
                sum_distr += 1
                if sum_distr == runs:
                    break
    return distributed


def estimate_no_proc(arr_size, factor):
    """
    Estimates number of processes the prep can be run on. Determined by number of available cpus and size
    of array.
    Parameters
    ----------
    arr_size : int
        size of array
    factor : int
        an estimate of how much memory is required to process comparing to array size
    Returns
    -------
    int
        number of processes
    """
    from multiprocessing import cpu_count
    import psutil

    ncpu = cpu_count()
    freemem = psutil.virtual_memory().available
    nmem = freemem / (factor * arr_size)
    # decide what limits, ncpu or nmem
    if nmem > ncpu:
        return ncpu
    else:
        return int(nmem)


def arr_property(arr):
    """
    Used only in development. Prints max value of the array and max coordinates.

    Parameters
    ----------
    arr : ndarray
        array to find max
    """
    arr1 = abs(arr)
    #print('norm', np.sum(pow(abs(arr), 2)))
    print('mean across all data', arr1.mean())
    print('std all data', arr1.std())
    print('mean non zeros only', arr1[np.nonzero(arr1)].mean())
    print('std non zeros only', arr1[np.nonzero(arr1)].std())
    # max_coordinates = list(np.unravel_index(np.argmax(arr1), arr.shape))
    # print('max coords, value', max_coordinates, arr[max_coordinates[0], max_coordinates[1], max_coordinates[2]])


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

