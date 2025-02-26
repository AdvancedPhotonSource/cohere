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
import GPUtil
from functools import reduce
import subprocess
import ast
import importlib


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c), UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['join',
           'get_logger',
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
           'get_avail_gpu_runs',
           'get_one_dev',
           'get_gpu_use',
           'estimate_no_proc',
           'get_lib',
           ]


def join(*args):
    return os.path.join(*args).replace(os.sep, '/')


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
    log_file = join(ldir, 'default.log')
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
                print(f'{param}: {value}\n  string value should be surrounded by "" ')
                raise

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
    with open(config.replace(os.sep, '/'), 'w+') as cf:
        cf.truncate(0)
        linesep = os.linesep
        for key, value in param_dict.items():
            if type(value) == str:
                value = f'"{value}"'
            cf.write(f'{key} = {str(value)}{linesep}')


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
    try:
        imagefile = join(read_dir, 'image.npy')
        image = np.load(imagefile)
    except:
        image = None

    try:
        supportfile = join(read_dir, 'support.npy')
        support = np.load(supportfile)
    except:
        support = None

    try:
        cohfile = join(read_dir, 'coherence.npy')
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
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    image_file = join(save_dir, 'image')
    np.save(image_file, image)
    support_file = join(save_dir, 'support')
    np.save(support_file, support)

    errs_file = join(save_dir, 'errors')
    np.save(errs_file, errs)
    if not coh is None:
        coh_file = join(save_dir, 'coherence')
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
            mf.write(f'str({er}) ')
    mf.close()


def write_plot_errors(save_dir):
    """
    Creates python executable that draw plot of error by iteration. It assumes that the given directory
    contains "errors.npy" file

    Parameters
    ----------
    save_dir : str
        directory containing errors.npy file
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
        while sub % 3 == 0:
            sub = sub / 3
        while sub % 5 == 0:
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
    :param fp:
    :return:
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


def get_central_object_extent(fp: np.ndarray) -> list:
    """
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
    shift = (np.array(arr.shape)/2) - np.unravel_index(np.argmax(arr), arr.shape) + np.array(center_shift)
    return np.roll(arr, shift.astype(int), tuple(range(arr.ndim))), shift.astype(int)


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


def normalize(vec):
    return vec / np.linalg.norm(vec)


def get_one_dev(ids):
    """
    Returns GPU ID that is included in the configuration, is on a local node, and has the most available memory.

    :param ids: list or string or dict
        list of gpu ids, or string 'all' indicating all GPUs included, or dict by hostname
    :return: int
        selected GPU ID
    """
    import socket

    # if cluster configuration, look only at devices on local machine
    if issubclass(type(ids), dict):  # a dict with cluster configuration
        ids = ids[socket.gethostname()]  # configured devices on local host

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getGPUs()
    dev = -1
    max_mem = 0
    # select one with the highest availbale memory
    for gpu in gpus:
        if ids == 'all' or gpu.id in ids:
            free_mem = gpu.memoryFree
            if free_mem > max_mem:
                dev = gpu.id
                max_mem = free_mem
    return dev


def get_avail_gpu_runs(devices, run_mem):
    """
    Finds how many jobs of run_mem size can run on configured GPUs on local host.

    :param devices: list or string
        list of GPU IDs or 'all' if configured to use all available GPUs
    :param run_mem: int
        size of GPU memory (in MB) needed for one job
    :return: dict
        pairs of GPU IDs, number of available jobs
    """
    import GPUtil

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    gpus = GPUtil.getGPUs()
    available = {}

    for gpu in gpus:
        if devices == 'all' or gpu.id in devices:
            available[gpu.id] = gpu.memoryFree // run_mem

    return available


def get_avail_hosts_gpu_runs(devices, run_mem):
    """
    This function is called in a cluster configuration case, i.e. devices parameter is configured as dictionary of hostnames and GPU IDs (either list of the IDs or 'all' for all GPUs per host).
    It starts mpi subprocess that targets each of the configured host. The subprocess returns tuples with hostname and available GPUs. The tuples are converted into dictionary and returned.

    :param devices:
    :param run_mem:
    :return:
    """
    hosts = ','.join(devices.keys())
    script = join(os.path.realpath(os.path.dirname(__file__)), 'host_utils.py')
    command = ['mpiexec', '-n', str(len(devices)), '--host', hosts, 'python', script, str(devices), str(run_mem)]
    result = subprocess.run(command, stdout=subprocess.PIPE, text=True).stdout
    mem_map = {}
    for entry in result.splitlines():
        host_devs = ast.literal_eval(entry)
        mem_map[host_devs[0]] = host_devs[1]
    return mem_map


def get_balanced_load(avail_runs, runs):
    """
    This function distributes the runs proportionally to the GPUs availability.
    If number of available runs is less or equal to the requested runs, the input parameter avail_runs becomes load.
    The function also returns number of available runs.

    :param avail_runs: dict
        keys are GPU IDs, and values are available runs
        for cluster configuration the keys are prepended with the hostnames
    :param runs: int
        number of requested jobs
    :return: dict, int
        a dictionary with the same structure as avail_runs input parameter, but with values indicating runs modified to achieve balanced distribution.
    """
    if len(avail_runs) == 0:
        return {}

    # if total number of available runs is less or equal runs, return the avail_runs,
    # and total number of available jobs
    total_available = reduce((lambda x, y: x + y), avail_runs.values())
    if total_available <= runs:
        return avail_runs, total_available

    # initialize variables for calculations
    need_runs = runs
    available = total_available
    load = {}

    # add one run from each available
    for k, v in avail_runs.items():
        if v > 0:
            load[k] = 1
            avail_runs[k] = v - 1
            need_runs -= 1
            if need_runs == 0:
                return load, runs
            available -= 1

    # use proportionally from available
    distributed = 0
    ratio = need_runs / available
    for k, v in avail_runs.items():
        if v > 0:
            share = int(v * ratio)
            load[k] = load[k] + share
            avail_runs[k] = v - share
            distributed += share
    need_runs -= distributed
    available -= distributed

    if need_runs > 0:
        # need to add the few remaining
        for k, v in avail_runs.items():
            if v > 0:
                load[k] = load[k] + 1
                need_runs -= 1
                if need_runs == 0:
                    break

    return load, runs


def get_gpu_use(devices, no_jobs, job_size):
    """
    Determines available GPUs that match configured devices, and selects the optimal distribution of jobs on available devices. If devices is configured as dict (i.e. cluster configuration) then a file "hosts" is created in the running directory. This file contains hosts names and number of jobs to run on that host.
    Parameters
    ----------
    devices : list or dict or 'all'
        Configured parameter. list of GPU ids to use for jobs or 'all' if all GPUs should be used. If cluster configuration, then
        it is dict with keys being host names.
    no_jobs : int
        wanted number of jobs
    job_size : float
        a GPU memory requirement to run one job
    Returns
    -------
    picked_devs : list or list of lists(if cluster conf)
        list of GPU ids that were selected for the jobs
    available jobs : int
        number of jobs allocated on all GPUs
    cluster_conf : boolean
        True is cluster configuration
    """

    def unpack_load(load):
        picked_devs = []
        for ds in [[k] * int(v) for k, v in load.items()]:
            picked_devs.extend(ds)
        return picked_devs

    if type(devices) != dict:  # a configuration for local host
        hostfile_name = None
        avail_jobs = get_avail_gpu_runs(devices, job_size)
        balanced_load, avail_jobs_no = get_balanced_load(avail_jobs, no_jobs)
        picked_devs = unpack_load(balanced_load)
    else:  # cluster configuration
        hosts_avail_jobs = get_avail_hosts_gpu_runs(devices, job_size)
        avail_jobs = {}
        # collapse the host dict into one dict by adding hostname in front of key (gpu id)
        for k, v in hosts_avail_jobs.items():
            host_runs = {(f'{k}_{str(kv)}'): vv for kv, vv in v.items()}
            avail_jobs.update(host_runs)
        balanced_load, avail_jobs_no = get_balanced_load(avail_jobs, no_jobs)

        # un-collapse the balanced load by hosts
        host_balanced_load = {}
        for k, v in balanced_load.items():
            idx = k.rfind('_')
            host = k[:idx]
            if host not in host_balanced_load:
                host_balanced_load[host] = {}
            host_balanced_load[host].update({int(k[idx + 1:]): v})

        # create hosts file and return corresponding picked devices
        hosts_picked_devs = [(k, unpack_load(v)) for k, v in host_balanced_load.items()]

        picked_devs = []
        hostfile_name = f'hostfile_{os.getpid()}'
        host_file = open(hostfile_name, mode='w+')
        linesep = os.linesep
        for h, ds in hosts_picked_devs:
            host_file.write(f'{h}:{str(len(ds))}{linesep}')
            picked_devs.append(ds)
        host_file.close()

    return picked_devs, int(min(avail_jobs_no, no_jobs)), hostfile_name


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


def get_lib(pkg):
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    elif pkg == 'torch':
        devlib = importlib.import_module('cohere_core.lib.torchlib').torchlib
    else:
        devlib = None

    return devlib


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
