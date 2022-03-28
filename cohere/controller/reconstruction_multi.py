# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This module controls the multi reconstruction process.
"""

import os
import numpy as np
import importlib
import cohere.utilities.utils as ut
import cohere.controller.phasing as calc
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Process
from functools import partial
import threading


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['single_rec_process',
           'assign_gpu',
           'multi_rec',
           'reconstruction']


class Devices:
    def __init__(self, devices):
        self.devices = devices
        self.index = 0

    def assign_gpu(self):
        thr = threading.current_thread()
        thr.gpu = self.devices[self.index]
        self.index = self.index + 1


def set_lib(pkg, ndim=None):
    global devlib
    if pkg == 'af':
        if ndim == 1:
            devlib = importlib.import_module('cohere.lib.aflib').aflib1
        elif ndim == 2:
            devlib = importlib.import_module('cohere.lib.aflib').aflib2
        elif ndim == 3:
            devlib = importlib.import_module('cohere.lib.aflib').aflib3
        else:
            raise NotImplementedError
    elif pkg == 'cp':
        devlib = importlib.import_module('cohere.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere.lib.nplib').nplib
    calc.set_lib(devlib, pkg=='af')


def single_rec_process(metric_type, gen, rec_attrs):
    """
    This function runs a single reconstruction process.

    Parameters
    ----------
    proc : str
        string defining library used 'cpu' or 'opencl' or 'cuda'

    pars : Object
        Params object containing parsed configuration

    data : numpy array
        data array

    req_metric : str
        defines metric that will be used if GA is utilized

    dirs : list
        tuple of two elements: directory that contain results of previous run or None, and directory where the results of this processing will be saved

    Returns
    -------
    metric : float
        a calculated characteristic of the image array defined by the metric
    """
    worker, prev_dir, save_dir = rec_attrs
    thr = threading.current_thread()
    if worker.init_dev(thr.gpu) < 0:
        worker = None
        metric = None
    else:
        worker.init(prev_dir, gen)

        if gen is not None and gen > 0:
            worker.breed()

        ret_code = worker.iterate()
        if ret_code == 0:
            worker.save_res(save_dir)
            metric = worker.get_metric(metric_type)
        else:    # bad reconstruction
            metric = None
    return metric


def multi_rec(lib, save_dir, devices, pars, datafile, prev_dirs, metric_type='chi', gen=None, q=None):
    """
    This function controls the multiple reconstructions.

    Parameters
    ----------
    save_dir : str
        a directory where the subdirectories will be created to save all the results for multiple reconstructions

    proc : str
        a string indicating the processor type (cpu, cuda or opencl)

    data : numpy array
        data array

    pars : Object
        Params object containing parsed configuration

    devices : list
        list of GPUs available for this reconstructions

    previous_dirs : list
        list directories that hols results of previous reconstructions if it is continuation

    metric : str
        a metric defining algorithm by which to evaluate the image array

    Returns
    -------
    save_dirs : list
        list of directories where to save results
    evals : list
        list of evaluation results of image arrays
    """
    evals = []

    def collect_result(result):
        for r in result:
            evals.append(r)

    if lib == 'af' or lib == 'cpu' or lib == 'opencl' or lib == 'cuda':
        if datafile.endswith('tif') or datafile.endswith('tiff'):
            try:
                data = ut.read_tif(datafile)
            except:
                print ('could not load data file', datafile)
                return
        elif datafile.endswith('npy'):
            try:
                data = np.load(datafile)
            except:
                print ('could not load data file', datafile)
                return
        else:
            print ('no data file found')
            return
        print('data shape', data.shape)
        set_lib('af', len(data.shape))
        if lib != 'af':
            devlib.set_backend(lib)
    else:
        set_lib(lib)

    if 'reconstructions' in pars:
        reconstructions = pars['reconstructions']
    else:
        reconstructions = 1
    workers = [calc.Rec(pars, datafile) for _ in range(reconstructions)]
    dev_obj = Devices(devices)
    iterable = []
    save_dirs = []

    for i in range(len(workers)):
        save_sub = os.path.join(save_dir, str(i))
        save_dirs.append(save_sub)
        iterable.append((workers[i], prev_dirs[i], save_sub))
    func = partial(single_rec_process, metric_type, gen)
    with Pool(processes=len(devices), initializer=dev_obj.assign_gpu, initargs=()) as pool:
        pool.map_async(func, iterable, callback=collect_result)
        pool.close()
        pool.join()
        pool.terminate()

    # remove the unsuccessful reconstructions
    for i, e in reversed(list(enumerate(evals))):
        if e is None:
            evals.pop(i)
            save_dirs.pop(i)

    if q is not None:
        q.put((save_dirs, evals))


def reconstruction(lib, conf_file, datafile, dir, devices):
    """
    This function controls multiple reconstructions.

    Parameters
    ----------
    proc : str
        processor to run on (cpu, opencl, or cuda)

    conf_file : str
        configuration file with reconstruction parameters

    datafile : str
        name of the file with initial data

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

    Returns
    -------
    nothing
    """
    pars = ut.read_config(conf_file)

    prev_dirs = []
    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'
    if pars['init_guess'] == 'continue':
        continue_dir = pars.continue_dir
        for sub in os.listdir(continue_dir):
            image, support, coh = ut.read_results(os.path.join(continue_dir, sub) + '/')
            if image is not None:
                prev_dirs.append(sub)
    elif pars['init_guess'] == 'AI_guess':
        print('multiple reconstruction do not support AI_guess initial guess')
        return
    else:
        if 'reconstructions' in pars:
            reconstructions = pars['reconstructions']
        else:
            reconstructions = 1
        for _ in range(reconstructions):
            prev_dirs.append(None)
    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        filename = conf_file.split('/')[-1]
        save_dir = os.path.join(dir, filename.replace('config_rec', 'results_phasing'))

    p = Process(target=multi_rec, args=(lib, save_dir, devices, pars, datafile, prev_dirs))
    p.start()
    p.join()
