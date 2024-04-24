# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_GA
=============================

This module controls a reconstructions using genetic algorithm (GA).
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI x or using command line scripts x.
"""

import numpy as np
import os
import cohere_core.utilities.utils as ut
import cohere_core.utilities.ga_utils as gaut
from multiprocessing import Queue
import shutil
import importlib
import cohere_core.controller.phasing as calc
from multiprocessing.pool import ThreadPool as Pool
from multiprocessing import Process
from functools import partial
import threading


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


class Devices:
    def __init__(self, devices):
        self.devices = devices
        self.index = 0

    def assign_gpu(self):
        thr = threading.current_thread()
        thr.gpu = self.devices[self.index]
        self.index = self.index + 1


def set_lib(pkg):
    global dvclib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    elif pkg == 'torch':
        devlib = importlib.import_module('cohere_core.lib.torchlib').torchlib
    calc.set_lib(devlib)


def single_rec_process(metric_type, gen, alpha_dir, rec_attrs):
    """
    This function runs a single reconstruction process.

    Parameters
    ----------
    proc : str
        string defining library, supported 'np' and 'cp'

    pars : Object
        Params object containing parsed configuration

    data : numpy array
        data array

    req_metric : str
        defines metric that will be used if GA is utilized

    dirs : list
        tuple of two elements: directory that contain results of previous run or None, and directory where the
        results of this processing will be saved

    Returns
    -------
    metric : float
        a calculated characteristic of the image array defined by the metric
    """
    worker, prev_dir, save_dir = rec_attrs
    thr = threading.current_thread()
    if worker.init_dev(thr.gpu) < 0:
        print(f'reconstruction failed, device not initialized to {thr.gpu}')
        metric = None
    else:
        ret_code = worker.init(prev_dir, alpha_dir, gen)
        if ret_code < 0:
            print('reconstruction failed, check algorithm sequence and triggers in configuration')
            metric = None
        else:
            if gen is not None and gen > 0:
                worker.breed()

            ret_code = worker.iterate()
            if ret_code == 0:
                worker.save_res(save_dir)
                metric = worker.get_metric(metric_type)
            else:    # bad reconstruction
                print('reconstruction failed during iterations')
                metric = None

    return [metric, save_dir]


def multi_rec(save_dir, devices, no_recs, pars, datafile, prev_dirs, metric_type='chi', gen=None, alpha_dir=None, q=None):
    """
    This function controls the multiple reconstructions.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy
        cp - to use cupy

    save_dir : str
        a directory where the subdirectories will be created to save all the results for multiple reconstructions

    devices : list
        list of GPUs available for this reconstructions

    no_recs : int
        number of reconstructions

    pars : dict
        parameters for reconstruction

    datafie : str
        name of file containing data for reconstruction

    previous_dirs : list
        directories that hols results of previous reconstructions if it is continuation or None(s)

    metric_type : str
        a metric defining algorithm by which to evaluate the quality of reconstructed array

    gen : int
        which generation is the reconstruction for

    q : queue
        if provided the results will be queued

    Returns
    -------
    None
    """
    results = []

    def collect_result(result):
        results.append(result)

    workers = [calc.Rec(pars, datafile) for _ in range(no_recs)]
    dev_obj = Devices(devices)
    iterable = []
    save_dirs = []

    for i in range(len(workers)):
        save_sub = ut.join(save_dir, str(i))
        save_dirs.append(save_sub)
        iterable.append((workers[i], prev_dirs[i], save_sub))
    func = partial(single_rec_process, metric_type, gen, alpha_dir)
    with Pool(processes=len(devices), initializer=dev_obj.assign_gpu, initargs=()) as pool:
        pool.map_async(func, iterable, callback=collect_result)
        pool.close()
        pool.join()
        pool.terminate()

    if q is not None:
        if len(results) == 0:
            q.put('failed')
        else:
            q.put(results[0])


def order_dirs(tracing, dirs, evals, metric_type):
    """
    Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

    Parameters
    ----------
    tracing : object
        Tracing object that keeps fields related to tracing
    dirs : list
        list of directories where the reconstruction results files are saved
    evals : list
        list of evaluations of the results saved in the directories matching dirs list. The evaluations are dict
        <metric type> : <eval result>
    metric_type : metric type to be applied for evaluation

    Returns
    -------
    list :
        a list of directories where results are saved ordered from best to worst
    dict :
        evaluations of the best results
    """
    # ranks keeps indexes of reconstructions from best to worst
    # for most of the metric types the minimum of the metric is best, but for
    # 'summed_phase' and 'area' it is opposite, so reversing the order
    metric_evals = [e[metric_type] for e in evals]
    ranks = np.argsort(metric_evals).tolist()
    if metric_type == 'summed_phase' or metric_type == 'area':
        ranks.reverse()
    best_metrics = evals[ranks[0]]

    # Add tracing for the generation results
    gen_ranks = {}
    for i in range(len(evals)):
        gen_ranks[ranks[i]] = (i, evals[ranks[i]])
    tracing.append_gen(gen_ranks)

    # find how the directories based on ranking, map to the initial start
    prev_map = tracing.map
    map = {}
    for i in range(len(ranks)):
        prev_seq = int(os.path.basename(dirs[ranks[i]]))
        inx = prev_map[prev_seq]
        map[i] = inx
    prev_map.clear()
    tracing.set_map(map)

    # order directories by rank
    rank_dirs = []
    # append "_<rank>" to each result directory name
    for i in range(len(ranks)):
        dest = f'{dirs[ranks[i]]}_{str(i)}'
        src = dirs[ranks[i]]
        shutil.move(src, dest)
        rank_dirs.append(dest)

    # remove the number preceding rank from each directory name, so the directories are numbered
    # according to rank
    current_dirs = []
    for dir in rank_dirs:
        last_sub = os.path.basename(dir)
        parent_dir = os.path.dirname(dir)
        dest = ut.join(parent_dir, last_sub.split('_')[-1])
        shutil.move(dir, dest)
        current_dirs.append(dest)
    return current_dirs, best_metrics


def cull(lst, no_left):
    if len(lst) <= no_left:
        return lst
    else:
        return lst[0:no_left]


def reconstruction(lib, conf_file, datafile, dir, devices):
    """
    Controls reconstruction that employs genetic algorith (GA).

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in the
    configuration file defines whether it is a random guess, AI algorithm determined (one reconstruction,
    the rest random), or starting from some saved state. It will set the initial guess accordingly and start
    GA algorithm. It will run multiple reconstructions for each generation in a loop. After each generation
    the best reconstruction, alpha is identified, and used for breeding. For each generation the results will
    be saved in g_x subdirectory, where x is the generation number, in configured 'save_dir' parameter or in
    'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,

    conf_file : str
        configuration file name

    datafile : str
        data file name

    dir : str
        a parent directory that holds the reconstructions. For example experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

    """
    pars = ut.read_config(conf_file)
    pars = gaut.set_ga_defaults(pars)

    if pars['ga_generations'] < 2:
        print("number of generations must be greater than 1")
        return

    if pars['reconstructions'] < 2:
        print("GA not implemented for a single reconstruction")
        return

    if pars['ga_fast']:
        reconstructions = min(pars['reconstructions'], len(devices))
    else:
        reconstructions = pars['reconstructions']

    if 'ga_cullings' in pars:
        cull_sum = sum(pars['ga_cullings'])
        if reconstructions - cull_sum < 2:
            print("At least two reconstructions should be left after culling. Number of starting reconstructions is", reconstructions, "but ga_cullings adds to", cull_sum)
            return

    if pars['init_guess'] == 'AI_guess':
        # run AI part first
        import cohere_core.controller.AI_guess as ai
        ai_dir = ai.start_AI(pars, datafile, dir)
        if ai_dir is None:
            return

    set_lib(lib)

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        # the config_rec might be an alternate configuration with a postfix that will be included in save_dir
        filename = conf_file.split('/')[-1]
        save_dir = ut.join(dir, filename.replace('config_rec', 'results_phasing'))

    # create alpha dir and placeholder for the alpha's metrics
    alpha_dir = ut.join(save_dir, 'alpha')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(alpha_dir):
        os.mkdir(alpha_dir)

    tracing = gaut.Tracing(reconstructions, pars, dir)

    q = Queue()
    prev_dirs = tracing.init_dirs
    tracing.set_map({i:i for i in range(len(prev_dirs))})
    for g in range(pars['ga_generations']):
        # delete previous-previous generation
        if g > 1:
            shutil.rmtree(ut.join(save_dir, f'g_{str(g-2)}'))
        print ('starting generation', g)
        gen_save_dir = ut.join(save_dir, f'g_{str(g)}')
        metric_type = pars['ga_metrics'][g]
        reconstructions = len(prev_dirs)
        p = Process(target=multi_rec, args=(gen_save_dir, devices, reconstructions, pars, datafile, prev_dirs, metric_type, g, alpha_dir, q))
        p.start()
        p.join()

        results = q.get()
        if results == 'failed':
            print('reconstruction failed')
            return

        evals = []
        temp_dirs = []
        for r in results:
            eval = r[0]
            if eval is not None:
                evals.append(eval)
                temp_dirs.append(r[1])

        # results are saved in a list of directories - save_dir
        # it will be ranked, and moved to temporary ranked directories
        current_dirs, best_metrics = order_dirs(tracing, temp_dirs, evals, metric_type)
        reconstructions = pars['ga_reconstructions'][g]
        current_dirs = cull(current_dirs, reconstructions)
        prev_dirs = current_dirs

        # compare current alpha and previous. If previous is better, set it as alpha.
        # no need to set alpha  for last generation
        if (g == 0
                or
                best_metrics[metric_type] >= alpha_metrics[metric_type] and
                (metric_type == 'summed_phase' or metric_type == 'area')
                or
                best_metrics[metric_type] < alpha_metrics[metric_type] and
                (metric_type == 'chi' or metric_type == 'sharpness')):
            shutil.copyfile(ut.join(current_dirs[0], 'image.npy'), ut.join(alpha_dir, 'image.npy'))
            alpha_metrics = best_metrics
    # remove the previous gen
    shutil.rmtree(ut.join(save_dir, 'g_' + str(pars['ga_generations'] - 2)))

    tracing.save(save_dir)

    print('done gen')