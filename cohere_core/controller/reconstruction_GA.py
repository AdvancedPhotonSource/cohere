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
import argparse
import numpy as np
import os
import cohere_core.utilities.utils as ut
import cohere_core.utilities.dvc_utils as dvut
import cohere_core.utilities.ga_utils as gaut
import cohere_core.controller.phasing as calc
from mpi4py import MPI
import datetime


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def order_ranks(tracing, proc_metrics, metric_type, last_alpha_metric):
    """
    Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

    Parameters
    ----------
    tracing : object
        Tracing object that keeps fields related to tracing
    proc_metrics : dict
        dictionary of <pid> : <evaluations> ; evaluations of the results saved in the directories matching dirs list. The evaluations are dict
        <metric type> : <eval result>
    metric_type : metric type to be applied for evaluation

    Returns
    -------
    list :
        a list of processes ids ordered from best to worst by the results the processes delivered
    dict :
        evaluations of the best results
    """
    last_alpha_best = False
    rank_eval = [(key, proc_metrics[key][metric_type]) for key in proc_metrics.keys()]
    ranked_proc = sorted(rank_eval, key=lambda x: x[1])

    # ranks keeps indexes of reconstructions from best to worst
    # for most of the metric types the minimum of the metric is best, but for
    # 'summed_phase' and 'area' it is opposite, so reversing the order
    if metric_type == 'summed_phase' or metric_type == 'area':
        ranked_proc.reverse()
        if last_alpha_metric is not None:
            last_alpha_best = last_alpha_metric[metric_type] > proc_metrics[ranked_proc[0][0]][metric_type]
    else:
        if last_alpha_metric is not None:
            last_alpha_best = last_alpha_metric[metric_type] < proc_metrics[ranked_proc[0][0]][metric_type]
    gen_ranks = {}
    for i in range(len(ranked_proc)):
        rk = ranked_proc[i][0]
        gen_ranks[rk] = (i, proc_metrics[rk])
    tracing.append_gen(gen_ranks)

    return ranked_proc, proc_metrics[ranked_proc[0][0]], last_alpha_best


def cull(lst, no_left):
    if len(lst) <= no_left:
        return lst
    else:
        return lst[0:no_left]


def write_log(rank: int, msg: str) -> None:
    """
    Use this to force writes for debugging. PBS sometimes doesn't flush
    std* outputs. MPI faults clobber greedy flushing of default python
    logs.
    """
    with open(f'{rank}.log', 'a') as log_f:
        log_f.write(f'{datetime.datetime.now()} | {msg}\n')


def reconstruction(pkg, conf_file, datafile, dir, devices):
    """
    Controls reconstruction that employs genetic algorith (GA).

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in
    the configuration file defines whether it is a random guess, AI algorithm determined
    (one reconstruction, the rest random), or starting from some saved state. It will set the initial guess
    accordingly and start GA algorithm. It will run multiple reconstructions for each generation in a loop.
    After each generation the best reconstruction, alpha is identified, and used for breeding.
    For each generation the results will be saved in g_x subdirectory, where x is the generation number,
    in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    pkg : str
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
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    dvut.set_lib_from_pkg(pkg)

    pars = ut.read_config(conf_file)
    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        # the config_rec might be an alternate configuration with a postfix that will be included in save_dir
        filename = conf_file.split('/')[-1]
        save_dir = ut.join(dir, filename.replace('config_rec', 'results_phasing'))
    last_alpha = None
    last_alpha_metric = None

    if rank == 0:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)

    pars = gaut.set_ga_defaults(pars)

    reconstructions = size

    if 'ga_cullings' in pars:
        cull_sum = sum(pars['ga_cullings'])

    worker = calc.Rec(pars, datafile, pkg)
    if rank == 0:
        active_ranks = list(np.linspace(0, size-1, num=size, dtype=int))
        tracing = gaut.Tracing(reconstructions, pars, dir)
        tracing_map = {r : r for r in active_ranks}
        tracing.set_map(tracing_map)

    active = True
    success = True
    for g in range(pars['ga_generations']):
        was_active = active
        if g == 0:
            ret = worker.init_dev(devices[rank])
            if ret < 0:
                print(f'rank {rank} failed initializing device {devices[rank]}')
                active = False
            if active:
                if pars['init_guess'] == 'AI_guess' and rank == 0:
                    # run AI part first
                    import cohere_core.controller.AI_guess as ai
                    ai_dir = ai.start_AI(pars, datafile, dir)
                    if ai_dir is None:
                        ret = worker.init_iter_loop(None, g)
                    else:
                        ret = worker.init_iter_loop(ai_dir, g)
                else:
                    ret = worker.init_iter_loop(None, g)
                if ret < 0:
                    print('failed init, check config')
                    break
        else:
            if active:
                ret = worker.init_iter_loop(None, g)
                if ret < 0:
                    print(f'rank {rank} reconstruction failed, check algorithm sequence and triggers in configuration')
                    active = False

            if active:
                def breed():
                    worker.ds_image = dvut.breed(pars['ga_breed_modes'][g], alpha, worker.ds_image)
                    worker.support = dvut.shrink_wrap(worker.ds_image, pars['ga_sw_thresholds'][g], pars['ga_sw_gauss_sigmas'][g])
                try:
                    breed()
                except:
                    # retry
                    breed()
        if active:
            ret = worker.iterate()
            if ret < 0:
                print(f'reconstruction for rank {rank} failed during iterations')
                active = False

        if active:
            metric_type = pars['ga_metrics'][g]
            metric = worker.get_metric()
        else:
            metric = None

        # send metric and rank to rank 0
        if rank > 0 and was_active:
            comm.send(metric, dest=0)

        # rank 0 receives metrics from ranks and creates metrics dict
        if rank == 0:
            metrics = {}
            if active:
                metrics[0] = metric
            elif was_active:
                active_ranks.remove(0)
            to_remove = []
            for r in active_ranks:
                if r != 0:
                    metric = comm.recv(source=r)
                    if metric is None:
                        to_remove.append(r)
                    else:
                        metrics[r] = metric
            for r in to_remove:
                active_ranks.remove(r)

#        comm.Barrier()

        if rank == 0:
            # order processes by metric
            proc_ranks, best_metrics, last_alpha_best = order_ranks(tracing, metrics, metric_type, last_alpha_metric)
            proc_ranks = [p[0] for p in proc_ranks]

            # cull
            culled_proc_ranks = cull(proc_ranks, pars['ga_reconstructions'][g])
            no_active = len(culled_proc_ranks)
            # if previous alpha was better, insert -1, so no rank will match
            if last_alpha_best:
                culled_proc_ranks = [-1] + culled_proc_ranks

            # send out the ordered active ranks, if alpha best with preceding -1
            for r in active_ranks:
                if r != 0:
                    comm.send(culled_proc_ranks, dest=r)
            # remove culled processes from active list
            to_remove = proc_ranks[no_active : len(proc_ranks)]
            for r in to_remove:
                if r in active_ranks:
                    active_ranks.remove(r)
        elif active:
            culled_proc_ranks = comm.recv(source=0)
            if rank not in culled_proc_ranks:
                active = False

        alpha_rank = max(0, culled_proc_ranks[0])

        if rank == alpha_rank:
            if culled_proc_ranks[0] == -1:
                # check if last alpha is the best
                alpha = last_alpha
            else:
                # check if this process is the best and find rank that would broadcast alpha
                alpha = worker.ds_image
        else:
            alpha = None
        # send the alpha image to oter workers
        alpha = comm.bcast(alpha, root=alpha_rank)

        if rank == 0:
            last_alpha = alpha
            last_alpha_metric = best_metrics

        if not active:
            worker = None

        if g == pars['ga_generations'] -1:
        # save alpha result for last generation
            if culled_proc_ranks[0] == -1:
                alpha_rank = culled_proc_ranks[1]
            if rank == alpha_rank:
                worker.save_res(save_dir)

    if rank == 0 and success:
        tracing.save(save_dir)


def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("lib", help="lib")
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    parser.add_argument('dir', help='dir')
    parser.add_argument('dev', help='dev')

    args = parser.parse_args()
    dev = ast.literal_eval(args.dev)
    reconstruction(args.lib, args.conf_file, args.datafile, args.dir, dev)


if __name__ == "__main__":
    exit(main())