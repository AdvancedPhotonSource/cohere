# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_multi
================================

This module controls a multiple reconstructions.
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI x or using command line scripts x.
"""

import os
import argparse
import cohere_core.utilities.utils as ut
import cohere_core.controller.phasing as calc
from mpi4py import MPI


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def reconstruction(pkg, conf_file, datafile, dir, devices):
    """
    Controls multiple reconstructions, the reconstructions run concurrently.

    This script is typically started with cohere_core-ui helper functions. The 'init_guess' parameter in the
    configuration file defines whether guesses are random, or start from some saved states. It will set the
    initial guesses accordingly and start phasing process, running each reconstruction in separate thread.
    The results will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if
    'save_dir' is not defined.

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
    rank = comm.Get_rank()
    pars = ut.read_config(conf_file)

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        # the config_rec might be an alternate configuration with a postfix that will be included in save_dir
        filename = conf_file.split('/')[-1]
        save_dir = ut.join(dir, filename.replace('config_rec', 'results_phasing'))
        if rank == 0:
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)

    comm.Barrier()

    if 'init_guess' in pars and pars['init_guess'] == 'AI_guess':
        print('multiple reconstruction do not support AI_guess initial guess')
        return

    prev_dir = None
    if 'init_guess' in pars and pars['init_guess'] == 'continue':
        if not 'continue_dir' in pars:
            print('continue directory must be defined if initial guess is continue')
            return

        continue_dir = pars['continue_dir']
        if os.path.isdir(continue_dir):
            prev_dirs = os.listdir(continue_dir)
            if len(prev_dirs) > rank:
                prev_dir = ut.join(continue_dir, prev_dirs[rank])
                if not os.path.isfile(ut.join(prev_dir, 'image.npy')):
                    prev_dir = None

    # assume for now the size can accommodate the no_recs
    worker = calc.Rec(pars, datafile, pkg)
    ret = worker.init_dev(devices[rank])
    if ret < 0:
        print(f'rank {rank} failed initializing device {devices[rank]}')
        return

    ret = worker.init_iter_loop(prev_dir)
    if ret < 0:
        print(f'rank {rank} failed, reconstruction failed, check algorithm sequence and triggers in configuration')
        return

    ret = worker.iterate()
    if ret < 0:
        print(f'reconstruction for rank {rank} failed during iterations')
        return

    save_sub = ut.join(save_dir, str(rank))
    if not os.path.isdir(save_sub):
        os.mkdir(save_sub)
    worker.save_res(save_sub)


def main():
    import ast

    parser = argparse.ArgumentParser()
    parser.add_argument("pkg", help="package to use for reconstruction calculations")
    parser.add_argument("conf_file", help="conf_file")
    parser.add_argument("datafile", help="datafile")
    parser.add_argument('dir', help='dir')
    parser.add_argument('dev', help='dev')

    args = parser.parse_args()
    dev = ast.literal_eval(args.dev)
    reconstruction(args.pkg, args.conf_file, args.datafile, args.dir, dev)


if __name__ == "__main__":
    exit(main())
