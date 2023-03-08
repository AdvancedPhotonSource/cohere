# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_single
=================================

This module controls a single reconstruction process.
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI or using command line scripts, see :ref:`use`.
"""

import numpy as np
import os
import importlib
import cohere_core.controller.phasing as calc
import cohere_core.utilities.utils as ut
from multiprocessing import Process


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def set_lib(pkg, ndim=None):
    global devlib
    if pkg == 'cp':
        devlib = importlib.import_module('cohere_core.lib.cplib').cplib
    elif pkg == 'np':
        devlib = importlib.import_module('cohere_core.lib.nplib').nplib
    calc.set_lib(devlib)


def rec_process(lib, pars, peak_dirs, dev, continue_dir):
    set_lib(lib)
    # with the calling script it is assumed that the calling script uses peak_dirs containing
    # peak orientation. It is parsed here, and passed to the CoupledRec constractor as list of
    # touples (<data directory>, <peak orientation>)
    peak_dir_orient = []
    for dir in peak_dirs:
        ornt_str = dir.split('_')[-1]
        orientation = [0, 0, 0]
        i = 0
        mult = 1
        for ind in range(len(ornt_str)):
            if ornt_str[ind] == '-':
                mult = -1
            else:
                orientation[i] = int(ornt_str[ind]) * mult
                i += 1
                mult = 1
        peak_dir_orient.append((dir, orientation))

    worker = calc.CoupledRec(pars, peak_dir_orient)
    if worker.init_dev(dev[0]) < 0:
        return
    worker.init(continue_dir)
    if worker.iterate() < 0:
        return
    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        save_dir = os.path.dirname(peak_dirs[0]) + '/results_phasing'
    worker.save_res(save_dir)


def reconstruction(lib, pars, peak_dirs, dev=None):
    """
    Controls single reconstruction.

    This script is typically started with cohere-ui helper functions. The 'init_guess' parameter in the configuration file defines whether it is a random guess, AI algorithm determined, or starting from some saved state. It will set the initial guess accordingly and start phasing process. The results will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,
        torch - to use pytorch,

    conf_map : dict
        configuration

    datafiles : list
        data files names

    dir : str
        a parent directory that holds the reconstructions. For example experiment directory or scan directory.

    dev : int
        id defining GPU the this reconstruction will be utilizing, or -1 if running cpu

    """
    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'
    if pars['init_guess'] == 'continue':
        continue_dir = pars['continue_dir']
    elif pars['init_guess'] == 'AI_guess':
        print('AI initial guess is not a valid choice for multi peak reconstruction')
        return -1
    else:
        continue_dir = None

    p = Process(target=rec_process, args=(lib, pars, peak_dirs, dev,
                                          continue_dir))
    p.start()
    p.join()
