# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.reconstruction_coupled
==================================

This module controls a multipeak reconstruction process.
Refer to cohere_core-ui suite for use cases. The reconstruction can be started from GUI or using command line scripts, see :ref:`use`.
"""

import os
import cohere_core.controller.phasing as calc
import cohere_core.utilities.utils as ut
from multiprocessing import Process


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def reconstruction(lib, pars, peak_dirs, dev, **kwargs):
    """
    Controls multipeak reconstruction.

    This script is typically started with cohere-ui helper functions. It will start based on the configuration. The config file must have multipeak parameter set to True. In addition the config_mp with the peaks parameters must be included. The results will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir' is not defined.

    Parameters
    ----------
    lib : str
        library acronym to use for reconstruction. Supported:
        np - to use numpy,
        cp - to use cupy,
        torch - to use pytorch,

    pars : dict
        parameters reflecting configuration

    peak_dirs : list
        list of directories with data taken at each peak

    dev : int
        id defining GPU this reconstruction will be utilizing
    kwargs : var parameters
        may contain:
        debug : if True the exceptions are not handled
    """
    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'
    elif pars['init_guess'] == 'AI_guess':
        print('AI initial guess is not a valid choice for multi peak reconstruction')
        return -1

    kwargs['rec_type'] = 'mp'
    worker = calc.create_rec(pars, peak_dirs, lib, dev[0], **kwargs)
    if worker is None:
        return

    if worker.iterate() < 0:
        return

    save_dir = pars.get('save_dir', ut.join(os.path.dirname(peak_dirs[0]), 'results_phasing'))
    worker.save_res(save_dir)
