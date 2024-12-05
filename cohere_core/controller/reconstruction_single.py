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

import cohere_core.controller.phasing as calc
import cohere_core.utilities.utils as ut
from multiprocessing import Process


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['reconstruction']


def reconstruction(pkg, conf_file, datafile, dir, dev, **kwargs):
    """
    Controls single reconstruction.

    This script is typically started with cohere-ui helper functions. The 'init_guess' parameter in the
    configuration file defines whether it is a random guess, AI algorithm determined, or starting from
    some saved state. It will set the initial guess accordingly and start phasing process. The results
    will be saved in configured 'save_dir' parameter or in 'results_phasing' subdirectory if 'save_dir'
    is not defined.

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

    dev : int
        id defining GPU the this reconstruction will be utilizing, or -1 if running cpu

    kwargs : var parameters
        may contain:
        debug : if True the exceptions are not handled

    """
    pars = ut.read_config(conf_file)

    pars['init_guess'] = pars.get('init_guess', 'random')
    if pars['init_guess'] == 'AI_guess':
        import cohere_core.controller.AI_guess as ai
        ai_dir = ai.start_AI(pars, datafile, dir)
        if ai_dir is None:
            return
        pars['continue_dir'] = ai_dir

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        filename = conf_file.split('/')[-1]
        save_dir = ut.join(dir, filename.replace('config_rec', 'results_phasing'))

    if dev is None:
        device = pars.get('device', -1)
    else:
        device = dev[0]

    worker = calc.create_rec(pars, datafile, pkg, device, **kwargs)
    if worker is None:
        return

    ret_code = worker.iterate()
    if ret_code < 0:
        print ('reconstruction failed during iterations')
        return

    worker.save_res(save_dir)
