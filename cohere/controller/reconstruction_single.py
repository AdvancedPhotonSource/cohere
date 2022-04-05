# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
This module controls a single reconstruction process.
"""

import numpy as np
import os
import importlib
import cohere.controller.reconstruction_common as common
import cohere.controller.phasing as calc
import cohere.utilities.utils as ut
from multiprocessing import Process


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['single_rec',
           'reconstruction']


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


def rec_process(lib, pars, datafile, dev, continue_dir, save_dir):
    if lib == 'af' or lib == 'cpu' or lib == 'opencl' or lib == 'cuda':
        if datafile.endswith('tif') or datafile.endswith('tiff'):
            try:
                data = ut.read_tif(datafile)
            except:
                print('could not load data file', datafile)
                return
        elif datafile.endswith('npy'):
            try:
                data = np.load(datafile)
            except:
                print('could not load data file', datafile)
                return
        else:
            print('no data file found')
            return

        set_lib('af', len(data.shape))
        if lib != 'af':
            devlib.set_backend(lib)
    else:
        set_lib(lib)

    worker = calc.Rec(pars, datafile)

    if dev is None:
        if 'device' in pars:
            device = pars['device']
        else:
            device = [-1]
    else:
        device = dev[0]
    if worker.init_dev(device) < 0:
        return

    worker.init(continue_dir)
    ret_code = worker.iterate()
    if ret_code == 0:
        worker.save_res(save_dir)


def reconstruction(lib, conf_file, datafile, dir, dev=None):
    """
    Controls single reconstruction.

    This function checks whether the reconstruction is continuation or initial reconstruction. If continuation, the arrays of image, support, coherence are read from cont_directory, otherwise they are initialized to None.
    It starts thr reconstruction and saves results.

    Parameters
    ----------
    proc : str
        a string indicating the processor type (cpu, cuda or opencl)

    conf_file : str
        configuration file name

    datafile : str
        data file name

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.

    dev : int
        id defining the GPU this reconstruction will be utilizing, or -1 if running cpu or the gpu assignment is left to OS


    Returns
    -------
    nothing
    """
    pars = ut.read_config(conf_file)

    if 'init_guess' not in pars:
        pars['init_guess'] = 'random'
    if pars['init_guess'] == 'continue':
        continue_dir = pars['continue_dir']
    elif pars['init_guess'] == 'AI_guess':
        ai_dir = common.start_AI(pars, datafile, dir)
        if ai_dir is None:
            return
        continue_dir = ai_dir
    else:
        continue_dir = None

    if 'save_dir' in pars:
        save_dir = pars['save_dir']
    else:
        filename = conf_file.split('/')[-1]
        save_dir = os.path.join(dir, filename.replace('config_rec', 'results_phasing'))

    p = Process(target=rec_process, args=(lib, pars, datafile, dev,
                                          continue_dir, save_dir))
    p.start()
    p.join()
