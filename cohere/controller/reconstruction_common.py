import os
import cohere.utilities.utils as ut
import numpy as np


def start_AI(pars, datafile, dir):
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
    if 'AI_trained_model' not in pars:
        print ('no AI_trained_model in config')
        return None
    if not os.path.isfile(pars['AI_trained_model']):
        print('there is no file', pars['AI_trained_model'])
        return None

    if datafile.endswith('tif') or datafile.endswith('tiff'):
        try:
            data = ut.read_tif(datafile)
        except:
            print('could not load data file', datafile)
            return None
    elif datafile.endswith('npy'):
        try:
            data = np.load(datafile)
        except:
            print('could not load data file', datafile)
            return None
    else:
        print('no data file found')
        return None

    import cohere.controller.AI_guess as ai

    # The results will be stored in the directory <experiment_dir>/AI_guess
    ai_dir = os.path.join(dir, 'results_AI')
    if os.path.exists(ai_dir):
        # for f in os.listdir(ai_dir):
        #     os.remove(os.path.join(ai_dir, f))
        pass
    else:
        os.makedirs(ai_dir)

    ai.run_AI(data, pars['AI_trained_model'], ai_dir)
    return ai_dir
