# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This module controls the genetic algoritm process.
"""

import numpy as np
import os
import cohere.src_py.controller.reconstruction as single
import cohere.src_py.controller.reconstruction_multi as multi
import cohere.src_py.utilities.utils as ut
import cohere.src_py.utilities.utils_ga as gut
import multiprocessing as mp
import shutil
from functools import partial
import time
from shutil import copyfile


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = [ 'Generation.next_gen',
            'Generation.get_data',
            'Generation.get_gmask',
            'Generation.order,'
            'Generation.breed_one',
            'Generation.breed',
            'reconstruction']


class Generation:
    """
    This class holds fields relevant to generations according to configuration and encapsulates generation functionality.
    """
    def __init__(self, config_map):
        """
        Constructor, parses GA parameters from configuration file and saves as class members.
        """
        self.current_gen = 0
        try:
            self.generations = config_map.generations
        except AttributeError:
            self.generations = 1

        try:
            self.metrics = list(config_map.ga_metrics)
            if len(self.metrics) == 1:
                self.metrics = self.metrics * self.generations
            elif len(self.metrics) < self.generations:
                self.metrics = self.metrics + ['chi',] * (self.generations - len(self.metrics))
        except AttributeError:
            self.metrics = ['chi',] * self.generations

        try:
            self.worst_remove_no = list(config_map.ga_cullings)
            if len(self.worst_remove_no) < self.generations:
                self.worst_remove_no = self.worst_remove_no + [0,] * (self.generations - len(self.worst_remove_no))
        except AttributeError:
            self.worst_remove_no = None

        try:
            self.ga_support_thresholds = list(config_map.ga_support_thresholds)
            if len(self.ga_support_thresholds) == 1:
                self.ga_support_thresholds = self.ga_support_thresholds * self.generations
            elif len(self.ga_support_thresholds) < self.generations:
                try:
                    support_threshold = config_map.support_threshold
                except:
                    support_threshold = .1
                self.ga_support_thresholds = self.ga_support_thresholds + [support_threshold,] * (self.generations - len(self.ga_support_thresholds))
        except AttributeError:
            try:
                support_threshold = config_map.support_threshold
            except:
                support_threshold = .1
            self.ga_support_thresholds = [support_threshold,] * (self.generations)

        try:
            self.ga_support_sigmas = list(config_map.ga_support_sigmas)
            if len(self.ga_support_sigmas) == 1:
                self.ga_support_sigmas = self.ga_support_sigmas * self.generations
            elif len(self.ga_support_sigmas) < self.generations:
                try:
                    support_sigma = config_map.support_sigma
                except:
                    support_sigma = 1.0
                self.ga_support_sigmas = self.ga_support_sigmas + [support_sigma,] * (self.generations - len(self.ga_support_sigmas))
        except AttributeError:
            try:
                support_sigma = config_map.support_sigma
            except:
                support_sigma = 1.0
            self.ga_support_sigmas = [support_sigma,] * (self.generations)

        try:
            self.breed_modes = list(config_map.ga_breed_modes)
            if len(self.breed_modes) == 1:
                self.breed_modes = self.breed_modes * self.generations
            elif len(self.breed_modes) < self.generations:
                self.breed_modes = self.breed_modes + ['sqrt_ab',] * (self.generations - len(self.breed_modes))
        except AttributeError:
            self.breed_modes = ['none',] * self.generations

        try:
            self.sigmas = config_map.ga_low_resolution_sigmas
            self.low_resolution_generations = len(self.sigmas)
        except AttributeError:
            self.low_resolution_generations = 0

        if self.low_resolution_generations > 0:
            try:
                self.low_resolution_alg = config_map.ga_low_resolution_alg
            except AttributeError:
                self.low_resolution_alg = 'GAUSS'

        
    def next_gen(self):
        """
        Advances to the next generation.

        Parameters
        ----------
        none

        Returns
        -------
        nothing
        """
        self.current_gen += 1

    def get_data(self, data):
        """
        If low resolution feature is enabled (i.e. when ga_low_resolution_sigmas parameter is defined) the data is modified in each generation that has the sigma provided in the ga_low_resolution_sigmas list. The data is multiplied by gaussian distribution calculated using the defined sigma.
        
        Parameters
        ----------
        data : ndarray
            experiment data array, prepared and formatted

        Returns
        -------
        data : ndarray
            modified data
        """
        if self.current_gen >= self.low_resolution_generations:
            return data
        else:
            gmask = self.get_gmask(data.shape)
            return data * gmask


    def get_gmask(self, shape):
        """
        It calculates the gaussian distribution for low resolution generations.

        Parameters
        ----------
        shape : tuple
            shape of the data array

        Returns
        -------
        mask : ndarray
            array of the same size as data containing Gaussian distribution accross all dimensions for low resolution generations, or array of 1s for other generations.
        """
        if self.low_resolution_alg == 'GAUSS':
            if self.sigmas[self.current_gen] < 1.0:
                ut.gaussian(shape, self.sigmas[self.current_gen])
            else:
                return np.ones(shape)


    def order(self, dirs, evals):
        """
        Orders results in generation directory in subdirectories numbered from 0 and up, the best result stored in the '0' subdirectory. The ranking is done by numbers in evals list, which are the results of the generation's metric to the image array.

        Parameters
        ----------
        dirs : list
            list of directories where the reconstruction results files are saved
        evals : list
            list of evaluation of the results in the directories from the dirs list. The evaluation is a number calculated for metric configured for this generation
        
        Returns
        -------
        nothing
        """
        metric = self.metrics[self.current_gen]
        # ranks keeps indexes of reconstructions from best to worst
        # for most of the metric types the minimum of the metric is best, but for
        # 'summed_phase' and 'area' it is oposite, so reversing the order
        ranks = np.argsort(evals).tolist()
        if metric == 'summed_phase' or metric == 'area':
            ranks.reverse()
         
        # all the generation directories are in the same parent directory
        parent_dir = os.path.abspath(os.path.join(dirs[0], os.pardir))
        rank_dirs = []
        # append "_<rank>" to each result directory name
        for i in range(len(ranks)):
            dest = os.path.join(parent_dir, str(i) + '_' + str(ranks[i]))
            shutil.move(dirs[i], dest)
            rank_dirs.append(dest)

        # remove the number preceding rank from each directory name, so the directories are numbered
        # according to rank
        for dir in rank_dirs:
            last_sub = os.path.basename(dir)
            dest = os.path.join(parent_dir, last_sub.split('_')[-1])
            shutil.move(dir, dest)


    def breed_one(self, alpha, breed_mode, incl_coh, dirs):
        """
        Aligns the image to breed from with the alpha image, and applies breed formula, to obtain a 'child' image.

        Parameters
        ----------
        alpha : ndarray
            the best image in the generation
        breed_mode : str
            literal defining the breeding process
        dirs : tuple
            a tuple containing two elements: directory where the image to breed from is stored, a 'parent', and a directory where the bred image, a 'child', will be stored.
            
        Returns
        -------
        nothing
        """
        (parent_dir, breed_dir) = dirs
        image_file = os.path.join(parent_dir, 'image.npy')
        beta = np.load(image_file)
        beta = gut.zero_phase(beta, 0)
        alpha = gut.check_get_conj_reflect(beta, alpha)
        alpha_s = gut.align_arrays(beta, alpha)
        alpha_s = gut.zero_phase(alpha_s, 0)
        ph_alpha = np.angle(alpha_s)
        beta = gut.zero_phase_cc(beta, alpha_s)
        ph_beta = np.angle(beta)
        if breed_mode == 'sqrt_ab':
            beta = np.sqrt(abs(alpha_s) * abs(beta)) * np.exp(0.5j * (ph_beta + ph_alpha))

        elif breed_mode == 'dsqrt':
            amp = pow(abs(beta), .5)
            beta = amp * np.exp(1j * ph_beta)

        elif breed_mode == 'pixel_switch':
            cond = np.random.random_sample(beta.shape)
            beta = np.where((cond > 0.5), beta, alpha_s)

        elif breed_mode == 'b_pa':
            beta = abs(beta) * np.exp(1j * (ph_alpha))

        elif breed_mode == '2ab_a_b':
            beta = 2 * (beta * alpha_s) / (beta + alpha_s)

        elif breed_mode == '2a_b_pa':
            beta = (2 * abs(alpha_s) - abs(beta)) * np.exp(1j * ph_alpha)

        elif breed_mode == 'sqrt_ab_pa':
            beta = np.sqrt(abs(alpha_s) * abs(beta)) * np.exp(1j * ph_alpha)

        elif breed_mode == 'sqrt_ab_pa_recip':
            temp1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(beta)))
            temp2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(alpha_s)))
            temp = np.sqrt(abs(temp1) * abs(temp2)) * np.exp(1j * np.angle(temp2))
            beta = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(temp)))

        elif breed_mode == 'sqrt_ab_recip':
            temp1 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(beta)))
            temp2 = np.fft.fftshift(np.fft.fftn(np.fft.fftshift(alpha_s)))
            temp = np.sqrt(abs(temp1) * abs(temp2)) * np.exp(.5j * np.angle(temp1)) * np.exp(.5j * np.angle(temp2))
            beta = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(temp)))

        elif breed_mode == 'max_ab':
            beta = np.maximum(abs(alpha_s), abs(beta)) * np.exp(.5j * (ph_beta + ph_alpha))

        elif breed_mode == 'max_ab_pa':
            beta = np.maximum(abs(alpha_s), abs(beta)) * np.exp(1j * ph_alpha)

        elif breed_mode == 'min_ab_pa':
            beta = np.minimum(abs(alpha_s), abs(beta)) * np.exp(1j * ph_alpha)

        elif breed_mode == 'avg_ab':
            beta = 0.5 * (alpha_s + beta)

        elif breed_mode == 'avg_ab_pa':
            beta = 0.5 * (abs(alpha_s) + abs(beta)) * np.exp(1j * (ph_alpha))

        # save beta in the breeding sub-dir
        if not os.path.exists(breed_dir):
            os.makedirs(breed_dir)

        np.save(os.path.join(breed_dir, 'image.npy'), beta)
        sigma = self.ga_support_sigmas[self.current_gen]
        threshold = self.ga_support_thresholds[self.current_gen]
        new_support = ut.shrink_wrap(beta, threshold, sigma)
        np.save(os.path.join(breed_dir, 'support.npy'), new_support)
        if incl_coh and os.path.isfile(os.path.join(parent_dir, 'coherence.npy')):
            shutil.copyfile(os.path.join(parent_dir, 'coherence.npy'), os.path.join(breed_dir, 'coherence.npy'))
            
        
    def breed(self, breed_dir, dirs, incl_coh):
        """
        Breeds next generation.
        
        Removes worst results from previous generation if configured, ans breeds according to the breeding mode. The breeding is processed concurrently.

        Parameters
        ----------
        breed_dir : str
            a directory where subdirectories with 'child' images will be created
        dirs : tuple
            list of directories where the image to breed from is stored, a 'parent', ordered from best to worst
            
        Returns
        -------
        breed_dirs : list
            list of directories containing 'child' images
        """
        breed_mode = self.breed_modes[self.current_gen]
        if breed_mode == 'none':
            return dirs
            
        print ('breeding generation ', (self.current_gen + 1))

        if (self.worst_remove_no is not None) and (self.worst_remove_no[self.current_gen] != 0):
            del dirs[-self.worst_remove_no[self.current_gen]:]
        alpha = np.load(os.path.join(dirs[0], 'image.npy'))
        alpha = gut.zero_phase(alpha, 0)

        # assign breed directory for each bred child
        iterable = []
        breed_dirs = []
        for i in range (len(dirs)):
            breed_dirs.append(os.path.join(breed_dir, str(i)))
            iterable.append((dirs[i], breed_dirs[i]))

    # copy the alpha to the first breeding sub-dir
        if not os.path.exists(breed_dirs[0]):
            os.makedirs(breed_dirs[0])

        shutil.copyfile(os.path.join(dirs[0], 'image.npy'), os.path.join(breed_dirs[0], 'image.npy'))
        shutil.copyfile(os.path.join(dirs[0], 'support.npy'), os.path.join(breed_dirs[0], 'support.npy'))
        if incl_coh and os.path.isfile(os.path.join(dirs[0], 'coherence.npy')):
            shutil.copyfile(os.path.join(dirs[0], 'coherence.npy'), os.path.join(breed_dirs[0], 'coherence.npy'))

        no_processes = min(len(dirs), mp.cpu_count())
        func = partial(self.breed_one, alpha, breed_mode, incl_coh)
        with mp.Pool(processes = no_processes) as pool:
            pool.map_async(func, iterable[1:])
            pool.close()
            pool.join()
            pool.terminate()

        return breed_dirs


def reconstruction(proc, conf_file, datafile, dir, devices):
    """
    This function controls reconstruction utilizing genetic algorithm.

    Parameters
    ----------
    proc : str
        processor to run on (cpu, opencl, or cuda)

    conf_file : str
        configuration file with reconstruction parameters

    datafile : str
        name of the file with initial data

    dir : str
        a parent directory that holds the generations. It can be experiment directory or scan directory.

    devices : list
        list of GPUs available for this reconstructions

    Returns
    -------
    nothing
    """
    if datafile.endswith('tif'):
        data = ut.read_tif(datafile)
    elif datafile.endswith('npy'):
        data = np.load(datafile)
        
    print ('data shape', data.shape)

    try:
        config_map = ut.read_config(conf_file)
        if config_map is None:
            print("can't read configuration file " + conf_file)
            return
        ut.prepare_config(conf_file)
    except:
        print('Cannot parse configuration file ' + conf_file + ' , check for matching parenthesis and quotations')
        return
    try:
        reconstructions = config_map.reconstructions
    except:
        reconstructions = 1

    gen_obj = Generation(config_map)

    try:
        save_dir = config_map.save_dir
    except AttributeError:
        filename = conf_file.split('/')[-1]
        save_dir = os.path.join(dir, filename.replace('config_rec', 'results'))
    temp_dir = os.path.join(save_dir, 'temp')
    try:
        generations = config_map.generations
    except:
        print ('generations not configured')
        return

    try:
        gen_pcdi_start = config_map.gen_pcdi_start
    except:
        gen_pcdi_start = 0
    try:
        pcdi_trigger = config_map.pcdi_trigger
    except:
        pcdi_trigger = None
    is_pcdi = pcdi_trigger is not None
    # init starting values
    # if multiple reconstructions configured (typical for genetic algorithm), use "reconstruction_multi" module
    if reconstructions > 1:
        rec = multi
        temp_dirs = []
        for _ in range(reconstructions):
            temp_dirs.append(None)
        for g in range(generations):
            gen_data = gen_obj.get_data(data)
            gen_save_dir = os.path.join(save_dir, 'g_' + str(g))
            m = gen_obj.metrics[g]
            first_pcdi = 0
            if is_pcdi and g == gen_pcdi_start:
                first_pcdi = 1
            save_dirs, evals = rec.multi_rec(gen_save_dir, proc, gen_data, conf_file, config_map, devices, temp_dirs, m, first_pcdi)
            
            # results are saved in a list of directories - save_dir
            # it will be ranked, and moved to temporary ranked directories
            gen_obj.order(save_dirs, evals)
            if g < generations - 1 and len(save_dirs) > 1:
                incl_coh = False
                if is_pcdi and g >= gen_pcdi_start:
                    incl_coh = True
                temp_dirs = gen_obj.breed(temp_dir, save_dirs, incl_coh)
                
            gen_obj.next_gen()
    # remove temp dir
        if os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir)
    else:
        image = None
        support = None
        coh = None
        rec = single

        for g in range(generations):
            gen_data = gen_obj.get_data(data)
            image, support, coh, err, flows, iter_arrs = rec.single_rec(proc, gen_data, conf_file, config_map, devices[0], image, support, coh)
            if image is None:
                return
            # save the generation results
            gen_save_dir = os.path.join(save_dir, 'g_' + str(g))
            ut.save_results(image, support, coh, err, flows, iter_arrs, gen_save_dir)
            gen_obj.next_gen()

    print ('done gen')

