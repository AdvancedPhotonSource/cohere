# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import os
import cohere_core.utilities.utils as ut
import numpy as np


def read_results(read_dir):
    """
    Reads results of reconstruction: image, support, and coherence if exists, and returns array representation.

    :param read_dir: directory to read the results from
    :return: image, support, and coherence arrays
    """
    try:
        imagefile = ut.join(read_dir, 'image.npy')
        image = np.load(imagefile)
    except:
        image = None

    try:
        supportfile = ut.join(read_dir, 'support.npy')
        support = np.load(supportfile)
    except:
        support = None

    try:
        cohfile = ut.join(read_dir, 'coherence.npy')
        coh = np.load(cohfile)
    except:
        coh = None

    return image, support, coh

class Tracing:
    def __init__(self, reconstructions, pars, dir):
        self.init_dirs = []
        self.report_tracing = []

        pars['init_guess'] = pars.get('init_guess', 'random')
        if pars['init_guess'] == 'continue':
            continue_dir = pars['continue_dir']
            for sub in os.listdir(continue_dir):
                image, support, coh = read_results(ut.join(continue_dir, sub))
                if image is not None:
                    self.init_dirs.append(ut.join(continue_dir, sub))
                    self.report_tracing.append([ut.join(continue_dir, sub)])
            if len(self.init_dirs) < reconstructions:
                for i in range(reconstructions - len(self.init_dirs)):
                    self.report_tracing.append([f'random{str(i)}'])
                self.init_dirs = self.init_dirs + (reconstructions - len(self.init_dirs)) * [None]
        elif pars['init_guess'] == 'AI_guess':
            self.report_tracing.append(['AI_guess'])
            for i in range(reconstructions - 1):
                self.report_tracing.append([f'random{str(i)}'])
            self.init_dirs = [ut.join(dir, 'results_AI')] + (reconstructions - 1) * [None]
        else:
            for i in range(reconstructions):
                self.init_dirs.append(None)
                self.report_tracing.append([f'random{str(i)}'])


    def append_gen(self, gen_ranks):
        for key in gen_ranks:
            self.report_tracing[self.map[key]].append(gen_ranks[key])


    def pretty_format_results(self):
        """
        Takes in a list of report traces and formats them into human-readable tables.
        Performs data conversion in 1 pass and formatting in a second to determine
        padding and spacing schemes.

        Parameters
        ----------
        none

        Returns
        -------
            report_output : str
            a string containing the formatted report
        """
        col_gap = 2

        num_gens = len(self.report_tracing[0]) - 1
        fitnesses = list(self.report_tracing[0][1][1].keys())

        report_table = []
        report_table.append(['start'] + [f'generation {i}' for i in range(num_gens)])
        report_table.append([''] * len(report_table[0]))

        data_col_width = 15
        start_col_width = 15
        for pop_data in self.report_tracing:
            report_table.append([str(pop_data[0])] + [str(ind_data[0]) for ind_data in pop_data[1:]])
            start_col_width = max(len(pop_data[0]), start_col_width)

            for fit in fitnesses:
                fit_row = ['']
                for ind_data in pop_data[1:]:
                    data_out = f'{fit} : {ind_data[1][fit]}'
                    data_col_width = max(len(data_out), data_col_width)
                    fit_row.append(data_out)
                report_table.append(fit_row)
            report_table.append([''] * len(report_table[0]))

        report_str = ''
        linesep = os.linesep
        for row in report_table:
            report_str += row[0].ljust(start_col_width + col_gap)
            report_str += (' ' * col_gap).join([cell.ljust(data_col_width) for cell in row[1:]]) + linesep

        return report_str


    def save(self, save_dir):
        try:
            report_str = self.pretty_format_results()
        except Exception as e:
            print(f'WARNING: Report formatting failed due to {type(e)}: {e}! Falling back to raw formatting.')
            report_str = '\n'.join([str(l) for l in self.report_tracing])

        with open(ut.join(save_dir, 'ranks.txt'), 'w+') as rank_file:
            rank_file.write(report_str)
            rank_file.flush()


    def set_map(self, map):
        self.map = map


def set_ga_defaults(pars):
    pars['reconstructions'] = pars.get('reconstructions', 1)
    pars['ga_generations'] = pars.get('ga_generations', 1)
    pars['init_guess'] = pars.get('init_guess', 'random')

    # check if pc feature is on
    if 'pc' in pars['algorithm_sequence'] and 'pc_interval' in pars:
        if not 'ga_gen_pc_start' in pars:
            pars['ga_gen_pc_start'] = 0
            pars['ga_gen_pc_start'] = min(pars['ga_gen_pc_start'], pars['ga_generations']-1)

    pars['ga_fast'] = pars.get('ga_fast', False)

    if 'ga_metrics' not in pars:
        metrics = ['chi'] * pars['ga_generations']
    else:
        metrics = pars['ga_metrics']
        if len(metrics) == 1:
            metrics = metrics * pars['ga_generations']
        elif len(metrics) < pars['ga_generations']:
            metrics = metrics + ['chi'] * (pars['ga_generations'] - len(metrics))
    pars['ga_metrics'] = metrics

    ga_reconstructions = []
    if 'ga_cullings' in pars:
        worst_remove_no = pars['ga_cullings']
        if len(worst_remove_no) < pars['ga_generations']:
            worst_remove_no = worst_remove_no + [0] * (pars['ga_generations'] - len(worst_remove_no))
    else:
        worst_remove_no = [0] * pars['ga_generations']
    pars['worst_remove_no'] = worst_remove_no
    # calculate how many reconstructions should continue
    reconstructions = pars['reconstructions']
    for culling in worst_remove_no:
        reconstructions = reconstructions - culling
        if reconstructions <= 0:
            return 'culled down to 0 reconstructions, check configuration'
        ga_reconstructions.append(reconstructions)
    pars['ga_reconstructions'] = ga_reconstructions

    sw_threshold = .1
    if 'ga_sw_thresholds' in pars:
        ga_sw_thresholds = pars['ga_sw_thresholds']
        if len(ga_sw_thresholds) == 1:
            ga_sw_thresholds = ga_sw_thresholds * pars['ga_generations']
        elif len(ga_sw_thresholds) < pars['ga_generations']:
            ga_sw_thresholds = ga_sw_thresholds + [sw_threshold] * (pars['ga_generations'] - len(ga_sw_thresholds))
    else:
        ga_sw_thresholds = [sw_threshold] * pars['ga_generations']
    pars['ga_sw_thresholds'] = ga_sw_thresholds

    sw_gauss_sigma = 1.0
    if 'ga_sw_gauss_sigmas' in pars:
        ga_sw_gauss_sigmas = pars['ga_sw_gauss_sigmas']
        if len(ga_sw_gauss_sigmas) == 1:
            ga_sw_gauss_sigmas = ga_sw_gauss_sigmas * pars['ga_generations']
        elif len(pars['ga_sw_gauss_sigmas']) < pars['ga_generations']:
            ga_sw_gauss_sigmas = ga_sw_gauss_sigmas + [sw_gauss_sigma] * (pars['ga_generations'] - len(ga_sw_gauss_sigmas))
    else:
        ga_sw_gauss_sigmas = [sw_gauss_sigma] * pars['ga_generations']
    pars['ga_sw_gauss_sigmas'] = ga_sw_gauss_sigmas

    if 'ga_breed_modes' not in pars:
        ga_breed_modes = ['sqrt_ab'] * pars['ga_generations']
    else:
        ga_breed_modes = pars['ga_breed_modes']
        if len(ga_breed_modes) == 1:
            ga_breed_modes = ga_breed_modes * pars['ga_generations']
        elif len(ga_breed_modes) < pars['ga_generations']:
            ga_breed_modes = ga_breed_modes + ['sqrt_ab'] * (pars['ga_generations'] - len(ga_breed_modes))
    pars['ga_breed_modes'] = ga_breed_modes

    if 'ga_lpf_sigmas' in pars:
        pars['low_resolution_generations'] = len(pars['ga_lpf_sigmas'])
    else:
        pars['low_resolution_generations'] = 0

    if pars['low_resolution_generations'] > 0:
        pars['low_resolution_alg'] = pars.get('low_resolution_alg', 'GAUSS')

    return pars

