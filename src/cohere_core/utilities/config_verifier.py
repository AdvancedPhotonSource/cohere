# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.config_verifier
===========================

Verification of configuration parameters.
"""

import os
from cohere_core.utilities.config_errors_dict import *
import cohere_core.utilities.utils as ut

__author__ = "Dave Cyl"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['verify']
           

def ver_list_int(param_name, param_value):
    """
    This function verifies if all elements in a given list are int.

    :param param_name: name of the parameter being evaluated
    :param param_value: the list to evaluate for int values 
    :return:  True if all elements are int, False otherwise
    """
    if not issubclass(type(param_value), list):
        print (f'{param_name} is not a list')
        return False
    for e in param_value:
        if type(e) != int:
            print (f'{param_name} should be list of integer values')
            return False
    return True


def ver_list_float(param_name, param_value):
    """
    This function verifies if all elements in a given list are float.

    :param param_name: name of the parameter being evaluated
    :param param_value: the list to evaluate for float values 
    :return:  True if all elements are float, False otherwise
    """
    if not issubclass(type(param_value), list):
        print (f'{param_name} is not a list')
        return False
    for e in param_value:
        if type(e) != float:
            print (f'{param_name} should be list of float values')
            return False
    return True


def get_config_error_message(config_file_name, map_file, config_parameter, config_error_no):
    """
    This function returns an error message string for this config file from the error map file using
    the parameter and error number as references for the error.

    :param config_file_name: The config file being verified
    :param map_file: The dictionary of error dictionary files
    :param config_parameter: The particular config file parameter being tested
    :param config_error_no: The error sequence in the test
    :return: An error string describing the error and where it was found
    """
    config_map_dic = config_map_names.get(map_file)
    error_string_message = config_map_dic.get(config_parameter)[config_error_no]
    # presented_message = "File=" + config_file_name, "Parameter=" + config_parameter, "Error=" + error_string_message

    return(error_string_message)


def ver_config(config_map):
    """
    This function verifies parameters from config file

    :param config_map: dict with main configuration parameters
    :return:  message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_error_map_file'
    fname = 'config'

    config_parameter = 'Workingdir'
    if 'working_dir' in config_map:
        working_dir = config_map['working_dir']
        if type(working_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return error_message

    config_parameter = 'ExperimentID'
    if 'experiment_id' in config_map:
        experiment_id = config_map['experiment_id']
        if type(experiment_id) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
    else:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print(error_message)
        return error_message

    config_parameter = 'Scan'
    if 'scan' in config_map:
        scan = config_map['scan']
        if type(scan) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Separatescans'
    if 'separate_scans' in config_map:
        separate_scans = config_map['separate_scans']
        if type(separate_scans) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Separatescanranges'
    if 'separate_scan_ranges' in config_map:
        separate_scan_ranges = config_map['separate_scan_ranges']
        if type(separate_scan_ranges) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Multipeak'
    if 'multipeak' in config_map:
        separate_scans = config_map['multipeak']
        if type(separate_scans) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    return ("")


def ver_config_prep(config_map):
    """
    This function verifies experiment config_prep file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_prep_error_map_file'
    fname = 'config_prep'

    config_parameter = 'Excludescans'
    if 'exclude_scans' in config_map:
        if not ver_list_int('exclude_scans', config_map['exclude_scans']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'MinFiles'
    if 'min_frames' in config_map:
        min_frames = config_map['min_frames']
        if type(min_frames) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Maxcrop'
    if 'max_crop' in config_map:
        if not ver_list_int('max_crop', config_map['max_crop']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        elif len(config_map['max_crop']) < 2:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    return ("")


def ver_config_rec(config_map):
    """
    This function verifies experiment config_rec file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    import string

    def get_no_iter(s):
        seq = []
        def parse_entry(ent):
            r_e = ent.split('*')
            # if r_e[1] not in algs:
            #     return r_e[1] + ' is not a valid entry in algorithm_sequence parameter'
            seq.append([int(r_e[0]), r_e[1]])
            return ''

        s = s.replace(' ','')
        entries = s.split('+')
        i = 0
        while i < len(entries):
            entry = entries[i]
            if '(' in entry:
                group = []
                rep_entry = entry.split('(')
                repeat = int(rep_entry[0][:-1])
                group.append(rep_entry[1])
                i += 1
                group_entry = entries[i]
                while ')' not in group_entry:
                    group.append(group_entry)
                    i += 1
                    group_entry = entries[i]
                group.append(group_entry[:-1])
                for _ in range(repeat):
                    for group_entry in group:
                        msg = parse_entry(group_entry)
                        if len(msg) > 0:
                            return msg, 0
                i += 1
            else:
                msg = parse_entry(entry)
                if len(msg) > 0:
                    return msg, 0
                i += 1
        return '', sum([e[0] for e in seq])


    def verify_trigger(trigger, no_iter, trigger_name):
        if not ver_list_int(trigger_name, trigger):
            return(f'{trigger_name} trigger type should be list of int')
        if len(trigger) == 0:
            return (f'empty {trigger_name} trigger {str(trigger)}')
        elif trigger[0] >= no_iter:
            return (f'{trigger_name} trigger start {str(trigger[0])} exceeds number of iterations {str(no_iter)}')
        if len(trigger) == 3:
            if trigger[2] >= no_iter:
                return (f'{trigger_name} trigger end {str(trigger[2])} exceeds number of iterations {str(no_iter)}')
        return ''


    config_map_file = 'config_rec_error_map_file'
    fname = 'config_rec'

    config_parameter = 'Datadir'
    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if not os.path.isdir(data_dir):
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if not os.path.isfile(ut.join(data_dir, 'data.tif')) and not os.path.isfile(ut.join(data_dir, 'data.npy')):
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Savedir'
    if 'save_dir' in config_map:
        save_dir = config_map['save_dir']
        if type(save_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Initguess'
    if 'init_guess' in config_map:
        init_guess = config_map['init_guess']
        init_guess_options = ['random', 'continue', 'AI_guess']
        if init_guess not in init_guess_options:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        elif init_guess == 'continue':
            config_parameter = 'Continuedir'
            if 'continue_dir' in config_map:
                continue_dir = config_map['continue_dir']
                if type(continue_dir) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return error_message
            else:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return error_message
        elif init_guess == 'AI_guess':
            config_parameter = 'Aitrainedmodel'
            if 'AI_trained_model' in config_map:
                AI_trained_model = config_map['AI_trained_model']
                if type(AI_trained_model) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(config_error)
                    return error_message
            else:
                config_error = 1
                print(fname, config_map_file, config_parameter, config_error)
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return error_message

    config_parameter = 'Reconstruction'
    if 'reconstructions' in config_map:
        reconstructions = config_map['reconstructions']
        if type(reconstructions) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Device'
    if 'device' in config_map:
        def ver_dev(d):
            if d == 'all' or ver_list_int('device', device):
                return ''
            else:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
               
        device = config_map['device']
        if issubclass(type(device), dict):
            for d in device.values():
                error_message =  ver_dev(d)
                if len(error_message) > 0:
                    print(error_message)
                    return (error_message)
        else:
            error_message = ver_dev(device)
            if len(error_message) > 0:
                print(error_message)
                return (error_message)

    config_parameter = 'Algorithmsequence'
    if 'algorithm_sequence' in config_map:
        algorithm_sequence = config_map['algorithm_sequence']
        if type(algorithm_sequence) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print (config_error)
            return (error_message)
        # check for supported characters
        alg_seq_chars = list(string.ascii_lowercase) + list(string.ascii_uppercase) + list(string.digits) + ['.','*', '+', '(', ')', ' ']
        if 0 in [c in alg_seq_chars for c in algorithm_sequence]:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(config_error)
            return (error_message)
        # check brackets, nested are not allowed
        br_count = 0
        for c in algorithm_sequence:
            if c == '(':
                br_count += 1
                if br_count > 1:
                    break
            elif c == ')':
                br_count -= 1
                if br_count < 0:
                    break
        if br_count != 0:
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(config_error)
            return (error_message)
        # calculate number of iterations
        try:
            msg, iter_no = get_no_iter(algorithm_sequence)
            if len(msg) > 0:
                print(msg)
                return msg
        except Exception as e:
            print('check algorithm_sequence')
            return ('check algorithm_sequence')
    else:
        config_error = 3
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print (error_message)
        return (error_message)

    config_parameter = 'Hiobeta'
    if 'hio_beta' in config_map:
        hio_beta = config_map['hio_beta']
        if type(hio_beta) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Initialsupportarea'
    if 'initial_support_area' in config_map:
        initial_support_area = config_map['initial_support_area']
        if not issubclass(type(initial_support_area), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        for e in initial_support_area:
            if type(e) != int and type(e) != float:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Generations'
    if 'ga_generations' in config_map:
        generations = config_map['ga_generations']
        if type(generations) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if 'reconstructions' in config_map:
            reconstructions = config_map['reconstructions']
        else:
            reconstructions = 1
        if reconstructions < 2:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Gametrics'
        if 'ga_metrics' in config_map:
            ga_metrics = config_map['ga_metrics']
            if not issubclass(type(ga_metrics), list):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (error_message)
                return (error_message)
            metrics_options = ['chi', 'sharpness', 'summed_phase', 'area']
            for metric in ga_metrics:
                if metric not in metrics_options:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print (error_message)
                    return ('') 

        config_parameter = 'Gabreedmodes'
        if 'ga_breed_modes' in config_map:
            ga_breed_modes = config_map['ga_breed_modes']
            if not issubclass(type(ga_breed_modes), list):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (error_message)
                return (error_message)
            breed_options = ['none', 'sqrt_ab', 'pixel_switch', 'b_pa', '2ab_a_b', '2a_b_pa', 'sqrt_ab_pa',\
'sqrt_ab_pa_recip', 'sqrt_ab_recip', 'max_ab', 'max_ab_pa', 'min_ab_pa', 'avg_ab', 'avg_ab_pa']
            for breed in ga_breed_modes:
                if breed not in breed_options:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print (error_message)
                    return ('')

        config_parameter = 'Gacullings'
        if 'ga_cullings' in config_map:
            ga_cullings = config_map['ga_cullings']
            if not ver_list_int('ga_cullings', ga_cullings):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
            if sum(ga_cullings) >= reconstructions:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Gashrinkwrapthresholds'
        if 'ga_sw_thresholds' in config_map:
            ga_sw_thresholds = config_map['ga_sw_thresholds']
            if not ver_list_float('ga_sw_thresholds', ga_sw_thresholds):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Gashrinkwrapgausssigmas'
        if 'ga_sw_gauss_sigmas' in config_map:
            ga_sw_gauss_sigmas = config_map['ga_sw_gauss_sigmas']
            if not ver_list_float('ga_sw_gauss_sigmas', ga_sw_gauss_sigmas):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Galowpassfiltersigmas'
        if 'ga_lpf_sigmas' in config_map:
            ga_lpf_sigmas = config_map['ga_lpf_sigmas']
            if not ver_list_float('ga_lpf_sigmas', ga_lpf_sigmas):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                return (error_message)

        config_parameter = 'Gagenpcstart'
        if 'ga_gen_pc_start' in config_map:
            ga_gen_pc_start = config_map['ga_gen_pc_start']
            if type(ga_gen_pc_start) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Twintrigger'
    if 'twin_trigger' in config_map:
        twin_trigger = config_map['twin_trigger']
        m = verify_trigger(twin_trigger, iter_no, 'twin')
        if len(m) > 0:
            print(m)
            return(m)
        if not ver_list_int('twin_trigger', twin_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Twinhalves'
        if 'twin_halves' in config_map:
            twin_halves = config_map['twin_halves']
            if not ver_list_int('twin_halves', twin_halves):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

    config_parameter = 'Shrinkwraptrigger'
    if 'shrink_wrap_trigger' in config_map:
        if '.SW' not in config_map['algorithm_sequence']:
            m = verify_trigger(config_map['shrink_wrap_trigger'], iter_no, 'shrink wrap')
            if len(m) > 0:
                print(m)
                return(m)
            if not ver_list_int('shrink_wrap_trigger', config_map['shrink_wrap_trigger']):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

            config_parameter = 'Shrinkwraptype'
            if 'shrink_wrap_type' in config_map:
                shrink_wrap_type = config_map['shrink_wrap_type']
                if type(shrink_wrap_type) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print (error_message)
                    return (error_message)
                if shrink_wrap_type != "GAUSS":
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print (error_message)
                    return (error_message)

            config_parameter = 'Shrinkwrapthreshold'
            if 'shrink_wrap_threshold' in config_map:
                shrink_wrap_threshold = config_map['shrink_wrap_threshold']
                if type(shrink_wrap_threshold) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Shrinkwrapgausssigma'
            if 'shrink_wrap_gauss_sigma' in config_map:
                shrink_wrap_gauss_sigma = config_map['shrink_wrap_gauss_sigma']
                if type(shrink_wrap_gauss_sigma) != float and type(shrink_wrap_gauss_sigma) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)
        else:
            for t in config_map['shrink_wrap_trigger']:
                if not ver_list_int('shrink_wrap_trigger', t):
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)
            config_parameter = 'Shrinkwrapthreshold'
            if 'shrink_wrap_threshold' in config_map:
                if not ver_list_float('shrink_wrap_threshold', config_map['shrink_wrap_threshold']):
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Shrinkwrapgausssigma'
            if 'shrink_wrap_gauss_sigma' in config_map:
                if not ver_list_float('shrink_wrap_gauss_sigma', config_map['shrink_wrap_gauss_sigma']):
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

    config_parameter = 'Phasesupporttrigger'
    if 'phc_trigger' in config_map:
        if '.PHC' not in config_map['algorithm_sequence']:
            m = verify_trigger(config_map['phc_trigger'], iter_no, 'phase constrain')
            if len(m) > 0:
                print(m)
                return(m)
            if not ver_list_int('phc_trigger', config_map['phc_trigger']):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

            config_parameter = 'Phcphasemin'
            if 'phc_phase_min' in config_map:
                phc_phase_min = config_map['phc_phase_min']
                if type(phc_phase_min) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Phcphasemax'
            if 'phc_phase_max' in config_map:
                phc_phase_max = config_map['phc_phase_max']
                if type(phc_phase_max) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)
        else:
            for t in config_map['phc_trigger']:
                if not ver_list_int('phc_trigger', t):
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)
            config_parameter = 'Phcphasemin'
            if 'phc_phase_min' in config_map:
                if not ver_list_float('phc_phase_min', config_map['phc_phase_min']):
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Phcphasemax'
            if 'phc_phase_max' in config_map:
                if not ver_list_float('phc_phase_max', config_map['phc_phase_max']):
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

    config_parameter = 'Pcinterval'
    if 'pc_interval' in config_map:
        pc_interval = config_map['pc_interval']
        if type(pc_interval) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if pc_interval >= iter_no:
            return('pc_interval', pc_interval, 'exceeds number of iterations', iter_no)

        config_parameter = 'Pctype'
        if 'pc_type' in config_map:
            pc_type = config_map['pc_type']
            if type(pc_type) != str:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
            if pc_type != "LUCY":
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Pclucyiterations'
        if 'pc_LUCY_iterations' in config_map:
            pc_LUCY_iterations = config_map['pc_LUCY_iterations']
            if type(pc_LUCY_iterations) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Pcnormalize'
        if 'pc_normalize' in config_map:
            pc_normalize = config_map['pc_normalize']
            if type(pc_normalize) != bool:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)

        config_parameter = 'Pclucykernel'
        if 'pc_LUCY_kernel' in config_map:
            pc_LUCY_kernel = config_map['pc_LUCY_kernel']
            if not ver_list_int('pc_LUCY_kernel', pc_LUCY_kernel):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        else:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Lpftrigger'
    if 'lowpass_filter_trigger' in config_map:
        m = verify_trigger(config_map['lowpass_filter_trigger'], iter_no, 'lowpass filter')
        if len(m) > 0:
            print(m)
            return(m)
        if not ver_list_int('lowpass_filter_trigger', config_map['lowpass_filter_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        if len(config_map['lowpass_filter_trigger']) < 3:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

        config_parameter = 'Lowpassfilterrange'
        if 'lowpass_filter_range' in config_map:
            lowpass_filter_range = config_map['lowpass_filter_range']
            if not ver_list_float('lowpass_filter_range', lowpass_filter_range):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        else:
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Averagetrigger'
    if 'average_trigger' in config_map:
        m = verify_trigger(config_map['average_trigger'], iter_no, 'average')
        if len(m) > 0:
            print(m)
            return(m)
        if not ver_list_int('average_trigger', config_map['average_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Progresstrigger'
    if 'progress_trigger' in config_map:
        m = verify_trigger(config_map['progress_trigger'], iter_no, 'progress')
        if len(m) > 0:
            return(m)
        if not ver_list_int('progress_trigger', config_map['progress_trigger']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    # return empty string if verified
    return ("")


def ver_config_data(config_map):
    """
    This function verifies experiment config_data file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_data_error_map_file'
    fname = 'config_data'

    config_parameter = 'Datadir'
    if 'data_dir' in config_map:
        data_dir = config_map['data_dir']
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'CropPad'
    if 'crop_pad' in config_map:
        if not ver_list_int('crop_pad', config_map['crop_pad']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Shift'
    if 'shift' in config_map:
        if not ver_list_int('shift', config_map['shift']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Binning'
    if 'binning' in config_map:
        if not ver_list_int('binning', config_map['binning']):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)

    config_parameter = 'Intensitythreshold'
    if 'intensity_threshold' in config_map:
        intensity_threshold = config_map['intensity_threshold']
        if type(intensity_threshold) != float and type(intensity_threshold) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return ''

    config_parameter = 'Alienalg'
    if 'alien_alg' in config_map:
        alien_alg = config_map['alien_alg']
        alien_alg_options = ['block_aliens', 'alien_file', 'AutoAlien1', 'none']
        if alien_alg not in alien_alg_options:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print(error_message)
            return (error_message)
        elif alien_alg == 'block_aliens':
            config_parameter = 'Aliens'
            if 'aliens' in config_map:
                aliens = config_map['aliens']
                if issubclass(type(aliens), list):
                    for a in aliens:
                        if not issubclass(type(a), list):
                            config_error = 0
                            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                            print (error_message)
                            return (error_message)
                        if not ver_list_int('aliens', a):
                            config_error = 1
                            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                            print (error_message)
                            return (error_message)
                        if (len(a) < 6):
                            config_error = 2
                            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                            print(error_message)
                            return (error_message)
            else:
                config_error = 3
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        elif alien_alg == 'alien_file':
            config_parameter = 'AlienFile'
            if 'alien_file' in config_map:
                alien_file = config_map['alien_file']
                if type(alien_file) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)
            else:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print(error_message)
                return (error_message)
        elif alien_alg == 'AutoAlien1':
            config_parameter = 'Aa1sizethreshold'
            if 'AA1_size_threshold' in config_map:
                AA1_size_threshold = config_map['AA1_size_threshold']
                if type(AA1_size_threshold) != float and type(AA1_size_threshold) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Aa1asymthreshold'
            if 'AA1_asym_threshold' in config_map:
                AA1_asym_threshold = config_map['AA1_asym_threshold']
                if type(AA1_asym_threshold) != float and type(AA1_asym_threshold) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Aa1minpts'
            if 'AA1_min_pts' in config_map:
                AA1_min_pts = config_map['AA1_min_pts']
                if type(AA1_min_pts) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(error_message)
                    return (error_message)

            config_parameter = 'Aa1eps'
            if 'AA1_eps' in config_map:
                AA1_eps = config_map['AA1_eps']
                if type(AA1_eps) != float and type(AA1_eps) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

            config_parameter = 'Aa1ampthreshold'
            if 'AA1_amp_threshold' in config_map:
                AA1_amp_threshold = config_map['AA1_amp_threshold']
                if type(AA1_amp_threshold) != float and type(AA1_amp_threshold) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

            config_parameter = 'Aa1savearrs'
            if 'AA1_save_arrs' in config_map:
                AA1_save_arrs = config_map['AA1_save_arrs']
                if type(AA1_save_arrs) != bool:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

            config_parameter = 'Aa1expandcleanedsigma'
            if 'AA1_expandcleanedsigma' in config_map:
                AA1_expandcleanedsigma = config_map['AA1_expandcleanedsigma']
                if type(AA1_expandcleanedsigma) != float and type(AA1_expandcleanedsigma) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print(AA1_size_threshold)
                    return (error_message)

    # return empty string if verified
    return ("")


def ver_config_disp(config_map):
    """
    This function verifies experiment config_disp file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
    error_message : str
        message describing parameter error or empty string if all parameters are verified
    """
    config_map_file = 'config_disp_error_map_file'
    fname = 'config_disp'

    config_parameter = 'Resultsdir'
    if 'results_dir' in config_map:
        results_dir = config_map['results_dir']
        if type(results_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('results_dir parameter should be string')
            return (error_message)

    config_parameter = 'Crop'
    if 'crop' in config_map:
        crop = config_map['crop']
        if not issubclass(type(crop), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('crop should be list')
            return (error_message)
        for e in crop:
            if type(e) != int and type(e) != float:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('crop should be a list of int or float')
                return (error_message)

    config_parameter = 'Rampups'
    if 'rampups' in config_map:
        rampups = config_map['rampups']
        if type(rampups) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('rampups should be float')
            return (error_message)

    return ("")


def verify(config_name, conf_map):
    """
    Verifies parameters.

    :param config_name: name of config to be verified. Supported: config_data, config_rec, config_disp.
    :param conf_map: dict with the parameters to verify.
        It defaults to None. 
    :return:  0 if successful, -1 otherwise. In debug mode will re-raise exception instead of returning -1.
    """
    if config_name == 'config':
        return ver_config(conf_map)
    if config_name == 'config_prep':
        return ver_config_prep(conf_map)
    elif config_name == 'config_data':
        return ver_config_data(conf_map)
    elif config_name == 'config_rec':
        return ver_config_rec(conf_map)
    elif config_name == 'config_disp':
        return ver_config_disp(conf_map)
    else:
        return ('verifier has no function to check config named', config_name)
