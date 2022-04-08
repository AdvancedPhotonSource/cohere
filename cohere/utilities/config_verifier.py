# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
verification of configuration files
"""

import cohere.utilities.utils as ut
import os
from config_errors_dict import *

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['ver_list_int',
           'ver_list_float',
           'get_config_error_message',
           'ver_config',
           'ver_config_rec',
           'ver_config_data',
           'ver_config_prep',
           'ver_config_disp']
           

def ver_list_int(param_name, param_value):
    """
    This function verifies if all elements in a given list are int.

    Parameters
    ----------
    param_name : str
        the parameter being evaluated

    param_value : list
        the list to evaluate for int values

    Returns
    -------
    eval : boolean
        True if all elements are int, False otherwise
    """
    if not issubclass(type(param_value), list):
        print (param_name + ' is not a list')
        return False
    for e in param_value:
        if type(e) != int:
            print (param_name + ' should be list of integer values')
            return False
    return True
def ver_list_float(param_name, param_value):
    """
    This function verifies if all elements in a given list are float.

    Parameters
    ----------
    param_name : str
        the parameter being evaluated

    param_value : list
        the list to evaluate for float values

    Returns
    -------
    eval : boolean
        True if all elements are float, False otherwise
    """
    if not issubclass(type(param_value), list):
        print (param_name + ' is not a list')
        return False
    for e in param_value:
        if type(e) != float:
            print (param_name + ' should be list of float values')
            return False
    return True



def get_config_error_message(config_file_name,map_file,config_parameter,config_error_no):
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
    presented_message = "File="+config_file_name, "Parameter="+config_parameter, "Error="+error_string_message

    return(presented_message)

def ver_config(fname):
    """
    This function verifies experiment main config file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
        True if configuration is correct, False otherwise
    """
    config_map_file = 'config_error_map_file'
    config_parameter = 'File'


    if not os.path.isfile(fname):
        config_error = 0
        error_message = get_config_error_message(fname,config_map_file,config_parameter,config_error)
        print ('no configuration file ' + fname + ' found')
#        print(error_message)
        return (error_message)

    try:
        config_map = ut.read_config(fname)
        if config_map is None:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ("can't read configuration file")
#            print(error_message)
            return (error_message)
    except Exception as e:
        config_error = 2
        print(str(e))
        print ('Cannot parse ' + fname + ' configuration file. Check paranthesis and quotations.')
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        return (error_message)
    config_parameter = 'Workingdir'
    try:
        working_dir = config_map.working_dir
        if type(working_dir) != str:
            config_error = 0
            print('working_dir parameter should be string')
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        print ('working_dir parameter parsing error')
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

        return (error_message)
    config_parameter = 'ExperimentID'
    try:
        experiment_id = config_map.experiment_id
        if type(experiment_id) != str:
            config_error = 0
            print('experiment_id parameter should be string')
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        print ('experiment_id parameter parsing error')
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

        return (error_message)

    config_parameter = 'Scan'

    try:
        scan = config_map.scan
        if type(scan) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            print('scan parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        print ('scan parameter parsing error')
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        return (error_message)

    config_parameter = 'Specfile'

    try:
        specfile = config_map.specfile
        if type(specfile) != str:
            config_error = 0
            print('specfile parameter should be string')
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        print ('specfile parameter parsing error')
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

        return (error_message)

    return ("")


def ver_config_rec(fname):
    """
    This function verifies experiment config_rec file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
        True if configuration is correct, False otherwise
    """
    config_map_file = 'config_rec_error_map_file'
    config_parameter = 'File'

    if not os.path.isfile(fname):
        config_error = 0
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('no configuration file ' + fname + ' found')
        return (error_message)

    try:
        config_map = ut.read_config(fname)
        if config_map is None:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ("can't read configuration file")
            return (error_message)
    except:
        config_error = 2
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('Cannot parse ' + fname + ' configuration file. Check parenthesis and quotations.')
        return (error_message)

    config_parameter = 'Datadir'

    try:
        data_dir = config_map.data_dir
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('data_dir parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('data_dir parameter parsing error')
        return (error_message)

    config_parameter = 'Savedir'

    try:
        save_dir = config_map.save_dir
        if type(save_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('save_dir parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('save_dir parameter parsing error')
        return (error_message)

    config_parameter = 'Cont'


    try:
        cont = config_map.cont
        if type(cont) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ('cont parameter should be true or false')
            return (error_message)
        config_parameter = 'Continuedir'
        try:
            continue_dir = config_map.continue_dir
            if type(continue_dir) != str:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('continue_dir parameter should be string')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('continue_dir parameter parsing error')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('cont parameter parsing error')
        return (error_message)

    config_parameter = 'Reconstruction'

    try:
        reconstructions = config_map.reconstructions
        if type(reconstructions) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('reconstructions parameter should be int')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('reconstructions parameter parsing error')
        return (error_message)
    config_parameter = 'Device'

    try:
        device = config_map.device
        if not ver_list_int('device', device):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('device parameter parsing error')
        return (error_message)

    config_parameter = 'Algorithmsequence'

    try:
        algorithm_sequence = config_map.algorithm_sequence

        if not issubclass(type(algorithm_sequence), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ('algorithm_sequence should be a list')
            return (error_message)
        for s in algorithm_sequence:
            for i in range(len(s)):
                # the first element in each sub-list is the repeat factor and should be int
                if i== 0 and type(s[i]) != int:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('algorithm_sequence configuration error, the repeat factor should be int')
                    return (error_message)
                if i > 0:
                    if not issubclass(type(s[i]), list):
                        config_error = 2
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print ('algorithm_sequence configuration error, the sequence element should be a list')
                        return (error_message)
                    algorithm = s[i][0]
                    if type(algorithm) != str:
                        config_error = 3
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print ('algorithm_sequence configuration error, algorithm should be str')
                        return (error_message)
                    algorithm_options = ["ER", "HIO"]
                    if algorithm not in algorithm_options:
                        config_error = 4
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print ('algorithm_sequence configuration error, algorithm should be "ER" or "HIO"')
                        return (error_message)
                    algorithm_repeat = s[i][1]
                    if type(algorithm_repeat) != int:
                        config_error = 5
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print ('algorithm_sequence configuration error, algorithm repeat should be int')
                        return (error_message)
    except AttributeError:
        config_error = 6
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('missing mandatory algorithm_sequence parameter')
        return (error_message)
    except:
        config_error = 7
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('algorithm_sequence parameter parsing error')
        return (error_message)
    config_parameter = 'Hiobeta'

    try:
        hiobeta = config_map.hio_beta
        if type(hiobeta) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('hiobeta parameter should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('beta parameter parsing error')
        return (error_message)
    config_parameter = 'Generations'

    try:
        generations = config_map.ga_generations
        if type(generations) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_generations parameter should be int')
            return (error_message)
        config_parameter = 'Gametrics'
        try:
            ga_metrics = config_map.ga_metrics
            if not issubclass(type(ga_metrics), list):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (ga_metrics + ' is not a list')
                return (error_message)
            metrics_options = ['chi', 'sharpness', 'summed_phase', 'area']
            for metric in ga_metrics:
                if metric not in metrics_options:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('non supported metric option used:', metric)
                    print ('ga_metrics list can include only following strings: "chi", "sharpness", "summed_phase", "area"')
                    return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_metrics parameter parsing error')
            return (error_message)
        config_parameter = 'Gabreedmodes'

        try:
            ga_breed_modes = config_map.ga_breed_modes
            if not issubclass(type(ga_breed_modes), list):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print (ga_breed_modes + ' is not a list')
                return (error_message)
            breed_options = ['none', 'sqrt_ab', 'dsqrt', 'pixel_switch', 'b_pa', '2ab_a_b', '2a_b_pa', 'sqrt_ab_pa',\
'sqrt_ab_pa_recip', 'sqrt_ab_recip', 'max_ab', 'max_ab_pa', 'min_ab_pa', 'avg_ab', 'avg_ab_pa']
            for breed in ga_breed_modes:
                if breed not in breed_options:
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('non supported breed mode used:', breed)
                    print ('ga_breed_modes list can include only following strings: “none”, “sqrt_ab”, “dsqrt”,\
#“pixel_switch”, “b_pa”, “2ab_a_b”, “2a_b_pa”, “sqrt_ab_pa”, “sqrt_ab_pa_recip”, “sqrt_ab_recip”,\
#“max_ab”, “max_ab_pa”, “min_ab_pa”, “avg_ab”, “avg_ab_pa”')
                    return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 2
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_breed_modes parameter parsing error')
            return (error_message)

        config_parameter = 'Gacullings'

        try:
            ga_cullings = config_map.ga_cullings
            if not ver_list_int('ga_cullings', ga_cullings):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            print('ga_cullings parameter parsing error')
            return (error_message)

        config_parameter = 'Gashrinkwrapthresholds'

        try:
            ga_shrink_wrap_thresholds = config_map.ga_shrink_wrap_thresholds
            if not ver_list_float('ga_shrink_wrap_thresholds', ga_shrink_wrap_thresholds):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_shrink_wrap_thresholds parameter parsing error')
            return (error_message)

        config_parameter = 'Gashrinkwrapgausssigmas'

        try:
            ga_shrink_wrap_gauss_sigmas = config_map.ga_shrink_wrap_gauss_sigmas
            if not ver_list_float('ga_shrink_wrap_gauss_sigmas', ga_shrink_wrap_gauss_sigmas):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_shrink_wrap_gauss_sigmas parameter parsing error')
            return (error_message)

        config_parameter = 'Galowpassfiltersigmas'

        try:
            ga_lowpass_filter_sigmas = config_map.ga_lowpass_filter_sigmas
            if not ver_list_float('ga_lowpass_filter_sigmas', ga_lowpass_filter_sigmas):
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_lowpass_filter_sigmas parameter parsing error')
            return (error_message)

        config_parameter = 'Gagenpcstart'

        try:
            ga_gen_pc_start = config_map.ga_gen_pc_start
            if type(ga_gen_pc_start) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('ga_gen_pc_start parameter should be int')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('ga_gen_pc_start parameter parsing error')
            return (error_message)

    except AttributeError:
        pass
    except:
        config_parameter = 'Generations'
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('generations parameter parsing error')
        return (error_message)

    config_parameter = 'Twintrigger'

    try:
        twin_trigger = config_map.twin_trigger
        if not ver_list_int('twin_trigger', twin_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
        else:
            config_parameter = 'Twinhalves'
            try:
                twin_halves = config_map.twin_halves
                if not ver_list_int('twin_halves', twin_halves):
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('twin_halves parameter parsing error')
                return (error_message)

    except AttributeError:
        pass

    config_parameter = 'Shrinkwraptrigger'

    try:
        if not ver_list_int('shrink_wrap_trigger', config_map.shrink_wrap_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
        else:
            config_parameter = 'Shrinkwraptype'
            try:
                shrink_wrap_type = config_map.shrink_wrap_type
                if type(shrink_wrap_type) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('shrink_wrap_type parameter should be string')
                    return (error_message)
                if shrink_wrap_type != "GAUSS":
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('supporting shrink_wrap_type "GAUSS"')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 2
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('shrink_wrap_type parameter parsing error')
                return (error_message)

            config_parameter = 'Shrinkwrapthreshold'

            try:
                shrink_wrap_threshold = config_map.shrink_wrap_threshold
                if type(shrink_wrap_threshold) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print('shrink_wrap_threshold should be float')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('shrink_wrap_threshold parameter parsing error')
                return (error_message)

            config_parameter = 'Shrinkwrapgausssigma'

            try:
                shrink_wrap_gauss_sigma = config_map.shrink_wrap_gauss_sigma
                if type(shrink_wrap_gauss_sigma) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print('shrink_wrap_gauss_sigma should be float')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('shrink_wrap_gauss_sigma parameter parsing error')
                return (error_message)

            config_parameter = 'Initialsupportarea'

            try:
                initial_support_area = config_map.initial_support_area
                if not issubclass(type(initial_support_area), list):
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print('initial_support_area should be list')
                    return (error_message)
                for e in initial_support_area:
                    if type(e) != int and type(e) !=float:
                        config_error = 1
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print('initial_support_area should be a list of int or float')
                        return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 2
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('initial_support_area parameter parsing error')
                return (error_message)

            config_parameter = 'Phasesupporttrigger'


    except AttributeError:
        pass

    try:
        if not ver_list_int('phase_support_trigger', config_map.phase_support_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            return (error_message)


        else:

            config_parameter = 'Phmphasemin'
            try:
                phm_phase_min = config_map.phm_phase_min
                if type(phm_phase_min) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

                    print('phm_phase_min should be float')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

                print('phm_phase_min parameter parsing error')
                return (error_message)

            try:
                config_parameter = 'Phmphasemax'
                phm_phase_max = config_map.phm_phase_max
                if type(phm_phase_max) != float:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print('phm_phase_max should be float')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('phm_phase_max parameter parsing error')
                return (error_message)


    except AttributeError:
        pass

    config_parameter = 'Pcinterval'

    try:
        if not ver_list_int('pc_interval', config_map.pc_interval):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
        else:
            config_parameter = 'Pctype'
            try:
                pc_type = config_map.pc_type
                if type(pc_type) != str:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('pc_type parameter should be string')
                    return (error_message)
                if pc_type != "LUCY":
                    config_error = 1
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('pc_type parameter can be configured "LUCY"')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 2
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('pc_type parameter parsing error')
                return (error_message)

            config_parameter = 'Pclucyiterations'

            try:
                pc_LUCY_iterations = config_map.pc_LUCY_iterations
                if type(pc_LUCY_iterations) != int:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print('pc_LUCY_iterations should be int')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('pc_LUCY_iterations parameter parsing error')
                return (error_message)

            config_parameter = 'Pcnormalize'

            try:
                pc_normalize = config_map.pc_normalize
                if type(pc_normalize) != bool:
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    print ('pc_normalize parameter should be true or false')
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('pc_normalize parameter parsing error')
                return (error_message)

            config_parameter = 'Pclucykernel'

            try:
                pc_LUCY_kernel = config_map.pc_LUCY_kernel
                if not ver_list_int('pc_LUCY_kernel', pc_LUCY_kernel):
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    return (error_message)
            except AttributeError:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print("'pc_LUCY_kernel' parameter must be configured when partial coherence feature in active")
                return (error_message)
            except:
                config_error = 2
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print("'pc_LUCY_kernel' parameter parsing error")
                return (error_message)

    except AttributeError:
        pass
    config_parameter = 'Resolutiontrigger'

    try:
        if not ver_list_int('resolution_trigger', config_map.resolution_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
        else:
            config_parameter = 'Lowpassfilterswsigmarange'
            try:
                lowpass_filter_sw_sigma_range = config_map.lowpass_filter_sw_sigma_range
                if not ver_list_float('lowpass_filter_sw_sigma_range', lowpass_filter_sw_sigma_range):
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print("'lowpass_filter_sw_sigma_range' parameter parsing error")
                return (error_message)

            config_parameter = 'Lowpassfilterrange'

            try:
                lowpass_filter_range = config_map.lowpass_filter_range
                if not ver_list_float('lowpass_filter_range', lowpass_filter_range):
                    config_error = 0
                    error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                    return (error_message)
            except AttributeError:
                pass
            except:
                config_error = 1
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print("'lowpass_filter_range' parameter parsing error")
                return (error_message)

    except AttributeError:
        pass
    config_parameter = 'Averagetrigger'
    try:
        if not ver_list_int('average_trigger', config_map.average_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass
    config_parameter = 'Progresstrigger'
    try:
        if not ver_list_int('progress_trigger', config_map.progress_trigger):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass

    return ("")


def ver_config_data(fname):
    """
    This function verifies experiment config_data file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
        True if configuration is correct, False otherwise
    """
    config_map_file = 'config_data_error_map_file'
    config_parameter = 'File'

    if not os.path.isfile(fname):
        config_error = 0
        error_message = get_config_error_message(fname,config_map_file,config_parameter,config_error)
        print ('no configuration file ' + fname + ' found')
        return (error_message)
    config_map = ut.read_config(fname)
    try:
        config_map = ut.read_config(fname)
        if config_map is None:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ("can't read configuration file")
            return (error_message)
    except:
        config_error = 2
        error_message = get_config_error_message(fname,config_map_file,config_parameter,config_error)
        print ('Cannot parse ' + fname + ' configuration file. Check paranthesis and quotations.')
        return False

    config_parameter = 'Datadir'

    try:
        data_dir = config_map.data_dir
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('data_dir parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname,config_map_file,config_parameter,config_error)
        print ('data_dir parameter parsing error')
        return (error_message)

    config_parameter = 'Adjustdimensions'


    try:
        if not ver_list_int('pad_crop', config_map.adjust_dimensions):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass

    config_parameter = 'Centershift'

    try:
        if not ver_list_int('center_shift', config_map.center_shift):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass

    config_parameter = 'Binning'
    try:
        if not ver_list_int('binning', config_map.binning):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass

    config_parameter = 'Intensitythreshold'

    try:
        intensity_threshold = config_map.intensity_threshold
        if type(intensity_threshold) != float and type(intensity_threshold) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('intensity_threshold should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('intensity_threshold parameter parsing error')
        return (error_message)

    config_parameter = 'Aliens'


    alien_alg = 'none'
    try:
        alien_alg = config_map.alien_alg
    except AttributeError:
        pass
    if alien_alg == 'none':
        pass
    elif alien_alg == 'block_aliens':
        try:
            aliens = config_map.aliens
            if issubclass(type(aliens), list):
                for a in aliens:
                    if not issubclass(type(a), list):
                        config_error = 0
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print ('aliens should be a list of alien blocks (lists) ')
                        return (error_message)
                    if not ver_list_int('aliens', a):
                        config_error = 1
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        return (error_message)
                    if (len(a) < 6):
                        config_error = 2
                        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                        print('misconfigured alien, each alien is defined by list of six int')
                        return (error_message)
        except AttributeError:
            pass
        except Exception as e:
            config_error = 3
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('aliens parameter parsing error ', str(e))
            return (error_message)

    elif alien_alg == 'alien_file':

        config_parameter = 'AlienFile'

        try:
            alien_file = config_map.alien_file
            if type(alien_file) != str:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('alien_file should be a string (mask file name)')
                return (error_message)
        except AttributeError:
            pass
    elif alien_alg == 'AutoAlien1':

        config_parameter = 'Aa1sizethreshold'
        try:
            AA1_size_threshold = config_map.AA1_asym_threshold
            if type(AA1_size_threshold) != float and type(AA1_asym_threshold) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_size_threshold should be float')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_size_threshold parameter parsing error')
            return (error_message)
        config_parameter = 'Aa1asymthreshold'
        try:
            AA1_asym_threshold = config_map.AA1_asym_threshold
            if type(AA1_asym_threshold) != float and type(AA1_asym_threshold) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_asym_threshold should be float')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_asym_threshold parameter parsing error')
            return (error_message)

        config_parameter = 'Aa1minpts'

        try:
            AA1_min_pts = config_map.AA1_min_pts
            if type(AA1_min_pts) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_min_pts should be int')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_min_pts parameter parsing error')
            return (error_message)

        config_parameter = 'Aa1eps'

        try:
            AA1_eps = config_map.AA1_eps
            if type(AA1_eps) != float and type(AA1_eps) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_eps should be float')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_eps parameter parsing error')
            return (error_message)
        config_parameter = 'Aa1ampthreshold'
        try:
            AA1_amp_threshold = config_map.AA1_amp_threshold
            if type(AA1_amp_threshold) != float and type(AA1_amp_threshold) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_amp_threshold should be float')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_amp_threshold parameter parsing error')
            return (error_message)
        config_parameter = 'Aa1savearrs'
        try:
            AA1_save_arrs = config_map.AA1_save_arrs
            if type(AA1_save_arrs) != bool:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_save_arrs parameter should be true or false')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_save_arrs parameter parsing error')
            return (error_message)
        config_parameter = 'Aa1expandcleanedsigma'
        try:
            AA1_expandcleanedsigma = config_map.AA1_expandcleanedsigma
            if type(AA1_expandcleanedsigma) != float and type(AA1_expandcleanedsigma) != int:
                config_error = 0
                error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
                print('AA1_expandcleanedsigma should be float')
                return (error_message)
        except AttributeError:
            pass
        except:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('AA1_expandcleanedsigma parameter parsing error')
            return (error_message)
    return ("")


def ver_config_prep(fname):
    """
    This function verifies experiment config_prep file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
        True if configuration is correct, False otherwise
    """
    config_map_file = 'config_prep_error_map_file'
    config_parameter = 'File'

    if not os.path.isfile(fname):
        config_error = 0
        error_message = get_config_error_message(fname,config_map_file,config_parameter,config_error)
        print ('no configuration file ' + fname + ' found')
        return (error_message)

    try:
        config_map = ut.read_config(fname)
        if config_map is None:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ("can't read configuration file")
            return (error_message)
    except:
        config_error = 2
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('Cannot parse ' + fname + ' configuration file. Check parenthesis and quotations.')
        return (error_message)

    config_parameter = 'Roi'

    try:
        if not ver_list_int('roi', config_map.roi):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            print ('roi parameter should be a list of int')
            return (error_message)
    except AttributeError:
        pass

    config_parameter = 'Data_dir'


    try:
        data_dir = config_map.data_dir
        if type(data_dir) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

            print('data_dir parameter should be string')
            return False
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

        print ('data_dir parameter parsing error')
        return (error_message)

    config_parameter = 'Darkfield'

    try:
        darkfield_filename = config_map.darkfield_filename
        if type(darkfield_filename) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('darkfield_filename parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('darkfield_filename parameter parsing error')
        return (error_message)

    config_parameter = 'Whitefield'

    try:
        whitefield_filename = config_map.whitefield_filename
        if type(whitefield_filename) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('whitefield_filename parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('whitefield_filename parameter parsing error')
        return (error_message)

    config_parameter = 'Excludescans'

    try:
        if not ver_list_int('exclude_scans', config_map.exclude_scans):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            return (error_message)
    except AttributeError:
        pass

    config_parameter = 'MinFiles'

    try:
        min_files = config_map.min_files
        if type(min_files) != int:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('min_files should be int')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('min_files parameter parsing error')
        return (error_message)

    config_parameter = 'Separatescans'


    try:
        separate_scans = config_map.separate_scans
        if type(separate_scans) != bool:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('separate_scans parameter should be true or false')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)

        print('separate_scans parameter parsing error')
        return (error_message)

    return ("")


def ver_config_disp(fname):
    """
    This function verifies experiment config_disp file

    Parameters
    ----------
    fname : str
        configuration file name

    Returns
    -------
        True if configuration is correct, False otherwise
    """
    config_map_file = 'config_disp_error_map_file'
    config_parameter = 'File'

    if not os.path.isfile(fname):
        config_error = 0
        error_message = get_config_error_message(fname,config_map_file,config_parameter,config_error)
        print ('no configuration file ' + fname + ' found')
        return (error_message)

    try:
        config_map = ut.read_config(fname)
        if config_map is None:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print ("can't read configuration file")
            return (error_message)
    except:
        config_error = 2
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('Cannot parse ' + fname + ' configuration file. Check paranthesis and quotations.')
        return (error_message)

    config_parameter = 'Diffractometer'


    try:
        diffractometer = config_map.diffractometer
        if type(diffractometer) != str:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('diffractometer parameter should be string')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print ('diffractometer parameter parsing error')
        return (error_message)

    config_parameter = 'Crop'

    try:
        crop = config_map.crop
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
    except AttributeError:
        pass
    except:
        config_error = 2
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('crop parameter parsing error')
        return (error_message)

    config_parameter = 'Energy'

    try:
        energy = config_map.energy

        if type(energy) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('energy should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('energy parameter parsing error')
        return (error_message)

    config_parameter = 'Delta'

    try:
        delta = config_map.delta
        if type(delta) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('delta should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('delta parameter parsing error')
        return (error_message)

    config_parameter = 'Gamma'

    try:
        gamma = config_map.gamma
        if type(gamma) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('gamma should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('gamma parameter parsing error')
        return (error_message)

    config_parameter = 'Detdist'

    try:
        detdist = config_map.detdist
        if type(detdist) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('detdist should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('detdist parameter parsing error')
        return (error_message)

    config_parameter = 'Dth'

    try:
        dth = config_map.dth
        if type(dth) != float:
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('dth should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 1
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('dth parameter parsing error')
        return (error_message)

    config_parameter = 'Pixel'

    try:
        pixel = config_map.pixel
        if not issubclass(type(pixel), list):
            config_error = 0
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('pixel should be a list')
            return (error_message)
        if type(pixel[0]) != float or type(pixel[1]) != float:
            config_error = 1
            error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
            print('pixel values should be float')
            return (error_message)
    except AttributeError:
        pass
    except:
        config_error = 2
        error_message = get_config_error_message(fname, config_map_file, config_parameter, config_error)
        print('pixel parameter parsing error')
        return (error_message)

    return ("")

