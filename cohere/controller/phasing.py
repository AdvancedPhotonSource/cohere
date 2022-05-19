# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This module controls the reconstruction process. The user has to provide parameters such as type of processor, data, and configuration.
The processor specifies which library will be used by FM (Fast Module) that performs the processor intensive calculations. The module can be run on cpu, or gpu. Depending on the gpu hardware and library, one can use opencl or cuda library.
"""

import time
import os
import importlib
import cohere.utilities.dvc_utils as dvut
import cohere.utilities.utils as ut
import cohere.utilities.config_verifier as ver
import cohere.controller.op_flow as of

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['get_norm',
           'reconstruction',
           'Rec']


def set_lib(dlib, is_af):
    global devlib
    devlib = dlib
    dvut.set_lib(devlib, is_af)


def get_norm(arr):
    return devlib.sum(devlib.square(devlib.absolute(arr)))


class Pcdi:
    def __init__(self, params, data, dir=None):
        self.params = params
        if dir is None:
            self.kernel = None
        else:
            try:
                self.kernel = devlib.load(os.path.join(dir, 'coherence.npy'))
            except:
                self.kernel = None

        self.dims = devlib.dims(data)
        self.roi_data = dvut.crop_center(devlib.fftshift(data), self.params['pc_LUCY_kernel'])
        if self.params['pc_normalize']:
            self.sum_roi_data = devlib.sum(devlib.square(self.roi_data))
        if self.kernel is None:
            self.kernel = devlib.full(self.params['pc_LUCY_kernel'], 0.5, dtype=devlib.dtype(data))

    def set_previous(self, abs_amplitudes):
        self.roi_amplitudes_prev = dvut.crop_center(devlib.fftshift(abs_amplitudes), self.params['pc_LUCY_kernel'])

    def apply_partial_coherence(self, abs_amplitudes):
        abs_amplitudes_2 = devlib.square(abs_amplitudes)
        converged_2 = devlib.fftconvolve(abs_amplitudes_2, self.kernel)
        converged_2 = devlib.where(converged_2 < 0, 0, converged_2)
        converged = devlib.sqrt(converged_2)
        return converged

    def update_partial_coherence(self, abs_amplitudes):
        roi_amplitudes = dvut.crop_center(devlib.fftshift(abs_amplitudes), self.params['pc_LUCY_kernel'])
        roi_combined_amp = 2 * roi_amplitudes - self.roi_amplitudes_prev
        if self.params['pc_normalize']:
            amplitudes_2 = devlib.square(roi_combined_amp)
            sum_ampl = devlib.sum(amplitudes_2)
            ratio = self.sum_roi_data / sum_ampl
            amplitudes = devlib.sqrt(amplitudes_2 * ratio)
        else:
            amplitudes = roi_combined_amp

        if self.params['pc_type'] == "LUCY":
            self.lucy_deconvolution(devlib.square(amplitudes), devlib.square(self.roi_data),
                                    self.params['pc_LUCY_iterations'])

    def lucy_deconvolution(self, amplitudes, data, iterations):
        data_mirror = devlib.flip(data)
        for i in range(iterations):
            conv = devlib.fftconvolve(self.kernel, data)
            devlib.where(conv == 0, 1.0, conv)
            relative_blurr = amplitudes / conv
            self.kernel = self.kernel * devlib.fftconvolve(relative_blurr, data_mirror)
        self.kernel = devlib.real(self.kernel)
        coh_sum = devlib.sum(devlib.absolute(self.kernel))
        self.kernel = devlib.absolute(self.kernel) / coh_sum


class Support:
    def __init__(self, params, dims, dir=None):
        self.params = params
        self.dims = dims

        if dir is None or not os.path.isfile(os.path.join(dir, 'support.npy')):
            initial_support_area = params['initial_support_area']
            init_support = []
            for i in range(len(initial_support_area)):
                if type(initial_support_area[0]) == int:
                    init_support.append(initial_support_area[i])
                else:
                    init_support.append(int(initial_support_area[i] * dims[i]))
            center = devlib.full(init_support, 1)
            self.support = dvut.pad_around(center, self.dims, 0)
        else:
            self.support = devlib.load(os.path.join(dir, 'support.npy'))

        # The sigma can change if resolution trigger is active. When it
        # changes the distribution has to be recalculated using the given sigma
        # At some iteration the low resolution become inactive, and the sigma
        # is set to shrink_wrap_gauss_sigma. The prev_sigma is kept to check if sigma changed
        # and thus the distribution must be updated
        self.distribution = None
        self.prev_sigma = 0

    def get_support(self):
        return self.support

    def update_amp(self, ds_image, sigma, threshold):
        self.support = dvut.shrink_wrap(ds_image, threshold, sigma)

    def update_phase(self, ds_image):
        phase = devlib.angle(ds_image)
        phase_condition = (phase > self.params['phm_phase_min']) & (phase < self.params['phm_phase_max'])
        self.support *= phase_condition


class Rec:
    def __init__(self, params, data_file):
        if 'init_guess' not in params:
            params['init_guess'] = 'random'
        elif params['init_guess'] == 'AI_guess':
            if 'AI_threshold' not in params:
                params['AI_threshold'] = params['shrink_wrap_threshold']
            if 'AI_sigma' not in params:
                params['AI_sigma'] = params['shrink_wrap_gauss_sigma']
        if 'reconstructions' not in params:
            params['reconstructions'] = 1
        if 'hio_beta' not in params:
            params['hio_beta'] = 0.9
        if 'initial_support_area' not in params:
            params['initial_support_area'] = (.5, .5, .5)
        if 'twin_trigger' in params and 'twin_halves' not in params:
            params['twin_halves'] = (0, 0)
        if 'shrink_wrap_gauss_sigma' not in params:
            params['shrink_wrap_gauss_sigma'] = 1.0
        if 'shrink_wrap_threshold' not in params:
            params['shrink_wrap_threshold'] = 0.1
        if 'shrink_wrap_trigger' in params:
            if 'shrink_wrap_type' not in params:
                params['shrink_wrap_type'] = "GAUSS"
        if 'phase_support_trigger' in params:
            if 'phm_phase_min' not in params:
                params['phm_phase_min'] = -1.57
            if 'phm_phase_max' not in params:
                params['phm_phase_max'] = 1.57
        if 'pc_interval' in params and 'pc' in params['algorithm_sequence']:
            self.is_pcdi = True
            if 'pc_type' not in params:
                params['pc_type'] = "LUCY"
            if 'pc_LUCY_iterations' not in params:
                params['pc_LUCY_iterations'] = 20
            if 'pc_normalize' not in params:
                params['pc_normalize'] = True
            if 'pc_LUCY_kernel' not in params:
                params['pc_LUCY_kernel'] = (16, 16, 16)
        else:
            self.is_pcdi = False
        self.ll_sigmas = None
        self.ll_dets = None
        if 'resolution_trigger' in params and len(params['resolution_trigger']) == 3:
            # linespacing the sigmas and dets need a number of iterations
            # The trigger should have three elements, the last one
            # meaning the last iteration when the low resolution is
            # applied. If the trigger does not have the limit,
            # it is misconfigured, and the trigger will not be
            # active
            ll_iter = params['resolution_trigger'][2]
            if 'shrink_wrap_trigger' not in params and 'lowpass_filter_sw_sigma_range' in params:
                # The sigmas are used to find support, if the shrink wrap
                # trigger is not active, it wil not be used
                sigma_range = params['lowpass_filter_sw_sigma_range']
                if len(sigma_range) > 0:
                    first_sigma = sigma_range[0]
                    if len(sigma_range) == 1:
                        last_sigma = params['shrink_wrap_gauss_sigma']
                    else:
                        last_sigma = sigma_range[1]
                    params['ll_sigmas'] = [first_sigma + x * (last_sigma - first_sigma) / ll_iter for x in range(ll_iter)]
            if 'lowpass_filter_range' in params:
                det_range = params['lowpass_filter_range']
                if type(det_range) ==int:
                    det_range = [det_range, 1.0]
                    params['ll_dets'] = [det_range[0] + x * (det_range[1] - det_range[0]) / ll_iter for x in range(ll_iter)]
            else:
                params['ll_dets'] = None
        # finished setting defaults
        self.params = params
        self.data_file = data_file
        self.ds_image = None
        self.need_save_data = False
        self.saved_data = None

    def init_dev(self, device_id):
        self.dev = device_id
        if device_id != -1:
            try:
                devlib.set_device(device_id)
            except Exception as e:
                print(e)
                print('may need to restart GUI')
                return -1
        if self.data_file.endswith('tif') or self.data_file.endswith('tiff'):
            try:
                data_np = ut.read_tif(self.data_file)
                data = devlib.from_numpy(data_np)
            except Exception as e:
                print(e)
                return -1
        elif self.data_file.endswith('npy'):
            try:
                data = devlib.load(self.data_file)
            except Exception as e:
                print(e)
                return -1
        else:
            print('no data file found')
            return -1

        # in the formatted data the max is in the center, we want it in the corner, so do fft shift
        self.data = devlib.fftshift(devlib.absolute(data))
        self.dims = devlib.dims(self.data)
        print('data shape', self.dims)

        if self.need_save_data:
            self.saved_data = devlib.copy(self.data)
            self.need_save_data = False

        return 0

    def fast_ga(self, worker_qin, worker_qout):
        if self.params['low_resolution_generations'] > 0:
            self.need_save_data = True

        functions_dict = {}
        functions_dict[self.init_dev.__name__] = self.init_dev
        functions_dict[self.init.__name__] = self.init
        functions_dict[self.breed.__name__] = self.breed
        functions_dict[self.iterate.__name__] = self.iterate
        functions_dict[self.get_metric.__name__] = self.get_metric
        functions_dict[self.save_res.__name__] = self.save_res

        done = False
        while not done:
            # cmd is a tuple containing name of the function, followed by arguments
            cmd = worker_qin.get()
            if cmd == 'done':
                done = True
            else:
                if type(cmd) == str:
                    # the function does not have any arguments
                    ret = functions_dict[cmd]()
                else:
                    # the function name is the first element in the cmd tuple, and the other elements are arguments
                    ret = functions_dict[cmd[0]](*cmd[1:])
                worker_qout.put(ret)

    def init(self, dir=None, gen=None):
        if self.ds_image is not None:
            first_run = False
        elif dir is None or not os.path.isfile(os.path.join(dir, 'image.npy')):
            self.ds_image = devlib.random(self.dims, dtype=self.data.dtype)
            first_run = True
        else:
            self.ds_image = devlib.load(os.path.join(dir, 'image.npy'))
            first_run = False
        iter_functions = [self.next,
                          self.resolution_trigger,
                          self.reset_resolution,
                          self.shrink_wrap_trigger,
                          self.phase_support_trigger,
                          self.to_reciprocal_space,
                          self.new_func_trigger,
                          self.pc_trigger,
                          self.pc_modulus,
                          self.modulus,
                          self.set_prev_pc_trigger,
                          self.to_direct_space,
                          self.er,
                          self.hio,
                          self.new_alg,
                          self.twin_trigger,
                          self.average_trigger,
                          self.progress_trigger]

        flow_items_list = []
        for f in iter_functions:
            flow_items_list.append(f.__name__)

        self.is_pc, flow = of.get_flow_arr(self.params, flow_items_list, gen, first_run)

        self.flow = []
        (op_no, iter_no) = flow.shape
        for i in range(iter_no):
            for j in range(op_no):
                if flow[j, i] == 1:
                    self.flow.append(iter_functions[j])

        self.aver = None
        self.iter = -1
        self.errs = []
        self.gen = gen
        self.prev_dir = dir
        self.sigma = self.params['shrink_wrap_gauss_sigma']
        self.support_obj = Support(self.params, self.dims, dir)
        if self.is_pc:
            self.pc_obj = Pcdi(self.params, self.data, dir)

        # for the fast GA the data needs to be saved, as it would be changed by each lr generation
        # for non-fast GA the Rec object is created in each generation with the initial data
        if self.saved_data is not None:
            if self.params['low_resolution_generations'] > self.gen:
                self.data = devlib.gaussian_filter(self.saved_data, self.params['ga_lowpass_filter_sigmas'][self.gen])
            else:
                self.data = self.saved_data
        else:
            if self.gen is not None and self.params['low_resolution_generations'] > self.gen:
                self.data = devlib.gaussian_filter(self.data, self.params['ga_lowpass_filter_sigmas'][self.gen])

        if 'll_sigma' not in self.params or not first_run:
            self.iter_data = self.data
        else:
            self.iter_data = self.data.copy()

        if (first_run):
            max_data = devlib.amax(self.data)
            self.ds_image *= get_norm(self.ds_image) * max_data

            # the line below are for testing to set the initial guess to support
            # self.ds_image = devlib.full(self.dims, 1.0) + 1j * devlib.full(self.dims, 1.0)

            self.ds_image *= self.support_obj.get_support()
        return 0

    def breed(self):
        breed_mode = self.params['ga_breed_modes'][self.gen]
        if breed_mode != 'none':
            self.ds_image = dvut.breed(breed_mode, self.prev_dir, self.ds_image)
            self.support_obj.params = dvut.shrink_wrap(self.ds_image, self.params['ga_shrink_wrap_thresholds'][self.gen],
                                                       self.params['ga_shrink_wrap_gauss_sigmas'][self.gen])
        return 0

    def iterate(self):
        start_t = time.time()
        for f in self.flow:
            f()

        print('iterate took ', (time.time() - start_t), ' sec')

        if devlib.hasnan(self.ds_image):
            print('reconstruction resulted in NaN')
            return -1

        if self.aver is not None:
            ratio = self.get_ratio(devlib.from_numpy(self.aver), devlib.absolute(self.ds_image))
            self.ds_image *= ratio / self.aver_iter

        mx = devlib.amax(devlib.absolute(self.ds_image))
        self.ds_image = self.ds_image / mx

        # it needs to return metric for the current generation, and can default to 'chi'.
        # for now returning 'chi'
        return 0

    def save_res(self, save_dir):
        from array import array

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        devlib.save(os.path.join(save_dir, 'image'), self.ds_image)
        devlib.save(os.path.join(save_dir, 'support'), self.support_obj.get_support())
        if self.is_pc:
            devlib.save(os.path.join(save_dir, 'coherence'), self.pc_obj.kernel)
        errs = array('f', self.errs)
        devlib.save(os.path.join(save_dir, 'errors'), errs)
        return 0

    def get_metric(self, metric_type):
        """
        Callculates array characteristic based on various formulas.

        Parameters
        ----------
        metric_type : str
            type of metric to apply

        Returns
        -------
        metric : float
            calculated metric
        """
        return dvut.get_metric(self.ds_image, self.errs, metric_type)

    def save_metrics(errs, dir, metrics=None):
        """
        Saves arrays metrics and errors by iterations in text file.

        Parameters
        ----------
        errs : list
            list of "chi" error by iteration

        dir : str
            directory to write the file containing array metrics

        metrics : dict
            dictionary with metric type keys, and metric values

        Returns
        -------
        nothing
        """
        metric_file = os.path.join(dir, 'summary')
        if os.path.isfile(metric_file):
            os.remove(metric_file)
        with open(metric_file, 'a') as f:
            if metrics is not None:
                f.write('metric     result\n')
                for key in metrics:
                    value = metrics[key]
                    f.write(key + ' = ' + str(value) + '\n')
            f.write('\nerrors by iteration\n')
            for er in errs:
                f.write(str(er) + ' ')
        f.close()

    def next(self):
        self.iter = self.iter + 1


    def resolution_trigger(self):
        if 'll_dets' in self.params:
            sigmas = [dim * self.params['ll_dets'][self.iter] for dim in self.dims]
            distribution = devlib.gaussian(self.dims, sigmas)
            max_el = devlib.amax(distribution)
            distribution = distribution / max_el
            data_shifted = devlib.fftshift(self.data)
            masked = data_shifted * distribution
            self.iter_data = devlib.fftshift(masked)

        if 'll_sigmas' in self.params:
            self.sigma = self.params['ll_sigmas'][self.iter]

    def reset_resolution(self):
        self.iter_data = self.data
        self.sigma = self.params['shrink_wrap_gauss_sigma']

    def shrink_wrap_trigger(self):
        self.support_obj.update_amp(self.ds_image, self.sigma, self.params['shrink_wrap_threshold'])

    def phase_support_trigger(self):
        self.support_obj.update_phase(self.ds_image)

    def to_reciprocal_space(self):
        self.rs_amplitudes = devlib.ifft(self.ds_image) * devlib.size(self.data)

    def new_func_trigger(self):
        print('in new_func_trigger, new_param', self.params['new_param'])

    def pc_trigger(self):
        self.pc_obj.update_partial_coherence(devlib.absolute(self.rs_amplitudes))

    def pc_modulus(self):
        abs_amplitudes = devlib.absolute(self.rs_amplitudes).copy()
        converged = self.pc_obj.apply_partial_coherence(abs_amplitudes)
        ratio = self.get_ratio(self.iter_data, devlib.absolute(converged))
        error = get_norm(
            devlib.where((converged > 0), (devlib.absolute(converged) - self.iter_data), 0)) / get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def modulus(self):
        ratio = self.get_ratio(self.iter_data, devlib.absolute(self.rs_amplitudes))
        error = get_norm(devlib.where((self.rs_amplitudes > 0), (devlib.absolute(self.rs_amplitudes) - self.iter_data),
                                      0)) / get_norm(self.iter_data)
        self.errs.append(error)
        self.rs_amplitudes *= ratio

    def set_prev_pc_trigger(self):
        self.pc_obj.set_previous(devlib.absolute(self.rs_amplitudes))

    def to_direct_space(self):
        self.ds_image_raw = devlib.fft(self.rs_amplitudes) / devlib.size(self.data)

    def er(self):
        self.ds_image = self.ds_image_raw * self.support_obj.get_support()

    def hio(self):
        combined_image = self.ds_image - self.ds_image_raw * self.params['hio_beta']
        support = self.support_obj.get_support()
        self.ds_image = devlib.where((support > 0), self.ds_image_raw, combined_image)

    def new_alg(self):
        self.ds_image = self.ds_image_raw * self.support_obj.get_support()

    def twin_trigger(self):
        # TODO this will work only for 3D array, but will the twin be used for 1D or 2D?
        com = devlib.center_of_mass(self.ds_image)
        sft = [int(self.dims[i] / 2 - com[i]) for i in range(len(self.dims))]
        self.ds_image = devlib.shift(self.ds_image, sft)
        dims = self.ds_image.shape
        half_x = int((dims[0] + 1) / 2)
        half_y = int((dims[1] + 1) / 2)
        if self.params['twin_halves'][0] == 0:
            self.ds_image[half_x:, :, :] = 0
        else:
            self.ds_image[: half_x, :, :] = 0
        if self.params['twin_halves'][1] == 0:
            self.ds_image[:, half_y:, :] = 0
        else:
            self.ds_image[:, : half_y, :] = 0

    def average_trigger(self):
        if self.aver is None:
            self.aver = devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter = 1
        else:
            self.aver = self.aver + devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter += 1

    def progress_trigger(self):
        print('------iter', self.iter, '   error ', self.errs[-1])

    def get_ratio(self, divident, divisor):
        divisor = devlib.where((divisor != 0.0), divisor, 1.0)
        ratio = divident / divisor
        return ratio

def reconstruction(datafile, **kwargs):
    """
    Reconstructs the image in datafile according to given parameters. The results: image.npy, support.npy, and
    errors.npy are saved in 'saved_dir' defined in kwargs, or if not defined, in the directory of datafile.

    Parameters
    ----------
    datafile : str
        filename of phasing data. Should be either .tif format or .npy

    kwargs : keyword arguments, option listed below
        save_dir  # directory where results of reconstruction are saved as npy files. If not present, the
                  # reconstruction outcome will be save in the same directory where datafile is.
        processing  # the library used when running reconstruction. When the 'auto' option is selected the
                    # program will use the best performing library that is available, in the following order:
                    # cupy, numpy. The 'cp' option will utilize cupy, and 'np' will utilize numpy. Default is auto.
        device  # IDs of the target devices. If not defined, it will default to -1 for the OS to select device.
        algorithm_sequence  # Mandatory, example: "3* (20*ER + 180*HIO) + 20*ER"
                            # defines algorithm applied in each iteration during modulus projection and
                            # during modulus. The "*" character means repeat, and the "+" means add to the sequence.
                            # The sequence may contain single brackets defining a group that will be repeated by the
                            # preceding multiplier. The alphabetic entries: 'ER', 'ERpc', 'HIO', 'HIOpc' define algorithms
                            # used in this iteration. The entries will invoke functions as follows:
                            # 'ER' definition will invoke 'er' and 'modulus' functions
                            # 'ERpc' will invoke 'er' and 'pc_modulus'
                            # 'HIO' will invoke 'hio' and 'modulus'
                            # 'HIOpc' will invoke 'hio' and 'pc_modulus'.
                            # The pc_modulus is implementation of modulus with partial coherence correction.
                            # If defining ERpc or HIOpc the pcdi feature must be activated. If not activated,
                            # the phasing will use modulus function instead.
        hio_beta  # used in hio algorithm
        twin_trigger # example: [2]. Defines at which iteration to cut half of the array(i.e. multiply by 0s),
        twin_halves  # defines which half of the array is zeroed out in x and y dimensions.
                     # If 0, the first half in that dimension is zeroed out, otherwise, the second half.
        shrink_wrap_trigger  # example: [1, 1]. Defines when to update support array using the parameters below.
        shrink_wrap_type  # supporting "GAUSS" only. Defines which algorithm to use for shrink wrap.
        shrink_wrap_threshold  # only point with relative intensity greater than the threshold are selected
        shrink_wrap_gauss_sigma  # used to calculate the Gaussian filter
        initial_support_area  # If the values are fractional, the support area will be calculated by multiplying
                              # by the data array dimensions. The support will be set to 1s to this dimensions centered.
        phase_support_trigger  # defines when to update support array using the parameters below by applaying phase
                               # constrain.
        phm_phase_min  # point with phase below this value will be removed from support area
        phm_phase_max  # point with phase over this value will be removed from support area
        pc_interval  # defines iteration interval to update coherence.
        pc_type  # partial coherence algorithm. 'LUCY' type is supported.
        pc_LUCY_iterations  # number of iterations used in Lucy algorithm
        pc_LUCY_kernel  # coherence kernel area.
        resolution_trigger  # defines when to apply low resolution filter using the parameters below.
        lowpass_filter_sw_sigma_range  # used when applying low resolution to replace support sigma.
                                       # The sigmas are linespaced for low resolution iterations from first value
                                       # to last. If only one number given, the last sigma will default to
                                       # shrink_wrap_gauss_sigma.
        lowpass_filter_range  # used when applying low resolution data filter while iterating.
                              # The det values are linespaced for low resolution iterations from first value to last.
                              # The filter is gauss with sigma of linespaced det. If only one number given,
                              # the last det will default to 1.
        average_trigger  # defines when to apply averaging. Negative start means it is offset from the last iteration.
        progress_trigger  # defines when to print info on the console. The info includes current iteration and error.

    example of the simplest kwargs parameters:
        algorithm_sequence='1*(20*ER+180*HIO)+20*ER',
        shrink_wrap_trigger = [1, 1],
        twin_trigger = [2],
        progress_trigger = [0, 20]

    Returns
    -------
    nothing
    """
    error_msg = ver.verify('config_rec', kwargs)
    if len(error_msg) > 0:
        print(error_msg)
        return
    if not os.path.isfile(datafile):
        print('no file found', datafile)
        return

    if 'processing' in kwargs:
        pkg = kwargs['processing']
    else:
        pkg = 'auto'
    if pkg == 'auto':
        try:
            import cupy as cp
            pkg = 'cp'
        except:
            pkg = 'np'
    if pkg == 'cp':
        devlib = importlib.import_module('cohere.lib.cplib').cplib
    elif pkg == 'np':
        print('np')
        devlib = importlib.import_module('cohere.lib.nplib').nplib
    else:
        print('supporting cp and np processing')
        return
    set_lib(devlib, False)

    worker = Rec(kwargs, datafile)
    if 'device' in kwargs:
        device = kwargs['device'][0]
    else:
        device = [-1]
    if worker.init_dev(device[0]) < 0:
        return

    if 'continue_dir' in kwargs:
        continue_dir = kwargs['continue_dir']
    else:
        continue_dir = None
    worker.init(continue_dir)
    ret_code = worker.iterate()
    if 'save_dir' in kwargs:
        save_dir = kwargs['save_dir']
    else:
        save_dir, filename = os.path.split(datafile)
    if ret_code == 0:
        worker.save_res(save_dir)