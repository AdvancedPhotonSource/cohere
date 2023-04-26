import os
import cohere_core.utilities.dvc_utils as dvut

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Pcdi',
           'Support',
           'LowPassFilter']


def set_lib(dlib):
    global devlib
    devlib = dlib
    dvut.set_lib(devlib)


class Pcdi:
    def __init__(self, params, data, dir=None):
        if 'pc_type' in params:
            self.type = params['pc_type']
        else:
            self.type = "LUCY"
        if 'pc_LUCY_iterations' in params:
            self.iterations = params['pc_LUCY_iterations']
        else:
            self.iterations = 20
        if 'pc_normalize' in params:
            self.normalize = params['pc_normalize']
        else:
            self.normalize = True
        if 'pc_LUCY_kernel' in params:
            self.kernel_area = params['pc_LUCY_kernel']
        else:
            self.kernel_area = (16, 16, 16)
        if dir is None:
            self.kernel = None
        else:
            try:
                self.kernel = devlib.load(dir + '/coherence.npy')
            except:
                self.kernel = None

        self.dims = devlib.dims(data)
        self.roi_data = dvut.crop_center(devlib.fftshift(data), self.kernel_area)
        if self.normalize:
            self.sum_roi_data = devlib.sum(devlib.square(self.roi_data))
        if self.kernel is None:
            self.kernel = devlib.full(self.kernel_area, 0.5, dtype=devlib.dtype(data))

    def set_previous(self, abs_amplitudes):
        self.roi_amplitudes_prev = dvut.crop_center(devlib.fftshift(abs_amplitudes), self.kernel_area)

    def apply_partial_coherence(self, abs_amplitudes):
        abs_amplitudes_2 = devlib.square(abs_amplitudes)
        converged_2 = devlib.fftconvolve(abs_amplitudes_2, self.kernel)
        # converged_2 = devlib.where(converged_2 < 0.0, 0.0, converged_2)
        converged = devlib.sqrt(converged_2)
        return converged

    def update_partial_coherence(self, abs_amplitudes):
        roi_amplitudes = dvut.crop_center(devlib.fftshift(abs_amplitudes), self.kernel_area)
        roi_combined_amp = 2 * roi_amplitudes - self.roi_amplitudes_prev
        if self.normalize:
            amplitudes_2 = devlib.square(roi_combined_amp)
            sum_ampl = devlib.sum(amplitudes_2)
            ratio = self.sum_roi_data / sum_ampl
            amplitudes = devlib.sqrt(amplitudes_2 * ratio)
        else:
            amplitudes = roi_combined_amp

        if self.type == "LUCY":
            self.lucy_deconvolution(devlib.square(amplitudes), devlib.square(self.roi_data),
                                    self.iterations)

    def lucy_deconvolution(self, amplitudes, data, iterations):
        data_mirror = devlib.flip(data)
        for i in range(iterations):
            conv = devlib.fftconvolve(self.kernel, data)
            conv = devlib.where(conv == 0.0, 1.0, conv)
            relative_blurr = amplitudes / conv
            self.kernel = self.kernel * devlib.fftconvolve(relative_blurr, data_mirror)
        self.kernel = devlib.real(self.kernel)
        coh_sum = devlib.sum(devlib.absolute(self.kernel))
        self.kernel = devlib.absolute(self.kernel) / coh_sum


class Support:
    def __init__(self, params, dims, sub_rows_trigs, dir=None):
        # create or get initial support
        if dir is None or not os.path.isfile(dir + '/support.npy'):
            initial_support_area = params['initial_support_area']
            init_support = []
            for i in range(len(initial_support_area)):
                if type(initial_support_area[0]) == int:
                    init_support.append(initial_support_area[i])
                else:
                    init_support.append(int(initial_support_area[i] * dims[i]))
            center = devlib.full(init_support, 1)
            self.support = dvut.pad_around(center, dims, 0)
        else:
            self.support = devlib.load(dir + '/support.npy')

        self.objs = {}
        self.f = {}
        self.create_objs(sub_rows_trigs, params)

    class GaussSW:
        def __init__(self, gauss_sigma, threshold):
            self.gauss_sigma = gauss_sigma
            self.threshold = threshold

        def update_amp(self, ds_image):
#            print('updating amp, sigma', self.gauss_sigma)
            return dvut.shrink_wrap(ds_image, self.threshold, self.gauss_sigma)

    class PhasePHM:
        def __init__(self, phm_phase_min, phm_phase_max):
            self.phm_phase_min = phm_phase_min
            self.phm_phase_max = phm_phase_max

        def update_phase(self, ds_image):
            phase = devlib.angle(ds_image)
#            print('in update_phase, ph_min', self.phm_phase_min)
            return (phase > self.phm_phase_min) & (phase < self.phm_phase_max)

    class GaussLPF:
        def __init__(self, sigma_range, iter_range, threshold):
            self.threshold = threshold
            self.iter_offset = iter_range[0]
            self.sigmas = [(sigma_range[1] - sigma_range[0]) / (iter_range[1] - iter_range[0]) *
                           (iter - self.iter_offset) + sigma_range[0]
                           for iter in range(iter_range[0], iter_range[1])]

        def apply_low_filter(self, ds_image, iter):
            sigma = self.sigmas[iter - self.iter_offset]
#            print('in apply_low_filter, sigma', sigma)
            return dvut.shrink_wrap(ds_image, self.threshold, sigma)

    def create_objs(self, sub_rows_trigs, params):
        def create_sw_obj(params, index=None, beg=None, end=None):
            if index is None:
                if 'shrink_wrap_type' not in params:
                    sw_type = 'GAUSS'
                else:
                    sw_type = params['shrink_wrap_type']
                if sw_type == 'GAUSS':
                    if 'shrink_wrap_gauss_sigma' not in params:
                        sigma = 1.0
                    else:
                        sigma = params['shrink_wrap_gauss_sigma']
                    if 'shrink_wrap_threshold' not in params:
                        threshold = 0.1
                    else:
                        threshold = params['shrink_wrap_threshold']
            else:
                if 'shrink_wrap_type' not in params or len(params['shrink_wrap_type']) - 1 < index:
                    sw_type = 'GAUSS'
                else:
                    sw_type = params['shrink_wrap_type'][index]
                if sw_type == 'GAUSS':
                    if 'shrink_wrap_gauss_sigma' not in params or len(params['shrink_wrap_gauss_sigma']) - 1 < index:
                        sigma = 1.0
                    else:
                        sigma = params['shrink_wrap_gauss_sigma'][index]
                    if 'shrink_wrap_threshold' not in params or len(params['shrink_wrap_threshold']) - 1 < index:
                        threshold = 0.1
                    else:
                        threshold = params['shrink_wrap_threshold'][index]
            return self.GaussSW(sigma, threshold)

        def create_phm_obj(params, index=None, beg=None, end=None):
            if index is None:
                if 'phm_phase_min' not in params:
                    phase_min = -1.57
                else:
                    phase_min = params['phm_phase_min']
                if 'phm_phase_max' not in params:
                    phase_max = 1.57
                else:
                    phase_max = params['phm_phase_max']
            else:
                if 'phm_phase_min' not in params or len(params['phm_phase_min']) - 1 < index:
                    phase_min = -1.57
                else:
                    phase_min = params['phm_phase_min'][index]
                if 'phm_phase_max' not in params or len(params['phm_phase_max']) - 1 < index:
                    phase_max = 1.57
                else:
                    phase_max = params['phm_phase_max'][index]
            return self.PhasePHM(phase_min, phase_max)

        def create_lpf_obj(params, index=None, beg=None, end=None):
            # in case of a all_iteration configuration the lowpass filter
            # iteration can be read from lowpass_filter_trigger
            if index is None:
                iter_range = (params['lowpass_filter_trigger'][0], params['lowpass_filter_trigger'][2])
                filter_range = params['lowpass_filter_range']
                if 'lowpass_filter_threshold' in params:
                    threshold = params['lowpass_filter_threshold']
                else:
                    threshold = .1
            # otherwise the iterations to apply this instance are given as arguments
            else:
                iter_range = (beg, end)
                filter_range = params['lowpass_filter_range'][index]
                if 'lowpass_filter_threshold' not in params or len(params['lowpass_filter_threshold']) - 1 < index:
                    threshold = .1
                else:
                    threshold = params['lowpass_filter_threshold'][index]

            if len(filter_range) == 1:
                filter_range.append(1.0)
            sigma_range = [1 / x for x in filter_range]
            return self.GaussLPF(sigma_range, iter_range, threshold)

        feats = {'SW':(self.update_amp_obj, self.update_amp_seq, create_sw_obj),
                 'PHM':(self.update_ph_obj, self.update_ph_seq, create_phm_obj),
                 'LPF':(self.update_lpf_obj, self.update_lpf_seq, create_lpf_obj)}

        trig_param = {'SW' : 'shrink_wrap_trigger',
                      'PHM' : 'phm_trigger',
                      'LPF' : 'lowpass_filter_trigger'}

        for key, funs in feats.items():
            if key in sub_rows_trigs.keys():
                row = sub_rows_trigs[key][0]
                sub_trigs = sub_rows_trigs[key][1]
                sub_objs = {}
                for sub_t in sub_trigs:
                    (beg, end, idx) = sub_t
                    index = int(idx)
                    sub_objs[index] = funs[2](params, index, beg, end)
                trigs = [i-1 for i in row.tolist() if i != 0]
                # the operation of creating object might fail
                # if conditions are not met, ex: the sw_type is not supported
                # This should be already verified by a verifier, so no checking here
                self.objs[key] = [sub_objs[idx] for idx in trigs]
                self.f[key] = funs[1]
            else:
                if trig_param[key] in params:
                    self.objs[key] = funs[2](params)
                    self.f[key] = funs[0]

    def get_support(self):
        return self.support

    def update_amp_obj(self, ds_image):
        self.support = self.objs['SW'].update_amp(ds_image)

    def update_amp_seq(self, ds_image):
        sub_obj = self.objs['SW'].pop(0)
        self.support = sub_obj.update_amp(ds_image)

    def update_amp(self, ds_image):
        # the f is either update_amp_seq function or update_amp_seq
        # The f is set in constructor depending on whether the trigger
        # was defined for the entire span of iterations or multiple
        # subtriggers were defined in algorithm sequence
        self.f['SW'](ds_image)

    def update_ph_obj(self, ds_image):
        self.support *= self.objs['PHM'].update_phase(ds_image)

    def update_ph_seq(self, ds_image):
        sub_obj = self.objs['PHM'].pop(0)
        self.support *= sub_obj.update_phase(ds_image)

    def update_phase(self, ds_image):
        # the f is either update_ph_seq function or update_ph_seq
        # The f is set in constructor depending on whether the trigger
        # was defined for the entire span of iterations or multiple
        # subtriggers were defined in algorithm sequence
        self.f['PHM'](ds_image)

    def update_lpf_obj(self, ds_image, iter):
        self.support = self.objs['LPF'].apply_lowpass_filter(ds_image, iter)

    def update_lpf_seq(self, ds_image, iter):
        sub_obj = self.objs['LPF'].pop(0)
        self.support = sub_obj.apply_lowpass_filter(ds_image, iter)

    def apply_lowpass_filter(self, ds_image, iter):
        # the f is either update_amp_seq function or update_amp_seq
        # The f is set in constructor depending on whether the trigger
        # was defined for the entire span of iterations or multiple
        # subtriggers were defined in algorithm sequence
        self.f['LPF'](ds_image, iter)

    def flip(self):
        self.support = devlib.flip(self.support)


class LowPassFilter:
    def __init__(self, params, sub_rows_trigs):
        self.objs = {}
        self.f = {}
        self.create_objs(sub_rows_trigs, params)

    class Filter:
        def __init__(self, filter_range, iter_range):
            self.iter_offset = iter_range[0]
            self.filter_sigmas = [(filter_range[1] - filter_range[0]) / (iter_range[1] - iter_range[0]) *
                                  (iter - self.iter_offset) + filter_range[0]
                                  for iter in range(iter_range[0], iter_range[1])]

        def apply_lowpass_filter(self, data, iter):
            filter_sigma = self.filter_sigmas[iter - self.iter_offset]
#            print('in apply_low_filter, filter_sigma', filter_sigma)
            return devlib.gaussian_filter(data, filter_sigma)


    def create_objs(self, sub_rows_trigs, params):
        def create_lpf_obj(params, index=None, beg=None, end=None):
            # in case of a all_iteration configuration the lowpass filter
            # iteration can be read from lowpass_filter_trigger
            if index is None:
                iter_range = (params['lowpass_filter_trigger'][0], params['lowpass_filter_trigger'][2])
                filter_range = params['lowpass_filter_range']
            # otherwise the iterations to apply this instance are given as arguments
            else:
                iter_range = (beg, end)
                filter_range = params['lowpass_filter_range'][index]
            if len(filter_range) == 1:
                filter_range.append(1.0)
            return self.Filter(filter_range, iter_range)

        feats = { 'LPF':(self.update_lpf_obj, self.update_lpf_seq, create_lpf_obj)}
        trig_param = {'LPF' : 'lowpass_filter_trigger'}

        for key, funs in feats.items():
            if key in sub_rows_trigs.keys():
                row = sub_rows_trigs[key][0]
                sub_trigs = sub_rows_trigs[key][1]
                sub_objs = {}
                for sub_t in sub_trigs:
                    (beg, end, idx) = sub_t
                    index = int(idx)
                    sub_objs[index] = funs[2](params, index, beg, end)
                trigs = [i-1 for i in row.tolist() if i != 0]
                # the operation of creating object might fail
                # if conditions are not met, ex: the sw_type is not supported
                # This should be already verified by a verifier, so no checking here
                self.objs[key] = [sub_objs[idx] for idx in trigs]
                self.f[key] = funs[1]
            else:
                if trig_param[key] in params:
                    self.objs[key] = funs[2](params)
                    self.f[key] = funs[0]

    def update_lpf_obj(self, data, iter):
        return self.objs['LPF'].apply_lowpass_filter(data, iter)

    def update_lpf_seq(self, data, iter):
        sub_obj = self.objs['LPF'].pop(0)
        return sub_obj.apply_lowpass_filter(data, iter)

    def apply_lowpass_filter(self, data, iter):
        # the f is either update_amp_seq function or update_amp_seq
        # The f is set in constructor depending on whether the trigger
        # was defined for the entire span of iterations or multiple
        # subtriggers were defined in algorithm sequence
        return self.f['LPF'](data, iter)

