# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

import cohere_core.utilities.utils as ut
import cohere_core.utilities.dvc_utils as dvut
from abc import ABC, abstractmethod

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['Pcdi',
           'TriggeredOp',
           'ShrinkWrap',
           'PhaseConstrain',
           'LowPassFilter']


def set_lib(dlib):
    global devlib
    devlib = dlib


class Pcdi:
    def __init__(self, params, data, dir=None):
        self.type = params.get('pc_type', 'LUCY')
        if 'pc_LUCY_iterations' in params:
            self.iterations = params['pc_LUCY_iterations']
        else:
            msg = 'pc_LUCY_iterations parameter not defined'
            raise ValueError(msg)
        self.normalize = params.get('pc_normalize', True)
        if 'pc_LUCY_kernel' in params:
            self.kernel_area = params['pc_LUCY_kernel']
        else:
            msg = 'pc_LUCY_kernel parameter not defined'
            raise ValueError(msg)
        if dir is None:
            self.kernel = None
        else:
            try:
                self.kernel = devlib.load(ut.join(dir, 'coherence.npy'))
            except:
                self.kernel = None

        self.dims = devlib.dims(data)
        self.roi_data = ut.crop_center(devlib.fftshift(data), self.kernel_area)
        if self.normalize:
            self.sum_roi_data = devlib.sum(devlib.square(self.roi_data))
        if self.kernel is None:
            self.kernel = devlib.full(self.kernel_area, 0.5, dtype=devlib.dtype(data))

    def set_previous(self, abs_amplitudes):
        self.roi_amplitudes_prev = ut.crop_center(devlib.fftshift(abs_amplitudes), self.kernel_area)

    def apply_partial_coherence(self, abs_amplitudes):
        abs_amplitudes_2 = devlib.square(abs_amplitudes)
        converged_2 = devlib.fftconvolve(abs_amplitudes_2, self.kernel)
        converged = devlib.sqrt(converged_2)
        return converged

    def update_partial_coherence(self, abs_amplitudes):
        roi_amplitudes = ut.crop_center(devlib.fftshift(abs_amplitudes), self.kernel_area)
        roi_combined_amp = 2 * roi_amplitudes - self.roi_amplitudes_prev
        if self.normalize:
            amplitudes_2 = devlib.square(roi_combined_amp)
            sum_ampl = devlib.sum(amplitudes_2)
            ratio = self.sum_roi_data / sum_ampl
            amplitudes = devlib.sqrt(amplitudes_2 * ratio)
        else:
            amplitudes = roi_combined_amp

        if self.type == "LUCY":
            self.kernel = dvut.lucy_deconvolution(devlib.square(amplitudes), devlib.square(self.roi_data),
                                    self.kernel, self.iterations)


class LowPassFilter():
    def __init__(self, params):
        iter_range = [params['lowpass_filter_trigger'][0], params['lowpass_filter_trigger'][2]]
        iter_diff = params['lowpass_filter_trigger'][2] - params['lowpass_filter_trigger'][0]
        if len(params['lowpass_filter_range']) == 1:
            end_sigma = 1.0
        else:
            end_sigma = params['lowpass_filter_range'][1]
        sigma_diff = end_sigma - params['lowpass_filter_range'][0]
        self.filter_sigmas = [sigma_diff / iter_diff * iter + params['lowpass_filter_range'][0]
                              for iter in range(iter_range[0], iter_range[1])]


    def apply_trigger(self, *args):
        # The filter is applied on data
        data = args[0]
        iter = args[1]

        filter_sigma = self.filter_sigmas[iter]
        return devlib.gaussian_filter(data, filter_sigma)


class TriggeredOp(ABC):
    """
    Base class for Triggered operation. This class creates feature objects and manages the trigger or subtrigger depending on
    configuration.
    """
    def __init__(self, trig_op_name):
        # the self.objs will hold either one concrete feature object if configuration is for general trigger,
        # i.e. spanning across all iterations,
        # or it will hold a list of objects, each being a concrete sub-feature, appearing in order that will be
        # executed during each trigger
        self.objs = None

        # the self.f points to either self.apply_trigger_obj function if configuration is for general trigger,
        # or to self.apply_trigger_seq if sub-triggers are configured
        self.f = None

        self.trig_op_name = trig_op_name  # ex: 'shrink_wrap_trigger'


    def apply_trigger_obj(self, *args):
        """
        This function is called when general trigger is used. In this case there is only one instance of the concrete
        feature. The trigger is then executed on this instance.
        :param args: variable parameters depending on the concrete feature
        :return: depends on the concrete feature
        """
        return self.objs.apply_trigger(*args)

    def apply_trigger_seq(self, *args):
        """
        This function is called when sub-triggers are used. the self.objs is a list for this case. Every time a trigger
        is applied the first object is removed from this list and the trigger is executed on this instance.
        :param args: variable parameters depending on the concrete feature
        :return: depends on the concrete feature
        """
        sub_obj = self.objs.pop(0)
        return sub_obj.apply_trigger(*args)

    def apply_trigger(self, *args):
        """
        Thhis function is called by a trigger. It then directs the trigger by calling the self.f function.
        :param args: variable parameters depending on the concrete feature
        :return: depends on the concrete feature
        """
        # the f is either update_amp_seq function or update_amp_seq
        # The f is set in constructor depending on whether the trigger
        # was defined for the entire span of iterations or multiple
        # subtriggers were defined in algorithm sequence
        return self.f(*args)

    @abstractmethod
    def create_obj(self, *args):
        """
        This class must be overriden by subclass. It creates a feature object.
        :param args: variable parameters depending on the concrete feature
        :return: an instance of the subclass
        """
        pass

    def create_objs(self, params, sub_rows_trigs):
        """
        The params map contains value of the feature trigger. If the trigger is configured as a general one,
        i.e. a single trigger, the self.objs is set to the created instance of the feature, and the self.f
        pointer is set accordingly, i.e. to the self.apply_trigger_obj function.
        If the trigger contains several sub-triggers, the sub_row_trigs contain info about the sequence the
        subtrigger should be executed. Based on this info the unique instances are created and are added to the
        in the order the triggers should be called. The self.f function is set to the self.apply_trigger_seq
        function.
        :param sub_rows_trigs: sub-triggers info if sub-triggers are configured
        :param params: configuration parameters
        :return:
        """
        trigger_name = f'{self.trig_op_name}_trigger'

        if trigger_name in sub_rows_trigs.keys():
            row = sub_rows_trigs[trigger_name][0]
            sub_trigs = sub_rows_trigs[trigger_name][1]
            sub_objs = {}
            for sub_t in sub_trigs:
                (beg, end, idx) = sub_t
                index = int(idx)
                if index not in sub_objs.keys():
                    sub_objs[index] = self.create_obj(params, index=index, beg=beg, end=end)
            trigs = [i-1 for i in row.tolist() if i != 0]
            # the operation of creating object might fail
            # if conditions are not met, ex: the sw_type is not supported
            # This should be already verified by a verifier, so no checking here
            self.objs = [sub_objs[idx] for idx in trigs]
            self.f = self.apply_trigger_seq
        else:
            if trigger_name in params:
                self.objs = self.create_obj(params)
                self.f = self.apply_trigger_obj


class ShrinkWrap(TriggeredOp):
    def __init__(self, trig_op):
        super().__init__(trig_op)

    class GaussSW:
        def __init__(self, gauss_sigma, threshold):
            self.gauss_sigma = gauss_sigma
            self.threshold = threshold

        def apply_trigger(self, *args):
            ds_image = args[0]
            return dvut.shrink_wrap(ds_image, self.threshold, self.gauss_sigma)


    class Gauss1SW:
        def __init__(self, gauss_sigma, threshold):
            self.gauss_sigma = gauss_sigma
            self.threshold = threshold

        def apply_trigger(self, *args):
            ds_image = args[0]
            return dvut.shrink_wrap(ds_image, self.threshold + .01, self.gauss_sigma)


    def create_obj(self, params, index=None, beg=None, end=None):
        def check_Gauss_type():
            # for now cohere supports only Gauss type, so the following parameters are mandatory
            if 'shrink_wrap_gauss_sigma' not in params:
                msg = 'shrink_wrap_gauss_sigma parameter not defined'
                raise ValueError(msg)
            if 'shrink_wrap_threshold' not in params:
                msg = 'shrink_wrap_threshold parameter not defined'
                raise ValueError(msg)

        def check_Gauss1_type():
            # for now cohere supports only Gauss type, so the following parameters are mandatory
            if 'shrink_wrap_gauss_sigma' not in params:
                msg = 'shrink_wrap_gauss_sigma parameter not defined'
                raise ValueError(msg)
            if 'shrink_wrap_threshold' not in params:
                msg = 'shrink_wrap_threshold parameter not defined'
                raise ValueError(msg)

        if 'shrink_wrap_type' not in params:
            msg = 'shrink_wrap_type parameter not defined'
            raise ValueError(msg)

        if index is None:
            sw_type = params['shrink_wrap_type']
            if sw_type == 'GAUSS':
                check_Gauss_type()
                sigma = params['shrink_wrap_gauss_sigma']
                threshold = params['shrink_wrap_threshold']
                return self.GaussSW(sigma, threshold)
            elif sw_type == 'GAUSS1':
                check_Gauss1_type()
                sigma = params['shrink_wrap_gauss_sigma']
                threshold = params['shrink_wrap_threshold']
                return self.Gauss1SW(sigma, threshold)
            else:
                msg = f'{sw_type} shrink wrap type is not supported'
                raise ValueError(msg)
        else:
            if len(params['shrink_wrap_type']) - 1 < index:
                msg = f'shrink_wrap_type not defined for sub-trigger {index}'
                raise ValueError(msg)
            sw_type = params['shrink_wrap_type'][index]
            if sw_type == 'GAUSS':
                check_Gauss_type()
                if len(params['shrink_wrap_gauss_sigma']) - 1 < index:
                    msg = f'shrink_wrap_gauss_sigma not defined for sub-trigger {index}'
                    raise ValueError(msg)
                sigma = params['shrink_wrap_gauss_sigma'][index]
                if len(params['shrink_wrap_threshold']) - 1 < index:
                    msg = f'shrink_wrap_threshold not defined for sub-trigger {index}'
                    raise ValueError(msg)
                threshold = params['shrink_wrap_threshold'][index]
                return self.GaussSW(sigma, threshold)
            elif sw_type == 'GAUSS1':
                check_Gauss1_type()
                if len(params['shrink_wrap_gauss_sigma']) - 1 < index:
                    msg = f'shrink_wrap_gauss_sigma not defined for sub-trigger {index}'
                    raise ValueError(msg)
                sigma = params['shrink_wrap_gauss_sigma'][index]
                if len(params['shrink_wrap_threshold']) - 1 < index:
                    msg = f'shrink_wrap_threshold not defined for sub-trigger {index}'
                    raise ValueError(msg)
                threshold = params['shrink_wrap_threshold'][index]
                return self.Gauss1SW(sigma, threshold)
            else:
                msg = f'{sw_type} shrink wrap type is not supported'
                raise ValueError(msg)


class PhaseConstrain(TriggeredOp):
    def __init__(self, trig_op):
        super().__init__(trig_op)

    class PhasePHC:
        def __init__(self, phc_phase_min, phc_phase_max):
            self.phc_phase_min = phc_phase_min
            self.phc_phase_max = phc_phase_max

        def apply_trigger(self, *args):
            ds_image = args[0]
            phase = devlib.angle(ds_image)
            return (phase > self.phc_phase_min) & (phase < self.phc_phase_max)


    def create_obj(self, params, index=None, beg=None, end=None):
        if 'phc_phase_min' not in params:
            msg = 'phc_phase_min parameter not defined'
            raise ValueError(msg)
        if 'phc_phase_max' not in params:
            msg = 'phc_phase_max parameter not defined'
            raise ValueError(msg)
        if index is None:
            phase_min = params['phc_phase_min']
            phase_max = params['phc_phase_max']
        else:
            if len(params['phc_phase_min']) - 1 < index:
                msg = f'phc_phase_min not defined for sub-trigger {index}'
                raise ValueError(msg)
            phase_min = params['phc_phase_min'][index]
            if len(params['phc_phase_max']) - 1 < index:
                msg = f'phc_phase_max not defined for sub-trigger {index}'
                raise ValueError(msg)
            phase_max = params['phc_phase_max'][index]
        return self.PhasePHC(phase_min, phase_max)


def create(trig_op, params, trig_op_info):
    if trig_op == 'shrink_wrap':
        to = ShrinkWrap(trig_op)
    if trig_op == 'phc':
        to = PhaseConstrain(trig_op)

    # this function sets self.objs and self.f and creates all objects
    # It may throw exception
    to.create_objs(params, trig_op_info)
    return to




