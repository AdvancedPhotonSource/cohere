# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
Provides phasing capabilities for the Bragg CDI data.

The reconstruction is based on known iterative algorithm, oscillating between direct and reciprocal space and applying projection algorithms. Cohere 
supports the following projection algorithms: ER (error reduction), HIO (hybrid input-output), SF (solvent flipping), and RAAR (relaxed averaged alternating reflectors). The algorithms are described in this publication: https://pubs.aip.org/aip/rsi/article/78/1/011301/349838/Invited-Article-A-unified-evaluation-of-iterative 

The cohere reconstruction can be tuned to specific cases. Typically every few iteration steps a shrink wrap is applied. 
And typically a twin blocking is performed after a few first iterations. Other factors differentiating reconstruction and supported by cohere are: low pass filter, phase constrain, averaging amplitudes. These are direct space methods.
Each of these features owns parameters that can be tuned to the projection algorithm. Cohere supports partial coherence feature that is performed in reciprocal space and is used to mitigate coherence of the light source. 

The software can run code on GPU or CPU, utilizing different library, such as cupy, numpy or torch. User configures the choice depending on hardware and installed software.

The reconstruction is utilized by the GA algorithm, where each generation creates multiple Rec objects, and executes parallel reconstructions. 

In addition two subclasses provide functionality for different types of reconstruction. The CoupleRec class supports multipeak reconstruction. 
The TeRec class supports reconstruction of time-evolving samples with improved temporal resolution.
"""

from pathlib import Path
import time
import os
from math import pi
import random
import tqdm

import numpy as np
from matplotlib import pyplot as plt
import tifffile as tf

import cohere_core.utilities.dvc_utils as dvut
import cohere_core.utilities.utils as ut
import cohere_core.controller.op_flow as of
import cohere_core.controller.features as ft

__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['set_lib_from_pkg',
           'create_rec',
           'Rec']


def set_lib_from_pkg(pkg):
    """
    Imports package specified in input and sets the device library to this package.

    :param pkg: supported values: 'cp' for cupy, 'np' for numpy, and 'torch' for torch
    :return:
    """
    global devlib

    # get the lib object
    devlib = ut.get_lib(pkg)
    # pass the lib object to the features associated with this reconstruction
    ft.set_lib(devlib)
    # the utilities are not associated with reconstruction and the initialization of lib is independent
    dvut.set_lib_from_pkg(pkg)


class Rec:
    """
    Performs phasing using iterative algorithm.

    The Rec object must have a device set, either to GPU id or set as CPU.
    The functions executed in each iteration are determined at the beginning, during iteration loop initialization. It is based on the reconstruction parameters and the order of functions internally defined in 'iter_functions' field. 
    A factory for this class, 'create_rec' returns initialized Rec object.
    The object is then called to run iterations. The results can be retrieved with getters or saved.
    """

    def __init__(self, params, data_file, pkg, **kwargs):
        """
        Constructor. Sets / initializes reconstruction parameters.

        :param params: dictionary containing configuration parameters
        :param data_file: path to the data file
        :param pkg: an acronym of the package that will be utilized for reconstruction: 
            "np" for numpy, "cp" for cupy, and "torch" for torch
        :param kwargs: can contain "debug" parameter. When activated, it will throw exception, if one happens. Otherwise, an error code of -1 is returned and error printed. The handling of the exception is needed in GA reconstruction.
        """
        set_lib_from_pkg(pkg)
        self.iter_functions = [self.next,
                               self.lowpass_filter_operation,
                               self.reset_resolution,
                               self.shrink_wrap_operation,
                               self.reset_phc_correction,
                               self.phc_operation,
                               self.to_reciprocal_space,
                               self.new_func_operation,
                               self.pc_operation,
                               self.pc_modulus,
                               self.modulus,
                               self.set_prev_pc,
                               self.to_direct_space,
                               self.er,
                               self.hio,
                               self.sf,
                               self.raar,
                               self.twin_operation,
                               self.average_operation,
                               self.progress_operation,
                               self.live_operation]

        params['init_guess'] = params.get('init_guess', 'random')
        if params['init_guess'] == 'AI_guess':
            if 'AI_threshold' not in params:
                params['AI_threshold'] = params['shrink_wrap_threshold']
            if 'AI_sigma' not in params:
                params['AI_sigma'] = params['shrink_wrap_gauss_sigma']
        params['reconstructions'] = params.get('reconstructions', 1)
        params['hio_beta'] = params.get('hio_beta', 0.9)
        params['raar_beta'] = params.get('raar_beta', 0.45)
        params['initial_support_area'] = params.get('initial_support_area', (.5, .5, .5))
        if 'twin_trigger' in params:
            params['twin_halves'] = params.get('twin_halves', (0, 0))
        if 'pc_interval' in params and 'pc' in params['algorithm_sequence']:
            self.is_pcdi = True
        else:
            self.is_pcdi = False
        # finished setting defaults
        self.params = params
        self.data_file = data_file
        self.ds_image = None
        self.need_save_data = False
        self.saved_data = None
        self.debug = kwargs.get('debug', False)
        # only when phc feature is configured this becomes array
        self.phc_correction = 1


    def init_dev(self, device_id):
        """
        This function initializes GPU device that will be used for reconstruction. When running on CPU, the device_id
        is set to -1, and the initialization is omitted.
        After the device is set, the data array is loaded on that device or cpu memory. The data file can be
        in tif or npy format.

        :param device_id: device id or -1 if cpu
        :return: 0 if successful, -1 otherwise. In debug mode will re-raise exception instead of returning -1.
        """
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if device_id != -1:
            self.dev = device_id
            if device_id != -1:
                try:
                    devlib.set_device(device_id)
                except Exception as e:
                    if self.debug:
                        raise
                    print(e)
                    print('may need to restart GUI')
                    return -1

        if self.data_file.endswith('tif') or self.data_file.endswith('tiff'):
            try:
                data_np = ut.read_tif(self.data_file)
                data = devlib.from_numpy(data_np)
            except Exception as e:
                if self.debug:
                    raise
                print(e)
                return -1
        elif self.data_file.endswith('npy'):
            try:
                data = devlib.load(self.data_file)
            except Exception as e:
                if self.debug:
                    raise
                print(e)
                return -1
        else:
            print('no data file found')
            return -1
        # in the formatted data the max is in the center, we want it in the corner, so do fft shift
        self.data = devlib.fftshift(data)
        self.dims = devlib.dims(self.data)
        print('data shape', self.dims)

        if self.params.get("live_trigger", None) is not None:
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 13), layout="constrained")
            plt.show(block=False)
        else:
            self.fig = None
            self.axs = None

        if self.need_save_data:
            self.saved_data = devlib.copy(self.data)
            self.need_save_data = False

        return 0


    def init_iter_loop(self, dir=None, gen=None):
        """
        Initializes the iteration loop.

        :param dir: directory containing 'image.npy' file that contains reconstructed object, if the reconstruction is continuation. 
            It defaults to None.
        :param gen: generation number, only used when running GA algorithm.
            It defaults to None. 
        :return:  0 if successful, -1 otherwise. In debug mode will re-raise exception instead of returning -1.
        """
        if self.ds_image is not None:
            first_run = False
        elif dir is None or not os.path.isfile(ut.join(dir, 'image.npy')):
            self.ds_image = devlib.random(self.dims, dtype=self.data.dtype)
            first_run = True
        else:
            self.ds_image = devlib.load(ut.join(dir, 'image.npy'))
            first_run = False

        # When running GA the lowpass filter, phc, and twin triggers should be active only during first
        # generation. The code below inactivates the triggers in subsequent generations.
        # This will be removed in the future when each generation will have own configuration.
        if not first_run:
            self.params.pop('lowpass_filter_trigger', None)
            self.params.pop('phc_trigger', None)
            self.params.pop('twin_trigger', None)

        self.flow_items_list = [f.__name__ for f in self.iter_functions]

        try:
            self.is_pc, flow, feats = of.get_flow_arr(self.params, self.flow_items_list, gen)
        except Exception as e:
            if self.debug:
                raise
            print(e)
            return -1

        self.flow = []
        (op_no, self.iter_no) = flow.shape
        for i in range(self.iter_no):
            for j in range(op_no):
                if flow[j, i] > 0:
                    self.flow.append(self.iter_functions[j])

        self.aver = None
        self.iter = -1
        self.errs = []
        self.prev_dir = dir

        # create or get initial support
        if dir is None or not os.path.isfile(ut.join(dir, 'support.npy')):
            init_support = [int(self.params['initial_support_area'][i] * self.dims[i]) for i in range(len(self.dims))]
            center = devlib.full(init_support, 1)
            self.support = dvut.pad_around(center, self.dims, 0)
        else:
            self.support = devlib.load(ut.join(dir, 'support.npy'))

        if self.is_pc:
            self.pc_obj = ft.Pcdi(self.params, self.data, dir)

        if 'lowpass_filter_trigger' in self.params:
            self.lowpass_filter_obj = ft.LowPassFilter( self.params)

        # create the trgger/sub-trigger objects
        try:
            if 'shrink_wrap_trigger' in self.params:
                self.shrink_wrap_obj = ft.create('shrink_wrap', self.params, feats)
            if 'phc_trigger' in self.params:
                self.phc_obj = ft.create('phc', self.params, feats)
        except Exception as e:
            if self.debug:
                raise
            print(e)
            return -1

        # for the fast GA the data needs to be saved, as it would be changed by each lr generation
        # for non-fast GA the Rec object is created in each generation with the initial data
        if self.saved_data is not None:
            if self.params['low_resolution_generations'] > gen:
                self.data = devlib.gaussian_filter(self.saved_data, self.params['ga_lpf_sigmas'][gen])
            else:
                self.data = self.saved_data
        else:
            if gen is not None and self.params['low_resolution_generations'] > gen:
                self.data = devlib.gaussian_filter(self.data, self.params['ga_lpf_sigmas'][gen])

        if 'lowpass_filter_range' not in self.params or not first_run:
            self.iter_data = self.data
        else:
            self.iter_data = devlib.copy(self.data)

        if (first_run):
            max_data = devlib.amax(self.data)
            self.ds_image *= dvut.get_norm(self.ds_image) * max_data

            # the line below are for testing to set the initial guess to support
            # self.ds_image = devlib.full(self.dims, 1.0) + 1j * devlib.full(self.dims, 1.0)
            # self.ds_image = devlib.full(self.dims, 1.0) + 1j * devlib.full(self.dims, 0.0)

            self.ds_image *= self.support
        return 0


    def iterate(self):
        """
        It iterates over function order that was determined during loop initialization. 

        During iteration execution an exception can happen. The exception is handled in non debug mode and thrown otherwise. 
    
        :return: 0 if successful, -1 otherwise. In debug mode will throw exception instead of returning -1.
        """
        self.iter = -1
        start_t = time.time()
        try:
            for f in self.flow:
                f()
        except Exception as error:
            if self.debug:
                raise
            print(error)
            return -1

        print('iterate took ', (time.time() - start_t), ' sec')

        if devlib.hasnan(self.ds_image):
            print('reconstruction resulted in NaN')
            return -1

        if self.aver is not None:
            ratio = self.get_ratio(devlib.from_numpy(self.aver), devlib.absolute(self.ds_image))
            self.ds_image *= ratio / self.aver_iter

        if self.params.get("live_trigger", None) is not None:
            plt.show()

        return 0


    def save_res(self, save_dir, only_image=False):
        """
        Saves reconstruction results.

        It centers the maximum of image array and saves this array, support array, coherence if present, and errors. 
        This function is typically called after iterations.

        :param save_dir: directory where results will be saved
        :param only_image: will save only image if True, otherwise all the results are saved. By default all results are saved.
        :return:
        """
        mx = devlib.amax(devlib.absolute(self.ds_image))
        self.ds_image = self.ds_image / mx

        # center image's center_of_mass and sync support
        self.ds_image, self.support = dvut.center_sync(self.ds_image, self.support)

        from array import array

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        devlib.save(ut.join(save_dir, 'image'), self.ds_image)
        ut.save_tif(devlib.to_numpy(self.ds_image), ut.join(save_dir, 'image.tif'))

        if only_image:
            return 0

        devlib.save(ut.join(save_dir, 'support'), self.support)
        if self.is_pc:
            devlib.save(ut.join(save_dir, 'coherence'), self.pc_obj.kernel)
        errs = array('f', self.errs)

        with open(ut.join(save_dir, 'errors.txt'), 'w+') as err_f:
            err_f.write('\n'.join(map(str, errs)))

        devlib.save(ut.join(save_dir, 'errors'), errs)

        metric = dvut.all_metrics(self.ds_image, self.errs)
        with open(ut.join(save_dir, 'metrics.txt'), 'w+') as f:
            f.write(str(metric))

        return 0

    def get_metric(self):
        """
        Returns metrics such as: chi (the error after last iteration), summed phase, area, sharpness of the ds_image array that holds reconstructed image.
        """
        return dvut.all_metrics(self.ds_image, self.errs)

    def next(self):
        self.iter = self.iter + 1

    def lowpass_filter_operation(self):
        args = (self.data, self.iter)
        self.iter_data = self.lowpass_filter_obj.apply_trigger(*args)

    def reset_resolution(self):
        self.iter_data = self.data

    def shrink_wrap_operation(self):
        args = (self.ds_image,)
        self.support = self.shrink_wrap_obj.apply_trigger(*args)

    def reset_phc_correction(self):
        self.phc_correction = 1

    def phc_operation(self):
        args = (self.ds_image,)
        self.phc_correction = self.phc_obj.apply_trigger(*args)

    def to_reciprocal_space(self):
        self.rs_amplitudes = devlib.ifft(self.ds_image)

    def new_func_operation(self):
        self.params['new_param'] = 1
        print(f'in new_func_trigger, new_param {self.params["new_param"]}')

    def pc_operation(self):
        self.pc_obj.update_partial_coherence(devlib.absolute(self.rs_amplitudes))

    def pc_modulus(self):
        abs_amplitudes = devlib.absolute(self.rs_amplitudes)
        converged = self.pc_obj.apply_partial_coherence(abs_amplitudes)
        ratio = self.get_ratio(self.iter_data, devlib.absolute(converged))
        error = dvut.get_norm(
            devlib.where(devlib.absolute(converged) != 0.0, devlib.absolute(converged) - self.iter_data, 0.0)) / dvut.get_norm(self.iter_data)
        self.errs.append(float(error))
        self.rs_amplitudes *= ratio

    def modulus(self):
        ratio = self.get_ratio(self.iter_data, devlib.absolute(self.rs_amplitudes))
        error = dvut.get_norm(devlib.where((self.rs_amplitudes != 0), (devlib.absolute(self.rs_amplitudes) - self.iter_data),
                                      0)) / dvut.get_norm(self.iter_data)
        self.errs.append(float(error))
        self.rs_amplitudes *= ratio

    def set_prev_pc(self):
        self.pc_obj.set_previous(devlib.absolute(self.rs_amplitudes))

    def to_direct_space(self):
        self.ds_image_proj = devlib.fft(self.rs_amplitudes)

    def er(self):
        self.ds_image = self.ds_image_proj * self.support * self.phc_correction

    def hio(self):
        combined_image = self.ds_image - self.ds_image_proj * self.params['hio_beta']
        self.ds_image = devlib.where((self.support * self.phc_correction > 0),
                                     self.ds_image_proj,
                                     combined_image)

    def sf(self):
        # solvent flipping
        self.ds_image = 2.0 * (self.ds_image_proj * self.support * self.phc_correction) - self.ds_image_proj

    def raar(self):
        # Relaxed averaged alternating reflectors
        self.ds_image = (self.params['raar_beta']
                         * (self.support * self.phc_correction * self.ds_image_proj + self.ds_image)
                         + (1 - 2 * self.params['raar_beta']) * self.ds_image_proj)

    def twin_operation(self):
        # TODO this will work only for 3D array, but will the twin be used for 1D or 2D?
        # com = devlib.center_of_mass(devlib.absolute(self.ds_image))
        # sft = [int(self.dims[i] / 2 - com[i]) for i in range(len(self.dims))]
        # self.ds_image = devlib.shift(self.ds_image, sft)
        dims = devlib.dims(self.ds_image)
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

    def average_operation(self):
        if self.aver is None:
            self.aver = devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter = 1
        else:
            self.aver = self.aver + devlib.to_numpy(devlib.absolute(self.ds_image))
            self.aver_iter += 1

    def progress_operation(self):
        print(f'------iter {self.iter}   error {self.errs[-1]}')

    def live_operation(self):
        self.shift_to_center()
        half = self.dims[0] // 2
        qtr = self.dims[0] // 4
        plt.suptitle(
            f"iteration: {self.iter}\n"
            f"error: {self.errs[-1]}\n"
        )
        [[ax.clear() for ax in row] for row in self.axs]
        if len(self.ds_image.shape) == 2:
            img = self.ds_image
        else:
            img = self.ds_image[qtr:-qtr, qtr:-qtr, half]
        self.axs[0][0].set(title="Amplitude", xticks=[], yticks=[])
        self.axs[0][0].imshow(devlib.absolute(img), cmap="gray")
        self.axs[0][1].set(title="Phase", xticks=[], yticks=[])
        self.axs[0][1].imshow(devlib.angle(img), cmap="hsv", interpolation_stage="rgba")

        if len(self.ds_image.shape) == 2:
            sup = self.support
        else:
            sup = self.support[qtr:-qtr, qtr:-qtr, half]
        self.axs[1][0].set(title="Error", xlim=(0,self.iter_no), xlabel="Iteration", yscale="log")
        self.axs[1][0].plot(self.errs[1:])
        self.axs[1][1].set(title="Support", xticks=[], yticks=[])
        self.axs[1][1].imshow(sup, cmap="gray")

        plt.draw()
        plt.pause(0.15)

    def get_ratio(self, divident, divisor):
        ratio = devlib.where((divisor > 1e-9), divident / divisor, 0.0)
        return ratio

    def shift_to_center(self):
        ind = devlib.center_of_mass(devlib.astype(self.support, 'int32'))
        shift_dist = (self.dims[0]//2) - devlib.round(devlib.array(ind))
        shift_dist = devlib.to_numpy(shift_dist).tolist()
        axis = tuple(range(len(self.ds_image.shape)))
        self.ds_image = devlib.roll(self.ds_image, shift_dist, axis=axis)
        self.support = devlib.roll(self.support, shift_dist, axis=axis)

    def get_image(self):
        """
        Returns image array.
        """
        return self.ds_image

    def get_support(self):
        """
        Returns support array.
        """
        return self.support

    def get_coherence(self):
        """
        returns coherence array, if exists.
        """
        if self.is_pc:
            return self.pc_obj.kernel
        else:
            return None


class Peak:
    """
    Holds parameters related to a peak.
    """

    def __init__(self, peak_dir):
        geometry = ut.read_config(f"{peak_dir}/geometry")
        self.hkl = geometry["peak_hkl"]
        self.order = max([abs(x) for x in self.hkl])
        self.weight = 1.0
        self.conf_hist = []
        self.conf_iter = []
        self.weight_iter = 0
        print(f"Loading peak {self.hkl}")

        # Geometry parameters
        try:
            a, b, c, alpha, beta, gamma = geometry["lattice"]
            alpha *= np.pi / 180
            beta *= np.pi / 180
            gamma *= np.pi / 180
        except TypeError:
            # If only a single value is given, assume a cubic lattice
            a = b = c = geometry["lattice"]
            alpha = beta = gamma = pi/2
        # The next part gets a little messy, so I'm going to predefine some things:
        cosa, cosb, sing, cosg = np.cos(alpha), np.cos(beta), np.sin(gamma), np.cos(gamma)

        # Convert the (possibly nonorthogonal) basis into cartesian coordinates
        # a1 is defined to be along the x-axis
        a1 = a * devlib.array([1, 0, 0])
        # a2 is defined to be in the xy-plane
        a2 = b * devlib.array([cosg, sing, 0])
        # a3 is defined by the following system of equations:
        # a1.a3 = a * c * cos(beta)
        # a2.a3 = b * c * cos(alpha)
        # a3.a3 = |c|^2
        # which can be solved to give:
        a3 = c * devlib.array([
            cosb,
            (cosa - cosb*cosg)/sing,
            np.sqrt(2*cosa*cosb*cosg + sing**2 - cosa**2 - cosb**2)/sing
        ])

        # Now convert the direct lattice vectors (a1, a2, a3) into reciprocal lattice vectors (b1, b2, b3)
        b1 = 2*pi*devlib.cross(a2, a3) / devlib.dot(a1, devlib.cross(a2, a3))
        b2 = 2*pi*devlib.cross(a3, a1) / devlib.dot(a1, devlib.cross(a2, a3))
        b3 = 2*pi*devlib.cross(a1, a2) / devlib.dot(a1, devlib.cross(a2, a3))

        self.g_vec = self.hkl[0]*b1 + self.hkl[1]*b2 + self.hkl[2]*b3
        # print(self.g_vec.get())
        self.gdotg = devlib.array(devlib.dot(self.g_vec, self.g_vec))
        self.size = geometry["final_size"]
        self.rs_lab_to_det = np.array(geometry["rmatrix"]) / geometry["rs_voxel_size"]
        # self.rs_lab_to_det = np.identity(3)  # FIXME
        self.rs_det_to_lab = np.linalg.inv(self.rs_lab_to_det)
        self.ds_lab_to_det = self.rs_det_to_lab.T
        self.ds_det_to_lab = self.rs_lab_to_det.T

        datafile = (Path(peak_dir) / "phasing_data/data.tif").as_posix()
        self.full_data = devlib.absolute(devlib.from_numpy(tf.imread(datafile)))
        self.full_data = devlib.moveaxis(self.full_data, 0, 2)
        self.full_data = devlib.moveaxis(self.full_data, 0, 1)
        self.full_size = np.max(self.full_data.shape)
        self.full_data = dvut.pad_to_cube(self.full_data, self.full_size)
        self.data = dvut.pad_to_cube(self.full_data, self.size)
        self.mask = devlib.zeros(self.data.shape)

        # Uncomment when phasing non-centered data
        # ind = devlib.center_of_mass(self.full_data)
        # shift_dist = (self.size // 2) - devlib.round(devlib.array(ind))
        # shift_dist = devlib.to_numpy(shift_dist).tolist()
        # self.full_data = devlib.roll(self.full_data, shift_dist, axis=(0, 1, 2))

        # Resample the diffraction patterns into the lab reference frame.
        self.res_data = dvut.resample(self.full_data, self.rs_det_to_lab)
        if self.size is not None:
            self.res_data = dvut.pad_to_cube(self.res_data, self.size)

        tf.imwrite((Path(peak_dir) / "phasing_data/data_resampled.tif").as_posix(), devlib.to_numpy(self.res_data))

        # Do the fftshift on all the arrays
        self.full_data = devlib.fftshift(self.full_data)
        self.data = devlib.fftshift(self.data)
        self.res_data = devlib.fftshift(self.res_data)
        self.total = devlib.sum(self.res_data**2)
        self.max_lk = devlib.sum(devlib.xlogy(self.res_data**2, self.res_data**2) - self.res_data**2)

    def normalize(self, norm):
        factor = norm / self.total
        self.full_data *= devlib.sqrt(factor)
        self.data *= devlib.sqrt(factor)
        self.res_data *= devlib.sqrt(factor)
        self.total = devlib.sum(self.res_data**2)
        self.max_lk = devlib.mean(devlib.xlogy(self.res_data**2, self.res_data**2) - self.res_data**2)

    def unlikelihood(self, proj):
        """Returns the difference between the maximum and current log-likelihood"""
        proj = devlib.absolute(proj)**2
        lk = devlib.mean(devlib.xlogy(self.res_data**2, proj) - proj)
        return self.max_lk - lk

    def chi_square(self, proj):
        error = devlib.where((self.res_data != 0), (devlib.absolute(proj) - self.res_data), 0)
        return dvut.get_norm(error) / dvut.get_norm(self.res_data)

    def get_error(self, proj):
        pass


class CoupledRec(Rec):
    """
    Performs a coupled reconstruction of multiple Bragg peaks using iterative phase retrieval. It alternates between a
    shared object with a density and atomic displacement field and a working object with an amplitude and phase. The
    general outline of this process is as follows:
    1. Initialize the shared object with random values.
    2. Randomly select a diffraction pattern and corresponding reciprocal lattice vector G from the collected data.
    3. Set the working object to the projection of the shared object onto G.
    4. Apply standard phase retrieval techniques to the working object for a set number of iterations.
    5. Update the G-projection of the shared object to be a weighted average of its current value with the working
        object.
    6. Repeat steps 2-5.

    params : dict
        parameters used in reconstruction. Refer to x for parameters description
    data_file : str
        name of file containing data for each peak to be reconstructed

    """
    __author__ = "Nick Porter"
    __all__ = []

    def __init__(self, params, peak_dirs, pkg, **kwargs):
        super().__init__(params, None, pkg, **kwargs)

        self.iter_functions = [self.adapt_operation] + self.iter_functions + [self.switch_peak_operation]

        self.params["switch_peak_trigger"] = self.params.get("switch_peak_trigger", [0, 5])
        self.params["adapt_trigger"] = self.params.get("adapt_trigger", [])
        self.params["calc_strain"] = self.params.get("calc_strain", False)

        self.peak_dirs = peak_dirs
        self.er_iter = False  # Indicates whether the last iteration done was ER

    def init_dev(self, device_id):
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        if device_id != -1:
            self.dev = device_id
            if device_id != -1:
                try:
                    devlib.set_device(device_id)
                except Exception as e:
                    if self.debug:
                        raise
                    print(e)
                    print('may need to restart GUI')
                    return -1

        # Allows for a control peak to be left out of the reconstruction for comparison
        all_peaks = [Peak(d) for d in self.peak_dirs]
        norm = max([p.total for p in all_peaks])
        for p in all_peaks:
            p.normalize(norm)
        # for pk in all_peaks:
        #     plt.figure()
        #     plt.imshow(np.log(devlib.fftshift(pk.res_data)[90].get() + 1))
        # plt.show()
        if "control_peak" in self.params:
            control_peak = self.params["control_peak"]
            self.peak_objs = [p for p in all_peaks if p.hkl != control_peak]
            try:
                self.ctrl_peak = next(p for p in all_peaks if p.hkl == control_peak)
                self.ctrl_error = []
            except StopIteration:
                self.ctrl_peak = None
                self.ctrl_error = None
        else:
            self.peak_objs = all_peaks
            self.ctrl_peak = None
            self.ctrl_error = None
        self.peak_confidence = [1.0 for _ in self.peak_objs]

        # Fast resampling means using the resampled data to reconstruct the object in a single common basis
        # When this is set to False, the reconstruction will need to be resampled each time the peak is switched
        self.fast_resample = True
        self.lab_frame = True

        self.num_peaks = len(self.peak_objs)
        self.pk = 0  # index in list of current peak being reconstructed
        self.data = self.peak_objs[self.pk].res_data
        self.iter_data = self.data
        self.dims = self.data.shape
        self.n_voxels = self.dims[0]*self.dims[1]*self.dims[2]

        if self.params.get("live_trigger", None) is not None:
            self.fig, self.axs = plt.subplots(2, 2, figsize=(12, 13), layout="constrained")
            plt.show(block=False)
        else:
            self.fig = None
            self.axs = None

        if self.need_save_data:
            self.saved_data = devlib.copy(self.data)
            self.need_save_data = False

        return 0

    def init_iter_loop(self, img_dir=None, gen=None):
        if super().init_iter_loop(img_dir, gen) == -1:
            return -1

        # Define the shared image
        self.rho_image = devlib.absolute(self.ds_image)
        self.u_image = devlib.absolute(self.ds_image[:, :, :, None]) * devlib.array([0, 0, 0])

        # Define the multipeak projection weighting and tapering
        self.proj_weight = devlib.array([self.params["weight_init"] for _ in range(self.iter_no)])
        for i, w in zip(self.params["weight_iters"], self.params["weight_vals"]):
            self.proj_weight[i:] = w

        self.adapt_on = False
        self.peak_threshold = devlib.array([self.params["adapt_threshold_init"] for _ in range(self.iter_no)])
        for i, w in zip(self.params["adapt_threshold_iters"], self.params["adapt_threshold_vals"]):
            self.peak_threshold[i:] = w
        self.peak_weights = []
        self.iter_weights = []
        return 0

    def save_res(self, save_dir):
        from array import array
        tight_support = devlib.binary_erosion(self.support)
        self.rho_image = self.rho_image * tight_support
        self.u_image = self.u_image * tight_support[:, :, :, None]
        # for i in range(3):
        #     self.u_image[:, :, :, i] = devlib.median_filter(self.u_image[:, :, :, i], 3)
        self.shift_to_center()

        crop = self.dims[0] / 4
        sup = self.support[crop:-crop, crop:-crop, crop:-crop]
        rho = self.rho_image[crop:-crop, crop:-crop, crop:-crop] * sup
        u_x = self.u_image[crop:-crop, crop:-crop, crop:-crop, 0] * sup
        u_y = self.u_image[crop:-crop, crop:-crop, crop:-crop, 1] * sup
        u_z = self.u_image[crop:-crop, crop:-crop, crop:-crop, 2] * sup

        if self.params["calc_strain"]:
            J = self.calc_jacobian()[crop:-crop, crop:-crop, crop:-crop]
            s_xx = J[:,:,:,0,0]
            s_yy = J[:,:,:,1,1]
            s_zz = J[:,:,:,2,2]
            s_xy = ((J[:,:,:,0,1] + J[:,:,:,1,0]) / 2)
            s_yz = ((J[:,:,:,1,2] + J[:,:,:,2,1]) / 2)
            s_zx = ((J[:,:,:,2,0] + J[:,:,:,0,2]) / 2)
        else:
            s_xx = s_yy = s_zz = s_xy = s_yz = s_zx = devlib.zeros(sup.shape)

        final_rec = devlib.stack((rho, u_x, u_y, u_z, s_xx, s_yy, s_zz, s_xy, s_yz, s_zx, sup))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        devlib.save(save_dir + "/reconstruction", final_rec)
        devlib.save(save_dir + "/image_nostrain", devlib.stack((rho, u_x, u_y, u_z)))
        devlib.save(save_dir + "/shared_density", rho)
        devlib.save(save_dir + "/shared_u1", u_x)
        devlib.save(save_dir + "/shared_u2", u_y)
        devlib.save(save_dir + "/shared_u3", u_z)
        devlib.save(save_dir + '/support', sup)

        errs = array('f', self.errs)
        with open(ut.join(save_dir, 'errors.txt'), 'w+') as err_f:
            err_f.write('\n'.join(map(str, errs)))
        devlib.save(ut.join(save_dir, 'errors'), errs)

        metric = dvut.all_metrics(self.ds_image, self.errs)
        with open(ut.join(save_dir, 'metrics.txt'), 'w+') as f:
            f.write(str(metric))

        for pk in self.peak_objs:
            np.savetxt(f"{save_dir}/conf_hist_{pk.hkl[0]}{pk.hkl[1]}{pk.hkl[2]}.txt", pk.conf_hist)
            np.savetxt(f"{save_dir}/conf_iter_{pk.hkl[0]}{pk.hkl[1]}{pk.hkl[2]}.txt", pk.conf_iter)

        if self.ctrl_error is not None:
            ctrl_error = np.array(self.ctrl_error)
            np.save(save_dir + '/ctrl_error', ctrl_error)
            np.savetxt(save_dir + '/ctrl_error.txt', ctrl_error)
        return 0

    def make_binary(self):
        self.support = devlib.absolute(self.support)
        self.support = devlib.where(self.support >= devlib.astype(0.9 * devlib.amax(self.support), 1, 0), ("?"))

    def get_phase_gradient(self):
        """Calculate the phase gradient for a given peak"""
        # Project onto the peak
        if self.fast_resample:
            self.iter_data = self.peak_objs[self.pk].res_data
        else:
            self.iter_data = self.peak_objs[self.pk].data

        self.to_working_image()

        # Calculate the continuous phase gradient using Hofmann's method
        # (https://doi.org/10.1103/PhysRevMaterials.4.013801)
        gradient = []
        voxel_size = self.params["ds_voxel_size"]
        dx0, dy0, dz0 = devlib.gradient(devlib.angle(self.ds_image), voxel_size)
        dx1, dy1, dz1 = devlib.gradient(devlib.angle(self.ds_image * devlib.exp(1j*pi/2)), voxel_size)
        dx2, dy2, dz2 = devlib.gradient(devlib.angle(self.ds_image * devlib.exp(-1j*pi/2)), voxel_size)
        for stack in ([dx0, dx1, dx2], [dy0, dy1, dy2], [dz0, dz1, dz2]):
            stack = devlib.array(stack)
            index = devlib.argmin(stack**2, axis=0)
            gradient.append(devlib.take_along_axis(stack, index[None, :, :, :], axis=0).squeeze(axis=0))
        return gradient

    def calc_jacobian(self):
        grad_phi = []
        for i, peak in enumerate(self.peak_objs):
            self.prev_pk = self.pk
            self.pk = i
            grad_phi.append(self.get_phase_gradient())
        grad_phi = devlib.array(grad_phi)
        G = devlib.array([peak.g_vec for peak in self.peak_objs])
        grad_phi = devlib.moveaxis(devlib.moveaxis(grad_phi, 0, -1), 0, -1)
        cond = devlib.where(self.support, True, False)
        final_shape = (grad_phi.shape[0], grad_phi.shape[1], grad_phi.shape[2], 3, 3)
        ind = devlib.moveaxis(devlib.indices(cond.shape), 0, -1).reshape((-1, 3))[cond.ravel()].get()
        jacobian = devlib.zeros(final_shape)
        print("Calculating strain...")
        for i, j, k in tqdm.tqdm(ind):
            jacobian[i, j, k] = devlib.lstsq(G, grad_phi[i, j, k])[0]
        return jacobian

    def iterate(self):
        self.iter = -1
        start_t = time.time()

        try:
            for f in self.flow:
                f()
        except Exception as error:
            if self.debug:
                raise
            print('error',error)
            return -1

        print('iterate took ', (time.time() - start_t), ' sec')

        if self.params.get("live_trigger", None) is not None:
            plt.show()

        if devlib.hasnan(self.ds_image):
            print('reconstruction resulted in NaN')
            return -1

        if self.aver is not None:
            ratio = self.get_ratio(devlib.from_numpy(self.aver), devlib.absolute(self.ds_image))
            self.ds_image *= ratio / self.aver_iter

        mx = devlib.amax(devlib.absolute(self.ds_image))
        self.ds_image = self.ds_image / mx

        return 0

    def switch_peak_operation(self):
        self.calc_confidence()
        self.to_shared_image()
        self.get_control_error()

        if self.adapt_on:
            print("Adapting weights")
            self.update_weights()

        if self.params.get("live_view", False):
            self.update_live()

        # Use the peak weights as probabilities for projecting to that peak. High-confidence = more likely
        p_weights = [self.peak_objs[x].weight for x in range(self.num_peaks)]
        p_weights[self.pk] = 0  # Don't project back onto the same peak we just used
        self.peak_weights.append(p_weights / np.sum(p_weights))
        self.iter_weights.append(self.iter)
        self.pk = random.choices([x for x in range(self.num_peaks)], p_weights)[0]

        self.to_working_image()

    def to_shared_image(self):
        pk = self.peak_objs[self.pk]  # shorthand

        if not self.fast_resample:
            # Resample the object
            if self.lab_frame:
                self.lab_frame = False
            else:
                self.shift_to_center()
                self.ds_image = dvut.resample(self.ds_image, self.peak_objs[self.pk].ds_det_to_lab)
                self.support = dvut.resample(self.support, self.peak_objs[self.pk].ds_det_to_lab,
                                                    plot=True)
                if self.dims is not None:
                    self.ds_image = dvut.pad_to_cube(self.ds_image, self.dims[0])
                    self.support = dvut.pad_to_cube(self.support, self.dims[0])

                self.make_binary()

        # The weighting is determined by the peak weight and the global weight. I'm leaving the option to give the
        # density and displacement their own weights, but currently they're just the same.
        wt_rho = pk.weight * self.proj_weight[self.iter]
        wt_u = pk.weight * self.proj_weight[self.iter]

        # Update density with a weighted average of the shared and working objects
        self.rho_image = (1-wt_rho) * self.rho_image + wt_rho * devlib.absolute(self.ds_image)

        # Update displacement with a weighted average.
        old_u = (devlib.dot(self.u_image, pk.g_vec) / pk.gdotg)[:, :, :, None] * pk.g_vec
        new_u = (devlib.angle(self.ds_image) / pk.gdotg)[:, :, :, None] * pk.g_vec
        self.u_image = self.u_image + wt_u * (new_u - old_u)

        if self.er_iter:
            # No shrinkwrap during ER blocks. Instead, the shared object is given an extra-tight support constraint,
            # and the density is given a 3-pixel smoothover.
            tight_support = devlib.binary_erosion(self.support)
            self.rho_image = devlib.median_filter(self.rho_image, 3)
            self.rho_image = self.rho_image * tight_support
            self.u_image = self.u_image * tight_support[:, :, :, None]
            self.shift_to_center()

        # Shift the displacement field so that the center of the image is at ux=uy=uz=0
        ctr = self.dims[0] / 2
        self.u_image = self.u_image[:, :, :] - self.u_image[ctr, ctr, ctr]

    def to_working_image(self):
        pk = self.peak_objs[self.pk]  # Shorthand for convenience

        # Define the exit wave for this peak based on the shared density/displacement
        self.ds_image = self.rho_image * devlib.exp(1j * devlib.dot(self.u_image, pk.g_vec))

        if self.fast_resample:
            # Use the resampled data
            self.iter_data = pk.res_data
        else:
            # Use the original sampling
            self.iter_data = pk.data

        if self.iter > self.params["adapt_alien_start"]:
            # Calculate the diffraction amplitude from the current exit wave (which is based ONLY on shared object).
            proj = devlib.absolute(devlib.ifft(self.ds_image))

            # Blur the projected and measured diffraction amplitudes by one pixel (to mitigate HF noise)
            proj_blurred = devlib.gaussian_filter(proj, 1)
            meas_blurred = devlib.gaussian_filter(self.iter_data, 1)

            # Map the absolute difference between the two blurred images.
            # Normalizing to the projected image makes bright artifacts in the measurement stand out:
            #   proj  |  meas  | diff_map
            # --------|--------|----------
            #   high  |  high  |  low
            #   low   |  low   |  low
            #   high  |  low   |  low
            #   low   |  high  |  HIGH
            diff_map = devlib.absolute(meas_blurred - proj_blurred) / proj_blurred

            # Mask voxels where the difference map is exceptionally high compared to its median. If there are no
            # outliers, the difference map will be tightly clustered around the median.
            pk.mask = diff_map > devlib.median(diff_map) * self.params["adapt_alien_threshold"]

            # For phasing, use the projected amplitude for masked voxels and the measurement for unmasked voxels.
            self.iter_data = devlib.where(pk.mask, proj, proj + pk.weight*(self.iter_data - proj))

        if not self.fast_resample:
            # Resample the object and support
            self.ds_image = dvut.resample(self.ds_image, pk.ds_lab_to_det)
            self.support = dvut.resample(self.support, pk.ds_lab_to_det)
            if self.dims is not None:
                self.ds_image = dvut.pad_to_cube(self.ds_image, self.dims[0])
                self.support = dvut.pad_to_cube(self.support, self.dims[0])
            self.make_binary()

    def calc_confidence(self):
        pk = self.peak_objs[self.pk]  # shorthand

        # Calculate the amplitude of the exit wave after phasing.
        rho = devlib.absolute(self.ds_image)
        cond = self.support > 0

        # Compare the exit wave amplitude before and after phasing (only within the support region)
        # conf = 1 / dvut.calc_ehd(self.rho_image[cond], rho[cond], log=False).get()
        conf = dvut.calc_nmi(self.rho_image[cond], rho[cond], log=False).get() - 1

        pk.conf_hist.append(conf)
        pk.conf_iter.append(self.iter)

    def adapt_operation(self):
        self.adapt_on = True

    def update_weights(self):
        # Grab the 75th %ile confidence for each peak. If a peak has not been phased, then its confidence history will
        # be empty, in which case it just gets a -1 and skips the weighting.
        confidences = [
            sorted(p.conf_hist, reverse=True)[len(p.conf_hist) // 4]
            if len(p.conf_hist) > 0 else -1
            for p in self.peak_objs
        ]
        for pk, confidence in zip(self.peak_objs, confidences):
            if confidence == -1:
                continue
            # Assign the weights based on the RELATIVE confidence raised to a power (currently hard-coded at 2). The
            # power determines how harshly bad peaks are punished.
            pk.weight = (confidence / max(confidences)) ** self.params["adapt_power"]
        print(f"New weights: {[round(p.weight, 2) for p in self.peak_objs]}")
        self.adapt_on = False

    def to_reciprocal_space(self):
        self.rs_amplitudes = devlib.ifft(self.ds_image)

    def to_direct_space(self):
        self.ds_image_proj = devlib.fft(self.rs_amplitudes)

    def modulus(self):
        ratio = self.get_ratio(self.iter_data, devlib.absolute(self.rs_amplitudes))
        error = dvut.get_norm(devlib.where((self.rs_amplitudes != 0), (devlib.absolute(self.rs_amplitudes) - self.iter_data),
                                      0)) / dvut.get_norm(self.iter_data)
        self.errs.append(float(error))
        self.rs_amplitudes *= ratio

    def er(self):
        self.er_iter = True
        self.ds_image = self.ds_image_proj * self.support

    def hio(self):
        self.er_iter = False
        combined_image = self.ds_image - self.ds_image_proj * self.params['hio_beta']
        support = self.support
        self.ds_image = devlib.where((support > 0), self.ds_image_proj, combined_image)

    def shrink_wrap_operation(self):
        if self.er_iter or self.peak_objs[self.pk].weight < self.peak_threshold[self.iter]:
            return
        super().shrink_wrap_operation()

    def get_control_error(self):
        pk = self.peak_objs[self.pk]
        if self.ctrl_peak is None:
            return
        phi = devlib.dot(self.u_image, pk.g_vec)
        rho = self.rho_image
        img = devlib.absolute(devlib.ifft(rho * devlib.exp(1j * phi)))

        lin_hist = devlib.histogram2d(self.ctrl_peak.res_data, img, log=False)
        nmi = dvut.calc_nmi(lin_hist).get()
        ehd = dvut.calc_ehd(lin_hist).get()

        log_hist = devlib.histogram2d(self.ctrl_peak.res_data, img, log=True)
        lnmi = dvut.calc_nmi(log_hist).get()
        lehd = dvut.calc_ehd(log_hist).get()

        self.ctrl_error.append([nmi, lnmi, ehd, lehd])

    def progress_operation(self):
        ornt = self.peak_objs[self.pk].hkl
        prg = f'|  iter {self.iter:>4}  ' \
              f'|  peak {self.pk}  [{ornt[0]:>2}, {ornt[1]:>2}, {ornt[2]:>2}]  ' \
              f'|  err {self.errs[-1]:0.6f}  ' \
              f'|  sup {devlib.sum(self.support):>8}  '
        if self.ctrl_error is not None:
            prg += f"|  NMI {self.ctrl_error[-1][0]:0.6f}  "
            prg += f"|  LNMI {self.ctrl_error[-1][1]:0.6f}  "
            prg += f"|  EHD {self.ctrl_error[-1][2]:0.6f}  "
            prg += f"|  LEHD {self.ctrl_error[-1][3]:0.6f}  "
        print(prg)

    def live_operation(self):
        half = self.dims[0] // 2
        qtr = self.dims[0] // 4
        plt.suptitle(
            f"iteration: {self.iter}\n"
            f"peak: {self.peak_objs[self.pk].hkl}\n"
            f"global weight: {self.proj_weight[self.iter]}\n"
            f"peak weight: {self.peak_objs[self.pk].weight:0.3f}\n"
            f"confidence threshold: {self.peak_threshold[self.iter]}\n"
        )
        [[ax.clear() for ax in row] for row in self.axs]
        self.axs[0][0].set_title("XY amplitude")
        self.axs[0][0].imshow(devlib.absolute(self.ds_image[qtr:-qtr, qtr:-qtr, half]).get(), cmap="gray")
        self.axs[0][1].set_title("XY phase")
        self.axs[0][1].imshow(devlib.angle(self.ds_image[qtr:-qtr, half, qtr:-qtr]).get(), cmap="hsv",
                              interpolation_stage="rgba")

        self.axs[1][0].set_title("Measurement")
        meas = devlib.sum(devlib.ifftshift(self.peak_objs[self.pk].res_data), axis=0)
        self.axs[1][0].imshow(devlib.sqrt(meas).get(), cmap="magma")
        self.axs[1][1].set_title("Fourier Constraint")
        data = devlib.sum(devlib.ifftshift(self.iter_data), axis=0)
        self.axs[1][1].imshow(devlib.sqrt(data).get(), cmap="magma")
        plt.setp(self.fig.get_axes(), xticks=[], yticks=[])

        plt.draw()
        plt.pause(0.15)

    def switch_resampling_operation(self):
        self.fast_resample = False

    def get_density(self):
        return self.rho_image

    def get_distortion(self):
        return self.u_image.swapaxes(3, 2).swapaxes(2, 1).swapaxes(1, 0)

    def flip(self):
        self.rho_image = devlib.flip(self.rho_image, axis=(0, 1, 2))
        self.u_image = -1 * devlib.flip(self.u_image, axis=(0, 1, 2))
        self.support = devlib.flip(self.support, axis=(0, 1, 2))

    def shift_to_center(self):
        ind = devlib.center_of_mass(self.rho_image)
        shift_dist = (self.dims[0]//2) - devlib.round(devlib.array(ind))
        shift_dist = devlib.to_numpy(shift_dist).tolist()
        axis = tuple(range(len(self.rho_image.shape)))
        self.rho_image = devlib.roll(self.rho_image, shift_dist, axis=axis)
        self.u_image = devlib.roll(self.u_image, shift_dist, axis=axis)
        self.ds_image = devlib.roll(self.ds_image, shift_dist, axis=axis)
        self.support = devlib.roll(self.support, shift_dist, axis=axis)


class TeRec(Rec):
    """
    Coherent diffractive imaging of time-evolving samples with improved temporal resolution

    params : dict
        parameters used in reconstruction. Refer to x for parameters description
    data_file : str
        name of file containing data

    """

    def __init__(self, params, data_file, pkg, comm):
        super().__init__(params, data_file, pkg)

        self.comm = comm
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.weight = params['weight']


    def exchange_data_info(self):
        # The data with fewer frames comparing to full data
        # has the missing frames filled with -1.
        # Since data collected by detector is greater than 0,
        # the data that does not have any negative values is full data,
        # and partial data otherwise.
        self.is_full_data = (self.data < 0).sum() == 0
        # send to previous if full_data
        if self.rank != 0:
            self.comm.send(self.is_full_data, dest=self.rank - 1)
        if self.rank != self.size - 1:
            next_full_data = self.comm.recv(source=self.rank+1)
        # send to next if full_data
        if self.rank != self.size - 1:
            self.comm.send(self.is_full_data, dest=self.rank + 1)
        if self.rank != 0:
            prev_full_data = self.comm.recv(source=self.rank-1)

        # figure out if rank needs to send prev/next
        self.send_to_prev = (self.rank > 0) and (not prev_full_data)
        self.send_to_next = (self.rank < self.size - 1) and (not next_full_data)


    def er(self):
        if self.send_to_prev:
            self.comm.send(self.ds_image, dest=self.rank - 1)
        if not self.is_full_data:
            ds_image_next = self.comm.recv(source=self.rank + 1)
        if self.send_to_next:
            self.comm.send(self.ds_image, dest=self.rank + 1)
        if not self.is_full_data:
            ds_image_prev = self.comm.recv(source=self.rank - 1)

        if self.is_full_data:
            # run the super er
            self.ds_image = self.ds_image_proj * self.support

        else:
            # use previous and next to run modified er
            self.ds_image = (1/(1+2*self.weight)) * self.support * (self.ds_image_proj +
            self.weight * (ds_image_prev + ds_image_next))


    def hio(self):
        if self.send_to_prev:
            self.comm.send(self.ds_image, dest=self.rank - 1)
        if not self.is_full_data:
            ds_image_next = self.comm.recv(source=self.rank + 1)
        if self.send_to_next:
            self.comm.send(self.ds_image, dest=self.rank + 1)
        if not self.is_full_data:
            ds_image_prev = self.comm.recv(source=self.rank - 1)

        if self.is_full_data:
            # run the super hio
            combined_image = self.ds_image - self.ds_image_proj * self.params['hio_beta']
            self.ds_image = devlib.where((self.support > 0), self.ds_image_proj, combined_image)
        else:
            # use previous and next to run modified hio
            combined_image = self.ds_image - self.ds_image_proj * self.params['hio_beta']
            corr = self.weight * self.support * (2 * (self.ds_image) - (ds_image_prev + ds_image_next))
            self.ds_image = devlib.where((self.support > 0), self.ds_image_proj, combined_image) - corr

    def modulus(self):
        ratio = self.get_ratio(self.iter_data, devlib.absolute(self.rs_amplitudes))
        error = dvut.get_norm(devlib.where((self.rs_amplitudes != 0), (devlib.absolute(self.rs_amplitudes) - self.iter_data),
                                      0)) / dvut.get_norm(self.iter_data)
        self.errs.append(float(error))
        # do not change the frames with -1 value.
        # This value is marking frames that were not collected, and the reconstruction
        # is taking neighboring results (in hio and er)
        self.rs_amplitudes[self.iter_data != -1.0] *= ratio[self.iter_data != -1.0]


def create_rec(params, datainfo, pkg, dev, **kwargs):
    """
    Factory for creating Rec object.

    If the rec_type is "mp", it will create CoupledRec object used for multipeak
    reconstruction, otherwise base Rec object.
    After the object is instantiated it will be set to a given device. This operation
    can fail, in which case no object is returned.
    The function continues with initialization of iteration loop. If successful, the
    Rec object is returned, otherwise None.

    :param params: dict, contains reconstruction parameters
    :param datainfo: 
        for 'basic' datainfo is data file name,
        and for 'mp' reconstruction it is a list of peak directories
    :param pkg:
        python package that will be used to run the reconstruction:
        'cp' for cupy, 'np' for numpy, 'torch' for torch
    :param dev: GPU device id or -1
    :param kwargs:
        var parameters:
        rec_type: 'mp' for multipeak,
        debug : if True the exceptions are not handled
    :return: created and initialized object if success, None if failure

    """

    rec_type = kwargs.pop('rec_type', 'basic')
    if rec_type == 'mp':
        worker = CoupledRec(params, datainfo, pkg, **kwargs)
    else:
        worker = Rec(params, datainfo, pkg, **kwargs)

    if worker.init_dev(dev) < 0:
        del worker
        return None

    continue_dir = params.get('continue_dir')
    ret_code = worker.init_iter_loop(continue_dir)
    if ret_code < 0:
        del worker
        return None

    return worker
