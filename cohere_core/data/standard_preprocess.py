# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################


"""
This script formats data for reconstruction according to configuration.
"""

import os
import numpy as np
import cohere_core.data.alien_tools as at
import cohere_core.utilities.utils as ut
import cohere_core.utilities.config_verifier as ver


__author__ = "Barbara Frosik"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['prep',
           ]


def prep(beamline_full_datafile_name, auto, **kwargs):
    """
    This function formats data for reconstruction and saves it in data.tif file. The preparation consists of the following steps:
        - removing the alien: aliens are areas that are effect of interference. The area is manually set in a configuration file after inspecting the data. It could be also a mask file of the same dimensions that data. Another option is AutoAlien1 algorithm that automatically removes the aliens.
        - clearing the noise: values below an amplitude threshold are set to zero
        - amplitudes are set to sqrt
        - cropping and padding. If the adjust_dimention is negative in any dimension, the array is cropped in this dimension. The cropping is followed by padding in the dimensions that have positive adjust dimension. After adjusting, the dimensions are adjusted further to find the smallest dimension that is supported by opencl library (multiplier of 2, 3, and 5).
        - centering - finding the greatest amplitude and locating it at a center of array. If shift center is defined, the center will be shifted accordingly.
        - binning - adding amplitudes of several consecutive points. Binning can be done in any dimension.

    Parameters
    ----------
    beamline_full_datafile_name : str
        full name of tif file containing beamline preprocessed data
    kwargs : keyword arguments
        data_dir : str
            directory where prepared data will be saved, default <experiment_dir>/phasing_data
        alien_alg : str
            Name of method used to remove aliens. Possible options are: ‘block_aliens’, ‘alien_file’, and ‘AutoAlien1’. The ‘block_aliens’ algorithm will zero out defined blocks, ‘alien_file’ method will use given file as a mask, and ‘AutoAlien1’ will use auto mechanism to remove aliens. Each of these algorithms require different parameters
        aliens : list
            Needed when the ‘block_aliens’ method is configured. Used when the data contains regions with intensity produced by interference. The regions needs to be zeroed out. The aliens can be defined as regions each defined by coordinates of starting point, and ending point (i.e. [[xb0,yb0,zb0,xe0,ye0,ze0],[xb1,yb1,zb1,xe1,ye1,ze1],…[xbn,ybn,zbn,xen,yen,zen]] ).
        alien_file : str
            Needed when the ‘alien_file’ method is configured. User can produce a file in npy format that contains table of zeros and ones, where zero means to set the pixel to zero, and one to leave it.
        AA1_size_threshold : float
            Used in the ‘AutoAliens1’ method. If not given it will default to 0.01. The AutoAlien1 algorithm will calculate relative sizes of all clusters with respect to the biggest cluster. The clusters with relative size smaller than the given threshold will be possibly deemed aliens. It also depends on asymmetry.
        AA1_asym_threshold : float
            Used in the ‘AutoAliens1’ method. If not given it will default to 1.75. The AutoAlien1 algorithm will calculate average asymmetry of all clusters. The clusters with average asymmetry greater than the given threshold will be possibly deemed aliens. It also depends on relative size.
        AA1_min_pts : int
            Used in the ‘AutoAliens1’ method. If not given it will default to 5. Defines minimum non zero points in neighborhood to count the area of data as cluster.
        AA1_eps : float
            Used in the ‘AutoAliens1’ method. If not given it will default to 1.1. Used in the clustering algorithm.
        AA1_amp_threshold : float
            Mandatory in the ‘AutoAliens1’ method. Used to zero data points below that threshold.
        AA1_save_arrs : boolean
            Used in the ‘AutoAliens1’ method, optional. If given and set to True multiple results of alien analysis will be saved in files.
        AA1_expandcleanedsigma : float
            Used in the ‘AutoAliens1’ method, optional. If given the algorithm will apply last step of cleaning the data using the configured sigma.
        intensity_threshold : float
            Mandatory, min data threshold. Intensity values below this are set to 0. The threshold is applied after removing aliens.
        adjust_dimensions : list
            Optional, a list of number to adjust the size at each side of 3D data. If number is positive, the array will be padded. If negative, cropped. The parameters correspond to [x left, x right, y left, y right, z left, z right] The final dimensions will be adjusted up to the good number for the FFT which also is compatible with opencl supported dimensions powers of 2 or a*2^n, where a is 3, 5, or 9
        center_shift : list
            Optional, enter center shift list the array maximum is centered before binning, and moved according to center_shift, [0,0,0] has no effect
        binning : list
            Optional, a list that defines binning values in respective dimensions, [1,1,1] has no effect.
        do_auto_binning : boolean
            Optional, mandatory if auto_data is True. is True the auto binning wil be done, and not otherwise.
        no_verify : boolean
            If True, ignores verifier error.
    """
    no_verify = kwargs.pop("no_verify", False)
    er_msg = ver.verify('config_data', kwargs)
    if len(er_msg) > 0:
        # the error message is printed in verifier
        if not no_verify:
            return

    beamline_full_datafile_name = beamline_full_datafile_name.replace(os.sep, '/')
    # The data has been transposed when saved in tif format for the ImageJ to show the right orientation
    beam_data = ut.read_tif(beamline_full_datafile_name)
    # if no_verify:
    #     print(f"Loaded array (max={int(data.max())}) as {beamline_full_datafile_name}")

    prep_data_dir, beamline_datafile_name = os.path.split(beamline_full_datafile_name)
    if 'data_dir' in kwargs:
        data_dir = kwargs['data_dir'].replace(os.sep, '/')
    else:
        # assuming the directory structure and naming follows cohere-ui mechanism
        data_dir = prep_data_dir.replace(os.sep, '/').replace('preprocessed_data', 'phasing_data')

    if 'alien_alg' in kwargs:
        data = at.remove_aliens(beam_data, kwargs, data_dir)
    else:
        data = beam_data

    if auto:
        # the formula for auto threshold was found empirically, may be
        # modified in the future if more tests are done
        auto_threshold_value = 0.141 * beam_data[np.nonzero(beam_data)].mean().item() - 3.062
        intensity_threshold = max(2.0, auto_threshold_value)
        print(f'auto intensity threshold: {intensity_threshold}')
    elif 'intensity_threshold' in kwargs:
        intensity_threshold = kwargs['intensity_threshold']
    else:
        print('define amplitude threshold. Exiting')
        return

    # zero out the noise
    data = np.where(data <= intensity_threshold, 0.0, data)

    # square root data
    data = np.sqrt(data)

    if 'adjust_dimensions' in kwargs:
        crops_pads = kwargs['adjust_dimensions']
        # the adjust_dimension parameter list holds adjustment in each direction. Append 0s, if shorter
        if len(crops_pads) < 6:
            for _ in range(6 - len(crops_pads)):
                crops_pads.append(0)
    else:
        # the size still has to be adjusted to the opencl supported dimension
        crops_pads = (0, 0, 0, 0, 0, 0)
    # adjust the size, either pad with 0s or crop array
    pairs = []
    for i in range(int(len(crops_pads) / 2)):
        pair = crops_pads[2 * i:2 * i + 2]
        pairs.append(pair)

    data = ut.adjust_dimensions(data, pairs)
    if data is None:
        print('check "adjust_dimensions" configuration')
        return

    if 'center_shift' in kwargs:
        center_shift = kwargs['center_shift']
        data, shift = ut.center_max(data, center_shift)
    else:
        data, shift = ut.center_max(data, [0, 0, 0])

    try:
        # assuming the mask file is in directory of preprocessed data
        mask = ut.read_tif(beamline_full_datafile_name.replace(beamline_datafile_name, 'mask.tif'))
        mask = np.roll(mask, shift, tuple(range(mask.ndim)))
        ut.save_tif(mask, ut.join(data_dir, 'mask.tif'))
    except FileNotFoundError:
        pass

    # if auto and kwargs['do_auto_binning']:
    #     # prepare data to make the oversampling ratio ~3
    #     wos = 3.0
    #     orig_os = ut.get_oversample_ratio(data)
    #     # match oversampling to wos
    #     wanted_os = [wos, wos, wos]
    #     change = [np.round(f / t).astype('int32') for f, t in zip(orig_os, wanted_os)]
    #     bins = [int(max([1, c])) for c in change]
    #     print(f'auto binning size: {bins}')
    #     data = ut.binning(data, bins)
    if 'binning' in kwargs:
        binsizes = kwargs['binning']
        try:
            bins = []
            for binsize in binsizes:
                bins.append(binsize)
            filler = len(data.shape) - len(bins)
            for _ in range(filler):
                bins.append(1)
            data = ut.binning(data, bins)
            kwargs['binning'] = bins
        except:
            print('check "binning" configuration')

    # save data
    data_file = ut.join(data_dir, 'data.tif')
    ut.save_tif(data, data_file)
    print(f'data ready for reconstruction, data dims: {data.shape}')

    # if auto save new config
    if auto:
        kwargs['intensity_threshold'] = intensity_threshold
    return kwargs
