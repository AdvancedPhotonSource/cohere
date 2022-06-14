# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere.viz_util
===============

This module is a suite of utility functions supporting visualization.
It supports 3D visualization.
"""

import numpy as np
import scipy.ndimage as ndi
import math as m
import cohere.utilities.utils as ut


__author__ = "Ross Harder"
__copyright__ = "Copyright (c) 2016, UChicago Argonne, LLC."
__docformat__ = 'restructuredtext en'
__all__ = ['shift',
           'crop_center',
           'get_zero_padded_centered',
           'remove_ramp',
           'center']


def shift(arr, s0, s1, s2):
    """
    Shifts-rolls 3D array.
    Parameters
    ----------
    arr : ndarray
        array to shift
    s0, s1, s2 : int, int, int
        shift in each dimension
    Returns
    -------
    ndarray
        shifted array
    """
    shifted = np.roll(arr, s0, axis=0)
    shifted = np.roll(shifted, s1, axis=1)
    return np.roll(shifted, s2, axis=2)


def crop_center(arr, new_size):
    """
    This function crops the array to the new size, leaving the array in the center.
    The dimensions of the new array is then adjusted to be supported by the opencl library.

    Parameters
    ----------
    arr : ndarray
        the array to crop

    new_size : tuple
        new size

    Returns
    -------
    cropped : ndarray
        the cropped array
    """
    size = arr.shape
    cropped = arr
    for i in range(len(size)):
        crop_front = int((size[i] - new_size[i]) / 2)
        crop_end = crop_front + new_size[i]
        splitted = np.split(cropped, [crop_front, crop_end], axis=i)
        cropped = splitted[1]

    return cropped


def get_zero_padded_centered(arr, new_shape):
    """
    This function pads the array with zeros to the new shape with the array in the center.

    Parameters
    ----------
    arr : ndarray
        the original array to be padded

    new_shape : tuple
        new dimensions

    Returns
    -------
    centered : ndarray
        the zero padded centered array
    """
    shape = arr.shape
    pad = []
    c_vals = []
    for i in range(len(new_shape)):
        pad.append((0, new_shape[i] - shape[i]))
        c_vals.append((0.0, 0.0))
    arr = np.lib.pad(arr, (pad), 'constant', constant_values=c_vals)

    centered = arr
    for i in range(len(new_shape)):
        centered = np.roll(centered, int((new_shape[i] - shape[i] + 1) / 2), i)

    return centered


# @measure
def remove_ramp(arr, ups=1):
    """
    Smooths image array (removes ramp) by applaying math formula.
    Parameters
    ----------
    arr : ndarray
        array to remove ramp
    ups : int
        upsample factor
    Returns
    -------
    ramp_removed : ndarray
        smoothed array
    """
    new_shape = list(arr.shape)
    # pad zeros around arr, to the size of 3 times (ups = 3) of arr size
    for i in range(len(new_shape)):
        new_shape[i] = ups * new_shape[i]
    padded = get_zero_padded_centered(arr, new_shape)
    padded_f = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(padded)))
    com = ndi.center_of_mass(np.power(np.abs(padded_f), 2))
    sub_pixel_shifted = ut.sub_pixel_shift(padded_f, new_shape[0] / 2.0 - com[0], new_shape[1] / 2.0 - com[1],
                                           new_shape[2] / 2.0 - com[2])
    ramp_removed_padded = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(sub_pixel_shifted)))
    ramp_removed = crop_center(ramp_removed_padded, arr.shape)

    return ramp_removed


def center(image, support):
    """
    Shifts the image and support arrays so the center of mass is in the center of array.
    Parameters
    ----------
    image, support : ndarray, ndarray
        image and support arrays to evaluate and shift
    Returns
    -------
    image, support : ndarray, ndarray
        shifted arrays
    """
    dims = image.shape
    image, support = ut.get_centered_both(image, support)

    # place center of mass image*support in the center
    for ax in range(len(dims)):
        com = ndi.center_of_mass(np.absolute(image) * support)
        image = shift(image, int(dims[0] / 2 - com[0]), int(dims[1] / 2 - com[1]), int(dims[2] / 2 - com[2]))
        support = shift(support, int(dims[0] / 2 - com[0]), int(dims[1] / 2 - com[1]), int(dims[2] / 2 - com[2]))

    # set center phase to zero, use as a reference
    phi0 = m.atan2(image.imag[int(dims[0] / 2), int(dims[1] / 2), int(dims[2] / 2)],
                   image.real[int(dims[0] / 2), int(dims[1] / 2), int(dims[2] / 2)])
    image = image * np.exp(-1j * phi0)

    return image, support
