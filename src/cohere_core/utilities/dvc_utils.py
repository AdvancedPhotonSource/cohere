# #########################################################################
# Copyright (c) , UChicago Argonne, LLC. All rights reserved.             #
#                                                                         #
# See LICENSE file.                                                       #
# #########################################################################

"""
cohere_core.utilities.dvc_utils
===============================

This module is a suite of utility functions. 

The function in this module operate on arrays loaded on device. The array can be in CPU memory or GPU memory, and is in format related to the package used to create the array. 
Therefore, before using functions from this module user must set the global variable devlib by calling function set_lib_from_pkg.
"""
import cmath
import math
from matplotlib import pyplot as plt
import cohere_core.utilities.utils as ut
import numpy as np

_docformat__ = 'restructuredtext en'
__all__ = [
           'align_arrays_pixel',
           'align_arrays_subpixel',
           'all_metrics',
           'arr_property',
           'breed',
           'calc_nmi',
           'calc_ehd',
           'center_sync',
           'check_get_conj_reflect',
           'conj_reflect',
           'correlation_err',
           'cross_correlation',
           'dftregistration',
           'dftups',
           'fast_shift',
           'gauss_conv_fft',
           'get_metric',
           'get_norm',
           'histogram2d',
           'lucy_deconvolution',
           'pad_around',
           'pad_to_cube',
           'register_3d_reconstruction',
           'remove_ramp',
           'resample',
           'set_lib_from_pkg',
           'shift_phase',
           'shrink_wrap',
           'sub_pixel_shift',
           'zero_phase',
           'zero_phase_cc',
           ]


def set_lib_from_pkg(pkg):
    """
    Sets the devlib according to the package specified in input.

    This function must be called before using functions from this module.
    The global variable devlib becomes the library that will be used in this file.
    The library is referencing to cohere_core.lib.cplib, orcohere_core.lib.nplib, orcohere_core.lib.torchlib. 

    :param pkg: acronym specifying library: 'cp' for cupy, 'np' for numpy, 'tprch' for torch
    :return:
    """
    global devlib

    # get the lib object
    devlib = ut.get_lib(pkg)



def align_arrays_pixel(ref, arr):
    """
    Aligns two arrays of the same dimensions with the pixel resolution. Used in reciprocal space.

    :param ref: ndarray, array to align with
    :param arr: ndarray, array to align

    :return: aligned array
    """
    CC = devlib.correlate(ref, arr, mode='same', method='fft')
    err = correlation_err(ref, arr, CC)
    CC_shifted = devlib.ifftshift(CC)
    shape = devlib.array(CC_shifted.shape)
    amp = devlib.absolute(CC_shifted)
    shift = devlib.unravel_index(devlib.argmax(amp), shape)
    if devlib.sum(shift) == 0:
        return [arr, err]
    intshift = devlib.array(shift)
    pixelshift = devlib.where(intshift >= shape / 2, intshift - shape, intshift)
    shifted_arr = fast_shift(arr, pixelshift)
    return [shifted_arr, err]


def align_arrays_subpixel(ref_arr, arr):
    """
    Shifts array to align with reference array.

    :param ref_arr: ndarray, reference array
    :param arr: ndarray, array to shift

    :return: aligned array
    """
    (shift_2, shift_1, shift_0) = register_3d_reconstruction(abs(ref_arr), abs(arr))
    return sub_pixel_shift(arr, shift_2, shift_1, shift_0)


def all_metrics(image, errs):
    """
    Calculates array characteristic based on various formulas and additional input.
    Is used to quntify result of image reconstruction.

    :param image: ndarray, array to get characteristic
    :param errs: list of floats
    :return: dict, calculated metrics for different formulas
    """
    metrics = {}
    eval = errs[-1]
    if type(eval) != float:
        eval = eval.item()
    metrics['chi'] = eval
    eval = devlib.sum(devlib.square(devlib.square(devlib.absolute(image))))
    if type(eval) != float:
        eval = eval.item()
    metrics['sharpness'] = eval
    eval = devlib.sum(abs(shift_phase(image, 0)))
    if type(eval) != float:
        eval = eval.item()
    metrics['summed_phase'] = eval
    eval = devlib.sum(shrink_wrap(image, .2, .5))
    if type(eval) != float:
        eval = eval.item()
    metrics['area'] = eval

    return metrics


def arr_property(arr):
    """
    Used only in development. Prints array statistics.

    :param arr: ndarray, array to find max
    """
    import numpy as np
    arr1 = abs(arr)
    print('norm', devlib.sum(pow(abs(arr), 2)))
    print('mean across all data', arr1.mean())
    print('std all data', arr1.std())
    print('sum', np.sum(arr))
    print('mean non zeros only', arr1[np.nonzero(arr1)].mean())
    print('std non zeros only', arr1[np.nonzero(arr1)].std())
    max_coordinates = list(devlib.unravel_index(devlib.argmax(devlib.absolute(arr1)), arr.shape))
    print('max coords, value', max_coordinates, arr[max_coordinates[0], max_coordinates[1], max_coordinates[2]])


def breed(breed_mode, alpha, image):
    """
    Aligns the image to breed from with the alpha image, and applies breed formula, to obtain a 'child' image.

    :param breed_mode: str, literal defining the breeding process
    :param alpha: array, the best image in the generation
    :param image: array, the image to breed with alpha
    :return: array, the bred child
    """
    if devlib.array_equal(image, alpha):
        # it is alpha, no breeding
        return zero_phase(image)
    alpha = zero_phase(alpha)

    # load image file
    beta = image
    beta = zero_phase(beta)
    alpha = check_get_conj_reflect(beta, alpha)
    alpha = align_arrays_subpixel(beta, alpha)
    alpha = zero_phase(alpha)
    ph_alpha = devlib.angle(alpha)
    beta = zero_phase_cc(beta, alpha)
    ph_beta = devlib.angle(beta)
    if breed_mode == 'sqrt_ab':
        beta = devlib.sqrt(devlib.absolute(alpha) * devlib.absolute(beta)) * devlib.exp(0.5j * (ph_beta + ph_alpha))

    elif breed_mode == 'pixel_switch':
        cond = devlib.random(beta.shape)
        beta = devlib.where((cond > 0.5), beta, alpha)

    elif breed_mode == 'b_pa':
        beta = devlib.absolute(beta) * devlib.exp(1j * ph_alpha)

    elif breed_mode == '2ab_a_b':
        beta = 2 * (beta * alpha) / (beta + alpha)

    elif breed_mode == '2a_b_pa':
        beta = (2 * devlib.absolute(alpha) - devlib.absolute(beta)) * devlib.exp(1j * ph_alpha)

    elif breed_mode == 'sqrt_ab_pa':
        beta = devlib.sqrt(devlib.absolute(alpha) * devlib.absolute(beta)) * devlib.exp(1j * ph_alpha)

    elif breed_mode == 'sqrt_ab_recip':
        temp1 = devlib.fftshift(devlib.fft(devlib.fftshift(beta)))
        temp2 = devlib.fftshift(devlib.fft(devlib.fftshift(alpha)))
        temp = devlib.sqrt(devlib.absolute(temp1) * devlib.absolute(temp2)) * devlib.exp(
            .5j * devlib.angle(temp1)) * devlib.exp(.5j * devlib.angle(temp2))
        beta = devlib.fftshift(devlib.ifft(devlib.fftshift(temp)))

    elif breed_mode == 'max_ab':
        beta = devlib.maximum(devlib.absolute(alpha), devlib.absolute(beta)) * devlib.exp(.5j * (ph_beta + ph_alpha))

    elif breed_mode == 'max_ab_pa':
        beta = devlib.maximum(devlib.absolute(alpha), devlib.absolute(beta)) * devlib.exp(1j * ph_alpha)

    elif breed_mode == 'avg_ab':
        beta = 0.5 * (alpha + beta)

    elif breed_mode == 'avg_ab_pa':
        beta = 0.5 * (devlib.absolute(alpha) + devlib.absolute(beta)) * devlib.exp(1j * ph_alpha)

    return beta


def calc_nmi(arr1, arr2=None, log=False):
    if arr2 is None:
        hgram = arr1
    else:
        hgram = histogram2d(arr1, arr2, log=log)
    h0 = devlib.entropy(devlib.sum(hgram, axis=0))
    h1 = devlib.entropy(devlib.sum(hgram, axis=1))
    h01 = devlib.entropy(devlib.reshape(hgram, -1))
    return (h0 + h1) / h01


def calc_ehd(arr1, arr2=None, log=False):
    if arr2 is None:
        hgram = arr1
    else:
        hgram = histogram2d(arr1, arr2, log=log)
    n = hgram.shape[0]
    x, y = devlib.meshgrid(devlib.arange(n), devlib.arange(n))
    return devlib.sum(hgram * devlib.abs(x - y)) / devlib.sum(hgram)


def center_sync(image, support):
    """
    Shifts the image and support arrays so the center of mass is in the center of array.

    :param image: array, the image to shift
    :param support: array, the support array to shift along with image
    :return: shifted arrays
    """
    shape = image.shape

    # set center phase to zero, use as a reference
    phi0 = math.atan2(image.flatten().imag[int(image.flatten().shape[0] / 2)],
                      image.flatten().real[int(image.flatten().shape[0] / 2)])
    image = image * cmath.exp(-1j * phi0)

    # roll the max to the center to avoid distributing the mass at edges by center of mass
    shift = [int(shape[i]/2 - devlib.unravel_index(devlib.argmax(devlib.absolute(image)), shape)[i]) for i in range(len(shape))]
    image = devlib.roll(image, shift, tuple(range(image.ndim)))
    support = devlib.roll(support, shift, tuple(range(image.ndim)))

    com = devlib.center_of_mass(devlib.absolute(image))
    # place center of mass in the center
    shift = [e[0] / 2.0 - e[1] + .5 for e in zip(shape, com)]

    axis = tuple([i for i in range(len(shape))])
    image = devlib.roll(image, shift, axis=axis)
    support =devlib.roll(support, shift, axis=axis)

    return image, support


def check_get_conj_reflect(arr1, arr2):
    """
    It returns the array of interest or conjugate reflection of that array depending whether it is reflection of the
    reference array.

    :param arr1: array, reference array
    :param arr2: array, array of interest
    :return: arr2 or conjugate reflection of it
    """
    support1 = shrink_wrap(abs(arr1), .1, .1)
    support2 = shrink_wrap(abs(arr2), .1, .1)
    cc1 = cross_correlation(support1, shrink_wrap(conj_reflect(arr2), .1, .1))
    cc2 = cross_correlation(support1, support2)
    if devlib.amax(devlib.absolute(cc1)) > devlib.amax(devlib.absolute(cc2)):
        return conj_reflect(arr2)
    else:
        return arr2


def conj_reflect(arr):
    """
    This function computes conjugate reflection of array.

    :param arr: array, input array
    :return: conjugate reflection array
    """
    F = devlib.ifftshift(devlib.fft(devlib.fftshift(arr)))
    return devlib.ifftshift(devlib.ifft(devlib.fftshift(devlib.conj(F))))


def correlation_err(refarr, arr, CC=None):
    """
    author: Paul Frosik
    The method is based on "Invariant error metrics for image reconstruction" by J. R. Fienup.
    Returns correlation error between two arrays.

    :param refarr: reference array
    :param arr:  array to measure likeness with reference array
    :param CC: cross-correlation, if computed before, defaults to None
    :return: float, correlation error
    """
    if CC is None:
        CC = devlib.correlate(refarr, arr, mode='same', method='fft')
    CCmax = CC.max()
    rfzero = devlib.sum(devlib.absolute(refarr) ** 2)
    rgzero = devlib.sum(devlib.absolute(arr) ** 2)
    e = abs(1 - CCmax * devlib.conj(CCmax) / (rfzero * rgzero)) ** 0.5
    return e


def cross_correlation(a, b):
    """
    This function computes cross correlation of two arrays.

    :param a: input array for cross correlation
    :param b: input array for cross correlation
    :return: cross correlation array
    """
    A = devlib.ifftshift(devlib.fft(devlib.fftshift(conj_reflect(a))))
    B = devlib.ifftshift(devlib.fft(devlib.fftshift(b)))
    CC = A * B
    return devlib.ifftshift(devlib.ifft(devlib.fftshift(CC)))


def dftregistration(ref_arr, arr, usfac=2):
    """
    Efficient subpixel image registration by crosscorrelation. Based on Matlab dftregistration by Manuel Guizar (Portions of this code were taken from code written by Ann M. Kowalczyk and James R. Fienup.)

    :param ref_arr: 2D array, Fourier transform of reference image
    :param arr: 2D array, Fourier transform of image to register
    :param usfac: int, upsampling factor
    :return: pixel shifts between images: (row_shift, col_shift)       
    """
    if usfac < 2:
        print('usfac less than 2 not supported')
        return
    # First upsample by a factor of 2 to obtain initial estimate
    # Embed Fourier data in a 2x larger array
    shape = ref_arr.shape
    large_shape = tuple(2 * x for x in ref_arr.shape)
    c_c = pad_around(devlib.fftshift(ref_arr) * devlib.conj(devlib.fftshift(arr)), large_shape, val=0j)

    # Compute crosscorrelation and locate the peak
    c_c = devlib.ifft(devlib.ifftshift(c_c))
    max_coord = devlib.unravel_index(devlib.argmax(devlib.absolute(c_c)), devlib.dims(c_c))

    if max_coord[0] > shape[0]:
        row_shift = max_coord[0] - large_shape[0]
    else:
        row_shift = max_coord[0]
    if max_coord[1] > shape[1]:
        col_shift = max_coord[1] - large_shape[1]
    else:
        col_shift = max_coord[1]

    row_shift = row_shift / 2
    col_shift = col_shift / 2

    # If upsampling > 2, then refine estimate with matrix multiply DFT
    if usfac > 2:
        # DFT computation
        # Initial shift estimate in upsampled grid
        row_shift = np.round(row_shift * usfac) / usfac
        col_shift = np.round(col_shift * usfac) / usfac
        dftshift = np.fix(np.ceil(usfac * 1.5) / 2)  # Center of output array at dftshift
        # Matrix multiply DFT around the current shift estimate
        # arr and ref_arr are on device
        # move the input to nparray for dftups calculations and convert result to device after
        combined_np = devlib.to_numpy(arr * devlib.conj(ref_arr))

        ups_np = dftups(combined_np, int(math.ceil(usfac * 1.5)), int(math.ceil(usfac * 1.5)), usfac,
                   int(dftshift - row_shift * usfac), int(dftshift - col_shift * usfac))

        ups = devlib.from_numpy(ups_np)
        c_c = devlib.conj(ups) / \
              (int(math.trunc(shape[0] / 2)) * int(math.trunc(shape[1] / 2)) * usfac ^ 2)
        # Locate maximum and map back to original pixel grid

        max_coord = devlib.unravel_index(devlib.argmax(devlib.absolute(c_c)), devlib.dims(c_c))
        [rloc, cloc] = max_coord

        rloc = rloc - dftshift
        cloc = cloc - dftshift
        row_shift = row_shift + rloc / usfac
        col_shift = col_shift + cloc / usfac

    return row_shift, col_shift


def dftups(arr, nor=-1, noc=-1, usfac=2, roff=0, coff=0):
    """
    Upsample DFT by array multiplies.

    :param arr: array, the subject of upsampling
    :param nor: int, number of pixels in the output upsampled DFT, in units of upsampled pixels 
    :param noc: int, number of pixels in the output upsampled DFT, in units of upsampled pixels
    :param usfac: int, upsampling factor (default usfac = 1)
    :param roff: int, row offset, allow to shift the output array to a region of interest on the DFT
    :param coff: int, column offsets, allow to shift the output array to a region of interest on the DFT
    :return: upsampled array
    """
    # arr is 2D
    [nr, nc] = list(arr.shape)
    if nor < 0:
        nor = nr
    if noc < 0:
        noc = nc

    # Compute kernels and obtain DFT by matrix products
    yl = list(range(-int(math.floor(nc / 2)), nc - int(math.floor(nc / 2))))
    y = np.fft.ifftshift(np.array(yl)) * (-2j * math.pi / (nc * usfac))
    xl = list(range(-coff, noc - coff))
    x = np.array(xl)
    yt = np.tile(y, (len(xl), 1))
    xt = np.tile(x, (len(yl), 1))
    kernc = np.exp(yt.T * xt)

    yl = list(range(-roff, nor - roff))
    y = np.array(yl)
    xl = list(range(-int(math.floor(nr / 2)), nr - int(math.floor(nr / 2))))
    x = np.fft.ifftshift(np.array(xl))
    yt = np.tile(y, (len(xl), 1))
    xt = np.tile(x, (len(yl), 1))
    kernr = np.exp(yt * xt.T * (-2j * math.pi / (nr * usfac)))

    return np.dot(np.dot(kernr.T, arr), kernc)


# supposedly this is faster than np.roll or scipy interpolation shift.
# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def fast_shift(arr, shifty, fill_val=0):
    """
    Shifts array by the shifty parameter.

    :param arr:  array to shift
    :param shifty: an array of integers/shifts in each dimension
    :param fill_val: float, value to fill shifted, defaults to 0
    :return: shifted array
    """
    dims = arr.shape
    result = devlib.full(dims, 1.0)
    result *= fill_val
    result_slices = []
    arr_slices = []
    for n in range(len(dims)):
        if shifty[n] > 0:
            result_slices.append(slice(shifty[n], dims[n]))
            arr_slices.append(slice(0, -shifty[n]))
        elif shifty[n] < 0:
            result_slices.append(slice(0, shifty[n]))
            arr_slices.append(slice(-shifty[n], dims[n]))
        else:
            result_slices.append(slice(0, dims[n]))
            arr_slices.append(slice(0, dims[n]))
    result_slices = tuple(result_slices)
    arr_slices = tuple(arr_slices)
    result[result_slices] = arr[arr_slices]
    return result


def gauss_conv_fft(arr, distribution):
    """
    Calculates convolution of array and gaussian distribution.

    :param arr: array
    :param distribution: gaussian distribution array
    :return: convolution array
    """
    dims = devlib.dims(arr)
    arr_sum = devlib.sum(arr)
    arr_f = devlib.ifftshift(devlib.fft(devlib.ifftshift(arr)))
    convag = arr_f * distribution
    convag = devlib.ifftshift(devlib.ifft(devlib.ifftshift(convag)))
    convag = devlib.real(convag)
    convag = devlib.where(convag > 0, convag, 0.0)
    correction = arr_sum / devlib.sum(convag)
    convag *= correction
    return convag


def get_metric(image, errs, metric_type):
    """
    Callculates array quantified characteristic for given metric type. 
    Is used to quntify result of image reconstruction.

    :param arr: array to get the metric
    :param errs: list of "chi" errors by iteration
    :param metric_type: str defining metric type
    :return: calculated metric for a given type
    """
    if metric_type == 'chi':
        eval = errs[-1]
    elif metric_type == 'sharpness':
        eval = devlib.sum(devlib.square(devlib.square(devlib.absolute(image))))
    elif metric_type == 'summed_phase':
        ph = shift_phase(image, 0)
        eval = devlib.sum(abs(ph))
    elif metric_type == 'area':
        eval = devlib.sum(shrink_wrap(image, .2, .5))

    if type(eval) != float:
        eval = eval.item()
    return eval


def get_norm(arr):
    """
    Calculates norm of array.

    :param arr: array to calculate norm for
    :return: float, norm value
    """
    return devlib.sum(devlib.square(devlib.absolute(arr)))


def histogram2d(arr1, arr2, n_bins=100, log=False):
    """
    Returns histogram.

    :param arr1: array
    :param arr2: array
    :param n_bins: number of bins, defaults to 100
    :param log: boolean, if True, return logarhitmic histogram, linear otherwise
    :return: histogram
    """
    norm = devlib.amax(arr1) / devlib.amax(arr2)
    if log:
        bins = devlib.logspace(devlib.log10(devlib.amin(arr1[arr1 != 0])), devlib.log10(devlib.amax(arr1)), n_bins + 1)
    else:
        bins = n_bins
    return devlib.histogram2d(devlib.ravel(arr1), devlib.ravel(norm * arr2), bins)


def lucy_deconvolution(pristine, blurred, kernel, iterations, diffbreak=0):
    """
    Performs Richardson-Lucy deconvolution.

    :param pristine: array, input degraded image
    :param blurred: array, blurred image
    :param kernel: array, point spread function
    :param iterations: number of iteration
    :diffbrake: float, if greater than 0, the iterations will brake if error reaches that value
    :return: deconvolved array
    """
    blurred_mirror = devlib.flip(blurred)
    for i in range(iterations):
        conv = devlib.fftconvolve(kernel, blurred)
        conv = devlib.where(conv == 0.0, 1.0, conv)
        relative_blurr = pristine / conv
        kernel = kernel * devlib.fftconvolve(relative_blurr, blurred_mirror)
        if diffbreak > 0:
            resblurred = devlib.fftconvolve(pristine, kernel)
            err = devlib.absolute(get_norm(resblurred) - get_norm(blurred)) / get_norm(blurred)
            if err < diffbreak:
                break
    kernel = devlib.real(kernel)
    coh_sum = devlib.sum(devlib.absolute(kernel))
    return devlib.absolute(kernel) / coh_sum


def pad_around(arr, shape, val=0):
    """
    Pads an array evenly at each side, to the given shape. It is assumed that the new shape is greater than shape of the array in each dimension.

    :param arr: array to pad
    :param shape: list or tuple, new shape
    :return: padded array
    """
    padded = devlib.full(shape, val)
    dims = devlib.dims(arr)
    principio = []
    finem = []
    for i in range(len(shape)):
        principio.append(int((shape[i] - dims[i]) / 2))
        finem.append(principio[i] + dims[i])
    if len(shape) == 1:
        padded[principio[0]: finem[0]] = arr
    elif len(shape) == 2:
        padded[principio[0]: finem[0], principio[1]: finem[1]] = arr
    elif len(shape) == 3:
        padded[principio[0]: finem[0], principio[1]: finem[1], principio[2]: finem[2]] = arr
    else:
        raise NotImplementedError
    return padded


def pad_to_cube(data, size):
    """
    Pads a given 3D array to the cube of a given size. If any dimension is greater than the given size, the array will be cropped.

    :param data: array to shape into cube
    :param size: int, size of the cube
    :return: the padded/cropped array to the cube
    """
    import numpy as np

    shp = np.array([size, size, size]) // 2
    # Pad the array to the largest dimensions
    shp1 = np.array(data.shape) // 2
    pad = shp - shp1
    pad[pad < 0] = 0
    data = devlib.pad(data, [(pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])])
    # Crop the array to the final dimensions
    shp1 = np.array(data.shape) // 2
    start, end = shp1 - shp, shp1 + shp
    data = data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
    return data


def register_3d_reconstruction(ref_arr, arr):
    """
    Finds pixel shifts between reconstructed images

    :param ref_arr: Fourier transform of reference image
    :param arr: Fourier transform of image to register
    :return: tuple of floats, pixel shifts between images
    """
    r_shift_2, c_shift_2 = dftregistration(devlib.fft(devlib.sum(ref_arr, 2)), devlib.fft(devlib.sum(arr, 2)), 4)
    r_shift_1, c_shift_1 = dftregistration(devlib.fft(devlib.sum(ref_arr, 1)), devlib.fft(devlib.sum(arr, 1)), 4)
    r_shift_0, c_shift_0 = dftregistration(devlib.fft(devlib.sum(ref_arr, 0)), devlib.fft(devlib.sum(arr, 0)), 4)

    shift_2 = sum([r_shift_2, r_shift_1]) * 0.5
    shift_1 = sum([c_shift_2, r_shift_0]) * 0.5
    shift_0 = sum([c_shift_1, c_shift_0]) * 0.5
    return shift_2, shift_1, shift_0


def remove_ramp(arr, ups=1):
    """
    Smooths image array.

    :param arr: array to remove ramp
    :param ups: int, upsample factor, defaults to 1
    :return: smoothed array
    """
    # the remove ramp is called (now) by visualization when arrays are on cpu
    # thus using functions from utilities will be ok for now
    import cohere_core.utilities.utils as ut
    shape = arr.shape
    new_shape = [dim * ups for dim in shape]
    padded = ut.pad_center(arr, new_shape)
    padded_f = devlib.fftshift(devlib.fft(devlib.ifftshift(padded)))
    com = devlib.center_of_mass(devlib.square(devlib.absolute(padded_f)))
    sft = [new_shape[i] / 2.0 - com[i] + .5 for i in range(len(shape))]
    sub_pixel_shifted = sub_pixel_shift(padded_f, *sft)
    ramp_removed_padded = devlib.fftshift(devlib.ifft(devlib.fftshift(sub_pixel_shifted)))
    ramp_removed = ut.crop_center(ramp_removed_padded, arr.shape)

    return ramp_removed


def resample(data, matrix, plot=False):
    """
    Resamples data and creates plot.

    :param data: data array
    :param matrix: matrix array
    :param plot: boolean, will plot if True, defaults to False.
    :return: smoothed array
    """
    import numpy as np
    s = data.shape
    corners = [
        [0, 0, 0],
        [s[0], 0, 0], [0, s[1], 0], [0, 0, s[2]],
        [0, s[1], s[2]], [s[0], 0, s[2]], [s[0], s[1], 0],
        [s[0], s[1], s[2]]
    ]
    new_corners = [np.linalg.inv(matrix)@c for c in corners]
    new_shape = np.ceil(np.max(new_corners, axis=0) - np.min(new_corners, axis=0)).astype(int)
    pad = np.max((new_shape - s) // 2)
    data = devlib.pad(data, np.max([pad, 0]))
    mid_shape = np.array(data.shape)

    offset = (mid_shape / 2) - matrix @ (mid_shape / 2)

    # The phasing data as saved is a magnitude, which is not smooth everywhere (there are non-differentiable points
    # where the complex amplitude crosses zero). However, the intensity (magnitude squared) is a smooth function.
    # Thus, in order to allow a high-order spline, we take the square root of the interpolated square of the image.
    data = devlib.affine_transform(devlib.square(data), devlib.array(matrix), order=1, offset=offset)
    data = devlib.sqrt(devlib.clip(data, 0))
    if plot:
        arr = devlib.to_numpy(data)
        plt.imshow(arr[np.argmax(np.sum(arr, axis=(1, 2)))])
        plt.title(np.linalg.det(matrix))
        plt.show()

    return data


def shift_phase(arr, val=0):
    """
    Adjusts phase accross the array so the relative phase stays intact, but it shifts towards given value.

    :param arr: array to adjust phase
    :param val: float, a value to adjust the phase toward
    :return: input array with adjusted phase
    """
    ph = devlib.angle(arr)
    support = shrink_wrap(devlib.absolute(arr), .2, .5)  # get just the crystal, i.e very tight support
    avg_ph = devlib.sum(ph * support) / devlib.sum(support)
    return arr * devlib.exp(-1j * (avg_ph - val))


def shrink_wrap(arr, threshold, sigma):
    """
    Calculates support array applying gaussian filter to the array. Support is array of 0 and 1 values, determined by threshold applyed to the filtered array.

    :param arr: subject array
    :param threshold: float, support is formed by points above this value
    :param sigma: float, sigma for gaussian filter
    :return: support array
    """
    filter = devlib.gaussian_filter(devlib.absolute(arr), sigma)
    max_filter = devlib.amax(filter)
    adj_threshold = max_filter * threshold
    support = devlib.where(filter >= adj_threshold, 1, 0)
    return support


def sub_pixel_shift(arr, row_shift, col_shift, z_shift):
    """
    Shifts pixels in a regularly sampled LR image with a subpixel precision according to local gradient.

    :param arr: array to shift
    :param row_shift: float, shift in x dimension
    :param col_shift: float, shift in y dimension
    :param z_shift: float, shift in z dimension
    :return: shifted array
    """
    buf2ft = devlib.fft(arr)
    shape = arr.shape
    Nr = devlib.ifftshift(
        devlib.array(list(range(-int(math.floor(shape[0] / 2)), shape[0] - int(math.floor(shape[0] / 2))))))
    Nc = devlib.ifftshift(
        devlib.array(list(range(-int(math.floor(shape[1] / 2)), shape[1] - int(math.floor(shape[1] / 2))))))
    Nz = devlib.ifftshift(
        devlib.array(list(range(-int(math.floor(shape[2] / 2)), shape[2] - int(math.floor(shape[2] / 2))))))
    [Nc, Nr, Nz] = devlib.meshgrid(Nc, Nr, Nz)
    mg = 1j * 2 * math.pi * (-row_shift * Nr / shape[0] - col_shift * Nc / shape[1] - z_shift * Nz / shape[2])
    # return as array on device
    Greg = buf2ft * devlib.exp(mg)
    return devlib.ifft(Greg)


def zero_phase(arr):
    """
    Adjusts phase accross the array so the relative phase stays intact, but it shifts towards zero.

    :param arr: array to adjust phase
    :return: input array with adjusted phase
    """
    ph = shift_phase(arr, 0)
    return devlib.absolute(arr) * devlib.exp(1j * ph)



def zero_phase_cc(arr1, arr2):
    """
    Sets avarage phase in array to reference array.

    :param arr1: array to adjust phase
    :param arr2: reference array
    :return: input array with adjusted phase
    """
    # will set array1 avg phase to array2
    c_c = devlib.conj(arr1) * arr2
    c_c_tot = devlib.sum(c_c)
    ph = devlib.angle(c_c_tot)
    arr = arr1 * devlib.exp(1j * ph)
    return arr
