import math
import cmath
import cohere_core.utilities.utils as ut
from matplotlib import pyplot as plt


_docformat__ = 'restructuredtext en'
__all__ = ['set_lib_from_pkg',
           'arr_property',
           'pad_around',
           'center_sync',
           'get_norm',
           'gauss_conv_fft',
           'shrink_wrap',
           'shift_phase',
           'zero_phase',
           'conj_reflect',
           'cross_correlation',
           'check_get_conj_reflect',
           'get_metric',
           'all_metrics',
           'dftups',
           'dftregistration',
           'register_3d_reconstruction',
           'sub_pixel_shift',
           'align_arrays_subpixel',
           'align_arrays_pixel',
           'fast_shift',
           'correlation_err',
           'remove_ramp',
           'zero_phase_cc',
           'breed',
           'resample',
           'pad_to_cube',
           ]


def set_lib_from_pkg(pkg):
    global devlib

    # get the lib object
    devlib = ut.get_lib(pkg)


def arr_property(arr):
    """
    Used only in development. Prints max value of the array and max coordinates.

    Parameters
    ----------
    arr : ndarray
        array to find max
    """
    import numpy as np
    arr1 = abs(arr)
    print('norm', devlib.sum(pow(abs(arr), 2)))
    print('mean across all data', arr1.mean())
    print('std all data', arr1.std())
    print('sum',np.sum(arr))
    # print('mean non zeros only', arr1[np.nonzero(arr1)].mean())
    # print('std non zeros only', arr1[np.nonzero(arr1)].std())
    max_coordinates = list(devlib.unravel_index(devlib.argmax(arr1), arr.shape))
    print('max coords, value', max_coordinates, arr[max_coordinates[0], max_coordinates[1], max_coordinates[2]])


def pad_around(arr, shape, val=0):
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


def center_sync(image, support):
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
    shape = image.shape

    # set center phase to zero, use as a reference
    phi0 = math.atan2(image.flatten().imag[int(image.flatten().shape[0] / 2)],
                   image.flatten().real[int(image.flatten().shape[0] / 2)])
    image = image * cmath.exp(-1j * phi0)

    com = devlib.center_of_mass(devlib.absolute(image))
    # place center of mass in the center
    sft = [e[0] / 2.0 - e[1] + .5 for e in zip(shape, com)]

    image = devlib.roll(image, sft)
    support =devlib.roll(support, sft)


    return image, support


def get_norm(arr):
    return devlib.sum(devlib.square(devlib.absolute(arr)))


def gauss_conv_fft(arr, distribution):
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


def shrink_wrap(arr, threshold, sigma):
    """
    Calculates support array.

    Parameters
    ----------
    arr : ndarray
        subject array

    threshold : float
        support is formed by points above this value
    sigma: float

    sigmas : list
        sigmas in all dimensions
    Returns
    -------
    support : ndarray
        support array
    """
    filter = devlib.gaussian_filter(devlib.absolute(arr), sigma)
    max_filter = devlib.amax(filter)
    adj_threshold = max_filter * threshold
    support = devlib.where(filter >= adj_threshold, 1, 0)
    return support


def shift_phase(arr, val=0):
    """
    Adjusts phase accross the array so the relative phase stays intact, but it shifts towards given value.

    Parameters
    ----------
    arr : ndarray
        array to adjust phase

    val : float
        a value to adjust the phase toward

    Returns
    -------
    ndarray
        input array with adjusted phase
    """
    ph = devlib.angle(arr)
    support = shrink_wrap(devlib.absolute(arr), .2, .5)  # get just the crystal, i.e very tight support
    avg_ph = devlib.sum(ph * support) / devlib.sum(support)
    return arr * devlib.exp(-1j * (avg_ph - val))


def zero_phase(arr):
    """
    Adjusts phase accross the array so the relative phase stays intact, but it shifts towards given value.

    Parameters
    ----------
    arr : ndarray
        array to adjust phase

    val : float
        a value to adjust the phase toward

    Returns
    -------
    ndarray
        input array with adjusted phase
    """
    ph = shift_phase(arr, 0)
    return devlib.absolute(arr) * devlib.exp(1j * ph)


def conj_reflect(arr):
    """
    This function computes conjugate reflection of array.
    Parameters
    ----------
    a : ndarray
        input array
    Returns
    -------
    ndarray
        conjugate reflection array
    """
    F = devlib.ifftshift(devlib.fft(devlib.fftshift(arr)))
    return devlib.ifftshift(devlib.ifft(devlib.fftshift(devlib.conj(F))))


def cross_correlation(a, b):
    """
    This function computes cross correlation of two arrays.
    Parameters
    ----------
    a : ndarray
        input array for cross correlation
    b : ndarray
        input array for cross correlation
    Returns
    -------
    ndarray
        cross correlation array
    """
    A = devlib.ifftshift(devlib.fft(devlib.fftshift(conj_reflect(a))))
    B = devlib.ifftshift(devlib.fft(devlib.fftshift(b)))
    CC = A * B
    return devlib.ifftshift(devlib.ifft(devlib.fftshift(CC)))


def check_get_conj_reflect(arr1, arr2):
    """
    It returns the array of interest or conjugate reflection of that array depending whether it is reflection of the
    reference array.
    Parameters
    ----------
    arr1 : ndarray
        reference array
    arr2 : ndarray
        array of interest
    Returns
    -------
    ndarray
        arr2 or conjugate reflection of it
    """
    support1 = shrink_wrap(abs(arr1), .1, .1)
    support2 = shrink_wrap(abs(arr2), .1, .1)
    cc1 = cross_correlation(support1, shrink_wrap(conj_reflect(arr2), .1, .1))
    cc2 = cross_correlation(support1, support2)
    if devlib.amax(cc1) > devlib.amax(cc2):
        return conj_reflect(arr2)
    else:
        return arr2


def get_metric(image, errs, metric_type):
    """
    Callculates array characteristic based on various formulas.

    Parameters
    ----------
    arr : ndarray
        array to get characteristic
    errs : list
        list of "chi" error by iteration
    metric_type : str

    Returns
    -------
    metric : float
        calculated metric for a given type
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


def all_metrics(image, errs):
    """
    Callculates array characteristic based on various formulas.

    Parameters
    ----------
    arr : ndarray
        array to get characteristic

    Returns
    -------
    metrics : dict
        calculated metrics for all types
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


def dftups(arr, nor=-1, noc=-1, usfac=2, roff=0, coff=0):
    """
    Upsample DFT by array multiplies.

    Parameters
    ----------
    arr : ndarray
        the subject of upsampling array
    nor, noc : int, int
        Number of pixels in the output upsampled DFT, in units of upsampled pixels

    usfac : int
        Upsampling factor (default usfac = 1)

    roff, coff : int, int
        Row and column offsets, allow to shift the output array to a region of interest on the DFT
    Returns
    -------
    ndarray
        upsampled array
    """
    # arr is 2D
    [nr, nc] = devlib.dims(arr)
    if nor < 0:
        nor = nr
    if noc < 0:
        noc = nc

    # Compute kernels and obtain DFT by matrix products
    yl = list(range(-int(math.floor(nc / 2)), nc - int(math.floor(nc / 2))))
    y = devlib.ifftshift(devlib.array(yl)) * (-2j * math.pi / (nc * usfac))
    xl = list(range(-coff, noc - coff))
    x = devlib.array(xl)
    yt = devlib.tile(y, (len(xl), 1))
    xt = devlib.tile(x, (len(yl), 1))
    kernc = devlib.exp(yt.T * xt)

    yl = list(range(-roff, nor - roff))
    y = devlib.array(yl)
    xl = list(range(-int(math.floor(nr / 2)), nr - int(math.floor(nr / 2))))
    x = devlib.ifftshift(devlib.array(xl))
    yt = devlib.tile(y, (len(xl), 1))
    xt = devlib.tile(x, (len(yl), 1))
    kernr = devlib.exp(yt * xt.T * (-2j * math.pi / (nr * usfac)))

    # return as device array
    return devlib.dot(devlib.dot(kernr.T, arr), kernc)


def dftregistration(ref_arr, arr, usfac=2):
    """
    Efficient subpixel image registration by crosscorrelation. Based on Matlab dftregistration by Manuel Guizar (Portions of this code were taken from code written by Ann M. Kowalczyk and James R. Fienup.)

    Parameters
    ----------
    ref_arr : 2D ndarray
        Fourier transform of reference image
    arr : 2D ndarray
        Fourier transform of image to register

    usfac : int
        Upsampling factor

    Returns
    -------
    row_shift, col_shift : float, float
        pixel shifts between images
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
    max_coord = devlib.unravel_index(devlib.argmax(c_c), devlib.dims(c_c))

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
        row_shift = devlib.round(row_shift * usfac) / usfac
        col_shift = devlib.round(col_shift * usfac) / usfac
        dftshift = devlib.fix(devlib.ceil(usfac * 1.5) / 2)  # Center of output array at dftshift
        # Matrix multiply DFT around the current shift estimate
        c_c = devlib.conj(dftups(arr * devlib.conj(ref_arr), int(math.ceil(usfac * 1.5)), int(math.ceil(usfac * 1.5)), usfac,
                             int(dftshift - row_shift * usfac), int(dftshift - col_shift * usfac))) / \
              (int(math.trunc(shape[0] / 2)) * int(math.trunc(shape[1] / 2)) * usfac ^ 2)
        # Locate maximum and map back to original pixel grid
        max_coord = devlib.unravel_index(devlib.argmax(c_c), devlib.dims(c_c))
        [rloc, cloc] = max_coord

        rloc = rloc - dftshift
        cloc = cloc - dftshift
        row_shift = row_shift + rloc / usfac
        col_shift = col_shift + cloc / usfac

    return row_shift, col_shift


def register_3d_reconstruction(ref_arr, arr):
    """
    Finds pixel shifts between reconstructed images

    Parameters
    ----------
    ref_arr : ndarray
        Fourier transform of reference image
    arr : ndarray
        Fourier transform of image to register

    Returns
    -------
    shift_2, shift_1, shift_0 : float, float
        pixel shifts between images
    """
    r_shift_2, c_shift_2 = dftregistration(devlib.fft(devlib.sum(ref_arr, 2)), devlib.fft(devlib.sum(arr, 2)), 4)
    r_shift_1, c_shift_1 = dftregistration(devlib.fft(devlib.sum(ref_arr, 1)), devlib.fft(devlib.sum(arr, 1)), 4)
    r_shift_0, c_shift_0 = dftregistration(devlib.fft(devlib.sum(ref_arr, 0)), devlib.fft(devlib.sum(arr, 0)), 4)

    shift_2 = sum([r_shift_2, r_shift_1]) * 0.5
    shift_1 = sum([c_shift_2, r_shift_0]) * 0.5
    shift_0 = sum([c_shift_1, c_shift_0]) * 0.5
    return shift_2, shift_1, shift_0


def sub_pixel_shift(arr, row_shift, col_shift, z_shift):
    """
    Shifts pixels in a regularly sampled LR image with a subpixel precision according to local gradient.


    Parameters
    ----------
    arr : ndarray
        array to shift

    row_shift, col_shift, z_shift : float, float, float
        shift in each dimension

    Returns
    -------
    ndarray
        shifted array
    """
    buf2ft = devlib.fft(arr)
    shape = arr.shape
    Nr = devlib.ifftshift(devlib.array(list(range(-int(math.floor(shape[0] / 2)), shape[0] - int(math.floor(shape[0] / 2))))))
    Nc = devlib.ifftshift(devlib.array(list(range(-int(math.floor(shape[1] / 2)), shape[1] - int(math.floor(shape[1] / 2))))))
    Nz = devlib.ifftshift(devlib.array(list(range(-int(math.floor(shape[2] / 2)), shape[2] - int(math.floor(shape[2] / 2))))))
    [Nc, Nr, Nz] = devlib.meshgrid(Nc, Nr, Nz)
    mg = 1j * 2 * math.pi * (-row_shift * Nr / shape[0] - col_shift * Nc / shape[1] - z_shift * Nz / shape[2])
    # return as array on device
    Greg = buf2ft * devlib.exp(mg)
    return devlib.ifft(Greg)


def align_arrays_subpixel(ref_arr, arr):
    """
    Shifts array to align with reference array.

    Parameters
    ----------
    ref_arr : ndarray
        reference array

    arr : ndarray
        array to shift

    Returns
    -------
    arr : ndarray
        shifted input array, aligned with reference array
    """
    (shift_2, shift_1, shift_0) = register_3d_reconstruction(abs(ref_arr), abs(arr))
    return sub_pixel_shift(arr, shift_2, shift_1, shift_0)


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


# supposedly this is faster than np.roll or scipy interpolation shift.
# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def fast_shift(arr, shifty, fill_val=0):
    """
    Shifts array by the shifty parameter.
    Parameters
    ----------
    arr : ndarray
        array to shift
    shifty : ndarray
        an array of integers/shifts in each dimension
    fill_val : float

    Returns
    -------
    ndarray, shifted array
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


def align_arrays_pixel(ref, arr):
    """
    Aligns two arrays of the same dimensions with the pixel resolution. Used in reciprocal space.
    Parameters
    ----------
    ref : ndarray
        array to align with
    arr : ndarray
        array to align

    Returns
    -------
    ndarray : aligned array
    """
    CC = devlib.correlate(ref, arr, mode='same', method='fft')
    err = correlation_err(ref, arr, CC)
    CC_shifted = devlib.ifftshift(CC)
    shape = devlib.array(CC_shifted.shape)
    amp = devlib.absolute(CC_shifted)
    shift = devlib.unravel_index(amp.argmax(), shape)
    if devlib.sum(shift) == 0:
        return [arr, err]
    intshift = devlib.array(shift)
    pixelshift = devlib.where(intshift >= shape / 2, intshift - shape, intshift)
    shifted_arr = fast_shift(arr, pixelshift)
    return [shifted_arr, err]


def correlation_err(refarr, arr, CC=None):
    """
    author: Paul Frosik
    The method is based on "Invariant error metrics for image reconstruction" by J. R. Fienup.
    Returns correlation error between two arrays.

    Parameters
    ----------
    refarr : ndarray
        reference array
    arr : ndarray
        array to measure likeness with reference array

    Returns
    -------
    float, correlation error
    """
    if CC is None:
        CC = devlib.correlate(refarr, arr, mode='same', method='fft')
    CCmax = CC.max()
    rfzero = devlib.sum(devlib.absolute(refarr) ** 2)
    rgzero = devlib.sum(devlib.absolute(arr) ** 2)
    e = abs(1 - CCmax * devlib.conj(CCmax) / (rfzero * rgzero)) ** 0.5
    return  e


def zero_phase_cc(arr1, arr2):
    """
    Sets avarage phase in array to reference array.

    Parameters
    ----------
    arr1 : ndarray
        array to adjust phase

    arr2 : ndarray
        reference array

    Returns
    -------
    arr : ndarray
        input array with adjusted phase
    """
    # will set array1 avg phase to array2
    c_c = devlib.conj(arr1) * arr2
    c_c_tot = devlib.sum(c_c)
    ph = devlib.angle(c_c_tot)
    arr = arr1 * devlib.exp(1j * ph)
    return arr


def breed(breed_mode, alpha_dir, image):
    """
    Aligns the image to breed from with the alpha image, and applies breed formula, to obtain a 'child' image.

    Parameters
    ----------
    alpha : ndarray
        the best image in the generation
    breed_mode : str
        literal defining the breeding process
    dirs : tuple
        a tuple containing two elements: directory where the image to breed from is stored, a 'parent', and a directory where the bred image, a 'child', will be stored.

    """
    # load alpha from alpha dir
    alpha = devlib.load(ut.join(alpha_dir, 'image.npy'))
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
        temp = devlib.sqrt(devlib.absolute(temp1) * devlib.absolute(temp2)) * devlib.exp(.5j * devlib.angle(temp1)) * devlib.exp(.5j * devlib.angle(temp2))
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


def histogram2d(arr1, arr2, n_bins=100, log=False):
    norm = devlib.max(arr1) / devlib.max(arr2)
    if log:
        bins = devlib.logspace(devlib.log10(devlib.amin(arr1[arr1 != 0])), devlib.log10(devlib.amax(arr1)), n_bins + 1)
    else:
        bins = n_bins
    return devlib.histogram2d(devlib.ravel(arr1), devlib.ravel(norm * arr2), bins)


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



def resample(data, matrix, plot=False):
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


def pad_to_cube(data, size):
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

