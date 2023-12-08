import math

_docformat__ = 'restructuredtext en'
__all__ = ['set_lib',
           'arr_property',
           'pad_around',
           'center_sync',
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
           'breed'
           ]


def set_lib(dlib):
    global dvclib
    dvclib = dlib


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
    print('norm', dvclib.sum(pow(abs(arr), 2)))
    print('mean across all data', arr1.mean())
    print('std all data', arr1.std())
    print('sum',np.sum(arr))
    # print('mean non zeros only', arr1[np.nonzero(arr1)].mean())
    # print('std non zeros only', arr1[np.nonzero(arr1)].std())
    max_coordinates = list(dvclib.unravel_index(dvclib.argmax(arr1), arr.shape))
    print('max coords, value', max_coordinates, arr[max_coordinates[0], max_coordinates[1], max_coordinates[2]])


def pad_around(arr, shape, val=0):
    padded = dvclib.full(shape, val)
    dims = dvclib.dims(arr)
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
    import numpy as np

    # set center phase to zero, use as a reference
    phi0 = math.atan2(image.flatten().imag[int(image.flatten().shape[0] / 2)],
                   image.flatten().real[int(image.flatten().shape[0] / 2)])
    image = image * dvclib.exp(-1j * phi0)

    com = dvclib.center_of_mass(dvclib.astype(support, dtype=np.int32))
    # place center of mass in the center
    sft = [shape[i] / 2.0 - com[i] + .5  for i in range(len(shape))]
    image = dvclib.roll(image, sft)
    support =dvclib.roll(support, sft)

    return image, support


def gauss_conv_fft(arr, distribution):
    dims = dvclib.dims(arr)
    arr_sum = dvclib.sum(arr)
    arr_f = dvclib.ifftshift(dvclib.fft(dvclib.ifftshift(arr)))
    convag = arr_f * distribution
    convag = dvclib.ifftshift(dvclib.ifft(dvclib.ifftshift(convag)))
    convag = dvclib.real(convag)
    convag = dvclib.where(convag > 0, convag, 0.0)
    correction = arr_sum / dvclib.sum(convag)
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
        support is formed by points above this valuue

    sigmas : list
        sigmas in all dimensions
    Returns
    -------
    support : ndarray
        support array
    """
    filter = dvclib.gaussian_filter(dvclib.absolute(arr), sigma)
    max_filter = dvclib.amax(filter)
    adj_threshold = max_filter * threshold
    support = dvclib.full(dvclib.dims(filter), 1)
    support = dvclib.where(filter >= adj_threshold, support, 0)
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
    ph = dvclib.angle(arr)
    support = shrink_wrap(dvclib.absolute(arr), .2, .5)  # get just the crystal, i.e very tight support
    avg_ph = dvclib.sum(ph * support) / dvclib.sum(support)
    ph = ph - avg_ph + val
    return ph


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
    return dvclib.absolute(arr) * dvclib.exp(1j * ph)


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
    F = dvclib.ifftshift(dvclib.fft(dvclib.fftshift(arr)))
    return dvclib.ifftshift(dvclib.ifft(dvclib.fftshift(dvclib.conj(F))))


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
    A = dvclib.ifftshift(dvclib.fft(dvclib.fftshift(conj_reflect(a))))
    B = dvclib.ifftshift(dvclib.fft(dvclib.fftshift(b)))
    CC = A * B
    return dvclib.ifftshift(dvclib.ifft(dvclib.fftshift(CC)))


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
    if dvclib.amax(cc1) > dvclib.amax(cc2):
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
        eval = dvclib.sum(dvclib.square(dvclib.square(dvclib.absolute(image))))
    elif metric_type == 'summed_phase':
        ph = shift_phase(image, 0)
        eval = dvclib.sum(abs(ph))
    elif metric_type == 'area':
        eval = dvclib.sum(shrink_wrap(image, .2, .5))

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
    eval = dvclib.sum(dvclib.square(dvclib.square(dvclib.absolute(image))))
    if type(eval) != float:
        eval = eval.item()
    metrics['sharpness'] = eval
    eval = dvclib.sum(abs(shift_phase(image, 0)))
    if type(eval) != float:
        eval = eval.item()
    metrics['summed_phase'] = eval
    eval = dvclib.sum(shrink_wrap(image, .2, .5))
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
    [nr, nc] = dvclib.dims(arr)
    if nor < 0:
        nor = nr
    if noc < 0:
        noc = nc

    # Compute kernels and obtain DFT by matrix products
    yl = list(range(-int(math.floor(nc / 2)), nc - int(math.floor(nc / 2))))
    y = dvclib.ifftshift(dvclib.array(yl)) * (-2j * math.pi / (nc * usfac))
    xl = list(range(-coff, noc - coff))
    x = dvclib.array(xl)
    yt = dvclib.tile(y, (len(xl), 1))
    xt = dvclib.tile(x, (len(yl), 1))
    kernc = dvclib.exp(yt.T * xt)

    yl = list(range(-roff, nor - roff))
    y = dvclib.array(yl)
    xl = list(range(-int(math.floor(nr / 2)), nr - int(math.floor(nr / 2))))
    x = dvclib.ifftshift(dvclib.array(xl))
    yt = dvclib.tile(y, (len(xl), 1))
    xt = dvclib.tile(x, (len(yl), 1))
    kernr = dvclib.exp(yt * xt.T * (-2j * math.pi / (nr * usfac)))

    # return as device array
    return dvclib.dot(dvclib.dot(kernr.T, arr), kernc)


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
    c_c = pad_around(dvclib.fftshift(ref_arr) * dvclib.conj(dvclib.fftshift(arr)), large_shape, val=0j)

    # Compute crosscorrelation and locate the peak
    c_c = dvclib.ifft(dvclib.ifftshift(c_c))
    max_coord = dvclib.unravel_index(dvclib.argmax(c_c), dvclib.dims(c_c))

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
        row_shift = dvclib.round(row_shift * usfac) / usfac
        col_shift = dvclib.round(col_shift * usfac) / usfac
        dftshift = dvclib.fix(dvclib.ceil(usfac * 1.5) / 2)  # Center of output array at dftshift
        # Matrix multiply DFT around the current shift estimate
        c_c = dvclib.conj(dftups(arr * dvclib.conj(ref_arr), int(math.ceil(usfac * 1.5)), int(math.ceil(usfac * 1.5)), usfac,
                             int(dftshift - row_shift * usfac), int(dftshift - col_shift * usfac))) / \
              (int(math.trunc(shape[0] / 2)) * int(math.trunc(shape[1] / 2)) * usfac ^ 2)
        # Locate maximum and map back to original pixel grid
        max_coord = dvclib.unravel_index(dvclib.argmax(c_c), dvclib.dims(c_c))
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
    r_shift_2, c_shift_2 = dftregistration(dvclib.fft(dvclib.sum(ref_arr, 2)), dvclib.fft(dvclib.sum(arr, 2)), 4)
    r_shift_1, c_shift_1 = dftregistration(dvclib.fft(dvclib.sum(ref_arr, 1)), dvclib.fft(dvclib.sum(arr, 1)), 4)
    r_shift_0, c_shift_0 = dftregistration(dvclib.fft(dvclib.sum(ref_arr, 0)), dvclib.fft(dvclib.sum(arr, 0)), 4)

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
    buf2ft = dvclib.fft(arr)
    shape = arr.shape
    Nr = dvclib.ifftshift(dvclib.array(list(range(-int(math.floor(shape[0] / 2)), shape[0] - int(math.floor(shape[0] / 2))))))
    Nc = dvclib.ifftshift(dvclib.array(list(range(-int(math.floor(shape[1] / 2)), shape[1] - int(math.floor(shape[1] / 2))))))
    Nz = dvclib.ifftshift(dvclib.array(list(range(-int(math.floor(shape[2] / 2)), shape[2] - int(math.floor(shape[2] / 2))))))
    [Nc, Nr, Nz] = dvclib.meshgrid(Nc, Nr, Nz)
    mg = 1j * 2 * math.pi * (-row_shift * Nr / shape[0] - col_shift * Nc / shape[1] - z_shift * Nz / shape[2])
    # return as array on device
    Greg = buf2ft * dvclib.exp(mg)
    return dvclib.ifft(Greg)


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
    padded_f = dvclib.fftshift(dvclib.fft(dvclib.ifftshift(padded)))
    com = dvclib.center_of_mass(dvclib.square(dvclib.absolute(padded_f)))
    sft = [new_shape[i] / 2.0 - com[i] + .5 for i in range(len(shape))]
    sub_pixel_shifted = sub_pixel_shift(padded_f, *sft)
    ramp_removed_padded = dvclib.fftshift(dvclib.ifft(dvclib.fftshift(sub_pixel_shifted)))
    ramp_removed = ut.crop_center(ramp_removed_padded, arr.shape)

    return ramp_removed


# supposedly this is faster than np.roll or scipy interpolation shift.
# https://stackoverflow.com/questions/30399534/shift-elements-in-a-numpy-array
def fast_shift(arr, shifty, fill_val=0):
    dims = arr.shape
    result = dvclib.full(dims, 1.0)
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
    CC = dvclib.correlate(ref, arr, mode='same', method='fft')
    CC_shifted = dvclib.ifftshift(CC)
    shape = dvclib.array(CC_shifted.shape)
    amp = dvclib.absolute(CC_shifted)
    shift = dvclib.unravel_index(amp.argmax(), shape)
    if dvclib.sum(shift) == 0:
        return arr
    intshift = dvclib.array(shift)
    pixelshift = dvclib.where(intshift >= shape / 2, intshift - shape, intshift)
    shifted_arr = fast_shift(arr, pixelshift)
    return shifted_arr


def correlation_err(refarr, arr):
    CC = dvclib.correlate(refarr, arr, mode='same', method='fft')
    CCmax = CC.max()
    rfzero = dvclib.sum(dvclib.absolute(refarr) ** 2)
    rgzero = dvclib.sum(dvclib.absolute(arr) ** 2)
    e = abs(1 - CCmax * dvclib.conj(CCmax) / (rfzero * rgzero)) ** 0.5
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
    c_c = dvclib.conj(arr1) * arr2
    c_c_tot = dvclib.sum(c_c)
    ph = dvclib.angle(c_c_tot)
    arr = arr1 * dvclib.exp(1j * ph)
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
    alpha = dvclib.load(alpha_dir + '/image.npy')
    if dvclib.array_equal(image, alpha):
        # it is alpha, no breeding
        return zero_phase(image)
    alpha = zero_phase(alpha)

    # load image file
    beta = image
    beta = zero_phase(beta)
    alpha = check_get_conj_reflect(beta, alpha)
    alpha = align_arrays_subpixel(beta, alpha)
    alpha = zero_phase(alpha)
    ph_alpha = dvclib.angle(alpha)
    beta = zero_phase_cc(beta, alpha)
    ph_beta = dvclib.angle(beta)
    if breed_mode == 'sqrt_ab':
        beta = dvclib.sqrt(dvclib.absolute(alpha) * dvclib.absolute(beta)) * dvclib.exp(0.5j * (ph_beta + ph_alpha))

    elif breed_mode == 'pixel_switch':
        cond = dvclib.random(beta.shape)
        beta = dvclib.where((cond > 0.5), beta, alpha)

    elif breed_mode == 'b_pa':
        beta = dvclib.absolute(beta) * dvclib.exp(1j * ph_alpha)

    elif breed_mode == '2ab_a_b':
        beta = 2 * (beta * alpha) / (beta + alpha)

    elif breed_mode == '2a_b_pa':
        beta = (2 * dvclib.absolute(alpha) - dvclib.absolute(beta)) * dvclib.exp(1j * ph_alpha)

    elif breed_mode == 'sqrt_ab_pa':
        beta = dvclib.sqrt(dvclib.absolute(alpha) * dvclib.absolute(beta)) * dvclib.exp(1j * ph_alpha)

    elif breed_mode == 'sqrt_ab_recip':
        temp1 = dvclib.fftshift(dvclib.fft(dvclib.fftshift(beta)))
        temp2 = dvclib.fftshift(dvclib.fft(dvclib.fftshift(alpha)))
        temp = dvclib.sqrt(dvclib.absolute(temp1) * dvclib.absolute(temp2)) * dvclib.exp(.5j * dvclib.angle(temp1)) * dvclib.exp(.5j * dvclib.angle(temp2))
        beta = dvclib.fftshift(dvclib.ifft(dvclib.fftshift(temp)))

    elif breed_mode == 'max_ab':
        beta = dvclib.maximum(dvclib.absolute(alpha), dvclib.absolute(beta)) * dvclib.exp(.5j * (ph_beta + ph_alpha))

    elif breed_mode == 'max_ab_pa':
        beta = dvclib.maximum(dvclib.absolute(alpha), dvclib.absolute(beta)) * dvclib.exp(1j * ph_alpha)

    elif breed_mode == 'avg_ab':
        beta = 0.5 * (alpha + beta)

    elif breed_mode == 'avg_ab_pa':
        beta = 0.5 * (dvclib.absolute(alpha) + dvclib.absolute(beta)) * dvclib.exp(1j * ph_alpha)

    return beta
