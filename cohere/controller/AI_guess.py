import numpy as np
import os
import cohere.utilities.utils as ut

import math
from typing import Union

# import tensorflow for trained model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.activations import sigmoid, tanh

def run_AI(data, threshold, sigma, dir):
    print('AI guess')

    print('original data shape', data.shape)
    # prepare data to make the oversampling ratio ~3
    wos = 3.0
    orig_os = get_oversample_ratio(data)
    # match oversampling to wos
    wanted_os = [wos, wos, wos]
    # match diff os
    new_data, inshape = match_oversample_diff(data, orig_os, wanted_os)
    print('processed data shape', new_data.shape)
    new_data = new_data[np.newaxis]

    # load trained network
    model = load_model('trained_model.hdf5',custom_objects={'loss_comb2_scale': loss_comb2_scale})
    print('successfully load the model')

    # get the outputs from amplitude and phase layers
    amp_layer_model = Model(inputs=model.input,
                            outputs=model.get_layer('amp').output)
    ph_layer_model = Model(inputs=model.input,
                        outputs=model.get_layer('phi').output)

    preds_amp = amp_layer_model.predict(new_data, verbose=1)

    preds_phi = ph_layer_model.predict(new_data, verbose=1)

    preds_amp, preds_phi = post_process(preds_amp[0,...,0],
                                    preds_phi[0,...,0],
                                    th=0.1,
                                    uw=0)

    pred_obj = preds_amp * np.exp(1j * preds_phi)
    
    # match object size with the input data
    pred_obj = Resize(pred_obj, inshape)
    guess = crop_pad(pred_obj, data.shape)
    print('initial guess shape', guess.shape)

    np.save(os.path.join(dir, 'image.npy'), guess)

    support = ut.shrink_wrap(guess, threshold, sigma)
    np.save(os.path.join(dir, 'support.npy'), support)

def threshold_by_edge(fp: np.ndarray) -> np.ndarray:
    # threshold by left edge value
    mask = np.ones_like(fp, dtype=bool)
    mask[tuple([slice(1, None)] * fp.ndim)] = 0
    zero = 1e-6
    cut = np.max(fp[mask])
    binary = np.zeros_like(fp)
    binary[(np.abs(fp) > zero) & (fp > cut)] = 1
    return binary

def select_central_object(fp: np.ndarray) -> np.ndarray:
    import scipy.ndimage as ndimage
    zero = 1e-6
    binary = np.abs(fp)
    binary[binary > zero] = 1
    binary[binary <= zero] = 0

    # cluster by connectivity
    struct = ndimage.morphology.generate_binary_structure(fp.ndim,
                                                          1).astype("uint8")
    label, nlabel = ndimage.label(binary, structure=struct)

    # select largest cluster
    select = np.argmax(np.bincount(np.ravel(label))[1:]) + 1

    binary[label != select] = 0

    fp[binary == 0] = 0
    return fp

def get_central_object_extent(fp: np.ndarray) -> list:
    fp_cut = threshold_by_edge(np.abs(fp))
    need = select_central_object(fp_cut)

    # get extend of cluster
    extent = [np.max(s) + 1 - np.min(s) for s in np.nonzero(need)]
    return extent

def get_oversample_ratio(fp: np.ndarray) -> np.ndarray:
    """ get oversample ratio
		fp = diffraction pattern
	"""
    # autocorrelation
    acp = np.fft.fftshift(np.fft.ifftn(np.abs(fp)**2.))
    aacp = np.abs(acp)

    # get extent
    blob = get_central_object_extent(aacp)

    # correct for underestimation due to thresholding
    correction = [0.025, 0.025, 0.0729][:fp.ndim]

    extent = [
        min(m, s + int(round(f * aacp.shape[i], 1)))
        for i, (s, f, m) in enumerate(zip(blob, correction, aacp.shape))
    ]

    # oversample ratio
    oversample = [
        2. * s / (e + (1 - s % 2)) for s, e in zip(aacp.shape, extent)
    ]
    return np.round(oversample, 3)

# copied from Mathew's code
def Bin(Array, Binsizes):
    odims = Array.shape
    ndims = []
    Binsizes = list(Binsizes)
    if len(Binsizes) < len(odims):
        while len(odims) > len(Binsizes):
            Binsizes.append(1)

    #playing interger math tricks
    n = 0
    for d in odims:
        ndims.append(int(d / Binsizes[n] * Binsizes[n]))
        n += 1

    def getindex(n, binsize, start, ndim):
        ind = []
        for i in range(ndim):
            if i < n:
                ind.append(slice(0, ndims[i]))
            elif i == n:
                #playing integer math games....
                ind.append(slice(a1, odims[i] // binsize * binsize, binsize))
            else:
                ind.append(slice(0, odims[i]))
        return ind

    # Bin each dim separately using a an indexing trick
    arrs = [Array.copy()]
    n = 0
    for b in Binsizes:
        for a1 in range(b):
            if a1 == 0:
                arrs.append(arrs[0][tuple(getindex(n, b, a1, len(odims)))])
            else:
                arrs[1] += arrs[0][tuple(getindex(n, b, a1, len(odims)))]
        n += 1
        del arrs[0]

    return arrs[0]

def crop_pad(IN, dim):
    [h, w, t] = IN.shape
    output = np.zeros(dim, dtype=IN.dtype)
    if h >= dim[0] and w >= dim[1] and t >= dim[2]:
        output = IN[h // 2 - dim[0] // 2:h // 2 + dim[0] // 2,
                    w // 2 - dim[1] // 2:w // 2 + dim[1] // 2,
                    t // 2 - dim[2] // 2:t // 2 + dim[2] // 2]
    else:
        ps = np.max([[h, w, t], list(dim)], axis=0)
        lb = ps // 2 - (np.floor(np.array([h, w, t]) / 2)).astype('int32')
        ub = ps // 2 + (np.ceil(np.array([h, w, t]) / 2)).astype('int32')

        IN_pad = np.zeros(ps, dtype=IN.dtype)
        IN_pad[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]] = IN

        lb = ps // 2 - (np.floor(np.array(dim) / 2)).astype('int32')
        ub = ps // 2 + (np.ceil(np.array(dim) / 2)).astype('int32')

        output = IN_pad[lb[0]:ub[0], lb[1]:ub[1], lb[2]:ub[2]]
    return output

def Resize(IN, dim):
    ft = np.fft.fftshift(np.fft.fftn(IN)) / np.prod(IN.shape)
    ft_resize = crop_pad(ft, dim)
    output = np.fft.ifftn(np.fft.ifftshift(ft_resize)) * np.prod(dim)
    return output

def match_oversample_diff(
    diff: np.ndarray,
    fr: Union[list, np.ndarray, None] = None,
    to: Union[list, np.ndarray, None] = None,
    shape: Union[list, np.ndarray, None] = [64, 64, 64],
):
    """ resize diff to match oversample ratios 
        diff = diffraction pattern
        fr = from oversample ratio
        to = to oversample ratio
        shape = output shape
    """
    # adjustment needed to match oversample ratio
    change = [np.round(f / t).astype('int32') for f, t in zip(fr, to)]
    change = [np.max([1, c]) for c in change]

    diff = Bin(diff, change)
    # crop diff to match output shape
    output = crop_pad(diff, shape)
    return output, diff.shape

def shift_com(amp, phi):
    from scipy.ndimage.measurements import center_of_mass as com
    from scipy.ndimage.interpolation import shift

    h,w,t = 64,64,64
    coms = com(amp)
    if (np.isnan(coms[0]) or np.isnan(coms[1]) or np.isnan(coms[2])):
        return 0
    deltas = (int(round(h / 2 - coms[0])), int(round(w / 2 - coms[1])),
              int(round(t / 2 - coms[2])))
    amp_shift = shift(amp, shift=deltas, mode='wrap')
    phi_shift = shift(phi, shift=deltas, mode='wrap')
    return amp_shift, phi_shift

def post_process(amp, phi, th=0.1, uw=0):
    if uw == 1:
        # phi = np.unwrap(np.unwrap(np.unwrap(phi,0),1),2)
        phi = unwrap_phase(phi)

    mask = np.where(amp > th, 1, 0)
    amp_out = mask * amp
    phi_out = mask * phi

    mean_phi = np.sum(phi_out) / np.sum(mask)
    phi_out = phi_out - mean_phi

    amp_out, phi_out = shift_com(amp_out, phi_out)

    mask = np.where(amp_out > th, 1, 0)
    amp_out = mask * amp_out
    phi_out = mask * phi_out
    return amp_out, phi_out

## funcions needed in tensorflow model
@tf.function
def combine_complex(amp, phi):
    import tensorflow as tf
    output = tf.cast(amp, tf.complex64) * tf.exp(
        1j * tf.cast(phi, tf.complex64))
    return output

@tf.function
def get_mask(input):
    import tensorflow as tf

    mask = tf.where(input >= 0.1, tf.ones_like(input), tf.zeros_like(input))
    return mask

@tf.function
def loss_comb2_scale(Y_true, Y_pred):  
    Y_pred = Y_pred/(tf.math.reduce_max(Y_pred, axis=(1,2,3), keepdims=True)+1e-6)*tf.math.reduce_max(Y_true, axis=(1,2,3), keepdims=True)
    loss_1 = tf.math.sqrt(loss_sq(Y_true, Y_pred))
    loss_2 = loss_pcc(Y_true, Y_pred)
    a1 = 1
    a2 = 1
    loss_value = (a1*loss_1+a2*loss_2)/(a1+a2)
    return loss_value

@tf.function
def ff_propagation(data):
    '''
    diffraction. Assume same x and y lengthss and uniform sampling
        data:        source plane field
        
    '''
    diff = _fourier_transform(data)

    # far-field amplitude
    intensity = tf.math.abs(diff)
    intensity = tf.cast(intensity, tf.float32)
    return intensity

@tf.function
# 3D fourier transform
def _fourier_transform(input):
    import tensorflow as tf
    # fft3d transform with channel unequal to 1
    perm_input = K.permute_dimensions(input, pattern=[4, 0, 1, 2, 3])
    perm_Fr = tf.signal.fftshift(tf.signal.fft3d(
        tf.signal.ifftshift(tf.cast(perm_input, tf.complex64),
                            axes=[-3, -2, -1])),
                                 axes=[-3, -2, -1])
    Fr = K.permute_dimensions(perm_Fr, pattern=[1, 2, 3, 4, 0])
    return Fr