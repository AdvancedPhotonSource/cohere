import os
import numpy as np
import cohere_core.utilities.utils as ut
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from skimage import restoration
from typing import Union

nconv = 32
INIT_SW = 0.07 #Initial shrink wrap
scale_I = 1 # normalize diff or not


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
    diff = ut.binning(diff, change)
    # crop diff to match output shape
    shape_arr = np.array(shape)
    diff_shape_arr = np.array(diff.shape)
    pad_value1 = shape_arr // 2 - diff_shape_arr // 2
    pad_value2 = shape_arr - diff_shape_arr -pad_value1
    pad = [[pad_value1[0], pad_value2[0]], [pad_value1[1], pad_value2[1]],
           [pad_value1[2], pad_value2[2]]]

    output = ut.adjust_dimensions(diff, pad)
    return output, diff.shape


def shift_com(amp, phi):
    from scipy.ndimage.measurements import center_of_mass as com
    from scipy.ndimage.interpolation import shift

    h, w, t = 64, 64, 64
    coms = com(amp)
    deltas = (int(round(h / 2 - coms[0])), int(round(w / 2 - coms[1])),
              int(round(t / 2 - coms[2])))
    amp_shift = shift(amp, shift=deltas, mode='wrap')
    phi_shift = shift(phi, shift=deltas, mode='wrap')
    return amp_shift, phi_shift


def post_process(amp, phi, th=0.1, uw=0):
    if uw == 1:
        # phi = np.unwrap(np.unwrap(np.unwrap(phi,0),1),2)
        phi = restoration.unwrap_phase(phi)

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


class MyDataset(Dataset):
    """Characterizes a dataset for PyTorch"""

    def __init__(
        self,
        data_ID,
    ):
        """Initialization"""
        self.data_ID = data_ID

    def load_array(self, data):
        """Load one sample (ndarray format) from the dataset"""
        diff = np.abs(data)
        realspace = np.fft.ifftn(np.fft.ifftshift(data))

        # fftshift the crystal if it is split up. This only checks the lower corner (0,0,0)
        ix_center = tuple([int(0.5 * ix) for ix in realspace.shape])
        if np.abs(realspace[ix_center]) < np.abs(realspace[0, 0, 0]):
            realspace = np.fft.fftshift(realspace)

        amp = np.abs(realspace)
        phi = np.angle(realspace)

        if scale_I > 0:
            max_I = diff.max()
            diff = diff / max_I * scale_I
            max_amp = amp.max()
            amp = amp / max_amp #Normalize amplitude to 1 always
        return diff, amp, phi

class recon_model(nn.Module):
    H, W = 32, 32

    def __init__(self):
        super(recon_model, self).__init__()

        self.sw_thresh = INIT_SW
        DROPOUT = 0.2

        # Order of layers guides :
        # https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        # https://sebastianraschka.com/faq/docs/dropout-activation.html
        self.encoder = nn.Sequential(
            # Appears sequential has similar functionality as TF avoiding need for separate model definition and activ
            nn.Conv3d(in_channels=1, out_channels=nconv, kernel_size=3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv),
            nn.Conv3d(in_channels=nconv, out_channels=nconv * 2, kernel_size=3, stride=2, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 2),

            nn.Conv3d(nconv * 2, nconv * 2, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 2),
            nn.Conv3d(nconv * 2, nconv * 4, 3, stride=2, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 4),

            nn.Conv3d(nconv * 4, nconv * 4, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 4),
            nn.Conv3d(nconv * 4, nconv * 8, 3, stride=2, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 8),

            nn.Conv3d(nconv * 8, nconv * 8, 3, stride=2, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 8),
        )

        self.decoder1 = nn.Sequential(
            nn.Conv3d(nconv * 8, nconv * 4, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 4),
            nn.Conv3d(nconv * 4, nconv * 4, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 4),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(nconv * 4, nconv * 2, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 2),
            nn.Conv3d(nconv * 2, nconv * 2, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 2),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(nconv * 2, nconv, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv),
            nn.Conv3d(nconv, nconv, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(nconv, 1, 3, stride=1, padding=1),
            nn.Sigmoid()  # Amplitude model
        )

        self.decoder2 = nn.Sequential(
            nn.Conv3d(nconv * 8, nconv * 4, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 4),
            nn.Conv3d(nconv * 4, nconv * 4, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 4),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(nconv * 4, nconv * 2, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 2),
            nn.Conv3d(nconv * 2, nconv * 2, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv * 2),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(nconv * 2, nconv, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv),
            nn.Conv3d(nconv, nconv, 3, stride=1, padding=1),
            nn.Dropout(DROPOUT),
            nn.LeakyReLU(negative_slope=0.01),
            nn.BatchNorm3d(nconv),
            nn.Upsample(scale_factor=2, mode='trilinear'),

            nn.Conv3d(nconv, 1, 3, stride=1, padding=1),
            #      nn.Tanh() #Phase model
        )

    def forward(self, x):
        x1 = self.encoder(x)

        amp = self.decoder1(x1)
        ph = self.decoder2(x1)

        # Normalize amp to max 1 before applying support

        amp = torch.clip(amp, min=0, max=1.0)

        # Apply the support to amplitude
        mask = torch.tensor([0, 1], dtype=amp.dtype, device=amp.device)
        amp = torch.where(amp < self.sw_thresh, mask[0], amp)

        # Restore -pi to pi range
        ph = ph * np.pi  # Using tanh activation (-1 to 1) for phase so multiply by pi

        # Pad the predictions to 2X
        pad = nn.ConstantPad3d(int(self.H / 2), 0)
        amp = pad(amp)
        ph = pad(ph)

        # Create the complex number
        complex_x = torch.complex(amp * torch.cos(ph), amp * torch.sin(ph))

        # Compute FT, shift and take abs
        y = torch.fft.fftn(complex_x, dim=(-3, -2, -1))
        y = torch.fft.fftshift(y, dim=(-3, -2, -1))  # FFT shift will move the wrong dimensions if not specified
        y = torch.abs(y)

        # Normalize to scale_I
        if scale_I > 0:
            max_I = torch.amax(y, dim=[-1, -2, -3], keepdim=True)
            y = scale_I * torch.div(y, max_I + 1e-6)  # Prevent zero div

        # get support for viz
        support = torch.where(amp < self.sw_thresh, mask[0], mask[1])

        return y, complex_x, amp, ph, support


def run_AI(data, model_file, dir):
    """
    Runs AI process.

    Parameters
    ----------
    data : ndarray
        data array

    model_file : str
        file name containing training model

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.
        Result of AI will be saved in dir/results_AI.

    Returns
    -------
    nothing
    """
    print('AI guess torch')

    # prepare data to make the oversampling ratio ~3
    wos = 3.0
    orig_os = ut.get_oversample_ratio(data)
    # match oversampling to wos
    wanted_os = [wos, wos, wos]
    # match diff os
    new_data, inshape = match_oversample_diff(data, orig_os, wanted_os)
    new_data = new_data[np.newaxis]
    new_data = np.expand_dims(new_data, axis=0)
    # scale max intensity to 1
    new_data = new_data / np.amax(new_data)
    # do thresholding
    new_data = np.where(new_data > 0.01, new_data, 0.0)
    new_data = torch.as_tensor(new_data, device="cpu")

    model = recon_model()
    model.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))
    y_ten, complex_x_ten, amp_ten, ph_ten, support_ten = model(new_data)

    amp = amp_ten.detach().to("cpu").numpy()
    ph = ph_ten.detach().to("cpu").numpy()
    pred_obj = np.squeeze(amp) * np.exp(1j * np.squeeze(ph))

    # match object size with the input data
    pred_obj = ut.Resize(pred_obj, inshape)

    pad_value = np.array(data.shape) // 2 - np.array(pred_obj.shape) // 2
    pad = [[pad_value[0], pad_value[0]], [pad_value[1], pad_value[1]],
           [pad_value[2], pad_value[2]]]
    guess = ut.adjust_dimensions(pred_obj, pad)
    guess = np.where(guess > 0.2 * np.amax(guess), guess, 0.0)

    np.save(dir + '/image.npy', guess)


def start_AI(pars, datafile, dir):
    """
    Starts AI process if all conditionas are met.

    Parameters
    ----------
    pars : dict
        parameters for reconstruction

    datafile : str
        file name containing data for reconstruction

    dir : str
        a parent directory that holds the reconstructions. It can be experiment directory or scan directory.
        Result of AI will be saved in dir/results_AI.

    Returns
    -------
    ai_dir : str
        directory where results were saved
    """
    if 'AI_trained_model' not in pars:
        print ('no AI_trained_model in config')
        return None
    if not os.path.isfile(pars['AI_trained_model']):
        print(f'there is no file {pars["AI_trained_model"]}')
        return None

    if datafile.endswith('tif') or datafile.endswith('tiff'):
        try:
            data = ut.read_tif(datafile)
        except:
            print('could not load data file', datafile)
            return None
    elif datafile.endswith('npy'):
        try:
            data = np.load(datafile)
        except:
            print('could not load data file', datafile)
            return None
    else:
        print('no data file found')
        return None

    # The results will be stored in the directory <experiment_dir>/AI_guess
    ai_dir = ut.join(dir, 'results_AI')
    if os.path.exists(ai_dir):
        pass
    else:
        os.makedirs(ai_dir)

    run_AI(data, pars['AI_trained_model'], ai_dir)
    return ai_dir

