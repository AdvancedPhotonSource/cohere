import abc
class cohlib(metaclass=abc.ABCMeta):
    # Interface
    @classmethod
    def __subclasshook__(cls, subclass):
        return (hasattr(subclass, 'array') and
                callable(subclass.array) and
                hasattr(subclass, 'dot') and
                callable(subclass.dot) and
                hasattr(subclass, 'set_device') and
                callable(subclass.set_device) and
                hasattr(subclass, 'set_backend') and
                callable(subclass.set_backend) and
                hasattr(subclass, 'to_numpy') and
                callable(subclass.to_numpy) and
                hasattr(subclass, 'save') and
                callable(subclass.save) and
                hasattr(subclass, 'load') and
                callable(subclass.load) and
                hasattr(subclass, 'from_numpy') and
                callable(subclass.from_numpy) and
                hasattr(subclass, 'dtype') and
                callable(subclass.dtype) and
                hasattr(subclass, 'astype') and
                callable(subclass.astype) and
                hasattr(subclass, 'size') and
                callable(subclass.size) and
                hasattr(subclass, 'hasnan') and
                callable(subclass.hasnan) and
                hasattr(subclass, 'copy') and
                callable(subclass.copy) and
                hasattr(subclass, 'roll') and
                callable(subclass.shift) and
                hasattr(subclass, 'roll') and
                callable(subclass.shift) and
                hasattr(subclass, 'fftshift') and
                callable(subclass.fftshift) and
                hasattr(subclass, 'ifftshift') and
                callable(subclass.ifftshift) and
                hasattr(subclass, 'fft') and
                callable(subclass.fft) and
                hasattr(subclass, 'ifft') and
                callable(subclass.ifft) and
                hasattr(subclass, 'fftconvolve') and
                callable(subclass.fftconvolve) and
                hasattr(subclass, 'correlate') and
                callable(subclass.correlate) and
                hasattr(subclass, 'where') and
                callable(subclass.where) and
                hasattr(subclass, 'dims') and
                callable(subclass.dims) and
                hasattr(subclass, 'absolute') and
                callable(subclass.absolute) and
                hasattr(subclass, 'square') and
                callable(subclass.square) and
                hasattr(subclass, 'sqrt') and
                callable(subclass.sqrt) and
                hasattr(subclass, 'sum') and
                callable(subclass.sum) and
                hasattr(subclass, 'real') and
                callable(subclass.real) and
                hasattr(subclass, 'imag') and
                callable(subclass.imag) and
                hasattr(subclass, 'amax') and
                callable(subclass.amax) and
                hasattr(subclass, 'argmax') and
                callable(subclass.argmax) and
                hasattr(subclass, 'unravel_index') and
                callable(subclass.unravel_index) and
                hasattr(subclass, 'maximum') and
                callable(subclass.maximum) and
                hasattr(subclass, 'ceil') and
                callable(subclass.ceil) and
                hasattr(subclass, 'fix') and
                callable(subclass.fix) and
                hasattr(subclass, 'round') and
                callable(subclass.round) and
                hasattr(subclass, 'random') and
                callable(subclass.random) and
                hasattr(subclass, 'full') and
                callable(subclass.full) and
                hasattr(subclass, 'print') and
                callable(subclass.print) and
                hasattr(subclass, 'angle') and
                callable(subclass.angle) and
                hasattr(subclass, 'flip') and
                callable(subclass.flip) and
                hasattr(subclass, 'tile') and
                callable(subclass.tile) and
                hasattr(subclass, 'expand_dims') and
                callable(subclass.expand_dims) and
                hasattr(subclass, 'squeeze') and
                callable(subclass.squeeze) and
                hasattr(subclass, 'gaussian_filter') and
                callable(subclass.gaussian_filter) and
                hasattr(subclass, 'gaussian') and
                callable(subclass.gaussian) and
                hasattr(subclass, 'center_of_mass') and
                callable(subclass.center_of_mass) and
                hasattr(subclass, 'meshgrid') and
                callable(subclass.meshgrid) and
                hasattr(subclass, 'exp') and
                callable(subclass.exp) and
                hasattr(subclass, 'conj') and
                callable(subclass.conj) and
                hasattr(subclass, 'array_equal') and
                callable(subclass.array_equal) and
                hasattr(subclass, 'cos') and
                callable(subclass.cos) and
                hasattr(subclass, 'linspace') and
                callable(subclass.linspace) and
                hasattr(subclass, 'clip') and
                callable(subclass.clip) and
                hasattr(subclass, 'gradient') and
                callable(subclass.gradient) and
                hasattr(subclass, 'argmin') and
                callable(subclass.argmin) and
                hasattr(subclass, 'take_along_axis') and
                callable(subclass.take_along_axis) and
                hasattr(subclass, 'moveaxis') and
                callable(subclass.moveaxis) and
                hasattr(subclass, 'lstsq') and
                callable(subclass.lstsq) and
                hasattr(subclass, 'zeros') and
                callable(subclass.zeros) and
                hasattr(subclass, 'indices') and
                callable(subclass.indices) and
                hasattr(subclass, 'diff') and
                callable(subclass.diff) and
                hasattr(subclass, 'concatenate') and
                callable(subclass.concatenate) and
                callable(subclass, 'clean_default_mem') and
                callable(subclass.clean_default_mem) or
                NotImplemented)

    @abc.abstractmethod
    def array(obj):
        pass

    @abc.abstractmethod
    def dot(arr1, arr2):
        pass

    @abc.abstractmethod
    def set_device(dev_id):
        pass

    @abc.abstractmethod
    def set_backend(proc):
        pass

    @abc.abstractmethod
    def to_numpy(arr):
        pass

    @abc.abstractmethod
    def save(filename, arr):
        pass

    @abc.abstractmethod
    def load(filename):
        pass

    @abc.abstractmethod
    def from_numpy(arr):
        pass

    @abc.abstractmethod
    def dtype(arr):
        pass

    @abc.abstractmethod
    def astype(arr):
        pass

    @abc.abstractmethod
    def size(arr):
        pass

    @abc.abstractmethod
    def hasnan(arr):
        pass

    @abc.abstractmethod
    def copy(arr):
        pass

    @abc.abstractmethod
    def roll(arr, sft, axis):
        pass

    @abc.abstractmethod
    def shift(arr, sft):
        pass

    @abc.abstractmethod
    def fftshift(arr):
        pass

    @abc.abstractmethod
    def ifftshift(arr):
        pass

    @abc.abstractmethod
    def fft(arr, norm='forward'):
        pass

    @abc.abstractmethod
    def ifft(arr, norm='forward'):
        pass

    @abc.abstractmethod
    def fftconvolve(arr1, kernel):
        pass

    @abc.abstractmethod
    def correlate(arr1, arr2, mode='same', method='fft'):
        pass

    @abc.abstractmethod
    def where(cond, x, y):
        pass

    @abc.abstractmethod
    def dims(arr):
        # get array dimensions
        pass

    @abc.abstractmethod
    def absolute(arr):
        pass

    @abc.abstractmethod
    def square(arr):
        pass

    @abc.abstractmethod
    def sqrt(arr):
        pass

    @abc.abstractmethod
    def sum(arr, axis=None):
        pass

    @abc.abstractmethod
    def real(arr):
        pass

    @abc.abstractmethod
    def imag(arr):
        pass

    @abc.abstractmethod
    def amax(arr):
        pass

    @abc.abstractmethod
    def argmax(arr, axis=None):
        pass

    @abc.abstractmethod
    def unravel_index(indices, shape):
        pass

    @abc.abstractmethod
    def maximum(arr1, arr2):
        pass

    @abc.abstractmethod
    def ceil(arr):
        pass

    @abc.abstractmethod
    def fix(arr):
        pass

    @abc.abstractmethod
    def round(val):
        pass

    @abc.abstractmethod
    def random(shape, **kwargs):
        pass

    @abc.abstractmethod
    def full(shape, fill_value, **kwargs):
        pass

    @abc.abstractmethod
    def print(arr, **kwargs):
        pass

    @abc.abstractmethod
    def angle(arr):
        pass

    @abc.abstractmethod
    def flip(arr, axis):
        pass

    @abc.abstractmethod
    def tile(arr, rep):
        pass

    @abc.abstractmethod
    def expand_dims(arr, axis):
        pass

    @abc.abstractmethod
    def squeeze(arr):
        pass

    @abc.abstractmethod
    def gaussian_filter(arr, sigma, **kwargs):
        pass

    @abc.abstractmethod
    def gaussian(dims, sigma, **kwargs):
        pass

    @abc.abstractmethod
    def center_of_mass(arr):
        pass

    @abc.abstractmethod
    def meshgrid(*xi):
        pass

    @abc.abstractmethod
    def exp(arr):
        pass

    @abc.abstractmethod
    def conj(arr):
        pass

    @abc.abstractmethod
    def array_equal(arr1, arr2):
        pass

    @abc.abstractmethod
    def cos(arr):
        pass

    @abc.abstractmethod
    def linspace(start, stop, num):
        pass

    @abc.abstractmethod
    def diff(arr, axis=None, prepend=0):
        pass

    @abc.abstractmethod
    def clip(arr, min, max=None):
        pass

    @abc.abstractmethod
    def gradient(arr, dx=1):
        pass

    @abc.abstractmethod
    def argmin(arr, axis=None):
        pass

    @abc.abstractmethod
    def take_along_axis(a, indices, axis):
        pass

    @abc.abstractmethod
    def moveaxis(arr, source, dest):
        pass

    @abc.abstractmethod
    def lstsq(A, B):
        pass

    @abc.abstractmethod
    def zeros(shape):
        pass

    @abc.abstractmethod
    def indices(dims):
        pass

    @abc.abstractmethod
    def concatenate(tup, axis=0):
        pass

    @abc.abstractmethod
    def amin(arr):
        pass

    @abc.abstractmethod
    def affine_transform(arr, matrix, order=3, offset=0):
        pass

    @abc.abstractmethod
    def pad(arr, padding):
        pass

    @abc.abstractmethod
    def histogram2d(meas, rec, n_bins=100, log=False):
        pass

    @abc.abstractmethod
    def calc_nmi(hgram):
        pass

    @abc.abstractmethod
    def calc_ehd(hgram):
        pass

    @abc.abstractmethod
    def clean_default_mem():
        pass
