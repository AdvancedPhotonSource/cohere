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
                hasattr(subclass, 'size') and
                callable(subclass.size) and
                hasattr(subclass, 'hasnan') and
                callable(subclass.hasnan) and
                hasattr(subclass, 'copy') and
                callable(subclass.copy) and
                hasattr(subclass, 'fftshift') and
                callable(subclass.fftshift) and
                hasattr(subclass, 'ifftshift') and
                callable(subclass.ifftshift) and
                hasattr(subclass, 'shift') and
                callable(subclass.shift) and
                hasattr(subclass, 'fft') and
                callable(subclass.fft) and
                hasattr(subclass, 'ifft') and
                callable(subclass.ifft) and
                hasattr(subclass, 'fftconvolve') and
                callable(subclass.fftconvolve) and
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
                callable(subclass.array_equal) or
                hasattr(subclass, 'cos') and
                callable(subclass.cos) or
                hasattr(subclass, 'linspace') and
                callable(subclass.linspace) or
                hasattr(subclass, 'clip') and
                callable(subclass.clip) or
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
    def size(arr):
        pass

    @abc.abstractmethod
    def hasnan(arr):
        pass

    @abc.abstractmethod
    def copy(arr):
        pass

    @abc.abstractmethod
    def fftshift(arr):
        pass

    @abc.abstractmethod
    def ifftshift(arr):
        pass

    @abc.abstractmethod
    def shift(arr, sft):
        pass

    @abc.abstractmethod
    def fft(arr):
        pass

    @abc.abstractmethod
    def ifft(arr):
        pass

    @abc.abstractmethod
    def fftconvolve(arr1, arr2):
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
    def clip(arr, min, max=None):
        pass
