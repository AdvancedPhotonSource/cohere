"""Tab implementations for the Jupyter GUI."""

from .base import BaseTab
from .data import DataTab
from .prep import PrepTab
from .rec import RecTab
from .disp import DispTab
from .instr import InstrTab

__all__ = ['BaseTab', 'DataTab', 'PrepTab', 'RecTab', 'DispTab', 'InstrTab']
