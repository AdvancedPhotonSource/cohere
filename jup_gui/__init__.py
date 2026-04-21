"""
Jupyter Notebook GUI for Cohere - Bragg CDI Reconstruction.

Usage:
    from jup_gui import CoherenceGUI
    gui = CoherenceGUI()
    gui.display()

    # After loading/running:
    gui.results.image  # numpy array
    gui.results.support
    gui.results.config  # all config dicts
"""

import sys
import os

# Fix cohere_beamlines import path. The editable install may point to the wrong
# location (root instead of src/), so we need to ensure the correct path is first.
_beamlines_src = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cohere_beamlines', 'src')
if os.path.isdir(_beamlines_src):
    # Remove any cached wrong import
    if 'cohere_beamlines' in sys.modules:
        _cached = sys.modules['cohere_beamlines']
        if hasattr(_cached, '__path__') and 'src' not in str(_cached.__path__[0]):
            del sys.modules['cohere_beamlines']
    # Ensure correct path is first
    if _beamlines_src not in sys.path:
        sys.path.insert(0, _beamlines_src)

from .core import CoherenceGUI
from .results import ResultsContainer
from .config import ConfigManager

__all__ = ['CoherenceGUI', 'ResultsContainer', 'ConfigManager']
__version__ = '0.1.0'
