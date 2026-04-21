"""
ResultsContainer for exposing analysis variables to notebook users.
"""

import os
from typing import Optional
import numpy as np

import cohere_core.utilities as ut


class ResultsContainer:
    """Container for lazy-loaded reconstruction results.

    Exposes experiment data and results as properties for interactive analysis.
    """

    def __init__(self, config_manager=None):
        self._config_manager = config_manager
        self._data = None
        self._image = None
        self._support = None
        self._coherence = None
        self._errors = None

    def set_config_manager(self, config_manager):
        """Set or update the config manager reference."""
        self._config_manager = config_manager
        self.reload()

    @property
    def experiment_dir(self) -> Optional[str]:
        """Current experiment directory path."""
        if self._config_manager:
            return self._config_manager.experiment_dir
        return None

    @property
    def config(self) -> dict:
        """All loaded configuration maps."""
        if self._config_manager:
            return self._config_manager.all_configs
        return {}

    @property
    def data(self) -> Optional[np.ndarray]:
        """Diffraction data from phasing_data/data.tif or data.npy."""
        if self._data is None:
            self._data = self._load_data()
        return self._data

    @property
    def image(self) -> Optional[np.ndarray]:
        """Reconstructed image from results_phasing/image.npy."""
        if self._image is None:
            self._image = self._load_result('image.npy')
        return self._image

    @property
    def support(self) -> Optional[np.ndarray]:
        """Support array from results_phasing/support.npy."""
        if self._support is None:
            self._support = self._load_result('support.npy')
        return self._support

    @property
    def coherence(self) -> Optional[np.ndarray]:
        """Coherence function from results_phasing/coherence.npy (if PCDI used)."""
        if self._coherence is None:
            self._coherence = self._load_result('coherence.npy')
        return self._coherence

    @property
    def errors(self) -> Optional[np.ndarray]:
        """Error metrics from results_phasing/errors.npy."""
        if self._errors is None:
            self._errors = self._load_result('errors.npy')
        return self._errors

    def _load_data(self) -> Optional[np.ndarray]:
        """Load diffraction data."""
        if not self.experiment_dir:
            return None

        data_dir = ut.join(self.experiment_dir, 'phasing_data')

        for filename in ['data.npy', 'data.tif']:
            filepath = ut.join(data_dir, filename)
            if os.path.isfile(filepath):
                return self._load_array(filepath)

        return None

    def _load_result(self, filename: str) -> Optional[np.ndarray]:
        """Load a result file from results_phasing directory."""
        if not self.experiment_dir:
            return None

        results_dir = ut.join(self.experiment_dir, 'results_phasing')
        filepath = ut.join(results_dir, filename)

        if os.path.isfile(filepath):
            return self._load_array(filepath)

        return None

    def _load_array(self, filepath: str) -> Optional[np.ndarray]:
        """Load an array from .npy or .tif file."""
        try:
            if filepath.endswith('.npy'):
                return np.load(filepath)
            elif filepath.endswith('.tif'):
                import tifffile as tf
                return tf.imread(filepath)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
        return None

    def reload(self):
        """Force reload all cached data."""
        self._data = None
        self._image = None
        self._support = None
        self._coherence = None
        self._errors = None

    def list_results(self) -> list:
        """List available result files in results_phasing directory."""
        if not self.experiment_dir:
            return []

        results_dir = ut.join(self.experiment_dir, 'results_phasing')
        if not os.path.isdir(results_dir):
            return []

        return [f for f in os.listdir(results_dir) if f.endswith(('.npy', '.tif'))]

    def __repr__(self):
        exp = self.experiment_dir or "(not set)"
        return f"ResultsContainer(experiment_dir={exp})"
