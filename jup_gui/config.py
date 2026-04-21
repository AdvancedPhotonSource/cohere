"""
Configuration management wrapping cohere_core.utilities and cohere_ui.api.common.
"""

import os
from typing import Optional

import cohere_core.utilities as ut


def _get_common():
    """Lazy import of cohere_ui.api.common to allow path setup first."""
    import cohere_ui.api.common as com
    return com


class ConfigManager:
    """Manages experiment configuration loading and saving."""

    def __init__(self, experiment_dir: Optional[str] = None):
        self.experiment_dir = experiment_dir
        self._config_maps = {}

    @property
    def conf_dir(self) -> Optional[str]:
        if self.experiment_dir:
            return ut.join(self.experiment_dir, 'conf')
        return None

    def set_experiment_dir(self, experiment_dir: str):
        """Set or change the experiment directory."""
        self.experiment_dir = experiment_dir
        self._config_maps = {}

    def load_configs(self, conf_list: list, no_verify: bool = False) -> dict:
        """Load multiple configuration files at once.

        Args:
            conf_list: List of config names like ['config_prep', 'config_data', 'config_rec']
            no_verify: If True, load even if validation fails

        Returns:
            Dictionary mapping config names to config dictionaries
        """
        if not self.experiment_dir:
            raise ValueError("Experiment directory not set")

        com = _get_common()
        maps, converted = com.get_config_maps(
            self.experiment_dir,
            conf_list,
            no_verify=no_verify
        )
        self._config_maps.update(maps)
        return maps

    def load_config(self, conf_name: str) -> Optional[dict]:
        """Load a single configuration file.

        Args:
            conf_name: Config name like 'config_rec' or 'config_data'

        Returns:
            Configuration dictionary or None if file doesn't exist
        """
        if not self.conf_dir:
            return None

        conf_path = ut.join(self.conf_dir, conf_name)
        if not os.path.isfile(conf_path):
            return None

        conf_map = ut.read_config(conf_path)
        if conf_map:
            self._config_maps[conf_name] = conf_map
        return conf_map

    def save_config(self, conf_name: str, conf_map: dict, no_verify: bool = False) -> str:
        """Save configuration to file.

        Args:
            conf_name: Config name like 'config_rec'
            conf_map: Configuration dictionary
            no_verify: If True, save even if validation fails

        Returns:
            Error message (empty string if successful)
        """
        if not self.conf_dir:
            return "Experiment directory not set"

        er_msg = self.verify(conf_name, conf_map)
        if er_msg and not no_verify:
            return er_msg

        conf_path = ut.join(self.conf_dir, conf_name)
        ut.write_config(conf_map, conf_path)
        self._config_maps[conf_name] = conf_map
        return ""

    def verify(self, conf_name: str, conf_map: dict) -> str:
        """Verify configuration dictionary.

        Args:
            conf_name: Config name for schema lookup
            conf_map: Configuration dictionary to verify

        Returns:
            Error message (empty string if valid)
        """
        return ut.verify(conf_name, conf_map)

    def get_cached(self, conf_name: str) -> Optional[dict]:
        """Get a previously loaded config from cache."""
        return self._config_maps.get(conf_name)

    @property
    def all_configs(self) -> dict:
        """Return all cached configuration maps."""
        return self._config_maps.copy()

    def ensure_conf_dir(self):
        """Create the conf directory if it doesn't exist."""
        if self.conf_dir and not os.path.exists(self.conf_dir):
            os.makedirs(self.conf_dir)

    def ensure_experiment_dir(self):
        """Create the experiment directory if it doesn't exist."""
        if self.experiment_dir and not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        self.ensure_conf_dir()
