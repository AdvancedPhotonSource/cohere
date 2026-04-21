"""
Base class for all tab implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import ipywidgets as widgets


class BaseTab(ABC):
    """Abstract base class defining the standard tab interface.

    Each tab must implement:
    - name: Display name for the tab
    - conf_name: Configuration file name (e.g., 'config_rec')
    - _build_ui(): Create the widget tree
    - load_tab(conf_map): Populate widgets from config dict
    - get_config(): Read widgets into config dict
    - clear_conf(): Reset all widgets to defaults
    - run_tab(): Execute the tab's backend function
    """

    name: str = "Tab"
    conf_name: str = "config"

    def __init__(self):
        self.main_gui = None
        self.output = widgets.Output()
        self._widget = None

    def init(self, main_gui):
        """Initialize the tab with reference to main GUI.

        Args:
            main_gui: CoherenceGUI instance
        """
        self.main_gui = main_gui
        self._widget = self._build_ui()

    @property
    def widget(self) -> widgets.Widget:
        """The tab's root widget."""
        if self._widget is None:
            self._widget = self._build_ui()
        return self._widget

    @abstractmethod
    def _build_ui(self) -> widgets.Widget:
        """Build and return the tab's widget tree."""
        pass

    @abstractmethod
    def load_tab(self, conf_map: dict):
        """Populate widgets from configuration dictionary.

        Args:
            conf_map: Configuration dictionary for this tab
        """
        pass

    @abstractmethod
    def get_config(self) -> dict:
        """Read current widget values into a configuration dictionary.

        Returns:
            Configuration dictionary with current values
        """
        pass

    @abstractmethod
    def clear_conf(self):
        """Reset all widgets to their default/empty state."""
        pass

    def save_conf(self) -> str:
        """Save current configuration to file.

        Returns:
            Error message (empty if successful)
        """
        if not self.main_gui or not self.main_gui.is_exp_exists():
            return "Experiment not set"

        conf_map = self.get_config()
        return self.main_gui.config_manager.save_config(
            self.conf_name,
            conf_map,
            no_verify=self.main_gui.no_verify
        )

    @abstractmethod
    def run_tab(self):
        """Execute the tab's backend processing function."""
        pass

    def log(self, message: str):
        """Log a message to the tab's output area."""
        with self.output:
            print(message)

    def clear_output(self):
        """Clear the output area."""
        self.output.clear_output()

    def _validate_experiment(self) -> Optional[str]:
        """Validate that experiment is ready for operations.

        Returns:
            Error message or None if valid
        """
        if not self.main_gui:
            return "Tab not initialized"
        if not self.main_gui.is_exp_exists():
            return "Experiment has not been created yet"
        if not self.main_gui.is_exp_set():
            return "Experiment has changed, press 'Set Experiment' button"
        return None
