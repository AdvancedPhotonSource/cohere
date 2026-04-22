"""
Base class for toggleable features in reconstruction and display tabs.
"""

from abc import ABC, abstractmethod
from typing import Optional
import ipywidgets as widgets
import ast

from ..widgets import checkbox, button, form_row, text_field


class Feature(ABC):
    """Base class for toggleable configuration features.

    Features have an 'active' checkbox that shows/hides parameter fields.
    Subclasses implement fill_active() to define their parameters.
    """

    name: str = "Feature"

    def __init__(self):
        self.active = checkbox(description='active')
        self.params_box = widgets.VBox()
        self.defaults_btn = button('Set Defaults', style='info', width='120px')

        self.active.observe(self._on_active_change, 'value')
        self.defaults_btn.on_click(lambda b: self.set_defaults())

        self._widget = widgets.VBox([
            self.active,
            self.params_box
        ])

    @property
    def widget(self) -> widgets.Widget:
        """The feature's root widget."""
        return self._widget

    def _on_active_change(self, change):
        if change['new']:
            self._show_params()
        else:
            self._hide_params()

    def _show_params(self):
        """Show parameter widgets when feature is activated."""
        param_widgets = self.fill_active()
        param_widgets.append(self.defaults_btn)
        self.params_box.children = param_widgets

    def _hide_params(self):
        """Hide parameter widgets when feature is deactivated."""
        self.params_box.children = []

    @abstractmethod
    def fill_active(self) -> list:
        """Create and return parameter widgets when feature becomes active.

        Returns:
            List of widgets to display
        """
        pass

    @abstractmethod
    def set_defaults(self):
        """Set all parameters to their default values."""
        pass

    @abstractmethod
    def init_config(self, conf_map: dict):
        """Initialize parameters from configuration dictionary.

        Args:
            conf_map: Configuration dictionary
        """
        pass

    @abstractmethod
    def add_config(self, conf_map: dict):
        """Add feature parameters to configuration dictionary.

        Only called when feature is active.

        Args:
            conf_map: Configuration dictionary to modify
        """
        pass

    def verify_active(self) -> str:
        """Verify that feature configuration is valid.

        Returns:
            Error message, or empty string if valid
        """
        return ""

    def clear(self):
        """Reset feature to inactive state."""
        self.active.value = False
        self._hide_params()

    @staticmethod
    def parse_value(text: str):
        """Parse a string value into Python object using ast.literal_eval."""
        text = text.strip()
        if not text:
            return None
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text

    @staticmethod
    def format_value(value) -> str:
        """Format a Python value as string for display."""
        if value is None:
            return ""
        return str(value).replace(" ", "")
