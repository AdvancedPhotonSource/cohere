"""
InstrTab - Instrument/beamline-specific configuration.
"""

import ast
import ipywidgets as widgets

from .base import BaseTab
from ..widgets import form_row, text_field, checkbox, button, output_area


class InstrTab(BaseTab):
    """Tab for beamline instrument configuration.

    Dynamically builds fields based on the beamline type.
    Supports: aps_34idc, aps_1ide, esrf_id01, Petra3_P10
    """

    name = "Instrument"
    conf_name = "config_instr"

    def __init__(self, beamline: str = None):
        super().__init__()
        self.beamline = beamline
        self._fields = {}

    def _build_ui(self) -> widgets.Widget:
        self.params_box = widgets.VBox()

        # Buttons with Qt-matching colors
        self.load_btn = button('Load Config', style='warning', width='120px', qt_style='load')
        self.save_btn = button('Save Config', style='success', width='120px', qt_style='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.save_btn.on_click(lambda b: self.save_conf())

        self.output = output_area(height='100px')

        layout = widgets.VBox([
            widgets.HTML(f'<b>Beamline: {self.beamline or "Not set"}</b>'),
            self.params_box,
            widgets.HBox([self.load_btn, self.save_btn]),
            self.output
        ])

        return layout

    def set_beamline(self, beamline: str):
        """Set beamline and rebuild UI fields."""
        self.beamline = beamline
        self._build_beamline_fields()

    def _build_beamline_fields(self):
        """Build fields based on beamline type."""
        self._fields = {}
        field_widgets = []

        if self.beamline == 'aps_34idc':
            fields = [
                ('diffractometer', 'e.g., 34idc'),
                ('specfile', 'Path to spec file'),
                ('data_dir', 'Data directory'),
                ('darkfield_filename', 'Dark field file'),
                ('whitefield_filename', 'White field file'),
            ]
        elif self.beamline == 'aps_1ide':
            fields = [
                ('diffractometer', 'e.g., 1ide'),
                ('data_dir', 'Data directory'),
                ('whitefield_filename', 'White field file'),
                ('roi', 'e.g., [y1, y2, x1, x2]'),
                ('energy', 'Energy in keV'),
            ]
        elif self.beamline == 'esrf_id01':
            fields = [
                ('detector', 'Detector name'),
                ('diffractometer', 'Diffractometer type'),
                ('h5file', 'HDF5 file path'),
                ('roi', 'e.g., [y1, y2, x1, x2]'),
            ]
        elif self.beamline == 'Petra3_P10':
            fields = [
                ('diffractometer', 'Diffractometer type'),
                ('data_dir', 'Data directory'),
                ('sample', 'Sample name'),
                ('darkfield_filename', 'Dark field file'),
                ('detector_module', 'Detector module'),
                ('energy', 'Energy in keV'),
                ('del', 'Delta angle'),
                ('gam', 'Gamma angle'),
                ('mu', 'Mu angle'),
                ('om', 'Omega angle'),
                ('chi', 'Chi angle'),
                ('phi', 'Phi angle'),
                ('detdist', 'Detector distance'),
                ('scanmot', 'Scan motor'),
                ('detector', 'Detector name'),
            ]
        else:
            fields = [
                ('diffractometer', 'Diffractometer type'),
                ('data_dir', 'Data directory'),
            ]

        for field_name, placeholder in fields:
            widget = text_field(placeholder=placeholder, width='300px')
            self._fields[field_name] = widget
            field_widgets.append(form_row(field_name, widget))

        self.params_box.children = field_widgets

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        # Build fields if not already done
        if not self._fields:
            self._build_beamline_fields()

        for field_name, widget in self._fields.items():
            if field_name in conf_map:
                value = conf_map[field_name]
                if isinstance(value, (list, dict)):
                    widget.value = str(value).replace(' ', '')
                else:
                    widget.value = str(value)

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        for field_name, widget in self._fields.items():
            if widget.value:
                # Try to parse as Python literal, otherwise keep as string
                try:
                    conf_map[field_name] = ast.literal_eval(widget.value)
                except (ValueError, SyntaxError):
                    conf_map[field_name] = widget.value

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        for widget in self._fields.values():
            widget.value = ''

    def run_tab(self):
        """Instrument tab doesn't have a run action - just saves config."""
        self.save_conf()
        with self.output:
            print("Instrument configuration saved.")

    def _load_config_dialog(self):
        """Placeholder for loading config from file."""
        with self.output:
            print("Enter config file path in working directory")
