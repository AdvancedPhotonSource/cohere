"""
DataTab - Data formatting configuration and processing.
"""

import ast
import ipywidgets as widgets

from .base import BaseTab
from ..widgets import form_row, text_field, dropdown, checkbox, button, output_area


class DataTab(BaseTab):
    """Tab for data formatting configuration.

    Handles alien removal, intensity threshold, binning, shift, crop/pad.
    """

    name = "Data"
    conf_name = "config_data"

    def _build_ui(self) -> widgets.Widget:
        # Alien algorithm selection
        self.alien_alg = dropdown(
            options=['none', 'block aliens', 'alien file', 'AutoAlien1'],
            value='none'
        )
        self.alien_params_box = widgets.VBox()
        self.alien_alg.observe(self._on_alien_change, 'value')

        # Standard fields
        self.auto_intensity_threshold = checkbox('auto intensity threshold')
        self.intensity_threshold = text_field(placeholder='e.g., 2.5')
        self.shift = text_field(placeholder='e.g., [0, 0, 0]')
        self.crop_pad = text_field(placeholder='e.g., [0, 0, 0, 0, 0, 0]')
        self.binning = text_field(placeholder='e.g., [1, 1, 1]')
        self.no_center_max = checkbox('not center max')

        # Buttons with Qt-matching colors
        self.load_btn = button('Load Config', style='warning', width='120px', qt_style='load')
        self.run_btn = button('Format Data', style='success', width='120px', qt_style='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())

        self.output = output_area(height='150px')

        layout = widgets.VBox([
            form_row('Alien Algorithm', self.alien_alg),
            self.alien_params_box,
            self.auto_intensity_threshold,
            form_row('Intensity Threshold', self.intensity_threshold),
            form_row('Shift', self.shift),
            form_row('Crop/Pad', self.crop_pad),
            form_row('Binning', self.binning),
            self.no_center_max,
            widgets.HBox([self.load_btn, self.run_btn]),
            self.output
        ])

        return layout

    def _on_alien_change(self, change):
        """Update alien parameters based on selected algorithm."""
        alg = change['new']
        self.alien_params_box.children = []

        if alg == 'block aliens':
            self.aliens = text_field(placeholder='e.g., [[x1,y1,z1,x2,y2,z2], ...]')
            self.alien_params_box.children = [form_row('Aliens', self.aliens)]

        elif alg == 'alien file':
            self.alien_file = text_field(placeholder='Path to alien file')
            self.alien_params_box.children = [form_row('Alien File', self.alien_file)]

        elif alg == 'AutoAlien1':
            self.AA1_size_threshold = text_field(placeholder='0.01')
            self.AA1_asym_threshold = text_field(placeholder='1.75')
            self.AA1_min_pts = text_field(placeholder='5')
            self.AA1_eps = text_field(placeholder='1.1')
            self.AA1_amp_threshold = text_field(placeholder='6.0')
            self.AA1_save_arrs = checkbox('save analysis arrays')
            self.AA1_expandcleanedsigma = text_field(placeholder='')

            self.AA1_defaults_btn = button('Set AA1 Defaults', style='info', width='140px', qt_style='info')
            self.AA1_defaults_btn.on_click(lambda b: self._set_AA1_defaults())

            self.alien_params_box.children = [
                form_row('Size Threshold', self.AA1_size_threshold),
                form_row('Asymmetry Threshold', self.AA1_asym_threshold),
                form_row('Min Points', self.AA1_min_pts),
                form_row('Cluster Eps', self.AA1_eps),
                form_row('Amp Threshold', self.AA1_amp_threshold),
                self.AA1_save_arrs,
                form_row('Expand Sigma', self.AA1_expandcleanedsigma),
                self.AA1_defaults_btn
            ]

    def _set_AA1_defaults(self):
        """Set AutoAlien1 parameters to defaults."""
        self.AA1_size_threshold.value = '0.01'
        self.AA1_asym_threshold.value = '1.75'
        self.AA1_min_pts.value = '5'
        self.AA1_eps.value = '1.1'
        self.AA1_amp_threshold.value = '6.0'
        self.AA1_save_arrs.value = False

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        # Alien algorithm
        alg = conf_map.get('alien_alg', 'random')
        if alg == 'random' or alg not in ['block_aliens', 'alien_file', 'AutoAlien1']:
            self.alien_alg.value = 'none'
        elif alg == 'block_aliens':
            self.alien_alg.value = 'block aliens'
            if 'aliens' in conf_map:
                self.aliens.value = str(conf_map['aliens']).replace(' ', '')
        elif alg == 'alien_file':
            self.alien_alg.value = 'alien file'
            if 'alien_file' in conf_map:
                self.alien_file.value = str(conf_map['alien_file'])
        elif alg == 'AutoAlien1':
            self.alien_alg.value = 'AutoAlien1'
            if 'AA1_size_threshold' in conf_map:
                self.AA1_size_threshold.value = str(conf_map['AA1_size_threshold'])
            if 'AA1_asym_threshold' in conf_map:
                self.AA1_asym_threshold.value = str(conf_map['AA1_asym_threshold'])
            if 'AA1_min_pts' in conf_map:
                self.AA1_min_pts.value = str(conf_map['AA1_min_pts'])
            if 'AA1_eps' in conf_map:
                self.AA1_eps.value = str(conf_map['AA1_eps'])
            if 'AA1_amp_threshold' in conf_map:
                self.AA1_amp_threshold.value = str(conf_map['AA1_amp_threshold'])
            if 'AA1_save_arrs' in conf_map:
                self.AA1_save_arrs.value = conf_map['AA1_save_arrs']
            if 'AA1_expandcleanedsigma' in conf_map:
                self.AA1_expandcleanedsigma.value = str(conf_map['AA1_expandcleanedsigma'])

        # Standard fields
        self.auto_intensity_threshold.value = conf_map.get('auto_intensity_threshold', False)
        if 'intensity_threshold' in conf_map:
            self.intensity_threshold.value = str(conf_map['intensity_threshold'])
        if 'binning' in conf_map:
            self.binning.value = str(conf_map['binning']).replace(' ', '')
        if 'shift' in conf_map:
            self.shift.value = str(conf_map['shift']).replace(' ', '')
        if 'crop_pad' in conf_map:
            self.crop_pad.value = str(conf_map['crop_pad']).replace(' ', '')
        self.no_center_max.value = conf_map.get('no_center_max', False)

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        # Alien algorithm
        alg = self.alien_alg.value
        if alg == 'block aliens':
            conf_map['alien_alg'] = 'block_aliens'
            if hasattr(self, 'aliens') and self.aliens.value:
                conf_map['aliens'] = self.aliens.value
        elif alg == 'alien file':
            conf_map['alien_alg'] = 'alien_file'
            if hasattr(self, 'alien_file') and self.alien_file.value:
                conf_map['alien_file'] = self.alien_file.value
        elif alg == 'AutoAlien1':
            conf_map['alien_alg'] = 'AutoAlien1'
            if hasattr(self, 'AA1_size_threshold') and self.AA1_size_threshold.value:
                conf_map['AA1_size_threshold'] = ast.literal_eval(self.AA1_size_threshold.value)
            if hasattr(self, 'AA1_asym_threshold') and self.AA1_asym_threshold.value:
                conf_map['AA1_asym_threshold'] = ast.literal_eval(self.AA1_asym_threshold.value)
            if hasattr(self, 'AA1_min_pts') and self.AA1_min_pts.value:
                conf_map['AA1_min_pts'] = ast.literal_eval(self.AA1_min_pts.value)
            if hasattr(self, 'AA1_eps') and self.AA1_eps.value:
                conf_map['AA1_eps'] = ast.literal_eval(self.AA1_eps.value)
            if hasattr(self, 'AA1_amp_threshold') and self.AA1_amp_threshold.value:
                conf_map['AA1_amp_threshold'] = ast.literal_eval(self.AA1_amp_threshold.value)
            if hasattr(self, 'AA1_save_arrs') and self.AA1_save_arrs.value:
                conf_map['AA1_save_arrs'] = True
            if hasattr(self, 'AA1_expandcleanedsigma') and self.AA1_expandcleanedsigma.value:
                conf_map['AA1_expandcleanedsigma'] = ast.literal_eval(self.AA1_expandcleanedsigma.value)

        # Standard fields
        if self.intensity_threshold.value:
            conf_map['intensity_threshold'] = ast.literal_eval(self.intensity_threshold.value)
        if self.binning.value:
            conf_map['binning'] = ast.literal_eval(self.binning.value)
        if self.shift.value:
            conf_map['shift'] = ast.literal_eval(self.shift.value)
        if self.crop_pad.value:
            conf_map['crop_pad'] = ast.literal_eval(self.crop_pad.value)
        if self.auto_intensity_threshold.value:
            conf_map['auto_intensity_threshold'] = True
        if self.no_center_max.value:
            conf_map['no_center_max'] = True

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.alien_alg.value = 'none'
        self.intensity_threshold.value = ''
        self.binning.value = ''
        self.shift.value = ''
        self.crop_pad.value = ''
        self.auto_intensity_threshold.value = False
        self.no_center_max.value = False

    def run_tab(self):
        """Execute data formatting."""
        import cohere_ui.standard_preprocess as run_dt

        self.output.clear_output()

        err = self._validate_experiment()
        if err:
            with self.output:
                print(f"Error: {err}")
            return

        # Check for prep_data.tif
        import os
        import cohere_core.utilities as ut
        found_file = False
        for p, d, f in os.walk(self.main_gui.experiment_dir):
            if 'prep_data.tif' in f:
                found_file = True
                break

        if not found_file:
            with self.output:
                print("Run data preparation in previous tab first")
            return

        conf_map = self.get_config()
        if conf_map:
            er_msg = self.main_gui.config_manager.save_config(self.conf_name, conf_map, self.main_gui.no_verify)
            if er_msg and not self.main_gui.no_verify:
                with self.output:
                    print(f"Config error: {er_msg}")
                return

        try:
            with self.output:
                print("Running data formatting...")
            run_dt.format_data(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            with self.output:
                print("Data formatting complete!")
        except Exception as e:
            with self.output:
                print(f"Error: {e}")

    def _load_config_dialog(self):
        """Placeholder for loading config from file."""
        with self.output:
            print("Enter config file path in working directory")
