"""
PrepTab - Beamline preprocessing configuration.
"""

import ast
import ipywidgets as widgets

from .base import BaseTab
from ..widgets import form_row, text_field, dropdown, checkbox, button, output_area


class PrepTab(BaseTab):
    """Tab for beamline preprocessing configuration.

    Handles min_frames, roi, exclude_scans, outlier removal.
    """

    name = "Prep Data"
    conf_name = "config_prep"

    def _build_ui(self) -> widgets.Widget:
        self.min_frames = text_field(placeholder='e.g., 10')
        self.exclude_scans = text_field(placeholder='e.g., [1, 2, 3]')
        self.roi = text_field(placeholder='e.g., [x1, y1, x2, y2]')
        self.roi_format = dropdown(
            options=['', 'center_point_dist', 'start_point_end_point', 'start_point_dist'],
            value=''
        )
        self.max_crop = text_field(placeholder='e.g., [100, 100]')
        self.remove_outliers = checkbox('remove outliers')
        self.outliers_scans = text_field(placeholder='Auto-populated after prep')

        # Buttons with Qt-matching colors
        self.load_btn = button('Load Config', style='warning', width='120px', qt_style='load')
        self.run_btn = button('Prepare', style='success', width='120px', qt_style='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())

        self.output = output_area(height='150px')

        # Tooltip for ROI format
        roi_tooltip = widgets.HTML(
            '<small style="color: #666;">center_point_dist: [cx, cy, dx, dy] | '
            'start_point_end_point: [x1, y1, x2, y2] | '
            'start_point_dist: [x1, dx, y1, dy]</small>'
        )

        layout = widgets.VBox([
            form_row('Min Frames', self.min_frames),
            form_row('Exclude Scans', self.exclude_scans),
            form_row('ROI', self.roi),
            form_row('ROI Format', self.roi_format),
            roi_tooltip,
            form_row('Max Crop', self.max_crop),
            self.remove_outliers,
            form_row('Outliers Scans', self.outliers_scans),
            widgets.HBox([self.load_btn, self.run_btn]),
            self.output
        ])

        return layout

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        if 'min_frames' in conf_map:
            self.min_frames.value = str(conf_map['min_frames']).replace(' ', '')
        if 'exclude_scans' in conf_map:
            self.exclude_scans.value = str(conf_map['exclude_scans']).replace(' ', '')
        if 'roi' in conf_map:
            self.roi.value = str(conf_map['roi']).replace(' ', '')
        if 'roi_format' in conf_map:
            self.roi_format.value = conf_map['roi_format']
        else:
            self.roi_format.value = ''
        if 'max_crop' in conf_map:
            self.max_crop.value = str(conf_map['max_crop']).replace(' ', '')
        self.remove_outliers.value = conf_map.get('remove_outliers', False)
        if 'outliers_scans' in conf_map:
            self.outliers_scans.value = str(conf_map['outliers_scans']).replace(' ', '')

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        if self.min_frames.value:
            conf_map['min_frames'] = ast.literal_eval(self.min_frames.value)
        if self.exclude_scans.value:
            conf_map['exclude_scans'] = ast.literal_eval(self.exclude_scans.value)
        if self.roi.value:
            conf_map['roi'] = ast.literal_eval(self.roi.value)
        if self.roi_format.value:
            conf_map['roi_format'] = self.roi_format.value
        if self.max_crop.value:
            conf_map['max_crop'] = ast.literal_eval(self.max_crop.value)
        if self.remove_outliers.value:
            conf_map['remove_outliers'] = True

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.min_frames.value = ''
        self.exclude_scans.value = ''
        self.roi.value = ''
        self.roi_format.value = ''
        self.max_crop.value = ''
        self.outliers_scans.value = ''
        self.remove_outliers.value = False

    def run_tab(self):
        """Execute beamline preprocessing."""
        import cohere_ui.beamline_preprocess as prep

        self.output.clear_output()

        err = self._validate_experiment()
        if err:
            with self.output:
                print(f"Error: {err}")
            return

        conf_map = self.get_config()
        er_msg = self.main_gui.config_manager.verify(self.conf_name, conf_map)
        if er_msg and not self.main_gui.no_verify:
            with self.output:
                print(f"Config error: {er_msg}")
            return

        # Handle outliers_scans preservation
        import cohere_core.utilities as ut
        if self.remove_outliers.value:
            current_prep = self.main_gui.config_manager.load_config(self.conf_name)
            if current_prep and 'outliers_scans' in current_prep:
                conf_map['outliers_scans'] = current_prep['outliers_scans']

        self.main_gui.config_manager.save_config(self.conf_name, conf_map, self.main_gui.no_verify)

        try:
            with self.output:
                print("Running beamline preprocessing...")
            prep.handle_prep(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            with self.output:
                print("Preprocessing complete!")

            # Reload config to get updated outliers_scans
            updated_conf = self.main_gui.config_manager.load_config(self.conf_name)
            if updated_conf:
                self.clear_conf()
                self.load_tab(updated_conf)

        except ValueError as e:
            with self.output:
                print(f"ValueError: {e}")
        except FileNotFoundError as e:
            with self.output:
                print(f"FileNotFoundError: {e}")
        except KeyError as e:
            with self.output:
                print(f"KeyError: {e}")

    def _load_config_dialog(self):
        """Placeholder for loading config from file."""
        with self.output:
            print("Enter config file path in working directory")
