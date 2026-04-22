"""
DispTab - Display/visualization configuration and processing.
"""

import ast
import ipywidgets as widgets

from .base import BaseTab
from ..widgets import form_row, text_field, dropdown, checkbox, button, output_area, FeaturePanel


class DispTab(BaseTab):
    """Tab for visualization/postprocessing configuration.

    Handles result display, cropping, interpolation, strain visualization.
    """

    name = "Display"
    conf_name = "config_disp"

    def _build_ui(self) -> widgets.Widget:
        self.result_dir = text_field(placeholder='Path to phasing results', width='350px')
        self.make_twin = checkbox('make twin')
        self.unwrap = checkbox('include unwrapped phase')
        self.rampups = text_field(placeholder='e.g., 1')
        self.complex_mode = dropdown(options=['AmpPhase', 'ReIm'], value='AmpPhase')

        # Features panel
        from ..features import DISP_FEATURES
        self.features = {name: cls() for name, cls in DISP_FEATURES.items()}
        self.feature_panel = FeaturePanel(self.features)

        # Action buttons with Qt-matching colors
        self.load_btn = button('Load Config', style='warning', width='120px', qt_style='load')
        self.run_btn = button('Process Display', style='success', width='140px', qt_style='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())

        self.output = output_area(height='150px')

        params_section = widgets.VBox([
            form_row('Results Directory', self.result_dir),
            self.make_twin,
            self.unwrap,
            form_row('Ramp Upscale', self.rampups),
            form_row('Complex Mode', self.complex_mode),
        ])

        layout = widgets.VBox([
            params_section,
            widgets.HTML('<h4>Features</h4>'),
            self.feature_panel.widget,
            widgets.HBox([self.load_btn, self.run_btn]),
            self.output
        ])

        return layout

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        if 'results_dir' in conf_map:
            self.result_dir.value = conf_map['results_dir'].replace('\\', '/')
        self.make_twin.value = conf_map.get('make_twin', False)
        self.unwrap.value = conf_map.get('unwrap', False)
        if 'rampups' in conf_map:
            self.rampups.value = str(conf_map['rampups'])
        if 'complex_mode' in conf_map:
            self.complex_mode.value = conf_map['complex_mode']

        # Features
        self.feature_panel.init_configs(conf_map)

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        if self.result_dir.value:
            conf_map['results_dir'] = self.result_dir.value
        if self.make_twin.value:
            conf_map['make_twin'] = True
        if self.unwrap.value:
            conf_map['unwrap'] = True
        if self.rampups.value:
            conf_map['rampups'] = ast.literal_eval(self.rampups.value)
        conf_map['complex_mode'] = self.complex_mode.value

        # Features
        self.feature_panel.add_configs(conf_map)

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.result_dir.value = ''
        self.make_twin.value = False
        self.unwrap.value = False
        self.rampups.value = ''
        self.complex_mode.value = 'AmpPhase'
        self.feature_panel.clear_all()

    def run_tab(self):
        """Execute visualization/postprocessing."""
        import cohere_ui.beamline_postprocess as dp

        self.output.clear_output()

        err = self._validate_experiment()
        if err:
            with self.output:
                print(f"Error: {err}")
            return

        # Set default results_dir if empty
        if not self.result_dir.value:
            self.result_dir.value = self.main_gui.experiment_dir
            with self.output:
                print(f"Setting results directory to: {self.result_dir.value}")

        conf_map = self.get_config()
        er_msg = self.main_gui.config_manager.verify(self.conf_name, conf_map)
        if er_msg and not self.main_gui.no_verify:
            with self.output:
                print(f"Config error: {er_msg}")
            return

        self.main_gui.config_manager.save_config(self.conf_name, conf_map, self.main_gui.no_verify)

        try:
            with self.output:
                print("Running visualization...")
            dp.handle_visualization(self.main_gui.experiment_dir, no_verify=self.main_gui.no_verify)
            with self.output:
                print("Visualization complete!")
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

    def update_tab(self, **kwargs):
        """Update tab from external notification (e.g., after reconstruction)."""
        if 'rec_id' in kwargs:
            import cohere_core.utilities as ut
            results_dir = ut.join(self.main_gui.experiment_dir, f'results_phasing_{kwargs["rec_id"]}')
            self.result_dir.value = results_dir
