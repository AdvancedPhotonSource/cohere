"""
RecTab - Reconstruction configuration and processing.
"""

import ast
import sys
import ipywidgets as widgets

from .base import BaseTab
from ..widgets import form_row, text_field, dropdown, checkbox, button, output_area, FeaturePanel


class RecTab(BaseTab):
    """Tab for reconstruction configuration.

    Handles algorithm sequence, initial guess, device settings, and features.
    """

    name = "Reconstruction"
    conf_name = "config_rec"

    def _build_ui(self) -> widgets.Widget:
        # Initial guess
        self.init_guess = dropdown(
            options=['random', 'continue', 'AI algorithm'],
            value='random'
        )
        self.init_guess_params = widgets.VBox()
        self.init_guess.observe(self._on_init_guess_change, 'value')

        # Processor settings
        proc_options = ['auto', 'np', 'torch']
        if sys.platform != 'darwin':
            proc_options.insert(1, 'cp')
        self.proc = dropdown(options=proc_options, value='auto')
        self.device = text_field(placeholder='e.g., [0] or "all"')
        self.reconstructions = text_field(placeholder='e.g., 1')

        # Algorithm settings
        self.alg_seq = text_field(placeholder='e.g., 3*(20*ER+180*HIO)+20*ER', width='300px')
        self.hio_beta = text_field(placeholder='0.9')
        self.raar_beta = text_field(placeholder='0.45')
        self.initial_support_area = text_field(placeholder='[0.5, 0.5, 0.5]')

        # Defaults button with Qt-matching color
        self.defaults_btn = button('Set Defaults', style='info', width='120px', qt_style='info')
        self.defaults_btn.on_click(lambda b: self._set_defaults())

        # Features panel
        from ..features import REC_FEATURES
        self.features = {name: cls() for name, cls in REC_FEATURES.items()}
        self.feature_panel = FeaturePanel(self.features)

        # Action buttons with Qt-matching colors
        self.load_btn = button('Load Config', style='warning', width='120px', qt_style='load')
        self.run_btn = button('Run Reconstruction', style='success', width='150px', qt_style='run')
        self.load_btn.on_click(lambda b: self._load_config_dialog())
        self.run_btn.on_click(lambda b: self.run_tab())

        self.output = output_area(height='150px')

        # Layout
        params_section = widgets.VBox([
            form_row('Initial Guess', self.init_guess),
            self.init_guess_params,
            form_row('Processor', self.proc),
            form_row('Device(s)', self.device),
            form_row('Reconstructions', self.reconstructions),
            form_row('Algorithm Sequence', self.alg_seq),
            form_row('HIO Beta', self.hio_beta),
            form_row('RAAR Beta', self.raar_beta),
            form_row('Initial Support Area', self.initial_support_area),
            self.defaults_btn,
        ])

        layout = widgets.VBox([
            params_section,
            widgets.HTML('<h4>Features</h4>'),
            self.feature_panel.widget,
            widgets.HBox([self.load_btn, self.run_btn]),
            self.output
        ])

        return layout

    def _on_init_guess_change(self, change):
        """Update init guess parameters based on selection."""
        guess = change['new']
        self.init_guess_params.children = []

        if guess == 'continue':
            self.cont_dir = text_field(placeholder='Path to continue directory')
            self.init_guess_params.children = [form_row('Continue Directory', self.cont_dir)]
        elif guess == 'AI algorithm':
            self.ai_model = text_field(placeholder='Path to trained model .hdf5')
            self.init_guess_params.children = [form_row('AI Model File', self.ai_model)]

    def _set_defaults(self):
        """Set reconstruction parameters to defaults."""
        self.reconstructions.value = '1'
        self.proc.value = 'auto'
        self.device.value = '[0]'
        self.alg_seq.value = '3*(20*ER+180*HIO)+20*ER'
        self.hio_beta.value = '.9'
        self.raar_beta.value = '.45'
        self.initial_support_area.value = '[0.5, 0.5, 0.5]'

    def load_tab(self, conf_map: dict):
        """Populate widgets from config dictionary."""
        # Initial guess
        init_guess = conf_map.get('init_guess', 'random')
        if init_guess == 'random':
            self.init_guess.value = 'random'
        elif init_guess == 'continue':
            self.init_guess.value = 'continue'
            if 'continue_dir' in conf_map and hasattr(self, 'cont_dir'):
                self.cont_dir.value = conf_map['continue_dir'].replace('\\', '/')
        elif init_guess == 'AI_guess':
            self.init_guess.value = 'AI algorithm'
            if 'AI_trained_model' in conf_map and hasattr(self, 'ai_model'):
                self.ai_model.value = conf_map['AI_trained_model'].replace('\\', '/')

        # Standard fields
        if 'processing' in conf_map:
            self.proc.value = conf_map['processing']
        if 'device' in conf_map:
            self.device.value = str(conf_map['device']).replace(' ', '')
        if 'reconstructions' in conf_map:
            self.reconstructions.value = str(conf_map['reconstructions'])
        if 'algorithm_sequence' in conf_map:
            self.alg_seq.value = str(conf_map['algorithm_sequence'])
        if 'hio_beta' in conf_map:
            self.hio_beta.value = str(conf_map['hio_beta'])
        if 'raar_beta' in conf_map:
            self.raar_beta.value = str(conf_map['raar_beta'])
        if 'initial_support_area' in conf_map:
            self.initial_support_area.value = str(conf_map['initial_support_area']).replace(' ', '')

        # Features
        self.feature_panel.init_configs(conf_map)

    def get_config(self) -> dict:
        """Read current widget values into config dictionary."""
        conf_map = {}

        # Initial guess
        if self.init_guess.value == 'continue':
            conf_map['init_guess'] = 'continue'
            if hasattr(self, 'cont_dir') and self.cont_dir.value:
                conf_map['continue_dir'] = self.cont_dir.value
        elif self.init_guess.value == 'AI algorithm':
            conf_map['init_guess'] = 'AI_guess'
            if hasattr(self, 'ai_model') and self.ai_model.value:
                conf_map['AI_trained_model'] = self.ai_model.value

        # Standard fields
        if self.proc.value:
            conf_map['processing'] = self.proc.value
        if self.device.value:
            dev = self.device.value.strip()
            if dev == 'all':
                conf_map['device'] = dev
            else:
                conf_map['device'] = ast.literal_eval(dev)
        if self.reconstructions.value:
            conf_map['reconstructions'] = ast.literal_eval(self.reconstructions.value)
        if self.alg_seq.value:
            conf_map['algorithm_sequence'] = self.alg_seq.value.strip()
        if self.hio_beta.value:
            conf_map['hio_beta'] = ast.literal_eval(self.hio_beta.value)
        if self.raar_beta.value:
            conf_map['raar_beta'] = ast.literal_eval(self.raar_beta.value)
        if self.initial_support_area.value:
            conf_map['initial_support_area'] = ast.literal_eval(self.initial_support_area.value)

        # Features
        self.feature_panel.add_configs(conf_map)

        return conf_map

    def clear_conf(self):
        """Reset all widgets to defaults."""
        self.init_guess.value = 'random'
        self.init_guess_params.children = []
        self.proc.value = 'auto'
        self.device.value = ''
        self.reconstructions.value = ''
        self.alg_seq.value = ''
        self.hio_beta.value = ''
        self.raar_beta.value = ''
        self.initial_support_area.value = ''
        self.feature_panel.clear_all()

    def run_tab(self):
        """Execute reconstruction."""
        import cohere_ui.cohere_reconstruction as run_rc
        import os
        import cohere_core.utilities as ut

        self.output.clear_output()

        err = self._validate_experiment()
        if err:
            with self.output:
                print(f"Error: {err}")
            return

        # Check for data.tif or data.npy
        found_file = False
        for p, d, f in os.walk(self.main_gui.experiment_dir):
            if 'data.tif' in f or 'data.npy' in f:
                found_file = True
                break

        if not found_file:
            with self.output:
                print("Run data formatting in previous tab first")
            return

        # Verify features
        for feat in self.features.values():
            err_msg = feat.verify_active()
            if err_msg:
                with self.output:
                    print(f"Feature error: {err_msg}")
                return

        conf_map = self.get_config()
        if not conf_map:
            return

        er_msg = self.main_gui.config_manager.verify(self.conf_name, conf_map)
        if er_msg and not self.main_gui.no_verify:
            with self.output:
                print(f"Config error: {er_msg}")
            return

        self.main_gui.config_manager.save_config(self.conf_name, conf_map, self.main_gui.no_verify)

        try:
            with self.output:
                print("Running reconstruction...")
            run_rc.manage_reconstruction(
                self.main_gui.experiment_dir,
                no_verify=self.main_gui.no_verify,
                debug=self.main_gui.debug
            )
            with self.output:
                print("Reconstruction complete!")
            self.main_gui.results.reload()
        except Exception as e:
            with self.output:
                print(f"Error: {e}")

    def _load_config_dialog(self):
        """Placeholder for loading config from file."""
        with self.output:
            print("Enter config file path in working directory")
