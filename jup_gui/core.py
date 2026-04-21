"""
Main CoherenceGUI class - the Jupyter notebook GUI for cohere reconstruction.
"""

import os
import ipywidgets as widgets
from IPython.display import display

from .config import ConfigManager
from .results import ResultsContainer
from .widgets import form_row, text_field, checkbox, button, dir_chooser, output_area, DirChooser
from .styles import inject_custom_css
import cohere_core.utilities as ut


class CoherenceGUI:
    """Main GUI container for Bragg CDI reconstruction.

    Provides a tab-based interface similar to the Qt cohere_gui.
    """

    def __init__(self):
        self.config_manager = ConfigManager()
        self.results = ResultsContainer(self.config_manager)

        self.no_verify = False
        self.debug = False

        self._loaded = False
        self._id = None
        self._exp_id = None

        self._tabs = {}
        self._widget = None
        self._build_ui()

    def _build_ui(self):
        """Build the main GUI widget tree."""
        # Working directory chooser (like Qt's QPushButton that opens file dialog)
        self.working_dir = dir_chooser(start_path=os.getcwd(), title='Select Working Directory')

        # Experiment directory chooser for loading (separate from working dir)
        self.experiment_chooser = dir_chooser(start_path=os.getcwd(), title='Select Experiment to Load')

        # Text fields for experiment settings
        self.experiment_id = text_field(placeholder='Experiment ID', width='200px')
        self.scan = text_field(placeholder='e.g., 2-7 or 54,56', width='150px')
        self.beamline = text_field(placeholder='e.g., aps_34idc', width='150px')

        self.separate_scans = checkbox('separate scans')
        self.separate_scan_ranges = checkbox('separate scan ranges')
        self.multipeak = checkbox('multi peak')

        # Section: Working Directory
        working_dir_section = widgets.VBox([
            widgets.HTML('<b>Working Directory</b> (for new experiments):'),
            self.working_dir.widget,
        ])

        # Section: Load Existing Experiment
        load_section = widgets.VBox([
            widgets.HTML('<b>Load Existing Experiment</b> (select experiment folder with conf/):'),
            self.experiment_chooser.widget,
        ])

        # Section: Experiment Settings
        settings_section = widgets.VBox([
            widgets.HTML('<b>Experiment Settings</b>:'),
            widgets.HBox([
                form_row('Experiment ID', self.experiment_id),
                form_row('Scan(s)', self.scan),
                form_row('Beamline', self.beamline),
            ]),
            widgets.HBox([self.separate_scans, self.separate_scan_ranges, self.multipeak])
        ])

        # Action buttons with Qt-matching colors
        self.load_btn = button('Load Selected Experiment', style='warning', width='200px', qt_style='load')
        self.set_btn = button('Set/Create Experiment', style='info', width='180px', qt_style='set')
        self.run_all_btn = button('Run All', style='success', width='100px', qt_style='run')

        self.load_btn.on_click(lambda b: self._load_from_chooser())
        self.set_btn.on_click(lambda b: self.set_experiment())
        self.run_all_btn.on_click(lambda b: self.run_all())

        buttons = widgets.HBox([self.load_btn, self.set_btn, self.run_all_btn])

        # Output area
        self.output = output_area(height='80px')

        # Tab container - built lazily when tabs are initialized
        self.tab_widget = widgets.Tab()

        # Main layout
        self._widget = widgets.VBox([
            load_section,
            buttons,
            self.output,
            widgets.HTML('<hr>'),
            working_dir_section,
            settings_section,
            self.tab_widget
        ])

    def _load_from_chooser(self):
        """Load experiment from the experiment chooser widget."""
        path = self.experiment_chooser.value
        if path:
            self.load_experiment(path)
        else:
            self.log("Please select an experiment directory first")

    def init_tabs(self, beamline: str = None):
        """Initialize all tabs after GUI is created.

        Args:
            beamline: Optional beamline name to include InstrTab
        """
        from .tabs import DataTab, PrepTab, RecTab, DispTab, InstrTab

        self._tabs = {}

        # Add InstrTab if beamline is specified
        if beamline:
            instr_tab = InstrTab(beamline)
            self._tabs['instr'] = instr_tab

        # Core tabs
        self._tabs['prep'] = PrepTab()
        self._tabs['data'] = DataTab()
        self._tabs['rec'] = RecTab()
        self._tabs['disp'] = DispTab()

        self._refresh_tab_widget()

    def _refresh_tab_widget(self):
        """Refresh the tab widget with current tabs."""
        tab_children = []
        tab_titles = []

        for name, tab in self._tabs.items():
            tab.init(self)
            tab_children.append(tab.widget)
            tab_titles.append(tab.name)

        self.tab_widget.children = tab_children
        for i, title in enumerate(tab_titles):
            self.tab_widget.set_title(i, title)

    def add_instr_tab(self, beamline: str):
        """Add or update the instrument tab for a beamline."""
        from .tabs import InstrTab

        if 'instr' in self._tabs:
            self._tabs['instr'].set_beamline(beamline)
        else:
            instr_tab = InstrTab(beamline)
            # Insert at beginning
            new_tabs = {'instr': instr_tab}
            new_tabs.update(self._tabs)
            self._tabs = new_tabs
            self._refresh_tab_widget()

    @property
    def widget(self) -> widgets.Widget:
        """The main GUI widget."""
        return self._widget

    @property
    def experiment_dir(self) -> str:
        """Current experiment directory path."""
        return self.config_manager.experiment_dir

    def display(self):
        """Display the GUI in the notebook."""
        inject_custom_css()  # Inject Qt-matching button colors
        if not self._tabs:
            self.init_tabs()
        display(self._widget)

    def log(self, message: str):
        """Log a message to the output area."""
        with self.output:
            print(message)

    def clear_log(self):
        """Clear the output area."""
        self.output.clear_output()

    def is_exp_exists(self) -> bool:
        """Check if experiment directory exists with required config."""
        if not self._exp_id or not self.working_dir.value:
            return False
        exp_dir = ut.join(self.working_dir.value, self._exp_id)
        return os.path.exists(exp_dir)

    def is_exp_set(self) -> bool:
        """Check if experiment settings haven't changed since last set."""
        if not self._exp_id or not self.working_dir.value:
            return False
        if self._id != self.experiment_id.value.strip():
            return False
        return True

    def reset_window(self):
        """Reset all GUI fields to empty state."""
        self._exp_id = None
        self._id = None
        self.working_dir.value = ''
        self.experiment_id.value = ''
        self.scan.value = ''
        self.beamline.value = ''
        self.separate_scans.value = False
        self.separate_scan_ranges.value = False
        self.multipeak.value = False

        for tab in self._tabs.values():
            tab.clear_conf()

    def load_experiment(self, path: str = None):
        """Load an existing experiment from directory.

        Args:
            path: Experiment directory path. If None, uses working_dir + exp_id
        """
        self.clear_log()

        if path:
            load_dir = path
        else:
            if not self.working_dir.value:
                self.log("Please enter a working directory")
                return
            load_dir = self.working_dir.value

        # Handle relative paths
        if not os.path.isabs(load_dir):
            load_dir = os.path.abspath(load_dir)

        config_path = ut.join(load_dir, 'conf', 'config')
        if not os.path.isfile(config_path):
            self.log(f"Missing conf/config file in {load_dir}")
            return

        try:
            self.config_manager.set_experiment_dir(load_dir)
            conf_list = ['config_prep', 'config_data', 'config_rec', 'config_disp', 'config_instr']
            conf_dicts = self.config_manager.load_configs(conf_list, no_verify=True)
        except Exception as e:
            self.log(f"Error loading configs: {e}")
            return

        self._load_main(conf_dicts.get('config', {}))
        self._loaded = True

        # Add InstrTab if beamline is set and config_instr exists
        beamline = conf_dicts.get('config', {}).get('beamline')
        if beamline and 'config_instr' in conf_dicts:
            if 'instr' not in self._tabs:
                self.add_instr_tab(beamline)

        # Load config into tabs
        for tab in self._tabs.values():
            if tab.conf_name in conf_dicts:
                tab.load_tab(conf_dicts[tab.conf_name])

        self.results.set_config_manager(self.config_manager)
        self.log(f"Loaded experiment: {load_dir}")

    def _load_main(self, conf_map: dict):
        """Load main config values into header widgets."""
        if 'working_dir' in conf_map:
            self.working_dir.value = conf_map['working_dir'].replace(os.sep, '/')
        if 'experiment_id' in conf_map:
            self.experiment_id.value = conf_map['experiment_id']
            self._id = conf_map['experiment_id']
        if 'scan' in conf_map:
            self.scan.value = str(conf_map['scan']).replace(' ', '')
        if 'beamline' in conf_map:
            self.beamline.value = conf_map['beamline']
        if conf_map.get('separate_scans'):
            self.separate_scans.value = True
        if conf_map.get('separate_scan_ranges'):
            self.separate_scan_ranges.value = True
        if conf_map.get('multipeak'):
            self.multipeak.value = True

        self._update_exp_id()

    def _update_exp_id(self):
        """Update the experiment ID from current field values."""
        self._id = self.experiment_id.value.strip()
        scan_text = self.scan.value.strip().replace(' ', '')
        if scan_text:
            self._exp_id = f"{self._id}_{scan_text}"
        else:
            self._exp_id = self._id

    def set_experiment(self):
        """Set/create experiment from current field values."""
        self.clear_log()

        if not self.working_dir.value:
            self.log("Please enter a working directory")
            return

        if not os.path.isdir(self.working_dir.value):
            self.log(f"Working directory does not exist: {self.working_dir.value}")
            return

        if not self.experiment_id.value.strip():
            self.log("Please enter an experiment ID")
            return

        self._update_exp_id()
        exp_dir = ut.join(self.working_dir.value, self._exp_id)

        self.config_manager.set_experiment_dir(exp_dir)
        self.config_manager.ensure_experiment_dir()

        self._save_main()

        for tab in self._tabs.values():
            tab.save_conf()

        self.results.set_config_manager(self.config_manager)
        self.log(f"Experiment set: {exp_dir}")

    def _save_main(self):
        """Save main config from header widgets."""
        conf_map = {
            'working_dir': self.working_dir.value,
            'experiment_id': self._id,
        }

        if self.scan.value.strip():
            conf_map['scan'] = self.scan.value.strip()
        if self.beamline.value.strip():
            conf_map['beamline'] = self.beamline.value.strip()
        if self.multipeak.value:
            conf_map['multipeak'] = True
        if self.separate_scans.value:
            conf_map['separate_scans'] = True
        if self.separate_scan_ranges.value:
            conf_map['separate_scan_ranges'] = True

        self.config_manager.save_config('config', conf_map, no_verify=self.no_verify)

    def run_all(self):
        """Run all tabs in sequence."""
        if not self.is_exp_exists():
            self.log("Experiment has not been created yet")
            return
        if not self.is_exp_set():
            self.log("Experiment has changed, press 'Set Experiment' button")
            return

        for tab in self._tabs.values():
            tab.run_tab()
