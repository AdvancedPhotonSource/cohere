"""
Widget factory functions and helpers for building ipywidgets-based forms.
"""

import os
import ipywidgets as widgets
from IPython.display import display

from .styles import inject_custom_css, apply_qt_style

try:
    from ipyfilechooser import FileChooser
    HAS_FILECHOOSER = True
except ImportError:
    HAS_FILECHOOSER = False


def form_row(label: str, widget, label_width: str = '180px') -> widgets.HBox:
    """Create a horizontal label-widget pair like QFormLayout.addRow()."""
    label_widget = widgets.Label(value=label, layout=widgets.Layout(width=label_width))
    return widgets.HBox([label_widget, widget])


def text_field(value: str = '', placeholder: str = '', width: str = '200px') -> widgets.Text:
    """Create a text input field."""
    return widgets.Text(value=value, placeholder=placeholder, layout=widgets.Layout(width=width))


def int_field(value: int = 0, width: str = '100px') -> widgets.IntText:
    """Create an integer input field."""
    return widgets.IntText(value=value, layout=widgets.Layout(width=width))


def float_field(value: float = 0.0, width: str = '100px') -> widgets.FloatText:
    """Create a float input field."""
    return widgets.FloatText(value=value, layout=widgets.Layout(width=width))


def dropdown(options: list, value=None, width: str = '200px') -> widgets.Dropdown:
    """Create a dropdown selection widget."""
    return widgets.Dropdown(
        options=options,
        value=value if value is not None else (options[0] if options else None),
        layout=widgets.Layout(width=width)
    )


def checkbox(description: str = '', value: bool = False) -> widgets.Checkbox:
    """Create a checkbox widget."""
    return widgets.Checkbox(value=value, description=description, indent=False)


def button(description: str, style: str = 'primary', width: str = '150px',
           qt_style: str = None) -> widgets.Button:
    """Create a button widget with predefined styles.

    Args:
        description: Button text
        style: ipywidgets button_style ('primary', 'success', 'warning', 'danger', 'info')
        width: CSS width string
        qt_style: Optional Qt-matching style ('load', 'set', 'run', 'info').
                  If provided, applies custom CSS class for Qt color matching.

    Styles: 'primary' (blue), 'success' (green), 'warning' (orange), 'danger' (red), 'info'
    """
    btn = widgets.Button(
        description=description,
        button_style=style,
        layout=widgets.Layout(width=width)
    )
    if qt_style:
        apply_qt_style(btn, qt_style)
    return btn


class DirChooser:
    """Directory chooser widget similar to Qt's QFileDialog.

    Uses ipyfilechooser if available, otherwise falls back to button + text display.
    """

    def __init__(self, start_path: str = None, title: str = 'Select Directory'):
        self.title = title
        self._value = ''
        self._callbacks = []

        start_path = start_path or os.getcwd()

        if HAS_FILECHOOSER:
            self._fc = FileChooser(
                path=start_path,
                select_default=True,
                show_only_dirs=True,
                title=title
            )
            self._fc.register_callback(self._on_select)
            self.widget = self._fc
        else:
            # Fallback: button + path display + manual entry
            self._path_display = widgets.Text(
                value='',
                placeholder='Click Browse or enter path...',
                layout=widgets.Layout(width='350px')
            )
            self._browse_btn = widgets.Button(
                description='Browse...',
                button_style='info',
                layout=widgets.Layout(width='80px')
            )
            self._browse_btn.on_click(self._show_browser)
            self._browser_output = widgets.Output()

            self.widget = widgets.VBox([
                widgets.HBox([self._path_display, self._browse_btn]),
                self._browser_output
            ])

    def _on_select(self, chooser):
        """Callback when directory is selected via ipyfilechooser."""
        self._value = chooser.selected_path or ''
        for cb in self._callbacks:
            cb(self._value)

    def _show_browser(self, b):
        """Show a simple directory browser in fallback mode."""
        with self._browser_output:
            self._browser_output.clear_output()
            current = self._path_display.value or os.getcwd()
            if not os.path.isdir(current):
                current = os.getcwd()

            print(f"Current: {current}")
            print("Subdirectories:")

            # Show parent
            parent = os.path.dirname(current)
            if parent and parent != current:
                print(f"  [..] (parent)")

            # Show subdirectories
            try:
                for item in sorted(os.listdir(current)):
                    full_path = os.path.join(current, item)
                    if os.path.isdir(full_path) and not item.startswith('.'):
                        print(f"  [{item}]")
            except PermissionError:
                print("  (permission denied)")

            print("\nEnter path in text field above and press Enter")

    @property
    def value(self) -> str:
        if HAS_FILECHOOSER:
            return self._fc.selected_path or ''
        return self._path_display.value

    @value.setter
    def value(self, val: str):
        self._value = val
        if HAS_FILECHOOSER:
            if val and os.path.isdir(val):
                self._fc.reset(path=val)
        else:
            self._path_display.value = val

    def register_callback(self, callback):
        """Register a callback for when selection changes."""
        self._callbacks.append(callback)
        if HAS_FILECHOOSER:
            pass  # Already registered in __init__
        else:
            self._path_display.observe(lambda c: callback(c['new']), 'value')


class FileChooserWidget:
    """File chooser widget similar to Qt's QFileDialog for files."""

    def __init__(self, start_path: str = None, filter_pattern: str = '*', title: str = 'Select File'):
        self.title = title
        self._value = ''
        self._callbacks = []

        start_path = start_path or os.getcwd()

        if HAS_FILECHOOSER:
            self._fc = FileChooser(
                path=start_path,
                filter_pattern=filter_pattern,
                select_default=False,
                title=title
            )
            self._fc.register_callback(self._on_select)
            self.widget = self._fc
        else:
            self._path_display = widgets.Text(
                value='',
                placeholder='Enter file path...',
                layout=widgets.Layout(width='400px')
            )
            self.widget = self._path_display

    def _on_select(self, chooser):
        self._value = chooser.selected or ''
        for cb in self._callbacks:
            cb(self._value)

    @property
    def value(self) -> str:
        if HAS_FILECHOOSER:
            return self._fc.selected or ''
        return self._path_display.value

    @value.setter
    def value(self, val: str):
        self._value = val
        if HAS_FILECHOOSER:
            if val and os.path.exists(val):
                self._fc.reset(path=os.path.dirname(val), filename=os.path.basename(val))
        else:
            self._path_display.value = val

    def register_callback(self, callback):
        self._callbacks.append(callback)


def dir_chooser(start_path: str = None, title: str = 'Select Directory') -> DirChooser:
    """Create a directory chooser widget like Qt's QFileDialog."""
    return DirChooser(start_path=start_path, title=title)


def file_chooser(start_path: str = None, filter_pattern: str = '*', title: str = 'Select File') -> FileChooserWidget:
    """Create a file chooser widget like Qt's QFileDialog."""
    return FileChooserWidget(start_path=start_path, filter_pattern=filter_pattern, title=title)


def output_area(height: str = '200px') -> widgets.Output:
    """Create an output area for messages and logs."""
    return widgets.Output(layout=widgets.Layout(
        border='1px solid #ccc',
        height=height,
        overflow='auto'
    ))


def section_header(text: str) -> widgets.HTML:
    """Create a section header."""
    return widgets.HTML(f'<h4 style="margin: 10px 0 5px 0; color: #333;">{text}</h4>')


class FeaturePanel:
    """A panel for displaying toggleable features with stacked parameter views.

    Similar to Qt's QListWidget + QStackedWidget combination.
    """

    def __init__(self, features: dict):
        """
        Args:
            features: Dict mapping feature names to Feature objects
        """
        self.features = features
        self.feature_names = list(features.keys())

        self.selector = widgets.Select(
            options=self.feature_names,
            value=self.feature_names[0] if self.feature_names else None,
            layout=widgets.Layout(width='150px', height='200px')
        )

        self.params_area = widgets.VBox(layout=widgets.Layout(
            width='350px',
            min_height='200px',
            padding='10px',
            border='1px solid #ddd'
        ))

        self.selector.observe(self._on_select, 'value')
        self._update_params_display()

        self.widget = widgets.HBox([self.selector, self.params_area])

    def _on_select(self, change):
        self._update_params_display()

    def _update_params_display(self):
        if self.selector.value and self.selector.value in self.features:
            feature = self.features[self.selector.value]
            self.params_area.children = [feature.widget]
        else:
            self.params_area.children = []

    def init_configs(self, conf_map: dict):
        """Initialize all features from config dictionary."""
        for feature in self.features.values():
            feature.init_config(conf_map)
        self._update_params_display()

    def add_configs(self, conf_map: dict):
        """Add all active feature configs to dictionary."""
        for feature in self.features.values():
            feature.add_config(conf_map)

    def clear_all(self):
        """Clear all feature configurations."""
        for feature in self.features.values():
            feature.clear()
        self._update_params_display()
