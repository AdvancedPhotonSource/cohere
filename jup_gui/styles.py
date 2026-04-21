"""
Custom styling for jup_gui to match Qt cohere_gui colors.

Qt GUI Button Colors (from cohere_gui.py):
- Load buttons: rgb(205, 178, 102) - gold/tan
- Set button: rgb(120, 180, 220) - light blue
- Run buttons: rgb(175, 208, 156) - light green
"""

from IPython.display import display, HTML

# Qt color definitions
QT_COLORS = {
    'load': 'rgb(205, 178, 102)',      # Gold/tan - load actions
    'set': 'rgb(120, 180, 220)',       # Light blue - set/create actions
    'run': 'rgb(175, 208, 156)',       # Light green - run/execute actions
    'info': 'rgb(0, 151, 167)',        # Teal - info/defaults (keep ipywidgets)
}

# CSS to inject for custom button styling
CUSTOM_CSS = """
<style>
/* Qt-matching button colors for jup_gui */

/* Load buttons - gold/tan rgb(205, 178, 102) */
.jup-gui-load {
    background-color: rgb(205, 178, 102) !important;
    border-color: rgb(185, 158, 82) !important;
    color: #000 !important;
}
.jup-gui-load:hover {
    background-color: rgb(185, 158, 82) !important;
}

/* Set/Create buttons - light blue rgb(120, 180, 220) */
.jup-gui-set {
    background-color: rgb(120, 180, 220) !important;
    border-color: rgb(100, 160, 200) !important;
    color: #000 !important;
}
.jup-gui-set:hover {
    background-color: rgb(100, 160, 200) !important;
}

/* Run buttons - light green rgb(175, 208, 156) */
.jup-gui-run {
    background-color: rgb(175, 208, 156) !important;
    border-color: rgb(155, 188, 136) !important;
    color: #000 !important;
}
.jup-gui-run:hover {
    background-color: rgb(155, 188, 136) !important;
}

/* Info/defaults buttons - keep teal */
.jup-gui-info {
    background-color: rgb(0, 151, 167) !important;
    border-color: rgb(0, 131, 147) !important;
    color: #fff !important;
}
.jup-gui-info:hover {
    background-color: rgb(0, 131, 147) !important;
}
</style>
"""

_css_injected = False


def inject_custom_css():
    """Inject custom CSS into the notebook for Qt-matching colors."""
    global _css_injected
    if not _css_injected:
        display(HTML(CUSTOM_CSS))
        _css_injected = True


def apply_qt_style(btn, style_type: str):
    """Apply Qt-matching style class to a button widget.

    Args:
        btn: ipywidgets Button
        style_type: One of 'load', 'set', 'run', 'info'
    """
    class_name = f'jup-gui-{style_type}'
    if hasattr(btn, 'add_class'):
        btn.add_class(class_name)
    else:
        # Fallback: modify the button's CSS classes via layout
        current = btn._dom_classes if hasattr(btn, '_dom_classes') else ()
        btn._dom_classes = tuple(set(current) | {class_name})
