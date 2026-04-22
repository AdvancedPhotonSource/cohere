"""
Display features: crop, interpolation, resolution, strain, etc.
"""

import ast
import ipywidgets as widgets

from .base import Feature
from ..widgets import form_row, text_field, checkbox, dropdown


class CropFeature(Feature):
    """Crop feature for display."""

    name = "crop"

    def fill_active(self) -> list:
        self.crop_type = dropdown(options=['tight', 'fraction'], value='fraction')
        self.crop_params = widgets.VBox()
        self.crop_type.observe(self._on_type_change, 'value')
        self._update_params()

        return [
            form_row('Crop Type', self.crop_type),
            self.crop_params,
        ]

    def _on_type_change(self, change):
        self._update_params()

    def _update_params(self):
        if self.crop_type.value == 'tight':
            self.margin = text_field(placeholder='10')
            self.thresh = text_field(placeholder='0.5')
            self.crop_params.children = [
                form_row('Margin', self.margin),
                form_row('Threshold', self.thresh),
            ]
        else:
            self.fraction = text_field(placeholder='[0.5, 0.5, 0.5]')
            self.crop_params.children = [
                form_row('Fraction', self.fraction),
            ]

    def set_defaults(self):
        self.crop_type.value = 'fraction'
        self._update_params()
        self.fraction.value = '[0.5, 0.5, 0.5]'

    def init_config(self, conf_map: dict):
        if 'crop_type' in conf_map:
            self.active.value = True
            self.crop_type.value = conf_map['crop_type']
            if conf_map['crop_type'] == 'tight':
                if 'crop_margin' in conf_map:
                    self.margin.value = str(conf_map['crop_margin'])
                if 'crop_thresh' in conf_map:
                    self.thresh.value = str(conf_map['crop_thresh'])
            else:
                if 'crop_fraction' in conf_map:
                    self.fraction.value = self.format_value(conf_map['crop_fraction'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        conf_map['crop_type'] = self.crop_type.value
        if self.crop_type.value == 'tight':
            if hasattr(self, 'margin') and self.margin.value:
                conf_map['crop_margin'] = self.parse_value(self.margin.value)
            if hasattr(self, 'thresh') and self.thresh.value:
                conf_map['crop_thresh'] = self.parse_value(self.thresh.value)
        else:
            if hasattr(self, 'fraction') and self.fraction.value:
                conf_map['crop_fraction'] = self.parse_value(self.fraction.value)


class InterpolationFeature(Feature):
    """Interpolation feature for display."""

    name = "interpolation"

    def fill_active(self) -> list:
        self.mode = dropdown(options=['AmpPhase', 'ReIm'], value='AmpPhase')
        self.resolution = text_field(placeholder='min_deconv_res')

        return [
            form_row('Mode', self.mode),
            form_row('Resolution', self.resolution),
        ]

    def set_defaults(self):
        self.mode.value = 'AmpPhase'
        self.resolution.value = 'min_deconv_res'

    def init_config(self, conf_map: dict):
        if 'interpolation_mode' in conf_map:
            self.active.value = True
            self.mode.value = conf_map['interpolation_mode']
            if 'interpolation_resolution' in conf_map:
                self.resolution.value = conf_map['interpolation_resolution']
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        conf_map['interpolation_mode'] = self.mode.value
        if self.resolution.value:
            conf_map['interpolation_resolution'] = self.resolution.value


class ResolutionFeature(Feature):
    """Resolution calculation feature."""

    name = "resolution"

    def fill_active(self) -> list:
        self.determine_type = text_field(placeholder='deconv')
        self.deconv_contrast = text_field(placeholder='0.25')

        return [
            form_row('Determine Type', self.determine_type),
            form_row('Deconv Contrast', self.deconv_contrast),
        ]

    def set_defaults(self):
        self.determine_type.value = 'deconv'
        self.deconv_contrast.value = '0.25'

    def init_config(self, conf_map: dict):
        if 'determine_resolution_type' in conf_map:
            self.active.value = True
            self.determine_type.value = conf_map['determine_resolution_type']
            if 'resolution_deconv_contrast' in conf_map:
                self.deconv_contrast.value = str(conf_map['resolution_deconv_contrast'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.determine_type.value:
            conf_map['determine_resolution_type'] = self.determine_type.value
        if self.deconv_contrast.value:
            conf_map['resolution_deconv_contrast'] = self.parse_value(self.deconv_contrast.value)


class ReciprocalFeature(Feature):
    """Reciprocal space visualization feature."""

    name = "reciprocal"

    def fill_active(self) -> list:
        self.write_recip = checkbox('write reciprocal space')

        return [
            self.write_recip,
        ]

    def set_defaults(self):
        self.write_recip.value = True

    def init_config(self, conf_map: dict):
        if 'write_recip' in conf_map:
            self.active.value = True
            self.write_recip.value = conf_map['write_recip']
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.write_recip.value:
            conf_map['write_recip'] = True


class StrainFeature(Feature):
    """Strain calculation feature."""

    name = "strain"

    def fill_active(self) -> list:
        self.compute_strain = checkbox('compute strain')

        return [
            self.compute_strain,
        ]

    def set_defaults(self):
        self.compute_strain.value = True

    def init_config(self, conf_map: dict):
        if 'compute_strain' in conf_map:
            self.active.value = True
            self.compute_strain.value = conf_map['compute_strain']
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.compute_strain.value:
            conf_map['compute_strain'] = True


class DisplacementFeature(Feature):
    """Bragg displacement feature."""

    name = "displacement"

    def fill_active(self) -> list:
        self.bragg_disp = checkbox('compute Bragg displacement')

        return [
            self.bragg_disp,
        ]

    def set_defaults(self):
        self.bragg_disp.value = True

    def init_config(self, conf_map: dict):
        if 'Bragg_displacement' in conf_map:
            self.active.value = True
            self.bragg_disp.value = conf_map['Bragg_displacement']
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.bragg_disp.value:
            conf_map['Bragg_displacement'] = True
