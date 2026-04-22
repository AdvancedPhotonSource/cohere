"""
Reconstruction features: GA, shrink_wrap, pcdi, twin, etc.
"""

import ast
import ipywidgets as widgets

from .base import Feature
from ..widgets import form_row, text_field, checkbox


class GAFeature(Feature):
    """Genetic Algorithm feature."""

    name = "GA"

    def fill_active(self) -> list:
        self.ga_fast = checkbox('fast processing (size limited)')
        self.generations = text_field(placeholder='e.g., 3')
        self.metrics = text_field(placeholder='["chi"]')
        self.breed_modes = text_field(placeholder='["sqrt_ab"]')
        self.cullings = text_field(placeholder='e.g., [2]')
        self.sw_thresholds = text_field(placeholder='[0.1]')
        self.sw_gauss_sigmas = text_field(placeholder='[1.0]')
        self.lpf_sigmas = text_field(placeholder='')
        self.gen_pc_start = text_field(placeholder='3')

        return [
            self.ga_fast,
            form_row('Generations', self.generations),
            form_row('Fitness Metrics', self.metrics),
            form_row('Breed Modes', self.breed_modes),
            form_row('Cullings', self.cullings),
            form_row('SW Thresholds', self.sw_thresholds),
            form_row('SW Gauss Sigmas', self.sw_gauss_sigmas),
            form_row('LPF Sigmas', self.lpf_sigmas),
            form_row('Gen PC Start', self.gen_pc_start),
        ]

    def set_defaults(self):
        self.generations.value = '3'
        self.metrics.value = '["chi"]'
        self.breed_modes.value = '["sqrt_ab"]'
        self.sw_thresholds.value = '[.1]'
        self.sw_gauss_sigmas.value = '[1.0]'
        self.gen_pc_start.value = '3'
        self.ga_fast.value = False

    def init_config(self, conf_map: dict):
        if 'ga_generations' in conf_map:
            self.active.value = True
            self.generations.value = str(conf_map['ga_generations'])
            if 'ga_fast' in conf_map:
                self.ga_fast.value = conf_map['ga_fast']
            if 'ga_metrics' in conf_map:
                self.metrics.value = self.format_value(conf_map['ga_metrics'])
            if 'ga_breed_modes' in conf_map:
                self.breed_modes.value = self.format_value(conf_map['ga_breed_modes'])
            if 'ga_cullings' in conf_map:
                self.cullings.value = self.format_value(conf_map['ga_cullings'])
            if 'ga_sw_thresholds' in conf_map:
                self.sw_thresholds.value = self.format_value(conf_map['ga_sw_thresholds'])
            if 'ga_sw_gauss_sigmas' in conf_map:
                self.sw_gauss_sigmas.value = self.format_value(conf_map['ga_sw_gauss_sigmas'])
            if 'ga_lpf_sigmas' in conf_map:
                self.lpf_sigmas.value = self.format_value(conf_map['ga_lpf_sigmas'])
            if 'ga_gen_pc_start' in conf_map:
                self.gen_pc_start.value = self.format_value(conf_map['ga_gen_pc_start'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.ga_fast.value:
            conf_map['ga_fast'] = True
        if self.generations.value:
            conf_map['ga_generations'] = self.parse_value(self.generations.value)
        if self.metrics.value:
            conf_map['ga_metrics'] = self.parse_value(self.metrics.value)
        if self.breed_modes.value:
            conf_map['ga_breed_modes'] = self.parse_value(self.breed_modes.value)
        if self.cullings.value:
            conf_map['ga_cullings'] = self.parse_value(self.cullings.value)
        if self.sw_thresholds.value:
            conf_map['ga_sw_thresholds'] = self.parse_value(self.sw_thresholds.value)
        if self.sw_gauss_sigmas.value:
            conf_map['ga_sw_gauss_sigmas'] = self.parse_value(self.sw_gauss_sigmas.value)
        if self.lpf_sigmas.value:
            conf_map['ga_lpf_sigmas'] = self.parse_value(self.lpf_sigmas.value)
        if self.gen_pc_start.value:
            conf_map['ga_gen_pc_start'] = self.parse_value(self.gen_pc_start.value)

    def verify_active(self) -> str:
        if self.active.value and not self.generations.value:
            return "GA is active but generations is not configured"
        return ""


class LowResolutionFeature(Feature):
    """Low pass filter / low resolution feature."""

    name = "low resolution"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.range = text_field(placeholder='[sigma_start, sigma_end]')

        return [
            form_row('Trigger', self.trigger),
            form_row('Range', self.range),
        ]

    def set_defaults(self):
        self.trigger.value = '[0, 1]'
        self.range.value = '[2.0, 0.5]'

    def init_config(self, conf_map: dict):
        if 'lowpass_filter_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['lowpass_filter_trigger'])
            if 'lowpass_filter_range' in conf_map:
                self.range.value = self.format_value(conf_map['lowpass_filter_range'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['lowpass_filter_trigger'] = self.parse_value(self.trigger.value)
        if self.range.value:
            conf_map['lowpass_filter_range'] = self.parse_value(self.range.value)

    def verify_active(self) -> str:
        if self.active.value and not self.trigger.value:
            return "Low resolution is active but trigger is not configured"
        return ""


class ShrinkWrapFeature(Feature):
    """Shrink wrap support feature."""

    name = "shrink wrap"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.sw_type = text_field(placeholder='GAUSS')
        self.threshold = text_field(placeholder='0.1')
        self.gauss_sigma = text_field(placeholder='1.0')

        return [
            form_row('Trigger', self.trigger),
            form_row('Type', self.sw_type),
            form_row('Threshold', self.threshold),
            form_row('Gauss Sigma', self.gauss_sigma),
        ]

    def set_defaults(self):
        self.trigger.value = '[1, 1]'
        self.sw_type.value = 'GAUSS'
        self.threshold.value = '0.1'
        self.gauss_sigma.value = '1.0'

    def init_config(self, conf_map: dict):
        if 'shrink_wrap_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['shrink_wrap_trigger'])
            if 'shrink_wrap_type' in conf_map:
                self.sw_type.value = conf_map['shrink_wrap_type']
            if 'shrink_wrap_threshold' in conf_map:
                self.threshold.value = str(conf_map['shrink_wrap_threshold'])
            if 'shrink_wrap_gauss_sigma' in conf_map:
                self.gauss_sigma.value = str(conf_map['shrink_wrap_gauss_sigma'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['shrink_wrap_trigger'] = self.parse_value(self.trigger.value)
        if self.sw_type.value:
            conf_map['shrink_wrap_type'] = self.sw_type.value
        if self.threshold.value:
            conf_map['shrink_wrap_threshold'] = self.parse_value(self.threshold.value)
        if self.gauss_sigma.value:
            conf_map['shrink_wrap_gauss_sigma'] = self.parse_value(self.gauss_sigma.value)

    def verify_active(self) -> str:
        if self.active.value and not self.trigger.value:
            return "Shrink wrap is active but trigger is not configured"
        return ""


class PhaseConstrainFeature(Feature):
    """Phase constrain feature."""

    name = "phase constrain"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.phase_min = text_field(placeholder='-1.57')
        self.phase_max = text_field(placeholder='1.57')

        return [
            form_row('Trigger', self.trigger),
            form_row('Phase Min', self.phase_min),
            form_row('Phase Max', self.phase_max),
        ]

    def set_defaults(self):
        self.trigger.value = '[0, 1]'
        self.phase_min.value = '-1.57'
        self.phase_max.value = '1.57'

    def init_config(self, conf_map: dict):
        if 'phc_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['phc_trigger'])
            if 'phc_phase_min' in conf_map:
                self.phase_min.value = str(conf_map['phc_phase_min'])
            if 'phc_phase_max' in conf_map:
                self.phase_max.value = str(conf_map['phc_phase_max'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['phc_trigger'] = self.parse_value(self.trigger.value)
        if self.phase_min.value:
            conf_map['phc_phase_min'] = self.parse_value(self.phase_min.value)
        if self.phase_max.value:
            conf_map['phc_phase_max'] = self.parse_value(self.phase_max.value)

    def verify_active(self) -> str:
        if self.active.value and not self.trigger.value:
            return "Phase constrain is active but trigger is not configured"
        return ""


class PCDIFeature(Feature):
    """Partial coherence (PCDI) feature."""

    name = "pcdi"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[iter_start, iter_step]')
        self.pc_type = text_field(placeholder='LUCY')
        self.lucy_iterations = text_field(placeholder='20')
        self.normalize = checkbox('normalize')
        self.lucy_kernel = text_field(placeholder='[16, 16, 16]')

        return [
            form_row('Trigger', self.trigger),
            form_row('Type', self.pc_type),
            form_row('LUCY Iterations', self.lucy_iterations),
            self.normalize,
            form_row('LUCY Kernel', self.lucy_kernel),
        ]

    def set_defaults(self):
        self.trigger.value = '[50, 50]'
        self.pc_type.value = 'LUCY'
        self.lucy_iterations.value = '20'
        self.normalize.value = True
        self.lucy_kernel.value = '[16, 16, 16]'

    def init_config(self, conf_map: dict):
        if 'pc_interval' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['pc_interval'])
            if 'pc_type' in conf_map:
                self.pc_type.value = conf_map['pc_type']
            if 'pc_LUCY_iterations' in conf_map:
                self.lucy_iterations.value = str(conf_map['pc_LUCY_iterations'])
            if 'pc_normalize' in conf_map:
                self.normalize.value = conf_map['pc_normalize']
            if 'pc_LUCY_kernel' in conf_map:
                self.lucy_kernel.value = self.format_value(conf_map['pc_LUCY_kernel'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['pc_interval'] = self.parse_value(self.trigger.value)
        if self.pc_type.value:
            conf_map['pc_type'] = self.pc_type.value
        if self.lucy_iterations.value:
            conf_map['pc_LUCY_iterations'] = self.parse_value(self.lucy_iterations.value)
        if self.normalize.value:
            conf_map['pc_normalize'] = True
        if self.lucy_kernel.value:
            conf_map['pc_LUCY_kernel'] = self.parse_value(self.lucy_kernel.value)

    def verify_active(self) -> str:
        if self.active.value and not self.trigger.value:
            return "PCDI is active but trigger is not configured"
        return ""


class TwinFeature(Feature):
    """Twin removal feature."""

    name = "twin"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[2]')
        self.halves = text_field(placeholder='[0, 0]')

        return [
            form_row('Trigger', self.trigger),
            form_row('Halves', self.halves),
        ]

    def set_defaults(self):
        self.trigger.value = '[2]'
        self.halves.value = '[0, 0]'

    def init_config(self, conf_map: dict):
        if 'twin_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['twin_trigger'])
            if 'twin_halves' in conf_map:
                self.halves.value = self.format_value(conf_map['twin_halves'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['twin_trigger'] = self.parse_value(self.trigger.value)
        if self.halves.value:
            conf_map['twin_halves'] = self.parse_value(self.halves.value)

    def verify_active(self) -> str:
        if self.active.value and not self.trigger.value:
            return "Twin is active but trigger is not configured"
        return ""


class AverageFeature(Feature):
    """Amplitude averaging feature."""

    name = "average"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[-1, 1]')

        return [
            form_row('Trigger', self.trigger),
        ]

    def set_defaults(self):
        self.trigger.value = '[-1, 1]'

    def init_config(self, conf_map: dict):
        if 'average_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['average_trigger'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['average_trigger'] = self.parse_value(self.trigger.value)

    def verify_active(self) -> str:
        if self.active.value and not self.trigger.value:
            return "Average is active but trigger is not configured"
        return ""


class ProgressFeature(Feature):
    """Progress reporting feature."""

    name = "progress"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[0, 20]')

        return [
            form_row('Trigger', self.trigger),
        ]

    def set_defaults(self):
        self.trigger.value = '[0, 20]'

    def init_config(self, conf_map: dict):
        if 'progress_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['progress_trigger'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['progress_trigger'] = self.parse_value(self.trigger.value)


class LiveFeature(Feature):
    """Live visualization feature."""

    name = "live"

    def fill_active(self) -> list:
        self.trigger = text_field(placeholder='[0, 20]')

        return [
            form_row('Trigger', self.trigger),
        ]

    def set_defaults(self):
        self.trigger.value = '[0, 20]'

    def init_config(self, conf_map: dict):
        if 'live_trigger' in conf_map:
            self.active.value = True
            self.trigger.value = self.format_value(conf_map['live_trigger'])
        else:
            self.active.value = False

    def add_config(self, conf_map: dict):
        if not self.active.value:
            return
        if self.trigger.value:
            conf_map['live_trigger'] = self.parse_value(self.trigger.value)
