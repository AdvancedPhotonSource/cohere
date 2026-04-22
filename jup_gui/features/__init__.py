"""Feature implementations for reconstruction and display tabs."""

from .base import Feature
from .rec_features import (
    GAFeature, LowResolutionFeature, ShrinkWrapFeature,
    PhaseConstrainFeature, PCDIFeature, TwinFeature,
    AverageFeature, ProgressFeature, LiveFeature
)
from .disp_features import (
    CropFeature, InterpolationFeature, ResolutionFeature,
    ReciprocalFeature, StrainFeature, DisplacementFeature
)

REC_FEATURES = {
    'GA': GAFeature,
    'low resolution': LowResolutionFeature,
    'shrink wrap': ShrinkWrapFeature,
    'phase constrain': PhaseConstrainFeature,
    'pcdi': PCDIFeature,
    'twin': TwinFeature,
    'average': AverageFeature,
    'progress': ProgressFeature,
    'live': LiveFeature,
}

DISP_FEATURES = {
    'crop': CropFeature,
    'interpolation': InterpolationFeature,
    'resolution': ResolutionFeature,
    'reciprocal': ReciprocalFeature,
    'strain': StrainFeature,
    'displacement': DisplacementFeature,
}

__all__ = [
    'Feature',
    'REC_FEATURES', 'DISP_FEATURES',
    'GAFeature', 'LowResolutionFeature', 'ShrinkWrapFeature',
    'PhaseConstrainFeature', 'PCDIFeature', 'TwinFeature',
    'AverageFeature', 'ProgressFeature', 'LiveFeature',
    'CropFeature', 'InterpolationFeature', 'ResolutionFeature',
    'ReciprocalFeature', 'StrainFeature', 'DisplacementFeature',
]
