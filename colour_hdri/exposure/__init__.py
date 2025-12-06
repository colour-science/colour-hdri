"""
Exposure Calculations
=====================

Photometric exposure value computations and conversions.

This subpackage provides algorithms for calculating exposure values from
luminance and illuminance measurements, photometric exposure scaling,
and camera exposure index computations based on digital still camera
standards.
"""

from .common import (
    adjust_exposure,
    average_illuminance,
    average_luminance,
    illuminance_to_exposure_value,
    luminance_to_exposure_value,
)

# isort: split

from .dsc import (
    arithmetic_mean_focal_plane_exposure,
    exposure_index_values,
    exposure_value_100,
    focal_plane_exposure,
    photometric_exposure_scale_factor_Lagarde2014,
    saturation_based_speed_focal_plane_exposure,
)

__all__ = [
    "adjust_exposure",
    "average_illuminance",
    "average_luminance",
    "illuminance_to_exposure_value",
    "luminance_to_exposure_value",
]
__all__ += [
    "arithmetic_mean_focal_plane_exposure",
    "exposure_index_values",
    "exposure_value_100",
    "focal_plane_exposure",
    "photometric_exposure_scale_factor_Lagarde2014",
    "saturation_based_speed_focal_plane_exposure",
]
