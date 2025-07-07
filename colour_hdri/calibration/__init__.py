"""
Camera Calibration
==================

Camera response function estimation and absolute luminance calibration.

This subpackage provides algorithms for determining camera response functions
from image sequences with known exposure values, and methods for absolute
luminance calibration using reference illumination measurements.
"""

# isort: skip_file

from .absolute_luminance import (
    upper_hemisphere_illuminance_weights_Lagarde2016,
    absolute_luminance_calibration_Lagarde2016,
)
from .debevec1997 import g_solve, camera_response_functions_Debevec1997

__all__ = [
    "upper_hemisphere_illuminance_weights_Lagarde2016",
    "absolute_luminance_calibration_Lagarde2016",
]
__all__ += [
    "g_solve",
    "camera_response_functions_Debevec1997",
]
