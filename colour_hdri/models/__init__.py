import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

from .datasets import *  # noqa
from . import datasets
from .dng import (
    xy_to_camera_neutral,
    camera_neutral_to_xy,
    matrix_XYZ_to_camera_space,
    matrix_camera_space_to_XYZ,
)
from .rgb import camera_space_to_RGB, camera_space_to_sRGB

__all__ = []
__all__ += datasets.__all__
__all__ += [
    "xy_to_camera_neutral",
    "camera_neutral_to_xy",
    "matrix_XYZ_to_camera_space",
    "matrix_camera_space_to_XYZ",
]
__all__ += [
    "camera_space_to_RGB",
    "camera_space_to_sRGB",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class models(ModuleAPI):
    """Define a class acting like the *models* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.2.1
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour_hdri.models.XYZ_to_camera_space_matrix",
            "colour_hdri.models.matrix_XYZ_to_camera_space",
        ],
        [
            "colour_hdri.models.camera_space_to_XYZ_matrix",
            "colour_hdri.models.matrix_camera_space_to_XYZ",
        ],
    ]
}
"""Defines the *colour_hdri.models* sub-package API changes."""

if not is_documentation_building():
    sys.modules["colour_hdri.models"] = models(  # type: ignore[assignment]
        sys.modules["colour_hdri.models"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
