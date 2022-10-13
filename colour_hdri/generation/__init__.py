import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

from .weighting_functions import (
    normal_distribution_function,
    hat_function,
    weighting_function_Debevec1997,
)
from .hdri import image_stack_to_HDRI

__all__ = []
__all__ += [
    "normal_distribution_function",
    "hat_function",
    "weighting_function_Debevec1997",
]
__all__ += [
    "image_stack_to_HDRI",
]


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class generation(ModuleAPI):
    """Define a class acting like the *generation* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.2.1
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour_hdri.generation.image_stack_to_radiance_image",
            "colour_hdri.generation.image_stack_to_HDRI",
        ],
    ]
}
"""Defines the *colour_hdri.generation* sub-package API changes."""

if not is_documentation_building():
    sys.modules["colour_hdri.generation"] = generation(  # type: ignore[assignment]
        sys.modules["colour_hdri.generation"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
