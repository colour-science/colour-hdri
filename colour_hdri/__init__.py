"""
Colour - HDRI
=============

HDRI processing algorithms for *Python*.

Subpackages
-----------
-   calibration: Camera calibration computations.
-   distortion: Lens vignette characterisation & correction.
-   exposure: Exposure computations.
-   examples: Examples for the sub-packages.
-   generation: HDRI Generation.
-   models: Colour models conversion.
-   plotting: Diagrams, figures, etc...
-   process: Image conversion helpers.
-   recovery: Clipped highlights recovery.
-   resources: Resources sub-modules.
-   sampling: Image sampling routines.
-   tonemapping: Tonemapping operators.
-   utilities: Various utilities and data structures.
"""

import sys

from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from colour.hints import Any

import numpy as np
import os
import subprocess  # nosec

import colour

from .utilities import (
    EXIF_EXECUTABLE,
    EXIFTag,
    Image,
    ImageStack,
    Metadata,
    copy_exif_tags,
    delete_exif_tags,
    filter_files,
    parse_exif_array,
    parse_exif_data,
    parse_exif_fraction,
    parse_exif_number,
    parse_exif_string,
    path_exists,
    read_exif_tag,
    read_exif_tags,
    update_exif_tags,
    vivification,
    vivified_to_dict,
    write_exif_tag,
)
from .distortion import (
    DataVignetteCharacterisation,
    VIGNETTE_CHARACTERISATION_METHODS,
    characterise_vignette,
    VIGNETTE_CORRECTION_METHODS,
    correct_vignette,
)
from .sampling import (
    light_probe_sampling_variance_minimization_Viriyothai2009,
    samples_Grossberg2003,
)
from .exposure import (
    adjust_exposure,
    arithmetic_mean_focal_plane_exposure,
    average_illuminance,
    average_luminance,
    exposure_index_values,
    exposure_value_100,
    photometric_exposure_scale_factor_Lagarde2014,
    focal_plane_exposure,
    illuminance_to_exposure_value,
    luminance_to_exposure_value,
    saturation_based_speed_focal_plane_exposure,
)
from .generation import (
    normal_distribution_function,
    hat_function,
    weighting_function_Debevec1997,
    image_stack_to_HDRI,
)
from .calibration import (
    absolute_luminance_calibration_Lagarde2016,
    camera_response_functions_Debevec1997,
    g_solve,
    upper_hemisphere_illuminance_weights_Lagarde2016,
)
from .models import (
    camera_neutral_to_xy,
    camera_space_to_RGB,
    camera_space_to_sRGB,
    matrix_camera_space_to_XYZ,
    matrix_XYZ_to_camera_space,
    xy_to_camera_neutral,
)
from .process import (
    DNG_CONVERTER,
    DNG_CONVERTER_ARGUMENTS,
    DNG_EXIF_TAGS_BINDING,
    RAW_CONVERTER,
    RAW_CONVERTER_ARGUMENTS_BAYER_CFA,
    RAW_CONVERTER_ARGUMENTS_DEMOSAICING,
    convert_dng_files_to_intermediate_files,
    convert_raw_files_to_dng_files,
    read_dng_files_exif_tags,
)
from .recovery import highlights_recovery_blend, highlights_recovery_LCHab
from .tonemapping import (
    tonemapping_operator_exponential,
    tonemapping_operator_exponentiation_mapping,
    tonemapping_operator_filmic,
    tonemapping_operator_gamma,
    tonemapping_operator_logarithmic,
    tonemapping_operator_logarithmic_mapping,
    tonemapping_operator_normalisation,
    tonemapping_operator_Reinhard2004,
    tonemapping_operator_Schlick1994,
    tonemapping_operator_simple,
    tonemapping_operator_Tumblin1999,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "EXIF_EXECUTABLE",
    "EXIFTag",
    "Image",
    "ImageStack",
    "Metadata",
    "copy_exif_tags",
    "delete_exif_tags",
    "filter_files",
    "parse_exif_array",
    "parse_exif_data",
    "parse_exif_fraction",
    "parse_exif_number",
    "parse_exif_string",
    "path_exists",
    "read_exif_tag",
    "read_exif_tags",
    "update_exif_tags",
    "vivification",
    "vivified_to_dict",
    "write_exif_tag",
]
__all__ += [
    "DataVignetteCharacterisation",
    "VIGNETTE_CHARACTERISATION_METHODS",
    "characterise_vignette",
    "VIGNETTE_CORRECTION_METHODS",
    "correct_vignette",
]
__all__ += [
    "light_probe_sampling_variance_minimization_Viriyothai2009",
    "samples_Grossberg2003",
]
__all__ += [
    "adjust_exposure",
    "arithmetic_mean_focal_plane_exposure",
    "average_illuminance",
    "average_luminance",
    "exposure_index_values",
    "exposure_value_100",
    "photometric_exposure_scale_factor_Lagarde2014",
    "focal_plane_exposure",
    "illuminance_to_exposure_value",
    "luminance_to_exposure_value",
    "saturation_based_speed_focal_plane_exposure",
]
__all__ += [
    "normal_distribution_function",
    "hat_function",
    "weighting_function_Debevec1997",
    "image_stack_to_HDRI",
]
__all__ += [
    "absolute_luminance_calibration_Lagarde2016",
    "camera_response_functions_Debevec1997",
    "g_solve",
    "upper_hemisphere_illuminance_weights_Lagarde2016",
]
__all__ += [
    "camera_neutral_to_xy",
    "camera_space_to_RGB",
    "camera_space_to_sRGB",
    "matrix_camera_space_to_XYZ",
    "matrix_XYZ_to_camera_space",
    "xy_to_camera_neutral",
]
__all__ += [
    "DNG_CONVERTER",
    "DNG_CONVERTER_ARGUMENTS",
    "DNG_EXIF_TAGS_BINDING",
    "RAW_CONVERTER",
    "RAW_CONVERTER_ARGUMENTS_BAYER_CFA",
    "RAW_CONVERTER_ARGUMENTS_DEMOSAICING",
    "convert_dng_files_to_intermediate_files",
    "convert_raw_files_to_dng_files",
    "read_dng_files_exif_tags",
]
__all__ += [
    "highlights_recovery_blend",
    "highlights_recovery_LCHab",
]
__all__ += [
    "tonemapping_operator_exponential",
    "tonemapping_operator_exponentiation_mapping",
    "tonemapping_operator_filmic",
    "tonemapping_operator_gamma",
    "tonemapping_operator_logarithmic",
    "tonemapping_operator_logarithmic_mapping",
    "tonemapping_operator_normalisation",
    "tonemapping_operator_Reinhard2004",
    "tonemapping_operator_Schlick1994",
    "tonemapping_operator_simple",
    "tonemapping_operator_Tumblin1999",
]

ROOT_RESOURCES = os.path.join(os.path.dirname(__file__), "resources")
ROOT_RESOURCES_EXAMPLES = os.path.join(
    ROOT_RESOURCES, "colour-hdri-examples-datasets"
)
ROOT_RESOURCES_TESTS = os.path.join(
    ROOT_RESOURCES, "colour-hdri-tests-datasets"
)

__application_name__ = "Colour - HDRI"

__major_version__ = "0"
__minor_version__ = "2"
__change_version__ = "1"
__version__ = ".".join(
    (__major_version__, __minor_version__, __change_version__)
)

try:
    _version: str = (
        subprocess.check_output(  # nosec
            ["git", "describe"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )
except Exception:
    _version: str = __version__  # type: ignore[no-redef]

colour.utilities.ANCILLARY_COLOUR_SCIENCE_PACKAGES["colour-hdri"] = _version

del _version

# TODO: Remove legacy printing support when deemed appropriate.
try:
    np.set_printoptions(legacy="1.13")
except TypeError:
    pass


# ----------------------------------------------------------------------------#
# ---                API Changes and Deprecation Management                ---#
# ----------------------------------------------------------------------------#
class colour_hdri(ModuleAPI):
    """Define a class acting like the *colour_hdri* module."""

    def __getattr__(self, attribute) -> Any:
        """Return the value from the attribute with given name."""

        return super().__getattr__(attribute)


# v0.2.1
API_CHANGES = {
    "ObjectRenamed": [
        [
            "colour_hdri.XYZ_to_camera_space_matrix",
            "colour_hdri.matrix_XYZ_to_camera_space",
        ],
        [
            "colour_hdri.camera_space_to_XYZ_matrix",
            "colour_hdri.matrix_camera_space_to_XYZ",
        ],
        [
            "colour_hdri.image_stack_to_radiance_image",
            "colour_hdri.image_stack_to_HDRI",
        ],
    ]
}
"""Defines the *colour_hdri* package API changes."""

if not is_documentation_building():
    sys.modules["colour_hdri"] = colour_hdri(  # type: ignore[assignment]
        sys.modules["colour_hdri"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
