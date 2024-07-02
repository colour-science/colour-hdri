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

from __future__ import annotations

import contextlib
import os
import subprocess
import sys

import colour
import numpy as np
from colour.hints import Any
from colour.utilities import is_matplotlib_installed
from colour.utilities.deprecation import ModuleAPI, build_API_changes
from colour.utilities.documentation import is_documentation_building

from .calibration import (
    absolute_luminance_calibration_Lagarde2016,
    camera_response_functions_Debevec1997,
    g_solve,
    upper_hemisphere_illuminance_weights_Lagarde2016,
)
from .distortion import (
    VIGNETTE_CHARACTERISATION_METHODS,
    VIGNETTE_CORRECTION_METHODS,
    DataVignetteCharacterisation,
    characterise_vignette,
    correct_vignette,
)
from .exposure import (
    adjust_exposure,
    arithmetic_mean_focal_plane_exposure,
    average_illuminance,
    average_luminance,
    exposure_index_values,
    exposure_value_100,
    focal_plane_exposure,
    illuminance_to_exposure_value,
    luminance_to_exposure_value,
    photometric_exposure_scale_factor_Lagarde2014,
    saturation_based_speed_focal_plane_exposure,
)
from .generation import (
    double_sigmoid_anchored_function,
    hat_function,
    image_stack_to_HDRI,
    normal_distribution_function,
    weighting_function_Debevec1997,
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
from .sampling import (
    light_probe_sampling_variance_minimization_Viriyothai2009,
    samples_Grossberg2003,
)
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
from .utilities import (
    EXIF_EXECUTABLE,
    EXIFTag,
    Image,
    ImageStack,
    Metadata,
    copy_exif_tags,
    delete_exif_tags,
    filter_files,
    is_lensfunpy_installed,
    is_opencv_installed,
    is_rawpy_installed,
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
from .network import (
    GraphBatchMergeHDRI,
    GraphHDRI,
    GraphMergeHDRI,
    GraphPostMergeHDRI,
    GraphRawProcessingCameraSensitivities,
    GraphRawProcessingDNG,
    InputTransform,
    NodeApplyInputTransformCameraSensitivities,
    NodeApplyInputTransformDNG,
    NodeComputeInputTransformCameraSensitivities,
    NodeComputeInputTransformDNG,
    NodeConvertRawFileToDNGFile,
    NodeCorrectLensAberrationLensFun,
    NodeCreateBatches,
    NodeCreateImageStack,
    NodeDownsample,
    NodeMergeImageStack,
    NodeNormaliseExposure,
    NodeProcessingMetadata,
    NodeProcessRawFileRawpy,
    NodeReadFileMetadataDNG,
    NodeReadImage,
    NodeRemoveFile,
    NodeWatermark,
    NodeWriteImage,
    NodeWritePreviewImage,
)

# Exposing "colour.plotting" sub-package if "Matplotlib" is available.
if is_matplotlib_installed():
    import colour_hdri.plotting as plotting  # noqa: F401, PLR0402
else:

    class MockPlotting:  # pragma: no cover
        """
        Mock object for :mod:`colour_hdri.plotting` sub-package raising an exception
        if the sub-package is accessed but *Matplotlib* is not installed.
        """

        def __getattr__(self, attribute) -> Any:
            """Return the value from the attribute with given name."""

            is_matplotlib_installed(raise_exception=True)

    globals()["plotting"] = MockPlotting()  # pragma: no cover


__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "absolute_luminance_calibration_Lagarde2016",
    "camera_response_functions_Debevec1997",
    "g_solve",
    "upper_hemisphere_illuminance_weights_Lagarde2016",
]
__all__ += [
    "VIGNETTE_CHARACTERISATION_METHODS",
    "VIGNETTE_CORRECTION_METHODS",
    "DataVignetteCharacterisation",
    "characterise_vignette",
    "correct_vignette",
]
__all__ += [
    "adjust_exposure",
    "arithmetic_mean_focal_plane_exposure",
    "average_illuminance",
    "average_luminance",
    "exposure_index_values",
    "exposure_value_100",
    "focal_plane_exposure",
    "illuminance_to_exposure_value",
    "luminance_to_exposure_value",
    "photometric_exposure_scale_factor_Lagarde2014",
    "saturation_based_speed_focal_plane_exposure",
]
__all__ += [
    "double_sigmoid_anchored_function",
    "hat_function",
    "image_stack_to_HDRI",
    "normal_distribution_function",
    "weighting_function_Debevec1997",
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
    "light_probe_sampling_variance_minimization_Viriyothai2009",
    "samples_Grossberg2003",
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
__all__ += [
    "EXIF_EXECUTABLE",
    "EXIFTag",
    "Image",
    "ImageStack",
    "Metadata",
    "copy_exif_tags",
    "delete_exif_tags",
    "filter_files",
    "is_rawpy_installed",
    "is_lensfunpy_installed",
    "is_opencv_installed",
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
    "GraphBatchMergeHDRI",
    "GraphHDRI",
    "GraphMergeHDRI",
    "GraphPostMergeHDRI",
    "GraphRawProcessingCameraSensitivities",
    "GraphRawProcessingDNG",
    "InputTransform",
    "NodeApplyInputTransformCameraSensitivities",
    "NodeApplyInputTransformDNG",
    "NodeComputeInputTransformCameraSensitivities",
    "NodeComputeInputTransformDNG",
    "NodeConvertRawFileToDNGFile",
    "NodeCorrectLensAberrationLensFun",
    "NodeCreateBatches",
    "NodeCreateImageStack",
    "NodeDownsample",
    "NodeMergeImageStack",
    "NodeNormaliseExposure",
    "NodeProcessRawFileRawpy",
    "NodeProcessingMetadata",
    "NodeReadFileMetadataDNG",
    "NodeReadImage",
    "NodeRemoveFile",
    "NodeWatermark",
    "NodeWriteImage",
    "NodeWritePreviewImage",
]

ROOT_RESOURCES: str = os.path.join(os.path.dirname(__file__), "resources")
ROOT_RESOURCES_EXAMPLES: str = os.path.join(
    ROOT_RESOURCES, "colour-hdri-examples-datasets"
)
ROOT_RESOURCES_TESTS: str = os.path.join(ROOT_RESOURCES, "colour-hdri-tests-datasets")

__all__ += ["ROOT_RESOURCES", "ROOT_RESOURCES_EXAMPLES", "ROOT_RESOURCES_TESTS"]

__application_name__ = "Colour - HDRI"

__major_version__ = "0"
__minor_version__ = "2"
__change_version__ = "5"
__version__ = ".".join((__major_version__, __minor_version__, __change_version__))

try:
    _version: str = (
        subprocess.check_output(
            ["git", "describe"],  # noqa: S603, S607
            cwd=os.path.dirname(__file__),
            stderr=subprocess.STDOUT,
        )
        .strip()
        .decode("utf-8")
    )
except Exception:
    _version: str = __version__

colour.utilities.ANCILLARY_COLOUR_SCIENCE_PACKAGES["colour-hdri"] = _version  # pyright: ignore

if is_rawpy_installed():
    import rawpy

    colour.utilities.ANCILLARY_RUNTIME_PACKAGES["rawpy"] = rawpy.__version__

    del rawpy

if is_lensfunpy_installed():
    import lensfunpy

    colour.utilities.ANCILLARY_RUNTIME_PACKAGES["lensfunpy"] = lensfunpy.__version__

    del lensfunpy

if is_opencv_installed():
    import cv2

    colour.utilities.ANCILLARY_RUNTIME_PACKAGES["OpenCV"] = cv2.__version__

    del cv2

del _version

# TODO: Remove legacy printing support when deemed appropriate.
with contextlib.suppress(TypeError):
    np.set_printoptions(legacy="1.13")


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
    sys.modules["colour_hdri"] = colour_hdri(  # pyright: ignore
        sys.modules["colour_hdri"], build_API_changes(API_CHANGES)
    )

    del ModuleAPI, is_documentation_building, build_API_changes, sys
