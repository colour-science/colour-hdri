"""
Network - Nodes
===============

Define a collection of :class:`colour.utilities.PortNode` nodes for raw and HDR
image processing.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from colour import (
    RGB_COLOURSPACES,
    CCT_to_uv,
    RGB_Colourspace,
    RGB_to_RGB,
    UCS_uv_to_xy,
    XYZ_to_RGB,
    linear_function,
    matrix_idt,
    sd_blackbody,
    sd_CIE_illuminant_D_series,
    uv_to_CCT,
    xy_to_UCS_uv,
)
from colour.algebra import vecmul
from colour.characterisation import RGB_CameraSensitivities
from colour.hints import Any, NDArray
from colour.io import (
    Image_Specification_Attribute,
    convert_bit_depth,
    read_image_OpenImageIO,
    write_image_OpenImageIO,
)
from colour.temperature import CCT_to_xy_CIE_D
from colour.utilities import (
    CanonicalMapping,
    ExecutionNode,
    as_float_array,
    batch,
    ones,
    required,
    slugify,
    zeros,
)

from colour_hdri import (
    ImageStack,
    convert_raw_files_to_dng_files,
    double_sigmoid_anchored_function,
    image_stack_to_HDRI,
    read_dng_files_exif_tags,
    tonemapping_operator_Reinhard2004,
)
from colour_hdri.models import (
    CCS_ILLUMINANT_ADOBEDNG,
    CCT_ILLUMINANTS_ADOBEDNG,
    LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS,
    camera_neutral_to_xy,
    matrix_XYZ_to_camera_space,
    xy_to_camera_neutral,
)
from colour_hdri.process import DNG_CONVERTER

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "JSONEncoderEXRAttribute",
    "InputTransform",
    "NodeConvertRawFileToDNGFile",
    "NodeReadImage",
    "NodeWriteImage",
    "NodeWritePreviewImage",
    "NodeRemoveFile",
    "NodeWatermark",
    "NodeProcessingMetadata",
    "NodeReadFileMetadataDNG",
    "NodeComputeInputTransformDNG",
    "NodeComputeInputTransformCameraSensitivities",
    "NodeProcessRawFileRawpy",
    "NodeCorrectLensAberrationLensFun",
    "NodeDownsample",
    "NodeApplyInputTransformDNG",
    "NodeApplyInputTransformCameraSensitivities",
    "NodeCreateBatches",
    "NodeCreateImageStack",
    "NodeMergeImageStack",
    "NodeNormaliseExposure",
]


class JSONEncoderEXRAttribute(json.JSONEncoder):
    """
    Define an encoder that serialize *JSON* data for storing inside an *EXR*
    attribute.
    """

    def default(self, object_: Any) -> Any:
        """
        Return a *JSON* serialisable object from given object.

        Parameters
        ----------
        object_
            Object to return a *JSON* serialisable object from.

        Returns
        -------
        :class:`object`
            *JSON* serialisable object
        """

        if isinstance(object_, CanonicalMapping):
            return dict(object_.items())
        elif isinstance(object_, (np.float32, np.float64)):
            return float(object_)
        elif isinstance(object_, (np.int32, np.int64)):
            return int(object_)
        elif isinstance(object_, np.ndarray):
            return object_.tolist()

        return super().default(object_)


@dataclass
class InputTransform:
    """
    Define an input transform for a camera.

    Parameters
    ----------
    M
        Matrix :math:`M`.
    RGB_w
        White balance multipliers :math:`RGB_w`.
    """

    M: NDArray = field(default_factory=lambda: np.identity(3))
    RGB_w: NDArray = field(default_factory=lambda: ones(3))

    def __eq__(self, other: Any) -> bool:
        """
        Return whether the input transform is equal to given other object.

        Parameters
        ----------
        other
            Object to test whether it is equal to the input transform.

        Returns
        -------
        :class:`bool`
            Whether given object is equal to the input transform.
        """

        if not isinstance(other, InputTransform):
            return False

        return np.all(self.M == other.M) and np.all(self.RGB_w == other.RGB_w)


class NodeConvertRawFileToDNGFile(ExecutionNode):
    """
    Convert given raw file, e.g., *CR2*, *CR3*, *NEF*, to *DNG*.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeConvertRawFileToDNGFile.__init__`
    -   :meth:`~colour_hdri.NodeConvertRawFileToDNGFile.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = 'Convert given raw file, e.g., "CR2", "CR3", "NEF", to "DNG"'

        self.add_input_port("raw_file_path")
        self.add_input_port("output_directory")
        self.add_input_port("dng_converter")
        self.add_input_port("dng_converter_arguments")
        self.add_output_port("dng_file_path")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        raw_file_path = self.get_input("raw_file_path")
        if raw_file_path is None:
            return

        if not os.path.exists(raw_file_path):
            self.log(f'"{raw_file_path}" file does not exist!', "error")
            return

        output_directory = self.get_input("output_directory")
        if output_directory is None:
            output_directory = os.path.dirname(raw_file_path)

        dng_file_path = next(
            iter(
                convert_raw_files_to_dng_files(
                    [raw_file_path],
                    output_directory,
                    self.get_input("dng_converter"),
                    self.get_input("dng_converter_arguments"),
                )
            )
        )

        if not os.path.exists(dng_file_path):
            self.log(
                f'"{dng_file_path}" file does not exist, conversion failed!',
                "error",
            )
            return

        self.set_output("dng_file_path", dng_file_path)

        self.dirty = False


def _is_linear_file_format(path: str) -> bool:
    """Return whether the file at given path is a linear file type."""

    return os.path.splitext(path)[-1].lower() in (".exr", ".hdr")


class NodeReadImage(ExecutionNode):
    """
    Read the image from input path and return its data and metadata. The
    decoding CCTF of the input colourspace is used to linearise the image if it
    is stored using non-linear file format.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeReadImage.__init__`
    -   :meth:`~colour_hdri.NodeReadImage.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            "Read the image from input path and return its data and metadata"
        )

        self.add_input_port("path")
        self.add_input_port("input_colourspace", "sRGB")
        self.add_output_port("image")
        self.add_output_port("metadata")

    @required("OpenImageIO")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        path = self.get_input("path")
        if path is None:
            return

        if not os.path.exists(path):
            self.log(f'"{path}" image does not exist!')
            return

        image, metadata = read_image_OpenImageIO(path, attributes=True)

        input_colourspace = self.get_input("input_colourspace")
        if isinstance(input_colourspace, str):
            input_colourspace = RGB_COLOURSPACES[input_colourspace]

        if not _is_linear_file_format(path):
            image = input_colourspace.cctf_decoding(image)

        self.set_output("image", image)
        self.set_output("metadata", metadata)

        self.dirty = False


class NodeWriteImage(ExecutionNode):
    """
    Write the input image to input path using the input metadata. The encoding
    CCTF of the output colourspace is used to non-linearly encode the image if
    it is to be stored using a non-linear file format.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeWriteImage.__init__`
    -   :meth:`~colour_hdri.NodeWriteImage.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            "Write the input image to input path using the input metadata"
        )

        self.add_input_port("image")
        self.add_input_port("metadata")
        self.add_input_port("path")
        self.add_input_port("output_colourspace")
        self.add_input_port("bypass", False)

    @required("OpenImageIO")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        if self.get_input("bypass"):
            return

        image = self.get_input("image")
        if image is None:
            return

        path = self.get_input("path")
        if path is None:
            return

        metadata = self.get_input("metadata")
        if metadata is None:
            metadata = {}

        attributes = []
        for key, value in metadata.items():
            attributes.append(
                Image_Specification_Attribute(
                    str(key), json.dumps(value, cls=JSONEncoderEXRAttribute)
                )
            )

        output_colourspace = self.get_input("output_colourspace")
        if isinstance(output_colourspace, str):
            output_colourspace = RGB_COLOURSPACES[output_colourspace]

        if not _is_linear_file_format(path):
            image = output_colourspace.cctf_encoding(image)

        write_image_OpenImageIO(image, path, attributes=attributes)

        self.dirty = False


def _cctf_encoding_preview(a):
    """
    Encode given image :math:`a` using the Reinhard (2004) global tonemapping
    function.
    """

    # Reinhard (2004) rendering is dim, increasing exposure by a stop.
    return tonemapping_operator_Reinhard2004(a * 2)


class NodeWritePreviewImage(ExecutionNode):
    """
    Write the image at input image path as a preview image.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeWritePreviewImage.__init__`
    -   :meth:`~colour_hdri.NodeWritePreviewImage.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Write the image at input image path as a preview image"

        self.add_input_port("path")
        self.add_input_port("cctf_encoding", _cctf_encoding_preview)
        self.add_input_port("bypass", False)
        self.add_output_port("preview_path")

    @required("OpenImageIO")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        if self.get_input("bypass"):
            return

        path = self.get_input("path")
        if path is None:
            return

        if not os.path.exists(path):
            self.log(f'"{path}" image does not exist!')
            return

        cctf_encoding = self.get_input("cctf_encoding")

        filename, extension = os.path.splitext(path)

        preview_path = f"{filename}.jpg"

        write_image_OpenImageIO(
            cctf_encoding(read_image_OpenImageIO(path)), preview_path
        )

        self.set_output("preview_path", preview_path)

        self.dirty = False


class NodeRemoveFile(ExecutionNode):
    """
    Remove the file at input path.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeRemoveFile.__init__`
    -   :meth:`~colour_hdri.NodeRemoveFile.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Remove the file at input path"

        self.add_input_port("path")
        self.add_input_port("bypass", False)

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        if self.get_input("bypass"):
            return

        path = self.get_input("path")
        if not os.path.exists(path):
            self.log(f'"{path}" file does not exist!', "error")
            return

        os.remove(path)

        self.dirty = False


class NodeWatermark(ExecutionNode):
    """
    Watermark the input image using given input metadata.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeWatermark.__init__`
    -   :meth:`~colour_hdri.NodeWatermark.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Watermark the input image using given input metadata"

        self.add_input_port("input_image")
        self.add_input_port("metadata")
        self.add_input_port("include_exposure_information", True)
        self.add_input_port("bypass", False)
        self.add_output_port("output_image")

    @required("OpenCV")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        input_image = self.get_input("input_image")
        if input_image is None:
            return

        self.set_output("output_image", input_image)

        if self.get_input("bypass"):
            return

        metadata = self.get_input("metadata")
        if metadata is None:
            metadata = {}

        exif_group = metadata.get("EXIF")
        if exif_group is None:
            self.log(
                'Could not read "EXIF" metadata from input metadata!',
                "error",
            )
            return

        import cv2

        text = (
            f"{exif_group['Camera Model Name']} - "
            f"{exif_group['Lens Model']} - "
            f"{exif_group['Focal Length']}mm"
        )

        if self.get_input("include_exposure_information"):
            text += (
                " - "
                f"{exif_group['Exposure Time']:.6f}\" "
                f"f{exif_group['F Number']} "
                f"{exif_group['ISO']}"
            )

        watermark = zeros(input_image.shape)

        cv2.putText(
            watermark,
            text,
            (10, input_image.shape[0] - 10),
            cv2.FONT_HERSHEY_PLAIN,
            fontScale=2.5,
            color=(255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        output_image = np.maximum(input_image, watermark)

        self.set_output("output_image", output_image)

        self.dirty = False


class NodeProcessingMetadata(ExecutionNode):
    """
    Add processing metadata to the input metadata.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeProcessingMetadata.__init__`
    -   :meth:`~colour_hdri.NodeProcessingMetadata.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Add processing metadata to the input metadata"

        self.add_input_port("input_metadata")
        self.add_input_port("namespace", "colour-science/colour-hdri/processing")
        self.add_input_port("input_transform", InputTransform())
        self.add_input_port("output_colourspace", "sRGB")
        self.add_input_port("orientation")
        self.add_input_port("type", "sdr")
        self.add_input_port("sources")
        self.add_output_port("output_metadata")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        input_metadata = self.get_input("input_metadata")
        if input_metadata is None:
            input_metadata = {}

        input_transform = self.get_input("input_transform")

        sources = self.get_input("sources")
        if isinstance(sources, str):
            sources = [sources]

        output_colourspace = self.get_input("output_colourspace")
        if isinstance(output_colourspace, RGB_Colourspace):
            output_colourspace = output_colourspace.name

        processing_metadata = {
            self.get_input("namespace"): {
                "type": self.get_input("type"),
                "sources": sources,
                "input_transform": {
                    "M": input_transform.M if input_transform is not None else None,
                    "RGB_w": (
                        input_transform.RGB_w if input_transform is not None else None
                    ),
                },
                "output_colourspace": output_colourspace,
                "orientation": self.get_input("orientation"),
                "script": sys.argv[0] if len(sys.argv) else __file__,
                "time": time.strftime("%Y/%m/%d-%H:%M:%S", time.gmtime()),
            }
        }

        output_metadata = dict(**input_metadata, **processing_metadata)

        self.set_output("output_metadata", output_metadata)

        self.dirty = False


class NodeReadFileMetadataDNG(ExecutionNode):
    """
    Return the metadata from the input *DNG* image.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeReadFileMetadataDNG.__init__`
    -   :meth:`~colour_hdri.NodeReadFileMetadataDNG.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = 'Return the metadata from the input "DNG" image'

        self.add_input_port("dng_file_path")
        self.add_output_port("metadata")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        dng_file_path = self.get_input("dng_file_path")
        if dng_file_path is None:
            return

        if not os.path.exists(dng_file_path):
            self.log(f'"{dng_file_path}" file does not exist!', "error")
            return

        metadata = next(iter(read_dng_files_exif_tags([dng_file_path])))

        self.set_output("metadata", metadata)

        self.dirty = False


class NodeComputeInputTransformDNG(ExecutionNode):
    """
    Compute the input transform from the input metadata using the *DNG* method.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeComputeInputTransformDNG.__init__`
    -   :meth:`~colour_hdri.NodeComputeInputTransformDNG.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            "Compute the input transform from the input metadata using the "
            '"DNG" method'
        )

        self.add_input_port("metadata")
        self.add_input_port("CCT_D_uv", [6500, 0])
        self.add_input_port("bypass", False)
        self.add_output_port("input_transform", InputTransform())

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        if self.get_input("bypass"):
            return

        metadata = self.get_input("metadata")
        if metadata is None:
            return

        exif_group = metadata.get("EXIF")
        if exif_group is None:
            self.log(
                'Could not read "EXIF" metadata from input metadata!',
                "error",
            )
            return

        as_shot_neutral = exif_group["As Shot Neutral"]
        self.log(
            f"As Shot Neutral (EXIF): {as_shot_neutral}",
        )

        CCT_D_uv = self.get_input("CCT_D_uv")
        camera_neutral = as_float_array(
            CCT_D_uv if CCT_D_uv is not None else as_shot_neutral
        )

        self.log(
            f"Camera Neutral: {camera_neutral}",
        )

        CCT_calibration_illuminant_1 = CCT_ILLUMINANTS_ADOBEDNG[
            LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS[exif_group["Calibration Illuminant 1"]]
        ]
        CCT_calibration_illuminant_2 = CCT_ILLUMINANTS_ADOBEDNG[
            LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS[exif_group["Calibration Illuminant 2"]]
        ]
        M_color_matrix_1 = exif_group["Color Matrix 1"]
        M_color_matrix_2 = exif_group["Color Matrix 2"]
        M_camera_calibration_1 = exif_group["Camera Calibration 1"]
        M_camera_calibration_2 = exif_group["Camera Calibration 2"]
        analog_balance = exif_group["Analog Balance"]
        as_shot_neutral = exif_group["As Shot Neutral"]

        if camera_neutral.size == 2:
            xy = UCS_uv_to_xy(CCT_to_uv(camera_neutral))
        else:
            xy = camera_neutral_to_xy(
                as_shot_neutral,
                CCT_calibration_illuminant_1,
                CCT_calibration_illuminant_2,
                M_color_matrix_1,
                M_color_matrix_2,
                M_camera_calibration_1,
                M_camera_calibration_2,
                analog_balance,
            )

        self.log(f'Camera Neutral "CIE xy" chromaticity coordinates: {xy}')

        CCT_D_uv = uv_to_CCT(xy_to_UCS_uv(xy))

        self.log(f'Camera Neutral "CCT": {CCT_D_uv}')

        M = matrix_XYZ_to_camera_space(
            xy,
            CCT_calibration_illuminant_1,
            CCT_calibration_illuminant_2,
            M_color_matrix_1,
            M_color_matrix_2,
            M_camera_calibration_1,
            M_camera_calibration_2,
            analog_balance,
        )

        self.log(f'"CIE XYZ D50" to "Camera Space" matrix "M": {M}')

        RGB_w = 1 / xy_to_camera_neutral(
            xy,
            CCT_calibration_illuminant_1,
            CCT_calibration_illuminant_2,
            M_color_matrix_1,
            M_color_matrix_2,
            M_camera_calibration_1,
            M_camera_calibration_2,
            analog_balance,
        )

        self.log(f'White balance multipliers "RGB": {RGB_w}')

        self.set_output("input_transform", InputTransform(M, RGB_w))

        self.dirty = False


class NodeComputeInputTransformCameraSensitivities(ExecutionNode):
    """
    Compute the input transform from the input metadata using the
    *Camera Sensitivities* method.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeComputeInputTransformCameraSensitivities.__init__`
    -   :meth:`~colour_hdri.NodeComputeInputTransformCameraSensitivities.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            "Compute the input transform from the input metadata using the "
            '"Camera Sensitivities" method'
        )

        self.add_input_port("metadata")
        self.add_input_port("CCT_D_uv", [6500, 0])
        self.add_input_port("camera_sensitivities")
        self.add_input_port("bypass", False)
        self.add_output_port("input_transform", InputTransform())

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        if self.get_input("bypass"):
            return

        camera_sensitivities = self.get_input("camera_sensitivities")
        if camera_sensitivities is None:
            return

        if isinstance(camera_sensitivities, RGB_CameraSensitivities):
            msds_sensitivities = camera_sensitivities
        else:
            metadata = self.get_input("metadata")
            if metadata is None:
                return

            exif_group = metadata.get("EXIF")
            if exif_group is None:
                self.log(
                    'Could not read "EXIF" metadata from input metadata!',
                    "error",
                )
                return

            camera_make = exif_group["Make"]
            camera_model = exif_group["Camera Model Name"]

            if DNG_CONVERTER == "dnglab":
                self.log(
                    f'"dnglab" used, prepending "{camera_make}" camera make.', "warning"
                )
                camera_model = f"{camera_make} {camera_model}"

            self.log(
                f'Using "{camera_model}" camera model sensitivities.',
            )

            try:
                msds_sensitivities = camera_sensitivities[slugify(camera_model)]
            except KeyError:
                self.log(
                    f'No sensitivities exist for "{camera_model}" camera model.',
                    "error",
                )
                return

        CCT_D_uv = self.get_input("CCT_D_uv")

        if CCT_D_uv[-1] != 0:
            self.log(
                f'Ignoring {CCT_D_uv[-1]} "D_uv" value!',
                "warning",
            )

        # TODO: Implement support for illuminant blending.
        if CCT_D_uv[0] < 3500:
            self.log('Using "Blackbobdy" illuminant!')

            sd_illuminant = sd_blackbody(CCT_D_uv[0])
        else:
            self.log(
                'Using "Daylight" illuminant!',
            )

            sd_illuminant = sd_CIE_illuminant_D_series(
                CCT_to_xy_CIE_D(CCT_D_uv[0] * 1.4388 / 1.4380)
            )

        M, RGB_w = matrix_idt(msds_sensitivities, sd_illuminant)

        self.log(f"Input Transform Matrix: {M}")

        self.log(f'White balance multipliers "RGB": {RGB_w}')

        self.set_output("input_transform", InputTransform(M, RGB_w))

        self.dirty = False


class NodeProcessRawFileRawpy(ExecutionNode):
    """
    Process given raw file, e.g., *CR2*, *CR3*, *NEF*, using *Rawpy*.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeProcessRawFileRawpy.__init__`
    -   :meth:`~colour_hdri.NodeProcessRawFileRawpy.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            'Process given raw file, e.g., "CR2", "CR3", "NEF", using "Rawpy"'
        )

        self.add_input_port("raw_file_path")
        self.add_input_port("input_transform", InputTransform())
        self.add_output_port("image")

    @required("rawpy")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        raw_file_path = self.get_input("raw_file_path")
        if raw_file_path is None:
            return

        if not os.path.exists(raw_file_path):
            self.log(f'"{raw_file_path}" file does not exist!', "error")
            return

        import rawpy

        input_transform = self.get_input("input_transform")

        with rawpy.imread(raw_file_path) as raw_file:
            self.log(f'Processing "{raw_file_path}" file...')

            image = raw_file.postprocess(
                gamma=(1, 1),
                no_auto_bright=True,
                demosaic_algorithm=rawpy.DemosaicAlgorithm(12),
                fbdd_noise_reduction=rawpy.FBDDNoiseReductionMode(2),
                highlight_mode=rawpy.HighlightMode(5),
                output_color=rawpy.ColorSpace(0),
                output_bps=16,
                user_wb=np.hstack(
                    [
                        input_transform.RGB_w,
                        input_transform.RGB_w[1],
                    ]
                ).tolist(),
            )

        image = convert_bit_depth(image, "float32")

        self.set_output("image", image)

        self.dirty = False


class NodeCorrectLensAberrationLensFun(ExecutionNode):
    """
    Correct the input image lens aberrations, i.e., vignette, distortion and
    chromatic aberration, using *LensFun*.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeCorrectLensAberrationLensFun.__init__`
    -   :meth:`~colour_hdri.NodeCorrectLensAberrationLensFun.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            "Correct the input image lens aberrations, i.e., vignette, "
            'distortion and chromatic aberration, using "LensFun"'
        )

        self.add_input_port("input_image")
        self.add_input_port("metadata", 1)
        self.add_input_port("correct_vignette", True)
        self.add_input_port("correct_chromatic_aberration", True)
        self.add_input_port("correct_distortion", True)
        self.add_input_port("focus_distance", 100)
        self.add_input_port("bypass", False)
        self.add_output_port("output_image")

    @required("lensfunpy", "OpenCV")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        input_image = self.get_input("input_image")
        if input_image is None:
            return

        self.set_output("output_image", input_image)

        if self.get_input("bypass"):
            return

        metadata = self.get_input("metadata")
        if metadata is None:
            return

        exif_group = metadata.get("EXIF")
        if exif_group is None:
            self.log(
                'Could not read "EXIF" metadata from input metadata!',
                "error",
            )
            return

        import cv2
        import lensfunpy

        database = lensfunpy.Database()

        camera_make = exif_group["Make"]
        camera_model = exif_group["Camera Model Name"]

        if DNG_CONVERTER == "dnglab":
            self.log(
                f'"dnglab" used, prepending "{camera_make}" camera make.', "warning"
            )
            camera_model = f"{camera_make} {camera_model}"

        self.log(
            f'Searching for "{camera_make}" "{camera_model}" camera model.',
        )
        camera = next(
            iter(
                database.find_cameras(
                    maker=camera_make, model=camera_model, loose_search=True
                )
            ),
            None,
        )
        self.log(
            f'Using "{camera}" camera for lens aberrations correction.',
        )

        lens_model = exif_group["Lens Model"]
        self.log(
            f'Searching for "{lens_model}" lens model.',
        )
        lens = next(
            iter(database.find_lenses(camera, lens=lens_model, loose_search=True)), None
        )
        self.log(
            f'Using "{lens}" lens for lens aberrations correction.',
        )

        focal_length = exif_group["Focal Length"]
        aperture = exif_group["F Number"]
        distance = self.get_input("focus_distance")
        width = input_image.shape[1]
        height = input_image.shape[0]

        modifier = lensfunpy.Modifier(lens, camera.crop_factor, width, height)
        modifier.initialize(
            focal_length,
            aperture,
            distance,
            pixel_format=np.float32,
            flags=lensfunpy.ModifyFlags.ALL,
        )

        output_image = input_image

        if self.get_input("correct_vignette"):
            self.log("Correcting lens vignette...")

            if modifier.apply_color_modification(output_image):
                self.log("Lens vignette was successfully corrected!")
            else:
                self.log(
                    "Lens vignette was not corrected, "
                    "the lens might be missing data."
                )

        if self.get_input("correct_chromatic_aberration"):
            self.log("Correcting lens chromatic aberration...")

            coordinates = modifier.apply_subpixel_distortion()
            if coordinates is not None:
                output_image[..., 0] = cv2.remap(
                    output_image[..., 0], coordinates[..., 0, :], None, cv2.INTER_CUBIC
                )
                output_image[..., 1] = cv2.remap(
                    output_image[..., 1], coordinates[..., 1, :], None, cv2.INTER_CUBIC
                )
                output_image[..., 2] = cv2.remap(
                    output_image[..., 2], coordinates[..., 2, :], None, cv2.INTER_CUBIC
                )
                self.log("Lens chromatic aberration was successfully corrected!")
            else:
                self.log(
                    "Lens chromatic aberration was not corrected, "
                    "the lens might be missing data."
                )

        if self.get_input("correct_distortion"):
            self.log("Correcting lens distortion...")

            coordinates = modifier.apply_geometry_distortion()
            if coordinates is not None:
                output_image = cv2.remap(
                    output_image, coordinates, None, cv2.INTER_CUBIC
                )
                self.log("Lens distortion was successfully corrected!")
            else:
                self.log(
                    "Lens distortion was not corrected, "
                    "the lens might be missing data."
                )

        self.set_output("output_image", output_image)

        self.dirty = False


class NodeDownsample(ExecutionNode):
    """
    Downsample the input image by the input downsampling factor.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeDownsample.__init__`
    -   :meth:`~colour_hdri.NodeDownsample.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Downsample the input image by the input downsampling factor"

        self.add_input_port("input_image")
        self.add_input_port("factor", 1)
        self.add_input_port("bypass", False)
        self.add_output_port("output_image")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        input_image = self.get_input("input_image")
        if input_image is None:
            return

        self.set_output("output_image", input_image)

        if self.get_input("bypass"):
            return

        factor = self.get_input("factor")
        output_image = input_image[::factor, ::factor, ...]

        self.set_output("output_image", output_image)

        self.dirty = False


class NodeApplyInputTransformDNG(ExecutionNode):
    """
    Apply the input transform to the input image using the *DNG* method.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeApplyInputTransformDNG.__init__`
    -   :meth:`~colour_hdri.NodeApplyInputTransformDNG.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            'Apply the input transform to the input image using the "DNG" method'
        )

        self.add_input_port("input_image")
        self.add_input_port("input_transform", InputTransform())
        self.add_input_port("output_colourspace", "sRGB")
        self.add_input_port("bypass", False)
        self.add_output_port("output_image")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        input_image = self.get_input("input_image")
        if input_image is None:
            return

        self.set_output("output_image", input_image)

        if self.get_input("bypass"):
            return

        input_transform = self.get_input("input_transform")

        RGB_w = vecmul(input_transform.M, ones(3))

        output_image = input_image * RGB_w * np.max(1 / RGB_w)

        output_image = XYZ_to_RGB(
            vecmul(
                np.linalg.inv(input_transform.M),
                output_image,
            ),
            self.get_input("output_colourspace"),
            CCS_ILLUMINANT_ADOBEDNG,
        )

        self.set_output("output_image", output_image)

        self.dirty = False


class NodeApplyInputTransformCameraSensitivities(ExecutionNode):
    """
    Apply the input transform to the input image using the
    *Camera Sensitivities* method.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeApplyInputTransformCameraSensitivities.__init__`
    -   :meth:`~colour_hdri.NodeApplyInputTransformCameraSensitivities.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            "Apply the input transform to the input image using the "
            '"Camera Sensitivities" method'
        )

        self.add_input_port("input_image")
        self.add_input_port("input_transform", InputTransform())
        self.add_input_port("output_colourspace", "sRGB")
        self.add_input_port("bypass", False)
        self.add_output_port("output_image")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        input_image = self.get_input("input_image")
        if input_image is None:
            return

        self.set_output("output_image", input_image)

        if self.get_input("bypass"):
            return

        input_transform = self.get_input("input_transform")

        output_image = input_image * np.max(input_transform.RGB_w)

        output_image = RGB_to_RGB(
            vecmul(
                input_transform.M,
                output_image,
            ),
            "ACES2065-1",
            self.get_input("output_colourspace"),
        )

        self.set_output("output_image", output_image)

        self.dirty = False


class NodeCreateBatches(ExecutionNode):
    """
    Create batches from the input array.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeCreateBatches.__init__`
    -   :meth:`~colour_hdri.NodeCreateBatches.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Create batches from the input array"

        self.add_input_port("array", [])
        self.add_input_port("batch_size", 3)
        self.add_output_port("batches", [])

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        array = self.get_input("array")
        if len(array) == 0:
            return

        self.set_output("batches", list(batch(array, self.get_input("batch_size"))))

        self.dirty = False


class NodeCreateImageStack(ExecutionNode):
    """
    Create an image stack from the input files.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeCreateImageStack.__init__`
    -   :meth:`~colour_hdri.NodeCreateImageStack.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Create an image stack from the input files"

        self.add_input_port("paths")
        self.add_input_port("cctf_decoding", linear_function)
        self.add_output_port("image_stack")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        paths = self.get_input("paths")
        if len(paths) == 0:
            return

        for path in paths:
            if not os.path.exists(path):
                self.log(f'"{path}" file does not exist!', "error")
                return

        self.set_output(
            "image_stack", ImageStack.from_files(paths, self.get_input("cctf_decoding"))
        )

        self.dirty = False


class NodeMergeImageStack(ExecutionNode):
    """
    Merge to HDRI the input image stack.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeMergeImageStack.__init__`
    -   :meth:`~colour_hdri.NodeMergeImageStack.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Merge to HDRI the input image stack"

        self.add_input_port("image_stack")
        self.add_input_port("weighting_function", double_sigmoid_anchored_function)
        self.add_output_port("image")

    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        image_stack = self.get_input("image_stack")
        if image_stack is None:
            return

        self.set_output(
            "image",
            image_stack_to_HDRI(image_stack, self.get_input("weighting_function")),
        )

        self.dirty = False


class NodeNormaliseExposure(ExecutionNode):
    """
    Normalise the exposure of the input images.

    Methods
    -------
    -   :meth:`~colour_hdri.NodeNormaliseExposure.__init__`
    -   :meth:`~colour_hdri.NodeNormaliseExposure.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Normalise the exposure of the input images"

        self.add_input_port("image_paths", [])
        self.add_input_port("scaling_factor", 0.2)

    @required("OpenImageIO")
    def process(self, **kwargs) -> None:  # noqa: ARG002
        """
        Process the node.
        """

        image_paths = self.get_input("image_paths")
        if len(image_paths) == 0:
            return

        median = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                self.log(f'"{image_path}" image does not exist!')
                return

            median.append(np.median(read_image_OpenImageIO(image_path)))

        normalising_factor = np.median(median)

        self.log(f"Normalising factor: {normalising_factor}")

        for image_path in image_paths:
            image, attributes = read_image_OpenImageIO(image_path, attributes=True)

            image /= normalising_factor
            image *= self.get_input("scaling_factor")

            write_image_OpenImageIO(image, image_path, attributes=attributes)

        self.dirty = False
