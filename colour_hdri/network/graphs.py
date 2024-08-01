"""
Network - Graphs
================

Define a collection of :class:`colour.utilities.PortGraph` graphs for raw and
HDR image processing.
"""

from __future__ import annotations

import os

from colour.utilities import (
    ExecutionNode,
    ParallelForMultiprocess,
    PortGraph,
)

from colour_hdri.network import (
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
    NodeOrient,
    NodeProcessingMetadata,
    NodeProcessRawFileRawpy,
    NodeReadFileMetadataDNG,
    NodeRemoveFile,
    NodeWatermark,
    NodeWriteImage,
    NodeWritePreviewImage,
)

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "BSD-3-Clause - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "GraphRawProcessingDNG",
    "GraphRawProcessingCameraSensitivities",
    "GraphMergeHDRI",
    "GraphPostMergeHDRI",
    "GraphBatchMergeHDRI",
    "GraphHDRI",
]


class GraphRawProcessingDNG(ExecutionNode, PortGraph):
    """
    Process given raw file, e.g., *CR2*, *CR3*, *NEF*, using the *DNG* method.

    Methods
    -------
    -   :meth:`~colour_hdri.GraphRawProcessingDNG.__init__`
    -   :meth:`~colour_hdri.GraphRawProcessingDNG.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            'Process given raw file, e.g., "CR2", "CR3", "NEF", using the'
            '"DNG" method'
        )

        self.add_input_port("index")
        self.add_input_port("element")

        self.add_input_port("raw_file_path", None)
        self.add_input_port("output_file_path", None)
        self.add_input_port("output_colourspace", "sRGB")
        self.add_input_port("downsample", 1)
        self.add_input_port("correct_vignette", True)
        self.add_input_port("correct_chromatic_aberration", True)
        self.add_input_port("correct_distortion", True)
        self.add_input_port("orientation")
        self.add_input_port("bypass_input_transform", False)
        self.add_input_port("bypass_correct_lens_aberration", False)
        self.add_input_port("bypass_watermark", False)
        self.add_input_port("bypass_orient", False)

        self.add_output_port("output")

        for node in [
            NodeConvertRawFileToDNGFile("ConvertRawFileToDNGFile"),
            NodeReadFileMetadataDNG("ReadFileMetadataDNG"),
            NodeComputeInputTransformDNG("ComputeInputTransformDNG"),
            NodeProcessRawFileRawpy("ProcessRawFileRawpy"),
            NodeRemoveFile("RemoveDNGFile"),
            NodeCorrectLensAberrationLensFun("CorrectLensAberrationLensFun"),
            NodeDownsample("Downsample"),
            NodeApplyInputTransformDNG("ApplyInputTransformDNG"),
            NodeProcessingMetadata("ProcessingMetadata"),
            NodeWatermark("Watermark"),
            NodeOrient("Orient"),
            NodeWriteImage("WriteImage"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("ConvertRawFileToDNGFile", "execution_output"),
                ("ReadFileMetadataDNG", "execution_input"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("ReadFileMetadataDNG", "dng_file_path"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("ProcessRawFileRawpy", "raw_file_path"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("RemoveDNGFile", "path"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("ProcessingMetadata", "sources"),
            ),
            (
                ("ReadFileMetadataDNG", "execution_output"),
                ("ComputeInputTransformDNG", "execution_input"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("ComputeInputTransformDNG", "metadata"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("ComputeInputTransformDNG", "metadata"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("CorrectLensAberrationLensFun", "metadata"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("ProcessingMetadata", "input_metadata"),
            ),
            (
                ("ComputeInputTransformDNG", "execution_output"),
                ("ProcessRawFileRawpy", "execution_input"),
            ),
            (
                ("ComputeInputTransformDNG", "input_transform"),
                ("ProcessRawFileRawpy", "input_transform"),
            ),
            (
                ("ComputeInputTransformDNG", "input_transform"),
                ("ApplyInputTransformDNG", "input_transform"),
            ),
            (
                ("ComputeInputTransformDNG", "input_transform"),
                ("ProcessingMetadata", "input_transform"),
            ),
            (
                ("ProcessRawFileRawpy", "execution_output"),
                ("RemoveDNGFile", "execution_input"),
            ),
            (
                ("ProcessRawFileRawpy", "image"),
                ("CorrectLensAberrationLensFun", "input_image"),
            ),
            (
                ("RemoveDNGFile", "execution_output"),
                ("CorrectLensAberrationLensFun", "execution_input"),
            ),
            (
                ("CorrectLensAberrationLensFun", "execution_output"),
                ("Downsample", "execution_input"),
            ),
            (
                ("CorrectLensAberrationLensFun", "output_image"),
                ("Downsample", "input_image"),
            ),
            (
                ("Downsample", "execution_output"),
                ("ApplyInputTransformDNG", "execution_input"),
            ),
            (
                ("Downsample", "output_image"),
                ("ApplyInputTransformDNG", "input_image"),
            ),
            (
                ("ApplyInputTransformDNG", "execution_output"),
                ("ProcessingMetadata", "execution_input"),
            ),
            (
                ("ApplyInputTransformDNG", "output_image"),
                ("Watermark", "input_image"),
            ),
            (
                ("ProcessingMetadata", "execution_output"),
                ("Watermark", "execution_input"),
            ),
            (
                ("ProcessingMetadata", "output_metadata"),
                ("Watermark", "metadata"),
            ),
            (
                ("ProcessingMetadata", "output_metadata"),
                ("WriteImage", "metadata"),
            ),
            (
                ("Watermark", "execution_output"),
                ("Orient", "execution_input"),
            ),
            (
                ("Watermark", "output_image"),
                ("Orient", "input_image"),
            ),
            (
                ("Orient", "execution_output"),
                ("WriteImage", "execution_input"),
            ),
            (
                ("Orient", "output_image"),
                ("WriteImage", "image"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self.nodes[input_node].connect(
                input_port,
                self.nodes[output_node],
                output_port,
            )

        self.connect(
            "raw_file_path",
            self.nodes["ConvertRawFileToDNGFile"],
            "raw_file_path",
        )
        self.connect(
            "output_colourspace",
            self.nodes["ApplyInputTransformDNG"],
            "output_colourspace",
        )
        self.connect(
            "output_colourspace",
            self.nodes["ProcessingMetadata"],
            "output_colourspace",
        )
        self.connect(
            "output_colourspace",
            self.nodes["WriteImage"],
            "output_colourspace",
        )
        self.connect(
            "correct_vignette",
            self.nodes["CorrectLensAberrationLensFun"],
            "correct_vignette",
        )
        self.connect(
            "correct_chromatic_aberration",
            self.nodes["CorrectLensAberrationLensFun"],
            "correct_chromatic_aberration",
        )
        self.connect(
            "correct_distortion",
            self.nodes["CorrectLensAberrationLensFun"],
            "correct_distortion",
        )
        self.connect(
            "bypass_correct_lens_aberration",
            self.nodes["CorrectLensAberrationLensFun"],
            "bypass",
        )
        self.connect(
            "downsample",
            self.nodes["Downsample"],
            "factor",
        )
        self.connect(
            "orientation",
            self.nodes["Orient"],
            "orientation",
        )
        self.connect(
            "orientation",
            self.nodes["ProcessingMetadata"],
            "orientation",
        )
        self.connect(
            "bypass_input_transform",
            self.nodes["ComputeInputTransformDNG"],
            "bypass",
        )
        self.connect(
            "bypass_input_transform",
            self.nodes["ApplyInputTransformDNG"],
            "bypass",
        )
        self.connect(
            "bypass_watermark",
            self.nodes["Watermark"],
            "bypass",
        )
        self.connect(
            "bypass_orient",
            self.nodes["Orient"],
            "bypass",
        )
        self.connect(
            "output_file_path",
            self.nodes["WriteImage"],
            "path",
        )

    def process(self, **kwargs) -> None:
        """
        Process the node-graph.
        """

        if self.get_input("element") is not None:
            self.set_input("raw_file_path", self.get_input("element"))
            extension = os.path.splitext(self.get_input("raw_file_path"))[-1]
            self.set_input(
                "output_file_path",
                self.get_input("raw_file_path").replace(extension, ".exr"),
            )

        super().process(**kwargs)

        self.set_output("output", self.get_input("output_file_path"))


class GraphRawProcessingCameraSensitivities(ExecutionNode, PortGraph):
    """
    Process given raw file, e.g., *CR2*, *CR3*, *NEF*, using the
    *Camera Sensitivities* method.

    Methods
    -------
    -   :meth:`~colour_hdri.GraphRawProcessingCameraSensitivities.__init__`
    -   :meth:`~colour_hdri.GraphRawProcessingCameraSensitivities.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            'Process given raw file, e.g., "CR2", "CR3", "NEF", using the'
            '"Camera Sensitivities" method'
        )

        self.add_input_port("index")
        self.add_input_port("element")

        self.add_input_port("raw_file_path", None)
        self.add_input_port("output_file_path", None)
        self.add_input_port("output_colourspace", "sRGB")
        self.add_input_port("camera_sensitivities")
        self.add_input_port("downsample", 1)
        self.add_input_port("correct_vignette", True)
        self.add_input_port("correct_chromatic_aberration", True)
        self.add_input_port("correct_distortion", True)
        self.add_input_port("orientation")
        self.add_input_port("bypass_input_transform", False)
        self.add_input_port("bypass_correct_lens_aberration", False)
        self.add_input_port("bypass_watermark", False)
        self.add_input_port("bypass_orient", False)

        self.add_output_port("output")

        for node in [
            NodeConvertRawFileToDNGFile("ConvertRawFileToDNGFile"),
            NodeReadFileMetadataDNG("ReadFileMetadataDNG"),
            NodeComputeInputTransformCameraSensitivities(
                "ComputeInputTransformCameraSensitivities"
            ),
            NodeProcessRawFileRawpy("ProcessRawFileRawpy"),
            NodeRemoveFile("RemoveDNGFile"),
            NodeCorrectLensAberrationLensFun("CorrectLensAberrationLensFun"),
            NodeDownsample("Downsample"),
            NodeApplyInputTransformCameraSensitivities(
                "ApplyInputTransformCameraSensitivities"
            ),
            NodeProcessingMetadata("ProcessingMetadata"),
            NodeWatermark("Watermark"),
            NodeOrient("Orient"),
            NodeWriteImage("WriteImage"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("ConvertRawFileToDNGFile", "execution_output"),
                ("ReadFileMetadataDNG", "execution_input"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("ReadFileMetadataDNG", "dng_file_path"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("ProcessRawFileRawpy", "raw_file_path"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("RemoveDNGFile", "path"),
            ),
            (
                ("ConvertRawFileToDNGFile", "dng_file_path"),
                ("ProcessingMetadata", "sources"),
            ),
            (
                ("ReadFileMetadataDNG", "execution_output"),
                ("ComputeInputTransformCameraSensitivities", "execution_input"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("ComputeInputTransformCameraSensitivities", "metadata"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("ComputeInputTransformCameraSensitivities", "metadata"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("CorrectLensAberrationLensFun", "metadata"),
            ),
            (
                ("ReadFileMetadataDNG", "metadata"),
                ("ProcessingMetadata", "input_metadata"),
            ),
            (
                ("ComputeInputTransformCameraSensitivities", "execution_output"),
                ("ProcessRawFileRawpy", "execution_input"),
            ),
            (
                ("ComputeInputTransformCameraSensitivities", "input_transform"),
                ("ProcessRawFileRawpy", "input_transform"),
            ),
            (
                ("ComputeInputTransformCameraSensitivities", "input_transform"),
                ("ApplyInputTransformCameraSensitivities", "input_transform"),
            ),
            (
                ("ComputeInputTransformCameraSensitivities", "input_transform"),
                ("ProcessingMetadata", "input_transform"),
            ),
            (
                ("ProcessRawFileRawpy", "execution_output"),
                ("RemoveDNGFile", "execution_input"),
            ),
            (
                ("ProcessRawFileRawpy", "image"),
                ("CorrectLensAberrationLensFun", "input_image"),
            ),
            (
                ("RemoveDNGFile", "execution_output"),
                ("CorrectLensAberrationLensFun", "execution_input"),
            ),
            (
                ("CorrectLensAberrationLensFun", "execution_output"),
                ("Downsample", "execution_input"),
            ),
            (
                ("CorrectLensAberrationLensFun", "output_image"),
                ("Downsample", "input_image"),
            ),
            (
                ("Downsample", "execution_output"),
                ("ApplyInputTransformCameraSensitivities", "execution_input"),
            ),
            (
                ("Downsample", "output_image"),
                ("ApplyInputTransformCameraSensitivities", "input_image"),
            ),
            (
                ("ApplyInputTransformCameraSensitivities", "execution_output"),
                ("ProcessingMetadata", "execution_input"),
            ),
            (
                ("ApplyInputTransformCameraSensitivities", "output_image"),
                ("Watermark", "input_image"),
            ),
            (
                ("ProcessingMetadata", "execution_output"),
                ("Watermark", "execution_input"),
            ),
            (
                ("ProcessingMetadata", "output_metadata"),
                ("Watermark", "metadata"),
            ),
            (
                ("ProcessingMetadata", "output_metadata"),
                ("WriteImage", "metadata"),
            ),
            (
                ("Watermark", "execution_output"),
                ("Orient", "execution_input"),
            ),
            (
                ("Watermark", "output_image"),
                ("Orient", "input_image"),
            ),
            (
                ("Orient", "execution_output"),
                ("WriteImage", "execution_input"),
            ),
            (
                ("Orient", "output_image"),
                ("WriteImage", "image"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self.nodes[input_node].connect(
                input_port,
                self.nodes[output_node],
                output_port,
            )

        self.connect(
            "raw_file_path",
            self.nodes["ConvertRawFileToDNGFile"],
            "raw_file_path",
        )
        self.connect(
            "output_colourspace",
            self.nodes["ApplyInputTransformCameraSensitivities"],
            "output_colourspace",
        )
        self.connect(
            "output_colourspace",
            self.nodes["ProcessingMetadata"],
            "output_colourspace",
        )
        self.connect(
            "output_colourspace",
            self.nodes["WriteImage"],
            "output_colourspace",
        )
        self.connect(
            "camera_sensitivities",
            self.nodes["ComputeInputTransformCameraSensitivities"],
            "camera_sensitivities",
        )
        self.connect(
            "correct_vignette",
            self.nodes["CorrectLensAberrationLensFun"],
            "correct_vignette",
        )
        self.connect(
            "correct_chromatic_aberration",
            self.nodes["CorrectLensAberrationLensFun"],
            "correct_chromatic_aberration",
        )
        self.connect(
            "correct_distortion",
            self.nodes["CorrectLensAberrationLensFun"],
            "correct_distortion",
        )
        self.connect(
            "bypass_correct_lens_aberration",
            self.nodes["CorrectLensAberrationLensFun"],
            "bypass",
        )
        self.connect(
            "downsample",
            self.nodes["Downsample"],
            "factor",
        )
        self.connect(
            "orientation",
            self.nodes["Orient"],
            "orientation",
        )
        self.connect(
            "orientation",
            self.nodes["ProcessingMetadata"],
            "orientation",
        )
        self.connect(
            "bypass_input_transform",
            self.nodes["ComputeInputTransformCameraSensitivities"],
            "bypass",
        )
        self.connect(
            "bypass_input_transform",
            self.nodes["ApplyInputTransformCameraSensitivities"],
            "bypass",
        )
        self.connect(
            "bypass_watermark",
            self.nodes["Watermark"],
            "bypass",
        )
        self.connect(
            "bypass_orient",
            self.nodes["Orient"],
            "bypass",
        )
        self.connect(
            "output_file_path",
            self.nodes["WriteImage"],
            "path",
        )

    def process(self, **kwargs) -> None:
        """
        Process the node-graph.
        """

        if self.get_input("element") is not None:
            self.set_input("raw_file_path", self.get_input("element"))
            extension = os.path.splitext(self.get_input("raw_file_path"))[-1]
            self.set_input(
                "output_file_path",
                self.get_input("raw_file_path").replace(extension, ".exr"),
            )

        super().process(**kwargs)

        self.set_output("output", self.get_input("output_file_path"))


class GraphMergeHDRI(ExecutionNode, PortGraph):
    """
    Merge the given *EXR* files to HDRI.

    Methods
    -------
    -   :meth:`~colour_hdri.GraphMergeHDRI.__init__`
    -   :meth:`~colour_hdri.GraphMergeHDRI.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = 'Merge the given "EXR" files to HDRI'

        self.add_input_port("index")
        self.add_input_port("element")

        self.add_input_port("exr_file_paths", None)
        self.add_input_port("metadata", None)
        self.add_input_port("input_transform", InputTransform())
        self.add_input_port("output_colourspace", "sRGB")
        self.add_input_port("output_file_path", None)
        self.add_input_port("bypass_watermark", False)

        self.add_output_port("output")

        for node in [
            NodeCreateImageStack("CreateImageStack"),
            NodeMergeImageStack("MergeImageStack"),
            NodeDownsample("Downsample"),
            NodeProcessingMetadata("ProcessingMetadata"),
            NodeWatermark("Watermark"),
            NodeWriteImage("WriteImage"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("CreateImageStack", "execution_output"),
                ("MergeImageStack", "execution_input"),
            ),
            (
                ("CreateImageStack", "image_stack"),
                ("MergeImageStack", "image_stack"),
            ),
            (
                ("MergeImageStack", "execution_output"),
                ("Downsample", "execution_input"),
            ),
            (
                ("MergeImageStack", "image"),
                ("Downsample", "input_image"),
            ),
            (
                ("Downsample", "execution_output"),
                ("ProcessingMetadata", "execution_input"),
            ),
            (
                ("Downsample", "output_image"),
                ("Watermark", "input_image"),
            ),
            (
                ("ProcessingMetadata", "execution_output"),
                ("Watermark", "execution_input"),
            ),
            (
                ("ProcessingMetadata", "output_metadata"),
                ("Watermark", "metadata"),
            ),
            (
                ("ProcessingMetadata", "output_metadata"),
                ("WriteImage", "metadata"),
            ),
            (
                ("Watermark", "execution_output"),
                ("WriteImage", "execution_input"),
            ),
            (
                ("Watermark", "output_image"),
                ("WriteImage", "image"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self._nodes[input_node].connect(
                input_port,
                self._nodes[output_node],
                output_port,
            )

        self.connect(
            "exr_file_paths",
            self.nodes["CreateImageStack"],
            "paths",
        )
        self.connect(
            "metadata",
            self.nodes["ProcessingMetadata"],
            "input_metadata",
        )
        self.connect(
            "exr_file_paths",
            self.nodes["ProcessingMetadata"],
            "sources",
        )
        self.connect(
            "input_transform",
            self.nodes["ProcessingMetadata"],
            "input_transform",
        )
        self.connect(
            "output_colourspace",
            self.nodes["ProcessingMetadata"],
            "output_colourspace",
        )
        self.connect(
            "output_colourspace",
            self.nodes["WriteImage"],
            "output_colourspace",
        )
        self.connect(
            "bypass_watermark",
            self.nodes["Watermark"],
            "bypass",
        )
        self.connect(
            "output_file_path",
            self.nodes["WriteImage"],
            "path",
        )

    def process(self, **kwargs) -> None:
        """
        Process the node-graph.
        """

        if self.get_input("element") is not None:
            element = self.get_input("element")

            exr_file_paths = [item[0] for item in element]
            metadata = [item[1] for item in element]
            input_transform = [item[2] for item in element]
            output_colourspace = [item[3] for item in element]

            self.set_input("exr_file_paths", exr_file_paths)
            self.set_input("metadata", next(iter(metadata)))
            self.set_input("input_transform", next(iter(input_transform)))
            self.set_input("output_colourspace", next(iter(output_colourspace)))

            filename = os.path.basename(exr_file_paths[0])
            filename, _extension = os.path.splitext(filename)

            self.set_input(
                "output_file_path",
                os.path.join(
                    os.path.dirname(exr_file_paths[0]),
                    f"{filename}_{len(exr_file_paths)}_HDR.exr",
                ),
            )

        self.nodes["ProcessingMetadata"].set_input("type", "hdr")
        self.nodes["Watermark"].set_input("include_exposure_information", False)

        super().process(**kwargs)

        self.set_output("output", self.get_input("output_file_path"))


def _task_multiprocess_post_merge_hdr(args):
    i, element, sub_graph, node = args

    node.log(f"Index {i}, Element {element}", "info")

    node.set_output("index", i)
    node.set_output("element", element)

    sub_graph.process()

    return i, sub_graph.get_output("preview_path")


class GraphPostMergeHDRI(ExecutionNode, PortGraph):
    """
    Normalise the exposure of the input *EXR* files at once and write
    corresponding preview image files.

    Methods
    -------
    -   :meth:`~colour_hdri.GraphPostMergeHDRI.__init__`
    -   :meth:`~colour_hdri.GraphPostMergeHDRI.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            'Normalise the exposure of the input "EXR" files at once and write '
            "corresponding preview image files"
        )

        self.add_input_port("array", [])
        self.add_output_port("output")
        self.add_input_port("processes")

        for node in [
            NodeNormaliseExposure("NormaliseExposure"),
            ParallelForMultiprocess("ParallelForMultiprocess"),
            NodeWritePreviewImage("WritePreviewImage"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("ParallelForMultiprocess", "loop_output"),
                ("WritePreviewImage", "execution_input"),
            ),
            (
                ("ParallelForMultiprocess", "element"),
                ("WritePreviewImage", "path"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self._nodes[input_node].connect(
                input_port,
                self._nodes[output_node],
                output_port,
            )

        self.connect(
            "array",
            self.nodes["NormaliseExposure"],
            "image_paths",
        )
        self.connect(
            "array",
            self.nodes["ParallelForMultiprocess"],
            "array",
        )
        self.connect(
            "processes",
            self.nodes["ParallelForMultiprocess"],
            "processes",
        )
        self.nodes["ParallelForMultiprocess"].set_input(
            "task", _task_multiprocess_post_merge_hdr
        )

    def process(self, **kwargs) -> None:
        """
        Process the node-graph.
        """

        super().process(**kwargs)

        self.set_output(
            "output", self.nodes["ParallelForMultiprocess"].get_output("results")
        )


class GraphBatchMergeHDRI(ExecutionNode, PortGraph):
    """
    Batch-merge to HDRI the input files.

    Methods
    -------
    -   :meth:`~colour_hdri.GraphBatchMergeHDRI.__init__`
    -   :meth:`~colour_hdri.GraphBatchMergeHDRI.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = "Batch-merge to HDRI the input files"

        self.add_input_port("array", [])
        self.add_input_port("batch_size", 3)
        self.add_input_port("bypass_watermark", False)
        self.add_input_port("processes")

        self.add_output_port("output")

        for node in [
            NodeCreateBatches("CreateBatches"),
            ParallelForMultiprocess("ParallelForMultiprocess"),
            GraphMergeHDRI("GraphMergeHDRI"),
            GraphPostMergeHDRI("GraphPostMergeHDRI"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("CreateBatches", "execution_output"),
                ("ParallelForMultiprocess", "execution_input"),
            ),
            (
                ("CreateBatches", "batches"),
                ("ParallelForMultiprocess", "array"),
            ),
            (
                ("ParallelForMultiprocess", "loop_output"),
                ("GraphMergeHDRI", "execution_input"),
            ),
            (
                ("ParallelForMultiprocess", "index"),
                ("GraphMergeHDRI", "index"),
            ),
            (
                ("ParallelForMultiprocess", "element"),
                ("GraphMergeHDRI", "element"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self._nodes[input_node].connect(
                input_port,
                self._nodes[output_node],
                output_port,
            )

        self.connect(
            "array",
            self.nodes["CreateBatches"],
            "array",
        )
        self.connect(
            "batch_size",
            self.nodes["CreateBatches"],
            "batch_size",
        )
        self.connect(
            "bypass_watermark",
            self.nodes["GraphMergeHDRI"],
            "bypass_watermark",
        )
        self.connect(
            "processes",
            self.nodes["ParallelForMultiprocess"],
            "processes",
        )
        self.nodes["ParallelForMultiprocess"].connect(
            "execution_output", self.nodes["GraphPostMergeHDRI"], "execution_input"
        )
        self.nodes["ParallelForMultiprocess"].connect(
            "results", self.nodes["GraphPostMergeHDRI"], "array"
        )

    def process(self, **kwargs) -> None:
        """
        Process the node-graph.
        """

        super().process(**kwargs)

        self.set_output(
            "output",
            list(
                zip(
                    self.nodes["ParallelForMultiprocess"].get_output("results"),
                    self.nodes["GraphPostMergeHDRI"].get_output("output"),
                )
            ),
        )


def _task_multiprocess_graph_hdri(args):
    i, element, sub_graph, node = args

    node.log(f"Index {i}, Element {element}", "info")

    node.set_output("index", i)
    node.set_output("element", element)

    sub_graph.process()

    return (
        i,
        (
            sub_graph.get_input("output_file_path"),
            sub_graph.nodes["ProcessingMetadata"].get_input("input_metadata"),
            sub_graph.nodes["ProcessingMetadata"].get_input("input_transform"),
            sub_graph.nodes["ProcessingMetadata"].get_input("output_colourspace"),
        ),
    )


class GraphHDRI(ExecutionNode, PortGraph):
    """
    Merge to HDRI the input files using the *Camera Sensitivities* method.

    Methods
    -------
    -   :meth:`~colour_hdri.GraphHDRI.__init__`
    -   :meth:`~colour_hdri.GraphHDRI.process`
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.description = (
            'Merge to HDRI the input files using the "Camera Sensitivities" method'
        )

        self.add_input_port("array", [])
        self.add_input_port("camera_sensitivities")
        self.add_input_port("downsample", 1)
        self.add_input_port("correct_vignette", True)
        self.add_input_port("correct_chromatic_aberration", True)
        self.add_input_port("correct_distortion", True)
        self.add_input_port("orientation", None)
        self.add_input_port("bypass_input_transform", False)
        self.add_input_port("bypass_correct_lens_aberration", False)
        self.add_input_port("bypass_watermark", False)
        self.add_input_port("bypass_orient", False)
        self.add_input_port("batch_size", 3)
        self.add_input_port("processes")

        self.add_output_port("output")

        for node in [
            ParallelForMultiprocess("ParallelForMultiprocess"),
            GraphRawProcessingCameraSensitivities(
                "GraphRawProcessingCameraSensitivities"
            ),
            GraphBatchMergeHDRI("GraphBatchMergeHDRI"),
        ]:
            self.add_node(node)

        for connection in [
            (
                ("ParallelForMultiprocess", "execution_output"),
                ("GraphBatchMergeHDRI", "execution_input"),
            ),
            (
                ("ParallelForMultiprocess", "results"),
                ("GraphBatchMergeHDRI", "array"),
            ),
            (
                ("ParallelForMultiprocess", "loop_output"),
                ("GraphRawProcessingCameraSensitivities", "execution_input"),
            ),
            (
                ("ParallelForMultiprocess", "index"),
                ("GraphRawProcessingCameraSensitivities", "index"),
            ),
            (
                ("ParallelForMultiprocess", "element"),
                ("GraphRawProcessingCameraSensitivities", "element"),
            ),
        ]:
            (input_node, input_port), (output_node, output_port) = connection
            self._nodes[input_node].connect(
                input_port,
                self._nodes[output_node],
                output_port,
            )

        self.connect(
            "array",
            self.nodes["ParallelForMultiprocess"],
            "array",
        )
        self.connect(
            "camera_sensitivities",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "camera_sensitivities",
        )
        self.connect(
            "downsample",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "downsample",
        )
        self.connect(
            "correct_vignette",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "correct_vignette",
        )
        self.connect(
            "correct_chromatic_aberration",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "correct_chromatic_aberration",
        )
        self.connect(
            "correct_distortion",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "correct_distortion",
        )
        self.connect(
            "orientation",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "orientation",
        )
        self.connect(
            "bypass_input_transform",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "bypass_input_transform",
        )
        self.connect(
            "bypass_correct_lens_aberration",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "bypass_correct_lens_aberration",
        )
        self.connect(
            "bypass_watermark",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "bypass_watermark",
        )
        self.connect(
            "bypass_watermark",
            self.nodes["GraphBatchMergeHDRI"],
            "bypass_watermark",
        )
        self.connect(
            "bypass_orient",
            self.nodes["GraphRawProcessingCameraSensitivities"],
            "bypass_orient",
        )
        self.connect(
            "batch_size",
            self.nodes["GraphBatchMergeHDRI"],
            "batch_size",
        )
        self.connect(
            "processes",
            self.nodes["ParallelForMultiprocess"],
            "processes",
        )
        self.nodes["ParallelForMultiprocess"].set_input(
            "task", _task_multiprocess_graph_hdri
        )
        self.nodes["GraphRawProcessingCameraSensitivities"].nodes[
            "Watermark"
        ].set_input("include_exposure_information", False)

    def process(self, **kwargs) -> None:
        """
        Process the node-graph.
        """

        super().process(**kwargs)

        self.set_output(
            "output", self.nodes["GraphBatchMergeHDRI"].get_output("output")
        )
