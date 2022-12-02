"""
Adobe DNG SDK Dataset
=====================

Defines various datasets objects for *Adobe DNG SDK*:

-   :attr:`colour_hdri.models.datasets.dng.CCS_ILLUMINANT_ADOBEDNG`
-   :attr:`colour_hdri.models.datasets.dng.\
CCT_ILLUMINANTS_ADOBEDNG`
-   :attr:`colour_hdri.models.datasets.dng.\
LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS`

References
----------
-   :cite:`AdobeSystems2015c` : Adobe Systems. (2015). Adobe DNG SDK 1.4 -
    dng_sdk_1_4/dng_sdk/source/dng_camera_profile.cpp -
    dng_camera_profile::IlluminantToTemperature.
    http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip
-   :cite:`AdobeSystems2015d` : Adobe Systems. (2015). Adobe DNG SDK 1.4.
    http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip
-   :cite:`AdobeSystems2015e` : Adobe Systems. (2015). Adobe DNG SDK 1.4 -
    dng_sdk_1_4/dng_sdk/source/dng_tag_values.h - LightSource tag.
    http://download.adobe.com/pub/adobe/dng/dng_sdk_1_4.zip
"""

from __future__ import annotations

from colour.colorimetry import CCS_ILLUMINANTS
from colour.hints import Dict, NDArray
from colour.utilities import CanonicalMapping

__author__ = "Colour Developers"
__copyright__ = "Copyright 2015 Colour Developers"
__license__ = "New BSD License - https://opensource.org/licenses/BSD-3-Clause"
__maintainer__ = "Colour Developers"
__email__ = "colour-developers@colour-science.org"
__status__ = "Production"

__all__ = [
    "CCS_ILLUMINANT_ADOBEDNG",
    "CCT_ILLUMINANTS_ADOBEDNG",
    "LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS",
]

CCS_ILLUMINANT_ADOBEDNG: NDArray = CCS_ILLUMINANTS[
    "CIE 1931 2 Degree Standard Observer"
]["D50"]
"""*Adobe DNG SDK* default illuminant for *CIE XYZ* colourspace."""

CCT_ILLUMINANTS_ADOBEDNG: CanonicalMapping = CanonicalMapping(
    {
        "Standard light A": 2850,
        "Tungsten (incandescent light)": 2850,
        "ISO studio tungsten": 3200,
        "D50": 5000,
        "D55": 5500,
        "Daylight": 5500,
        "Fine weather": 5500,
        "Flash": 5500,
        "Standard light B": 5500,
        "D65": 6500,
        "Standard light C": 6500,
        "Cloudy weather": 6500,
        "D75": 7500,
        "Shade": 7500,
        "Daylight fluorescent (D 5700 - 7100K)": (5700 + 7100) * 0.5,
        "Day white fluorescent (N 4600 - 5500K)": (4600 + 5400) * 0.5,
        "Cool white fluorescent (W 3800 - 4500K)": (3900 + 4500) * 0.5,
        "Fluorescent": (3900 + 4500) * 0.5,
        "White fluorescent (WW 3250 - 3800K)": (3200 + 3700) * 0.5,
        "Warm white fluorescent (L 2600 - 3250K)": (2600 + 3250) * 0.5,
    }
)
"""
*Adobe DNG SDK* illuminants correlated colour temperature.

References
----------
:cite:`AdobeSystems2015c`

Notes
-----
-   The correlated colour temperature are given for the
    *CIE 1931 2 Degree Standard Observer*.
"""

LIGHT_SOURCE_TAG_TO_DNG_ILLUMINANTS: Dict = {
    1: "Daylight",
    2: "Fluorescent",
    3: "Tungsten (incandescent light)",
    4: "Flash",
    9: "Fine weather",
    10: "Cloudy weather",
    11: "Shade",
    12: "Daylight fluorescent (D 5700 - 7100K)",
    13: "Day white fluorescent (N 4600 - 5500K)",
    14: "Cool white fluorescent (W 3800 - 4500K)",
    15: "White fluorescent (WW 3250 - 3800K)",
    16: "Warm white fluorescent (L 2600 - 3250K)",
    17: "Standard light A",
    18: "Standard light B",
    19: "Standard light C",
    20: "D55",
    21: "D65",
    22: "D75",
    23: "D50",
    24: "ISO studio tungsten",
    255: "Other",
}
"""
*Adobe DNG SDK* *LightSource Tag* indexes mapping to illuminants.

References
----------
:cite:`AdobeSystems2015e`
"""
