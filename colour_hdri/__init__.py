#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour - HDRI
=============

HDRI - Radiance image processing algorithms for *Python*.

Subpackages
-----------
-   calibration: Camera calibration computations.
-   examples: Examples for the sub-packages.
-   generation: HDRI / radiance image generation.
-   models: Colour models conversion.
-   plotting: Diagrams, figures, etc...
-   process: Image conversion helpers.
-   recovery: Clipped highlights recovery.
-   resources: Resources sub-modules.
-   sampling: Image sampling routines.
-   tonemapping: Tonemapping operators.
-   utilities: Various utilities and data structures.
"""

from __future__ import absolute_import

import os

from .utilities import *  # noqa
from . import utilities
from .sampling import *  # noqa
from . import sampling
from .generation import *  # noqa
from . import generation
from .calibration import *  # noqa
from . import calibration
from .models import *  # noqa
from . import models
from .process import *  # noqa
from . import process
from .recovery import *  # noqa
from . import recovery
from .tonemapping import *  # noqa
from . import tonemapping

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []
__all__ += utilities.__all__
__all__ += sampling.__all__
__all__ += generation.__all__
__all__ += calibration.__all__
__all__ += models.__all__
__all__ += process.__all__
__all__ += recovery.__all__
__all__ += tonemapping.__all__

RESOURCES_DIRECTORY = os.path.join(
    os.path.dirname(__file__), 'resources')
EXAMPLES_RESOURCES_DIRECTORY = os.path.join(
    RESOURCES_DIRECTORY, 'colour-hdri-examples-dataset')
TESTS_RESOURCES_DIRECTORY = os.path.join(
    RESOURCES_DIRECTORY, 'colour-hdri-tests-dataset')

__application_name__ = 'Colour - HDRI'

__major_version__ = '0'
__minor_version__ = '1'
__change_version__ = '2'
__version__ = '.'.join((__major_version__,
                        __minor_version__,
                        __change_version__))
