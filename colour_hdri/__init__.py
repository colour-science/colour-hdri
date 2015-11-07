#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Colour - HDRI
=============

HDRI processing algorithms for *Python*.

Subpackages
-----------
-   calibration
-   generation
-   models
-   plotting
-   process
-   recovery
-   utilities
"""

from __future__ import absolute_import

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
from .plotting import *  # noqa
from . import plotting
from .process import *  # noqa
from . import process
from .recovery import *  # noqa
from . import recovery

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
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
__all__ += plotting.__all__
__all__ += process.__all__
__all__ += recovery.__all__

__application_name__ = 'Colour - HDRI'

__major_version__ = '0'
__minor_version__ = '1'
__change_version__ = '0'
__version__ = '.'.join((__major_version__,
                        __minor_version__,
                        __change_version__))
