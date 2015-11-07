#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

from .constants import *  # noqa
from . import constants
from .camera_response_functions import *  # noqa
from . import camera_response_functions
from .exif import *  # noqa
from . import exif
from .exposure import *  # noqa
from . import exposure
from .image import *  # noqa
from . import image
from .plotting import *  # noqa
from . import plotting
from .process import *  # noqa
from . import process
from .radiance import *  # noqa
from . import radiance
from .recovery import *  # noqa
from . import recovery
from .rgb import *  # noqa
from . import rgb
from .utilities import *  # noqa
from . import utilities
from .weighting_functions import *  # noqa
from . import weighting_functions

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2015 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = []
__all__ += constants.__all__
__all__ += camera_response_functions.__all__
__all__ += exif.__all__
__all__ += exposure.__all__
__all__ += image.__all__
__all__ += plotting.__all__
__all__ += process.__all__
__all__ += radiance.__all__
__all__ += recovery.__all__
__all__ += rgb.__all__
__all__ += utilities.__all__
__all__ += weighting_functions.__all__

__application_name__ = 'Colour - HDRI'

__major_version__ = '0'
__minor_version__ = '1'
__change_version__ = '0'
__version__ = '.'.join((__major_version__,
                        __minor_version__,
                        __change_version__))
