Colour - HDRI
=============

A `Python <https://www.python.org/>`_ package implementing various
HDRI / Radiance image processing algorithms.

It is open source and freely available under the
`New BSD License <https://opensource.org/licenses/BSD-3-Clause>`_ terms.

..  image:: https://raw.githubusercontent.com/colour-science/colour-hdri/master/docs/_static/Radiance_001.png

.. contents:: **Table of Contents**
    :backlinks: none
    :depth: 3

.. sectnum::

Features
--------

The following features are available:

-   HDRI / Radiance Image Generation
-   Debevec (1997) Camera Response Function Computation
-   Grossberg (2003) Histogram Based Image Sampling
-   Variance Minimization Light Probe Sampling
-   Global Tonemapping Operators
-   Adobe DNG SDK Colour Processing
-   Absolute Luminance Calibration
-   Raw Processing Helpers

Installation
------------

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`_
when cloning the
`repository <https://github.com/colour-science/colour-hdri>`_.

Primary Dependencies
^^^^^^^^^^^^^^^^^^^^

**Colour - HDRI** requires various dependencies in order to run:

-  `Python 2.7 <https://www.python.org/download/releases/>`_ or
   `Python 3.7 <https://www.python.org/download/releases/>`_
-  `Colour Science <https://www.colour-science.org>`_

Optional Features Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   `colour-demosaicing <https://github.com/colour-science/colour-demosaicing>`_
-   `Adobe DNG Converter <https://www.adobe.com/support/downloads/product.jsp?product=106&platform=Mac>`_
-   `dcraw <https://www.cybercom.net/~dcoffin/dcraw/>`_
-   `ExifTool <http://www.sno.phy.queensu.ca/~phil/exiftool/>`_
-   `rawpy <https://github.com/neothemachine/rawpy>`_

Pypi
^^^^

Once the dependencies satisfied, **Colour - HDRI** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-hdri>`_ by
issuing this command in a shell::

	pip install colour-hdri

The optional features dependencies are installed as follows::

    pip install 'colour-hdri[optional]'

The figures plotting dependencies are installed as follows::

    pip install 'colour-hdri[plotting]'

The tests suite dependencies are installed as follows::

    pip install 'colour-hdri[tests]'

The documentation building dependencies are installed as follows::

    pip install 'colour-hdri[docs]'

The overall development dependencies are installed as follows::

    pip install 'colour-hdri[development]'

Usage
-----

API
^^^

The main reference for `Colour - HDRI <https://github.com/colour-science/colour-hdri>`_
is the manual:

.. toctree::
    :maxdepth: 4

    manual

Examples
^^^^^^^^

Various usage examples are available from the
`examples directory <https://github.com/colour-science/colour-hdri/tree/master/colour_hdri/examples>`_.

Contributing
------------

If you would like to contribute to `Colour - HDRI <https://github.com/colour-science/colour-hdri>`_,
please refer to the following `Contributing <https://www.colour-science.org/contributing/>`_
guide for `Colour <https://github.com/colour-science/colour>`_.

Bibliography
------------

The bibliography is available in the repository in
`BibTeX <https://github.com/colour-science/colour-hdri/blob/develop/BIBLIOGRAPHY.bib>`_
format.

Code of Conduct
---------------

The *Code of Conduct*, adapted from the `Contributor Covenant 1.4 <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>`_,
is available on the `Code of Conduct <https://www.colour-science.org/code-of-conduct/>`_ page.

About
-----

| **Colour - HDRI** by Colour Developers
| Copyright © 2015-2019 – Colour Developers – `colour-science@googlegroups.com <colour-science@googlegroups.com>`_
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour-hdri <https://github.com/colour-science/colour-hdri>`_
