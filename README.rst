Colour - HDRI
=============

.. start-badges

|actions| |coveralls| |codacy| |version|

.. |actions| image:: https://github.com/colour-science/colour-hdri/workflows/Continuous%20Integration/badge.svg
    :target: https://github.com/colour-science/colour-hdri/actions
    :alt: Develop Build Status
.. |coveralls| image:: http://img.shields.io/coveralls/colour-science/colour-hdri/develop.svg?style=flat-square
    :target: https://coveralls.io/r/colour-science/colour-hdri
    :alt: Coverage Status
.. |codacy| image:: https://img.shields.io/codacy/grade/290ad2c23b0749b99a1d548ca47d9062/develop.svg?style=flat-square
    :target: https://www.codacy.com/app/colour-science/colour-hdri
    :alt: Code Grade
.. |version| image:: https://img.shields.io/pypi/v/colour-hdri.svg?style=flat-square
    :target: https://pypi.org/project/colour-hdri
    :alt: Package Version

.. end-badges

A `Python <https://www.python.org/>`__ package implementing various
HDRI / Radiance image processing algorithms.

It is open source and freely available under the
`New BSD License <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

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
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__
when cloning the
`repository <https://github.com/colour-science/colour-hdri>`__.

Primary Dependencies
^^^^^^^^^^^^^^^^^^^^

**Colour - HDRI** requires various dependencies in order to run:

-  `Python >= 2.7 <https://www.python.org/download/releases/>`__ or
   `Python >= 3.5 <https://www.python.org/download/releases/>`__
-  `Colour Science <https://www.colour-science.org>`__
-  `Recordclass <https://pypi.org/project/recordclass/>`__

Optional Features Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-   `colour-demosaicing <https://github.com/colour-science/colour-demosaicing>`__
-   `Adobe DNG Converter <https://www.adobe.com/support/downloads/product.jsp?product=106&platform=Mac>`__
-   `dcraw <https://www.cybercom.net/~dcoffin/dcraw/>`__
-   `ExifTool <http://www.sno.phy.queensu.ca/~phil/exiftool/>`__
-   `rawpy <https://github.com/neothemachine/rawpy>`__

Pypi
^^^^

Once the dependencies satisfied, **Colour - HDRI** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-hdri>`__ by
issuing this command in a shell::

	pip install colour-hdri

The optional features dependencies are installed as follows::

    pip install 'colour-hdri[optional]'

The figures plotting dependencies are installed as follows::

    pip install 'colour-hdri[plotting]'

The overall development dependencies are installed as follows::

    pip install 'colour-hdri[development]'

Usage
-----

API
^^^

The main reference for `Colour - HDRI <https://github.com/colour-science/colour-hdri>`__
is the `Colour - HDRI Manual <https://colour-hdri.readthedocs.io/en/latest/manual.html>`__.

Examples
^^^^^^^^

Various usage examples are available from the
`examples directory <https://github.com/colour-science/colour-hdri/tree/master/colour_hdri/examples>`__.

Contributing
------------

If you would like to contribute to `Colour - HDRI <https://github.com/colour-science/colour-hdri>`__,
please refer to the following `Contributing <https://www.colour-science.org/contributing/>`__
guide for `Colour <https://github.com/colour-science/colour>`__.

Bibliography
------------

The bibliography is available in the repository in
`BibTeX <https://github.com/colour-science/colour-hdri/blob/develop/BIBLIOGRAPHY.bib>`__
format.

Code of Conduct
---------------

The *Code of Conduct*, adapted from the `Contributor Covenant 1.4 <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>`__,
is available on the `Code of Conduct <https://www.colour-science.org/code-of-conduct/>`__ page.

About
-----

| **Colour - HDRI** by Colour Developers
| Copyright © 2015-2019 – Colour Developers – `colour-science@googlegroups.com <colour-science@googlegroups.com>`__
| This software is released under terms of New BSD License: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour-hdri <https://github.com/colour-science/colour-hdri>`__
