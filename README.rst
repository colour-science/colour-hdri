Colour - HDRI
=============

.. start-badges

|actions| |coveralls| |codacy| |version|

.. |actions| image:: https://img.shields.io/github/actions/workflow/status/colour-science/colour-hdri/.github/workflows/continuous-integration-quality-unit-tests.yml?branch=develop&style=flat-square
    :target: https://github.com/colour-science/colour-hdri/actions
    :alt: Develop Build Status
.. |coveralls| image:: http://img.shields.io/coveralls/colour-science/colour-hdri/develop.svg?style=flat-square
    :target: https://coveralls.io/r/colour-science/colour-hdri
    :alt: Coverage Status
.. |codacy| image:: https://img.shields.io/codacy/grade/f422dc0703dd4653b2b766217c745813/develop.svg?style=flat-square
    :target: https://app.codacy.com/gh/colour-science/colour-hdri
    :alt: Code Grade
.. |version| image:: https://img.shields.io/pypi/v/colour-hdri.svg?style=flat-square
    :target: https://pypi.org/project/colour-hdri
    :alt: Package Version

.. end-badges

A `Python <https://www.python.org>`__ package implementing various
HDRI processing algorithms.

It is open source and freely available under the
`BSD-3-Clause <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

..  image:: https://raw.githubusercontent.com/colour-science/colour-hdri/master/docs/_static/Radiance_001.png

.. contents:: **Table of Contents**
    :backlinks: none
    :depth: 2

.. sectnum::

Features
--------

The following features are available:

- HDRI Generation
- Debevec (1997) Camera Response Function Computation
- Grossberg (2003) Histogram Based Image Sampling
- Variance Minimization Light Probe Sampling
- Global Tonemapping Operators
- Adobe DNG SDK Colour Processing
- Absolute Luminance Calibration
- Digital Still Camera (DSC) Exposure Model
- Raw Processing Helpers
- Vignette Characterisation & Correction

Examples
^^^^^^^^

Various usage examples are available from the
`examples directory <https://github.com/colour-science/colour-hdri/tree/master/colour_hdri/examples>`__.

User Guide
----------

Installation
^^^^^^^^^^^^

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__
when cloning the
`repository <https://github.com/colour-science/colour-hdri>`__.

Primary Dependencies
~~~~~~~~~~~~~~~~~~~~

**Colour - HDRI** requires various dependencies in order to run:

- `python >= 3.10, < 3.14 <https://www.python.org/download/releases>`__
- `colour-science >= 4.4 <https://pypi.org/project/colour-science>`__
- `imageio >= 2, < 3 <https://imageio.github.io>`__
- `numpy >= 1.24, < 3 <https://pypi.org/project/numpy>`__
- `scipy >= 1.10, < 2 <https://pypi.org/project/scipy>`__

Optional Features Dependencies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `colour-demosaicing <https://pypi.org/project/colour-demosaicing>`__
- `Adobe DNG Converter <https://helpx.adobe.com/nz/camera-raw/using/adobe-dng-converter.html>`__
- `dcraw <https://dechifro.org/dcraw/>`__
- `ExifTool <http://www.sno.phy.queensu.ca/~phil/exiftool>`__
- `rawpy <https://pypi.org/project/rawpy>`__

Pypi
~~~~

Once the dependencies are satisfied, **Colour - HDRI** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-hdri>`__ by
issuing this command in a shell::

    pip install --user colour-hdri

The optional features dependencies are installed as follows::

    pip install --user 'colour-hdri[optional]'

The figures plotting dependencies are installed as follows::

    pip install --user 'colour-hdri[plotting]'

The overall development dependencies are installed as follows::

    pip install --user 'colour-hdri[development]'

Contributing
^^^^^^^^^^^^

If you would like to contribute to `Colour - HDRI <https://github.com/colour-science/colour-hdri>`__,
please refer to the following `Contributing <https://www.colour-science.org/contributing>`__
guide for `Colour <https://github.com/colour-science/colour>`__.

Bibliography
^^^^^^^^^^^^

The bibliography is available in the repository in
`BibTeX <https://github.com/colour-science/colour-hdri/blob/develop/BIBLIOGRAPHY.bib>`__
format.

API Reference
-------------

The main technical reference for `Colour - HDRI <https://github.com/colour-science/colour-hdri>`__
is the `API Reference <https://colour-hdri.readthedocs.io/en/latest/reference.html>`__.

See Also
--------

Publications
^^^^^^^^^^^^

- `Advanced High Dynamic Range Imaging: Theory and Practice <https://dl.acm.org/doi/book/10.5555/1996408>`__ by Banterle, F. et al.

*Advanced High Dynamic Range Imaging: Theory and Practice* was used as a
reference for some of the algorithms of **Colour - HDRI**.

Software
^^^^^^^^

**C/C++**

- `OpenCV <https://opencv.org>`__ by Bradski, G.
- `Piccante <https://github.com/cnr-isti-vclab/piccante>`__ by Banterle, F. and Benedetti, L.,

*Piccante* was used to verify the Grossberg (2003) Histogram Based Image Sampling.

**Matlab**

- `HDR Toolbox <https://github.com/banterle/HDR_Toolbox>`__ by Banterle, F. et al.

Code of Conduct
---------------

The *Code of Conduct*, adapted from the `Contributor Covenant 1.4 <https://www.contributor-covenant.org/version/1/4/code-of-conduct.html>`__,
is available on the `Code of Conduct <https://www.colour-science.org/code-of-conduct>`__ page.

Contact & Social
----------------

The *Colour Developers* can be reached via different means:

- `Email <mailto:colour-developers@colour-science.org>`__
- `Facebook <https://www.facebook.com/python.colour.science>`__
- `Github Discussions <https://github.com/colour-science/colour-hdri/discussions>`__
- `Gitter <https://gitter.im/colour-science/colour>`__
- `Twitter <https://twitter.com/colour_science>`__

About
-----

| **Colour - HDRI** by Colour Developers
| Copyright 2015 Colour Developers – `colour-developers@colour-science.org <colour-developers@colour-science.org>`__
| This software is released under terms of BSD-3-Clause: https://opensource.org/licenses/BSD-3-Clause
| `https://github.com/colour-science/colour-hdri <https://github.com/colour-science/colour-hdri>`__
