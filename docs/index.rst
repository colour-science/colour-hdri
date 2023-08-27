Colour - HDRI
=============

A `Python <https://www.python.org>`__ package implementing various
HDRI processing algorithms.

It is open source and freely available under the
`BSD-3-Clause <https://opensource.org/licenses/BSD-3-Clause>`__ terms.

..  image:: https://raw.githubusercontent.com/colour-science/colour-hdri/master/docs/_static/Radiance_001.png

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

.. toctree::
    :maxdepth: 2

    user-guide

API Reference
-------------

.. toctree::
    :maxdepth: 2

    reference

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
- `Discourse <https://colour-science.discourse.group>`__
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
