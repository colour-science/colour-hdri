Colour - HDRI
=============

..  image:: https://raw.githubusercontent.com/colour-science/colour-hdri/master/docs/_static/Radiance_001.png

.. list-table::
    :stub-columns: 1

    * - Status
      - |waffle| |travis| |coveralls| |scrutinizer| |landscape| |gemnasium|
    * - Package
      - |version| |downloads|

.. |waffle| image:: https://badge.waffle.io/colour-science/colour-hdri.svg?label=ready&title=Ready
    :target: https://github.com/colour-science/colour-hdri/issues
    :alt: Issues Ready
.. |travis| image:: https://img.shields.io/travis/colour-science/colour-hdri/develop.svg
    :target: https://travis-ci.org/colour-science/colour-hdri
    :alt: Develop Build Status
.. |coveralls| image:: http://img.shields.io/coveralls/colour-science/colour-hdri/develop.svg
    :target: https://coveralls.io/r/colour-science/colour-hdri
    :alt: Coverage Status
.. |scrutinizer| image:: https://img.shields.io/scrutinizer/g/colour-science/colour-hdri/develop.svg
    :target: https://scrutinizer-ci.com/g/colour-science/colour-hdri/
    :alt: Code Quality
.. |landscape| image:: https://landscape.io/github/colour-science/colour-hdri/master/landscape.png
    :target: https://landscape.io/github/colour-science/colour-hdri
    :alt: Code Quality
.. |gemnasium| image:: https://img.shields.io/gemnasium/colour-science/colour-hdri.svg
    :target: https://gemnasium.com/colour-science/colour-hdri
    :alt: Dependencies Status
.. |version| image:: https://badge.fury.io/py/colour-hdri.svg
    :target: https://pypi.python.org/pypi/colour-hdri
    :alt: Package Version
.. |downloads| image:: https://img.shields.io/pypi/dm/colour-hdri.svg
    :target: https://pypi.python.org/pypi/colour-hdri
    :alt: Package Downloads

.. end-badges

A `Python <https://www.python.org/>`_ package implementing various
HDRI / Radiance image processing algorithms.

It is open source and freely available under the
`New BSD License <http://opensource.org/licenses/BSD-3-Clause>`_ terms.

Features
--------

The following features are available:

-   HDRI / Radiance Image Generation
-   Debevec (1997) Camera Response Function Computation
-   Grossberg (2003) Histogram Based Image Sampling
-   Variance Minimization Light Probe Sampling
-   Global Tonemapping Operators
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
   `Python 3.5 <https://www.python.org/download/releases/>`_
-  `NumPy <http://www.numpy.org/>`_
-  `OpenImageIO <https://github.com/OpenImageIO/oiio>`_

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

Usage
-----

API
^^^

The main reference for `Colour - HDRI <https://github.com/colour-science/colour-hdri>`_
is the Sphinx `API Reference <http://colour-hdri.readthedocs.io/en/latest/>`_.

Examples
^^^^^^^^

Various usage examples are available from the
`examples directory <https://github.com/colour-science/colour-hdri/tree/master/colour_hdri/examples>`_.

Contributing
------------

If you would like to contribute to `Colour - HDRI <https://github.com/colour-science/colour-hdri>`_,
please refer to the following `Contributing <http://colour-science.org/contributing/>`_
guide for `Colour <https://github.com/colour-science/colour>`_.

Bibliography
------------

The bibliography is available in the repository in either
`BibTeX <https://github.com/colour-science/colour-hdri/blob/develop/BIBLIOGRAPHY.bib>`_
format or `reStructuredText <https://github.com/colour-science/colour-hdri/blob/develop/BIBLIOGRAPHY.rst>`_.

About
-----

| **Colour - HDRI** by Colour Developers
| Copyright © 2015-2016 – Colour Developers – `colour-science@googlegroups.com <colour-science@googlegroups.com>`_
| This software is released under terms of New BSD License: http://opensource.org/licenses/BSD-3-Clause
| `http://github.com/colour-science/colour-hdri <http://github.com/colour-science/colour-hdri>`_
