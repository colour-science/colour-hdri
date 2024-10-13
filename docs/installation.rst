Installation Guide
==================

Because of their size, the resources dependencies needed to run the various
examples and unit tests are not provided within the Pypi package. They are
separately available as
`Git Submodules <https://git-scm.com/book/en/v2/Git-Tools-Submodules>`__
when cloning the
`repository <https://github.com/colour-science/colour-hdri>`__.

Primary Dependencies
--------------------

**Colour - HDRI** requires various dependencies in order to run:

- `python >= 3.10, < 3.14 <https://www.python.org/download/releases>`__
- `colour-science >= 4.4 <https://pypi.org/project/colour-science>`__
- `imageio >= 2, < 3 <https://imageio.github.io>`__
- `numpy >= 1.24, < 3 <https://pypi.org/project/numpy>`__
- `scipy >= 1.10, < 2 <https://pypi.org/project/scipy>`__

Optional Features Dependencies
------------------------------

- `colour-demosaicing <https://pypi.org/project/colour-demosaicing>`__
- `Adobe DNG Converter <https://helpx.adobe.com/nz/camera-raw/using/adobe-dng-converter.html>`__
- `dcraw <https://dechifro.org/dcraw/>`__
- `ExifTool <http://www.sno.phy.queensu.ca/~phil/exiftool>`__
- `rawpy <https://pypi.org/project/rawpy>`__

Pypi
----

Once the dependencies are satisfied, **Colour - HDRI** can be installed from
the `Python Package Index <http://pypi.python.org/pypi/colour-hdri>`__ by
issuing this command in a shell::

    pip install --user colour-hdri

The optional features dependencies are installed as follows::

    pip install --user 'colour-hdri[optional]'


The overall development dependencies are installed as follows::

    pip install --user 'colour-hdri[development]'
