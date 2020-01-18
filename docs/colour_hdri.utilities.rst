Utilities
=========

.. contents:: :local:

Common
------

``colour_hdri``

.. currentmodule:: colour_hdri

.. autosummary::
    :toctree: generated/

    vivification
    vivified_to_dict
    path_exists
    filter_files

EXIF Data Manipulation
----------------------

``colour_hdri``

.. currentmodule:: colour_hdri

.. autosummary::
    :toctree: generated/

    EXIF_EXECUTABLE
    ExifTag
    parse_exif_string
    parse_exif_numeric
    parse_exif_fraction
    parse_exif_array
    parse_exif_data
    read_exif_tags
    copy_exif_tags
    update_exif_tags
    delete_exif_tags
    read_exif_tag
    write_exif_tag

Exposure Value Computation
--------------------------

``colour_hdri``

.. currentmodule:: colour_hdri

.. autosummary::
    :toctree: generated/

    average_luminance
    average_illuminance
    luminance_to_exposure_value
    illuminance_to_exposure_value
    adjust_exposure

Image Data & Metadata Utilities
-------------------------------

``colour_hdri``

.. currentmodule:: colour_hdri

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Metadata
    Image
    ImageStack
