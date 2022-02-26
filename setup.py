"""
Colour - HDRI - Setup
=====================
"""

import codecs
from setuptools import setup

packages = [
    "colour_hdri",
    "colour_hdri.calibration",
    "colour_hdri.calibration.tests",
    "colour_hdri.exposure",
    "colour_hdri.exposure.tests",
    "colour_hdri.generation",
    "colour_hdri.generation.tests",
    "colour_hdri.models",
    "colour_hdri.models.datasets",
    "colour_hdri.models.tests",
    "colour_hdri.plotting",
    "colour_hdri.process",
    "colour_hdri.process.tests",
    "colour_hdri.recovery",
    "colour_hdri.recovery.tests",
    "colour_hdri.sampling",
    "colour_hdri.sampling.tests",
    "colour_hdri.tonemapping",
    "colour_hdri.tonemapping.global_operators",
    "colour_hdri.tonemapping.global_operators.tests",
    "colour_hdri.utilities",
    "colour_hdri.utilities.tests",
]

package_data = {
    "": ["*"],
    "colour_hdri": ["examples/*", "resources/colour-hdri-tests-datasets/*"],
}

install_requires = [
    "colour-science>=0.4.0",
    "imageio>=2,<3",
    "numpy>=1.19,<2",
    "scipy>=1.5,<2",
    "typing-extensions>=4,<5",
]

extras_require = {
    "development": [
        "biblib-simple",
        "black",
        "coverage!=6.3",
        "coveralls",
        "flake8",
        "flynt",
        "invoke",
        "jupyter",
        "mypy",
        "pre-commit",
        "pydata-sphinx-theme",
        "pydocstyle",
        "pytest",
        "pytest-cov",
        "restructuredtext-lint",
        "sphinx>=4,<5",
        "sphinxcontrib-bibtex",
        "toml",
        "twine",
    ],
    "optional": ["colour-demosaicing>=0.2.0"],
    "plotting": ["matplotlib>=3.2,!=3.5.0,!=3.5.1"],
    "read-the-docs": [
        "matplotlib>=3.2,!=3.5.0,!=3.5.1",
        "pydata-sphinx-theme",
        "sphinxcontrib-bibtex",
    ],
}

setup(
    name="colour-hdri",
    version="0.2.0",
    description="HDRI / Radiance image processing algorithms for Python",
    long_description=codecs.open("README.rst", encoding="utf8").read(),
    author="Colour Developers",
    author_email="colour-developers@colour-science.org",
    maintainer="Colour Developers",
    maintainer_email="colour-developers@colour-science.org",
    url="https://www.colour-science.org/",
    packages=packages,
    package_data=package_data,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8,<3.11",
)
