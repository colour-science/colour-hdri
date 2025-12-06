"""
Lens Distortion Correction
==========================

Vignetting characterisation and correction algorithms.

This subpackage provides methods for characterising and correcting lens
vignetting effects using various mathematical models including 2D functions,
bivariate splines, and radial basis functions.
"""

from .vignette import (
    VIGNETTE_CHARACTERISATION_METHODS,
    VIGNETTE_CORRECTION_METHODS,
    DataVignetteCharacterisation,
    apply_radial_gradient,
    characterise_vignette,
    characterise_vignette_2D_function,
    characterise_vignette_bivariate_spline,
    characterise_vignette_RBF,
    correct_vignette,
    correct_vignette_2D_function,
    correct_vignette_bivariate_spline,
    correct_vignette_RBF,
    hyperbolic_cosine_2D_function,
    parabolic_2D_function,
)

__all__ = [
    "VIGNETTE_CHARACTERISATION_METHODS",
    "VIGNETTE_CORRECTION_METHODS",
    "DataVignetteCharacterisation",
    "apply_radial_gradient",
    "characterise_vignette",
    "characterise_vignette_2D_function",
    "characterise_vignette_bivariate_spline",
    "characterise_vignette_RBF",
    "correct_vignette",
    "correct_vignette_2D_function",
    "correct_vignette_bivariate_spline",
    "correct_vignette_RBF",
    "hyperbolic_cosine_2D_function",
    "parabolic_2D_function",
]
