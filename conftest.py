"""Configure stable *NumPy* representations for the documented examples."""

from collections.abc import Generator

import numpy as np
import pytest


@pytest.fixture(autouse=True, scope="session")
def numpy_print_options() -> Generator[None]:
    """Preserve the array and scalar representation used in the doctests."""
    with np.printoptions(legacy="1.13"):
        yield
