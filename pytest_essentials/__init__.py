"""
Pytest Essentials - A collection of essential utilities for Pytest.
"""

from .soft_assert import SoftAssert, SoftAssertBrokenTestError

__version__ = "0.1.0"

__all__ = [
    "SoftAssert",
    "SoftAssertBrokenTestError",
    "__version__",
]