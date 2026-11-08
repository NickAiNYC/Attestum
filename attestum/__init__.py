"""Commercial layer for the assessment business.

This package holds everything that is NOT the open-source runtime: the
brand record, the contractual template models, and their renderers.
The frozen runtime package (robocore) lives alongside it, unchanged.
"""

from __future__ import annotations

from attestum.brand import BRAND, Brand

__version__ = BRAND.package_version

__all__ = ["BRAND", "Brand", "__version__"]
