# -*- coding: utf-8 -*-

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("phonopy_spectroscopy")
except PackageNotFoundError:
    # If the package is not installed, don't add __version__
    pass
