"""
kynera
======
Python library for downloading and processing ERA5 reanalysis data
from the Copernicus Climate Data Store (CDS).

Authors: Davide Galluzzo
Course:  Geospatial Processing 2025/2026 — Politecnico di Milano
"""

from .kynera import (
    __version__,
    VARIABLE_CATALOGUE,
    list_variables,
    download_era5,
    load_era5,
    convert_units,
    compute_derived,
    plot_field,
)

__all__ = [
    "__version__",
    "VARIABLE_CATALOGUE",
    "list_variables",
    "download_era5",
    "load_era5",
    "convert_units",
    "compute_derived",
    "plot_field",
]