"""
Kynera
======
A Python library for downloading, processing, and validating ERA5 reanalysis
data from the Copernicus Climate Data Store (CDS).

Designed for geospatial analysis and ground-truth validation in weather
forecasting tasks.

Course: Geospatial Processing 2025/2026 — Politecnico di Milano

Functions
---------
download_era5           Download ERA5 data from CDS (handles new ZIP format)
load_era5               Load ERA5 NetCDF or ZIP into xarray.Dataset
extract_at_stations     Extract ERA5 values at in-situ station coordinates
convert_units           Convert raw ERA5 units to human-readable units
compute_derived         Compute derived variables (wind speed, RH, ...)
validate_vs_observations Validate ERA5 against in-situ observations (RMSE, MAE, bias)
plot_field              Plot a 2D map of an ERA5 variable
"""

from .kynera import (
    download_era5,
    load_era5,
    extract_at_stations,
    convert_units,
    compute_derived,
    validate_vs_observations,
    plot_field,
)

__all__ = [
    "download_era5",
    "load_era5",
    "extract_at_stations",
    "convert_units",
    "compute_derived",
    "validate_vs_observations",
    "plot_field",
]

__version__ = "0.1.0"
__author__  = "Kynera Contributors"
