"""
kynera.py
=========
Kynera is a Python library for downloading and processing ERA5 reanalysis data
from the Copernicus Climate Data Store (CDS).

Kynera improves on era5cli (https://github.com/eWaterCycle/era5cli) by providing:
  - A Python API instead of a CLI, for direct integration in scripts and notebooks
  - Native handling of the new CDS Beta ZIP format (instant + accum NetCDF files)
  - Automatic multi-year split downloads from a single function call
  - Unit conversion from raw CDS units to human-readable values
  - Derived meteorological variables (wind speed/direction, relative humidity)
  - Georeferenced 2D map visualisation with Cartopy
  - A built-in variable catalogue with units and descriptions

Authors: [your name]
Course:  Geospatial Processing 2025/2026 — Politecnico di Milano
"""

import os
import glob
import zipfile

import numpy as np
import xarray as xr
import cdsapi
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==============================================================================
# VARIABLE CATALOGUE
# era5cli provides an `info` CLI command; Kynera exposes the same information
# as a Python dictionary, queryable at runtime via list_variables().
#
# Structure: short_name → (long_name, raw_unit, converted_unit, conversion_fn)
# ==============================================================================
VARIABLE_CATALOGUE = {
    # --- Surface: instantaneous ---
    "t2m":   ("2m Temperature",                        "K",       "°C",      lambda x: x - 273.15),
    "d2m":   ("2m Dewpoint Temperature",               "K",       "°C",      lambda x: x - 273.15),
    "u10":   ("10m U-component of wind",               "m/s",     "m/s",     lambda x: x),
    "v10":   ("10m V-component of wind",               "m/s",     "m/s",     lambda x: x),
    "msl":   ("Mean Sea Level Pressure",               "Pa",      "hPa",     lambda x: x / 100),
    "sp":    ("Surface Pressure",                      "Pa",      "hPa",     lambda x: x / 100),
    "sst":   ("Sea Surface Temperature",               "K",       "°C",      lambda x: x - 273.15),
    "tcc":   ("Total Cloud Cover",                     "0-1",     "0-1",     lambda x: x),
    "lcc":   ("Low Cloud Cover",                       "0-1",     "0-1",     lambda x: x),
    "mcc":   ("Medium Cloud Cover",                    "0-1",     "0-1",     lambda x: x),
    "hcc":   ("High Cloud Cover",                      "0-1",     "0-1",     lambda x: x),
    "blh":   ("Boundary Layer Height",                 "m",       "m",       lambda x: x),
    "cape":  ("Convective Available Potential Energy", "J/kg",    "J/kg",    lambda x: x),
    "cin":   ("Convective Inhibition",                 "J/kg",    "J/kg",    lambda x: x),
    "tcwv":  ("Total Column Water Vapour",             "kg/m²",   "kg/m²",   lambda x: x),
    "i10fg": ("Instantaneous 10m Wind Gust",           "m/s",     "m/s",     lambda x: x),
    "mx2t":  ("Maximum 2m Temperature",                "K",       "°C",      lambda x: x - 273.15),
    "mn2t":  ("Minimum 2m Temperature",                "K",       "°C",      lambda x: x - 273.15),
    # --- Surface: accumulated ---
    "tp":    ("Total Precipitation",                   "m",       "mm",      lambda x: x * 1000),
    "lsp":   ("Large-scale Precipitation",             "m",       "mm",      lambda x: x * 1000),
    "cp":    ("Convective Precipitation",              "m",       "mm",      lambda x: x * 1000),
    "sf":    ("Snowfall",                              "m",       "mm",      lambda x: x * 1000),
    "ssrd":  ("Surface Solar Radiation Downwards",     "J/m²",    "J/m²",    lambda x: x),
    "sshf":  ("Surface Sensible Heat Flux",            "J/m²",    "J/m²",    lambda x: x),
    "slhf":  ("Surface Latent Heat Flux",              "J/m²",    "J/m²",    lambda x: x),
    # --- Vertical integrals ---
    "viwvd": ("Vert. Integral Divergence of Moisture Flux", "kg/m²/s", "kg/m²/s", lambda x: x),
    # --- Wave ---
    "swh":   ("Significant Wave Height",               "m",       "m",       lambda x: x),
    "pp1d":  ("Peak Wave Period",                      "s",       "s",       lambda x: x),
    "mwd":   ("Mean Wave Direction",                   "°",       "°",       lambda x: x),
    "hmax":  ("Maximum Individual Wave Height",        "m",       "m",       lambda x: x),
}

# CDS long-name → short-name lookup (used in download validation)
_CDS_NAME_TO_SHORT = {
    "2m_temperature":                          "t2m",
    "2m_dewpoint_temperature":                 "d2m",
    "10m_u_component_of_wind":                 "u10",
    "10m_v_component_of_wind":                 "v10",
    "mean_sea_level_pressure":                 "msl",
    "surface_pressure":                        "sp",
    "sea_surface_temperature":                 "sst",
    "total_cloud_cover":                       "tcc",
    "low_cloud_cover":                         "lcc",
    "medium_cloud_cover":                      "mcc",
    "high_cloud_cover":                        "hcc",
    "boundary_layer_height":                   "blh",
    "convective_available_potential_energy":   "cape",
    "convective_inhibition":                   "cin",
    "total_column_water_vapour":               "tcwv",
    "instantaneous_10m_wind_gust":             "i10fg",
    "maximum_2m_temperature_since_previous_post_processing": "mx2t",
    "minimum_2m_temperature_since_previous_post_processing": "mn2t",
    "total_precipitation":                     "tp",
    "large_scale_precipitation":               "lsp",
    "convective_precipitation":                "cp",
    "snowfall":                                "sf",
    "surface_solar_radiation_downwards":       "ssrd",
    "surface_sensible_heat_flux":              "sshf",
    "surface_latent_heat_flux":                "slhf",
    "vertical_integral_of_divergence_of_moisture_flux": "viwvd",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "peak_wave_period":                        "pp1d",
    "mean_wave_direction":                     "mwd",
    "maximum_individual_wave_height":          "hmax",
}


# ==============================================================================
# 1. LIST VARIABLES
# Improvement over era5cli: callable from Python, filterable by category,
# returns structured data instead of plain CLI text output.
# ==============================================================================
def list_variables(category=None):
    """
    List ERA5 variables available in the Kynera catalogue.

    Improvement over era5cli `info` command: returns a structured dictionary
    usable programmatically, with optional filtering by variable category.

    Parameters
    ----------
    category : str, optional
        Filter by category. One of: 'surface', 'accumulated', 'wave', 'vertical'.
        If None, all variables are returned.

    Returns
    -------
    dict
        Filtered catalogue: {short_name: (long_name, raw_unit, converted_unit)}.

    Examples
    --------
    >>> import kynera
    >>> kynera.list_variables()
    >>> kynera.list_variables(category='accumulated')
    """
    _categories = {
        "surface":     ["t2m", "d2m", "u10", "v10", "msl", "sp", "sst", "tcc",
                        "lcc", "mcc", "hcc", "blh", "cape", "cin", "i10fg",
                        "mx2t", "mn2t", "tcwv"],
        "accumulated": ["tp", "lsp", "cp", "sf", "ssrd", "sshf", "slhf"],
        "vertical":    ["viwvd"],
        "wave":        ["swh", "pp1d", "mwd", "hmax"],
    }

    if category is not None:
        if category not in _categories:
            raise ValueError(
                f"[Kynera] Categoria non valida: '{category}'. "
                f"Scegli tra: {list(_categories.keys())}"
            )
        keys = _categories[category]
    else:
        keys = list(VARIABLE_CATALOGUE.keys())

    result = {k: VARIABLE_CATALOGUE[k][:3] for k in keys if k in VARIABLE_CATALOGUE}

    print(f"\n{'Short':8} {'Long name':48} {'Raw unit':12} {'Converted unit'}")
    print("-" * 90)
    for short, (long, raw, conv) in result.items():
        print(f"{short:8} {long:48} {raw:12} {conv}")
    print()
    return result


# ==============================================================================
# 2. DOWNLOAD
# Improvements over era5cli:
#   - Python API (no CLI required)
#   - Multi-year split: one file per year from a single function call
#   - Automatic ZIP handling (new CDS Beta format)
#   - dry_run mode to preview the request without downloading
#   - Skip existing files (with optional overwrite flag)
# ==============================================================================
def download_era5(
    variables,
    years,
    months,
    days,
    times,
    area,
    output_dir=".",
    output_prefix="era5",
    cds_url="https://cds.climate.copernicus.eu/api",
    cds_key=None,
    product_type="reanalysis",
    dataset="reanalysis-era5-single-levels",
    split_by_year=True,
    dry_run=False,
    overwrite=False,
):
    """
    Download ERA5 data from the Copernicus Climate Data Store.

    Improvements over era5cli:
    - Python API callable from scripts and notebooks (no shell required)
    - Multi-year split: downloads one file per year automatically from a
      single call, avoiding CDS "Request Too Large" errors
    - Handles the new CDS Beta ZIP format natively
    - dry_run mode: prints the CDS request without submitting it
    - Skips existing files unless overwrite=True

    Parameters
    ----------
    variables : list of str
        CDS variable names, e.g. ['2m_temperature', 'total_precipitation'].
        Use list_variables() to explore available variables.
    years : int, str, or list
        Year(s) to download. E.g. 2024 or [2023, 2024].
        If split_by_year=True, one file is produced per year.
    months : str or list of str
        Month(s), e.g. '10' or ['10', '11'].
    days : str, list of str, or 'all'
        Day(s), e.g. ['01', '15'] or 'all' for every day of the month.
    times : list of str
        UTC times, e.g. ['00:00', '06:00', '12:00', '18:00'].
    area : list of float
        Bounding box [North, West, South, East].
    output_dir : str
        Directory where downloaded files are saved. Default: current directory.
    output_prefix : str
        Prefix for output filenames. Files are named: <prefix>_<year>.zip
    cds_url : str
        CDS API endpoint URL.
    cds_key : str, optional
        CDS API key. If None, reads from ~/.cdsapirc.
    product_type : str
        ERA5 product type. Default: 'reanalysis'.
    dataset : str
        CDS dataset ID. Default: 'reanalysis-era5-single-levels'.
    split_by_year : bool
        If True (default), one file per year. If False, one file for all years.
    dry_run : bool
        If True, prints the request without submitting it. Default: False.
    overwrite : bool
        If True, re-downloads even if the output file already exists.

    Returns
    -------
    list of str
        Paths to the downloaded files (empty list if dry_run=True).

    Examples
    --------
    >>> import kynera
    >>> # Preview without downloading
    >>> kynera.download_era5(
    ...     variables=['2m_temperature', 'total_precipitation'],
    ...     years=[2023, 2024], months=['10', '11'],
    ...     days='all', times=['00:00', '06:00', '12:00', '18:00'],
    ...     area=[46.5, 12.0, 39.0, 20.0],
    ...     dry_run=True
    ... )
    >>> # Actual download split by year
    >>> paths = kynera.download_era5(
    ...     variables=['2m_temperature', 'total_precipitation'],
    ...     years=[2023, 2024], months='10',
    ...     days=['25', '26', '27'],
    ...     times=['00:00', '06:00', '12:00', '18:00'],
    ...     area=[46.5, 12.0, 39.0, 20.0],
    ...     output_dir='data/',
    ...     cds_key='your-api-key'
    ... )
    """
    # --- Normalise inputs ---
    if isinstance(years, (int, str)):
        years = [years]
    years = [str(y) for y in years]

    if days == "all":
        days = [f"{d:02d}" for d in range(1, 32)]

    months    = months    if isinstance(months,    list) else [months]
    days      = days      if isinstance(days,      list) else [days]
    variables = variables if isinstance(variables, list) else [variables]

    os.makedirs(output_dir, exist_ok=True)
    downloaded = []

    year_groups = [[y] for y in years] if split_by_year else [years]

    for year_group in year_groups:
        label       = year_group[0] if len(year_group) == 1 else f"{year_group[0]}-{year_group[-1]}"
        output_path = os.path.join(output_dir, f"{output_prefix}_{label}.zip")

        request = {
            "product_type": product_type,
            "variable":     variables,
            "year":         year_group,
            "month":        months,
            "day":          days,
            "time":         times,
            "data_format":  "netcdf",
            "area":         area,
        }

        if dry_run:
            print(f"\n[Kynera] DRY RUN — request for {label}:")
            for k, v in request.items():
                print(f"  {k}: {v}")
            print(f"  → would save to: {output_path}")
            continue

        if os.path.exists(output_path) and not overwrite:
            print(f"[Kynera] File già presente, skip: {output_path}")
            downloaded.append(output_path)
            continue

        print(f"[Kynera] Downloading {label} → {output_path} ...")
        client_kwargs = {"url": cds_url}
        if cds_key:
            client_kwargs["key"] = cds_key
        client = cdsapi.Client(**client_kwargs)
        client.retrieve(dataset, request, output_path)
        print(f"[Kynera] Done: {output_path}")
        downloaded.append(output_path)

    return downloaded


# ==============================================================================
# 3. LOAD
# Improvement over era5cli: era5cli only downloads; Kynera also loads the data
# into a ready-to-use xarray.Dataset, handling both plain .nc and the new
# CDS Beta ZIP format (instant + accum files merged automatically).
# ==============================================================================
def load_era5(path, extract_dir=None):
    """
    Load an ERA5 dataset from a NetCDF file or a CDS Beta ZIP archive.

    era5cli only downloads data; Kynera additionally loads it into an
    xarray.Dataset. Handles the new CDS Beta format where the server returns
    a ZIP containing two separate NetCDF files (instant and accum variables),
    merging them transparently into a single dataset.

    Parameters
    ----------
    path : str
        Path to a .nc file or a .zip archive downloaded from CDS.
    extract_dir : str, optional
        Directory where ZIP contents are extracted.
        Defaults to '<path without .zip>_extracted/'.

    Returns
    -------
    xarray.Dataset
        Dataset with all available variables and dimensions intact.

    Examples
    --------
    >>> import kynera
    >>> ds = kynera.load_era5('era5_2024.zip')
    >>> print(ds)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[Kynera] File non trovato: {path}")

    if not zipfile.is_zipfile(path):
        print(f"[Kynera] Loading NetCDF: {path}")
        return xr.open_dataset(path)

    # --- Handle ZIP (new CDS Beta format) ---
    if extract_dir is None:
        extract_dir = path.replace(".zip", "_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)

    nc_files = sorted(glob.glob(os.path.join(extract_dir, "*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"[Kynera] No .nc files found inside: {path}")

    if len(nc_files) == 1:
        print(f"[Kynera] Loaded: {nc_files[0]}")
        return xr.open_dataset(nc_files[0])

    # Merge instant + accum (shared lat/lon/time grid)
    datasets = [xr.open_dataset(f) for f in nc_files]
    merged   = xr.merge(datasets, compat="override")
    print(f"[Kynera] Merged {len(nc_files)} NetCDF files from: {os.path.basename(path)}")
    return merged


# ==============================================================================
# 4. CONVERT UNITS
# Improvement over era5cli: era5cli delivers raw CDS units (K, Pa, m).
# Kynera converts them automatically to human-readable units, operating
# directly on the xarray.Dataset (not on a flattened DataFrame).
# ==============================================================================
def convert_units(ds):
    """
    Convert ERA5 variables from raw CDS units to human-readable units.

    era5cli delivers data in raw CDS units (K, Pa, m). Kynera applies
    standard meteorological conversions automatically, adding converted
    variables alongside the originals in the xarray.Dataset.

    Conversions applied:
        t2m, d2m, sst, mx2t, mn2t : K   → °C   (new vars: *_c)
        msl, sp                   : Pa  → hPa  (new vars: *_hpa)
        tp, lsp, cp, sf           : m   → mm   (new vars: *_mm)

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset as returned by load_era5().

    Returns
    -------
    xarray.Dataset
        Dataset with additional converted DataArrays.

    Examples
    --------
    >>> import kynera
    >>> ds = kynera.convert_units(kynera.load_era5('era5_2024.zip'))
    >>> print(ds['t2m_c'])   # temperature in °C
    """
    _conversions = {
        "t2m":  ("t2m_c",   lambda x: x - 273.15),
        "d2m":  ("d2m_c",   lambda x: x - 273.15),
        "sst":  ("sst_c",   lambda x: x - 273.15),
        "mx2t": ("mx2t_c",  lambda x: x - 273.15),
        "mn2t": ("mn2t_c",  lambda x: x - 273.15),
        "msl":  ("msl_hpa", lambda x: x / 100),
        "sp":   ("sp_hpa",  lambda x: x / 100),
        "tp":   ("tp_mm",   lambda x: x * 1000),
        "lsp":  ("lsp_mm",  lambda x: x * 1000),
        "cp":   ("cp_mm",   lambda x: x * 1000),
        "sf":   ("sf_mm",   lambda x: x * 1000),
    }

    ds = ds.copy()
    for raw_var, (new_var, fn) in _conversions.items():
        if raw_var in ds:
            ds[new_var] = fn(ds[raw_var])
            meta = VARIABLE_CATALOGUE.get(raw_var)
            if meta:
                ds[new_var].attrs["long_name"] = meta[0]
                ds[new_var].attrs["units"]     = meta[2]
    return ds


# ==============================================================================
# 5. COMPUTE DERIVED VARIABLES
# New in Kynera — not present in era5cli.
# Computes meteorological variables derived from ERA5 base fields,
# operating on the xarray.Dataset to preserve spatial dimensions.
# ==============================================================================
def compute_derived(ds):
    """
    Compute meteorological derived variables from ERA5 base fields.

    This function is not present in era5cli. It adds commonly used derived
    meteorological quantities directly to the xarray.Dataset, preserving
    all spatial and temporal dimensions.

    Derived variables added:
        wind_speed_10m : 10m wind speed [m/s]   — sqrt(u10² + v10²)
        wind_dir_10m   : 10m wind direction [°, meteorological] — arctan2(-u10,-v10)
        rh_2m          : 2m relative humidity [%] — Magnus formula (t2m_c, d2m_c)

    Requires convert_units() to have been applied first (needs t2m_c, d2m_c).

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset, after convert_units() for RH calculation.

    Returns
    -------
    xarray.Dataset
        Input dataset with additional derived variable DataArrays.

    Examples
    --------
    >>> import kynera
    >>> ds = kynera.compute_derived(kynera.convert_units(kynera.load_era5('data.zip')))
    >>> print(ds['wind_speed_10m'])
    """
    ds = ds.copy()

    # --- Wind speed and direction ---
    if "u10" in ds and "v10" in ds:
        ds["wind_speed_10m"] = np.sqrt(ds["u10"] ** 2 + ds["v10"] ** 2)
        ds["wind_speed_10m"].attrs = {"long_name": "10m Wind Speed",                         "units": "m/s"}
        ds["wind_dir_10m"]  = (np.degrees(np.arctan2(-ds["u10"], -ds["v10"])) + 360) % 360
        ds["wind_dir_10m"].attrs  = {"long_name": "10m Wind Direction (meteorological conv.)", "units": "degrees"}

    # --- Relative humidity via Magnus formula ---
    if "t2m_c" in ds and "d2m_c" in ds:
        T  = ds["t2m_c"]
        Td = ds["d2m_c"]
        ds["rh_2m"] = 100 * (
            np.exp(17.625 * Td / (243.04 + Td)) /
            np.exp(17.625 * T  / (243.04 + T))
        )
        ds["rh_2m"].attrs = {"long_name": "2m Relative Humidity (Magnus formula)", "units": "%"}

    return ds


# ==============================================================================
# 6. PLOT FIELD
# New in Kynera — not present in era5cli.
# Produces georeferenced 2D maps of any ERA5 variable using Cartopy.
# ==============================================================================
def plot_field(ds, variable, time_index=0, title=None, cmap="RdYlBu_r", output_path=None):
    """
    Plot a georeferenced 2D map of an ERA5 variable at a given timestep.

    This visualisation capability is not present in era5cli.
    Uses Cartopy for proper geographic projection with coastlines and borders.
    Variable metadata (units, long name) are read directly from the Dataset
    attributes set by convert_units() and compute_derived().

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset as returned by load_era5(), convert_units(), or compute_derived().
    variable : str
        Short name of the variable to plot (e.g. 't2m_c', 'tp_mm', 'wind_speed_10m').
    time_index : int
        Index along the time dimension. Default: 0 (first timestep).
    title : str, optional
        Custom plot title. Auto-generated from variable metadata if None.
    cmap : str
        Matplotlib colormap. Default: 'RdYlBu_r'.
    output_path : str, optional
        If provided, saves the figure to this path (PNG/PDF). Otherwise displays it.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import kynera
    >>> ds  = kynera.compute_derived(kynera.convert_units(kynera.load_era5('data.zip')))
    >>> fig = kynera.plot_field(ds, 't2m_c', time_index=0, output_path='t2m.png')
    >>> fig = kynera.plot_field(ds, 'wind_speed_10m', cmap='viridis')
    """
    if variable not in ds:
        raise KeyError(
            f"[Kynera] Variable '{variable}' not found in dataset. "
            f"Available: {list(ds.data_vars)}"
        )

    # Select timestep along whichever time dimension exists
    time_dims = [d for d in ds[variable].dims if "time" in d.lower()]
    da = ds[variable].isel({time_dims[0]: time_index}) if time_dims else ds[variable]

    fig, ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        add_colorbar=True,
        cbar_kwargs={"shrink": 0.7},
    )

    units = ds[variable].attrs.get("units", "")
    name  = ds[variable].attrs.get("long_name", variable)
    label = f"{name} [{units}]" if units else name
    ax.set_title(title or f"{label} — timestep {time_index}", fontsize=13)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"[Kynera] Figure saved: {output_path}")
    return fig
