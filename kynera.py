"""
era5kit.py
==========
A Python library for downloading, processing, and validating ERA5 reanalysis data
from the Copernicus Climate Data Store (CDS).

Inspired by era5cli (https://github.com/eWaterCycle/era5cli), this library focuses
on post-download processing and validation against in-situ observations.

Compatible with the new CDS Beta format (ZIP containing separate instant/accum NetCDF files).

Authors: [your name]
Course:  Geospatial Processing 2025/2026 — Politecnico di Milano
"""

import os
import glob
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import cdsapi
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ==============================================================================
# UNIT CONVERSION CONSTANTS
# ==============================================================================
KELVIN_OFFSET = 273.15
PA_TO_HPA     = 1 / 100
M_TO_MM       = 1000

# Mapping: ERA5 short name → (long name, raw unit, converted unit, conversion)
VARIABLE_META = {
    't2m':  ('2m Temperature',            'K',   '°C',  lambda x: x - KELVIN_OFFSET),
    'msl':  ('Mean Sea Level Pressure',   'Pa',  'hPa', lambda x: x * PA_TO_HPA),
    'tp':   ('Total Precipitation',       'm',   'mm',  lambda x: x * M_TO_MM),
    'u10':  ('10m U-wind component',      'm/s', 'm/s', lambda x: x),
    'v10':  ('10m V-wind component',      'm/s', 'm/s', lambda x: x),
    'd2m':  ('2m Dewpoint Temperature',   'K',   '°C',  lambda x: x - KELVIN_OFFSET),
}


# ==============================================================================
# 1. DOWNLOAD
# ==============================================================================
def download_era5(
    variables,
    year,
    months,
    days,
    times,
    area,
    output_path,
    cds_url="https://cds.climate.copernicus.eu/api",
    cds_key=None,
    product_type="reanalysis",
    dataset="reanalysis-era5-single-levels",
):
    """
    Download ERA5 data from the Copernicus Climate Data Store.

    Handles the new CDS Beta format, which returns a ZIP archive containing
    separate NetCDF files for instant and accumulated variables.

    Parameters
    ----------
    variables : list of str
        ERA5 variable names, e.g. ['2m_temperature', 'total_precipitation'].
    year : str or int
        Year to download, e.g. '2025'.
    months : str or list of str
        Month(s) to download, e.g. '10' or ['10', '11'].
    days : str or list of str
        Day(s) to download, e.g. ['25', '26', '27'].
    times : list of str
        UTC times to download, e.g. ['00:00', '06:00', '12:00', '18:00'].
    area : list of float
        Bounding box [North, West, South, East].
    output_path : str
        Path where the downloaded file (ZIP or NetCDF) will be saved.
    cds_url : str
        CDS API endpoint URL.
    cds_key : str
        CDS API key. If None, reads from ~/.cdsapirc.
    product_type : str
        ERA5 product type. Default: 'reanalysis'.
    dataset : str
        CDS dataset ID. Default: 'reanalysis-era5-single-levels'.

    Returns
    -------
    str
        Path to the downloaded file.

    Examples
    --------
    >>> path = download_era5(
    ...     variables=['2m_temperature', 'total_precipitation'],
    ...     year='2025', months='10', days=['25', '26'],
    ...     times=['00:00', '12:00'],
    ...     area=[46.5, 12.0, 39.0, 20.0],
    ...     output_path='era5_data.zip',
    ...     cds_key='your-api-key'
    ... )
    """
    if os.path.exists(output_path):
        print(f"File {output_path} già presente. Salto il download.")
        return output_path

    client_kwargs = {"url": cds_url}
    if cds_key:
        client_kwargs["key"] = cds_key
    client = cdsapi.Client(**client_kwargs)

    request = {
        "product_type": product_type,
        "variable":     variables if isinstance(variables, list) else [variables],
        "year":         str(year),
        "month":        months if isinstance(months, list) else [months],
        "day":          days if isinstance(days, list) else [days],
        "time":         times,
        "data_format":  "netcdf",
        "area":         area,
    }

    print("Scaricamento ERA5 in corso (potrebbe richiedere qualche minuto)...")
    client.retrieve(dataset, request, output_path)
    print(f"Download completato: {output_path}")
    return output_path


# ==============================================================================
# 2. LOAD
# ==============================================================================
def load_era5(path, extract_dir=None):
    """
    Load an ERA5 dataset from a NetCDF file or a ZIP archive.

    The new Copernicus CDS Beta returns a ZIP containing two separate NetCDF files:
    - *instant.nc  → instantaneous variables (t2m, msl, u10, v10, d2m, ...)
    - *accum.nc    → accumulated variables (tp, ...)
    This function detects the format automatically and merges the files if needed.

    Parameters
    ----------
    path : str
        Path to a .nc file or a .zip archive from the CDS.
    extract_dir : str, optional
        Directory where ZIP contents are extracted.
        Defaults to '<path>_extracted/' next to the ZIP.

    Returns
    -------
    xarray.Dataset
        Merged dataset with all available variables.

    Examples
    --------
    >>> ds = load_era5('era5_adriatico_sample.zip')
    >>> print(ds)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato: {path}")

    if not zipfile.is_zipfile(path):
        # Already a plain NetCDF
        return xr.open_dataset(path)

    # --- Handle ZIP (new CDS Beta format) ---
    if extract_dir is None:
        extract_dir = path.replace(".zip", "_extracted")
    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(extract_dir)

    nc_files = sorted(glob.glob(os.path.join(extract_dir, "*.nc")))
    if not nc_files:
        raise FileNotFoundError(f"Nessun file .nc trovato dentro {path}")

    if len(nc_files) == 1:
        return xr.open_dataset(nc_files[0])

    # Merge instant + accum datasets (they share lat/lon/time dimensions)
    datasets = [xr.open_dataset(f) for f in nc_files]
    merged = xr.merge(datasets, compat="override")
    print(f"Caricati e uniti {len(nc_files)} file NetCDF da {path}")
    return merged


# ==============================================================================
# 3. EXTRACT AT STATIONS
# ==============================================================================
def extract_at_stations(ds, stations, method="nearest"):
    """
    Extract ERA5 values at the locations of a list of stations.

    Uses nearest-neighbour interpolation on the ERA5 grid.

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset as returned by load_era5().
    stations : pandas.DataFrame
        DataFrame with at least columns: 'station_id', 'station_name', 'lat', 'lon'.
    method : str
        Interpolation method for xr.Dataset.sel(). Default: 'nearest'.

    Returns
    -------
    pandas.DataFrame
        Long-format DataFrame with one row per (station, timestep) and
        one column per ERA5 variable (raw units).

    Examples
    --------
    >>> stations = pd.read_csv('stations.csv')
    >>> df = extract_at_stations(ds, stations)
    """
    required_cols = {"lat", "lon"}
    if not required_cols.issubset(stations.columns):
        raise ValueError(f"stations deve contenere le colonne: {required_cols}")

    results = []
    for _, row in stations.iterrows():
        point = ds.sel(latitude=row["lat"], longitude=row["lon"], method=method)
        df_point = point.to_dataframe().reset_index()
        df_point["station_id"]   = row.get("station_id",   "unknown")
        df_point["station_name"] = row.get("station_name", "unknown")
        df_point["lat_station"]  = row["lat"]
        df_point["lon_station"]  = row["lon"]
        results.append(df_point)

    if not results:
        raise ValueError("Nessuna stazione processata. Controlla il DataFrame in input.")

    return pd.concat(results, ignore_index=True)


# ==============================================================================
# 4. CONVERT UNITS
# ==============================================================================
def convert_units(df):
    """
    Convert ERA5 variables from raw units to human-readable units.

    Conversions applied (based on VARIABLE_META):
        t2m  : K   → °C   (subtract 273.15)
        d2m  : K   → °C   (subtract 273.15)
        msl  : Pa  → hPa  (divide by 100)
        tp   : m   → mm   (multiply by 1000)
        u10  : m/s → m/s  (no change)
        v10  : m/s → m/s  (no change)

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with ERA5 raw variable columns (e.g. 't2m', 'msl', 'tp').

    Returns
    -------
    pandas.DataFrame
        DataFrame with added converted columns (e.g. 't2m_celsius', 'mslp_hpa').

    Examples
    --------
    >>> df_converted = convert_units(df_raw)
    >>> df_converted[['t2m_celsius', 'mslp_hpa', 'tp_mm']].head()
    """
    df = df.copy()
    rename_map = {
        "t2m": "t2m_celsius",
        "d2m": "d2m_celsius",
        "msl": "mslp_hpa",
        "tp":  "tp_mm",
    }
    for raw_col, new_col in rename_map.items():
        if raw_col in df.columns:
            df[new_col] = VARIABLE_META[raw_col][3](df[raw_col])
    return df


# ==============================================================================
# 5. COMPUTE DERIVED VARIABLES  [funzione originale]
# ==============================================================================
def compute_derived(df):
    """
    Compute meteorological derived variables from ERA5 base variables.

    Derived variables:
        wind_speed_10m : sqrt(u10² + v10²)  [m/s]
        wind_dir_10m   : arctan2(-u10, -v10) [degrees, meteorological convention]
        rh_2m          : Relative Humidity at 2m from t2m and d2m  [%]
                         Magnus formula: RH = 100 * exp(17.625*Td/(243.04+Td))
                                                  / exp(17.625*T/(243.04+T))

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with ERA5 converted columns (output of convert_units()).
        Requires: u10, v10 for wind; t2m_celsius + d2m_celsius for RH.

    Returns
    -------
    pandas.DataFrame
        Input DataFrame with additional derived columns.

    Examples
    --------
    >>> df_derived = compute_derived(convert_units(df_raw))
    >>> df_derived[['wind_speed_10m', 'wind_dir_10m', 'rh_2m']].describe()
    """
    df = df.copy()

    # Wind speed and direction
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed_10m"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        df["wind_dir_10m"]   = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) + 360) % 360

    # Relative humidity via Magnus formula (requires converted °C columns)
    if "t2m_celsius" in df.columns and "d2m_celsius" in df.columns:
        T  = df["t2m_celsius"]
        Td = df["d2m_celsius"]
        df["rh_2m"] = 100 * (
            np.exp(17.625 * Td / (243.04 + Td)) /
            np.exp(17.625 * T  / (243.04 + T))
        )

    return df


# ==============================================================================
# 6. VALIDATE AGAINST OBSERVATIONS  [funzione originale — ground truth analysis]
# ==============================================================================
def validate_vs_observations(df_era5, df_obs, variable_era5, variable_obs, station_col="station_id", time_col="valid_time"):
    """
    Validate ERA5 reanalysis against in-situ observations.

    Merges ERA5 extracted values with observational data on (station, time),
    then computes per-station error statistics useful for ground-truth assessment
    in weather forecasting tasks.

    Parameters
    ----------
    df_era5 : pandas.DataFrame
        ERA5 data as returned by extract_at_stations() + convert_units().
    df_obs : pandas.DataFrame
        Observational data with columns: station_id, valid_time, <variable_obs>.
    variable_era5 : str
        Column name in df_era5 to validate (e.g. 't2m_celsius').
    variable_obs : str
        Column name in df_obs to validate against (e.g. 'temp_obs').
    station_col : str
        Column used to match stations. Default: 'station_id'.
    time_col : str
        Column used to match timestamps. Default: 'valid_time'.

    Returns
    -------
    pandas.DataFrame
        Per-station error metrics:
        - bias  : mean(ERA5 - obs)
        - mae   : mean absolute error
        - rmse  : root mean squared error
        - r     : Pearson correlation coefficient
        - n     : number of matched observations

    Examples
    --------
    >>> stats = validate_vs_observations(
    ...     df_era5, df_obs,
    ...     variable_era5='t2m_celsius',
    ...     variable_obs='temp_obs'
    ... )
    >>> print(stats.sort_values('rmse'))
    """
    # Align time columns to datetime
    df_era5 = df_era5.copy()
    df_obs  = df_obs.copy()
    df_era5[time_col] = pd.to_datetime(df_era5[time_col])
    df_obs[time_col]  = pd.to_datetime(df_obs[time_col])

    merged = pd.merge(
        df_era5[[station_col, time_col, variable_era5]],
        df_obs[[station_col, time_col, variable_obs]],
        on=[station_col, time_col],
        how="inner",
    )

    if merged.empty:
        raise ValueError("Nessun dato in comune tra ERA5 e osservazioni. Controlla station_id e valid_time.")

    stats = []
    for station, grp in merged.groupby(station_col):
        era5_vals = grp[variable_era5].values
        obs_vals  = grp[variable_obs].values
        diff      = era5_vals - obs_vals
        r = np.corrcoef(era5_vals, obs_vals)[0, 1] if len(grp) > 1 else np.nan
        stats.append({
            station_col: station,
            "bias":      float(np.mean(diff)),
            "mae":       float(np.mean(np.abs(diff))),
            "rmse":      float(np.sqrt(np.mean(diff ** 2))),
            "r":         float(r),
            "n":         len(grp),
        })

    return pd.DataFrame(stats).sort_values("rmse").reset_index(drop=True)


# ==============================================================================
# 7. PLOT FIELD  (bonus — visualizzazione)
# ==============================================================================
def plot_field(ds, variable, time_index=0, title=None, cmap="RdYlBu_r", output_path=None):
    """
    Plot a 2D map of an ERA5 variable at a given timestep.

    Parameters
    ----------
    ds : xarray.Dataset
        ERA5 dataset as returned by load_era5().
    variable : str
        Variable to plot (e.g. 't2m', 'tp').
    time_index : int
        Index along the time dimension to plot. Default: 0.
    title : str, optional
        Plot title. Auto-generated if None.
    cmap : str
        Matplotlib colormap. Default: 'RdYlBu_r'.
    output_path : str, optional
        If provided, saves the figure to this path instead of displaying it.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = plot_field(ds, 't2m', time_index=0, output_path='t2m_map.png')
    """
    if variable not in ds:
        raise KeyError(f"Variabile '{variable}' non trovata nel dataset. Disponibili: {list(ds.data_vars)}")

    da = ds[variable].isel(valid_time=time_index) if "valid_time" in ds[variable].dims else ds[variable]

    fig, ax = plt.subplots(
        figsize=(10, 7),
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS,   linewidth=0.5, linestyle="--")
    ax.add_feature(cfeature.LAND,      facecolor="lightgray", alpha=0.3)
    ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.5)

    im = da.plot(
        ax=ax, transform=ccrs.PlateCarree(),
        cmap=cmap, add_colorbar=True,
        cbar_kwargs={"shrink": 0.7}
    )

    meta  = VARIABLE_META.get(variable, (variable, "", "", None))
    label = f"{meta[0]} [{meta[1]}]" if meta[1] else variable
    ax.set_title(title or f"{label} — t={time_index}", fontsize=13)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Mappa salvata: {output_path}")
    return fig
