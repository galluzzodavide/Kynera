# Kynera

**Kynera** is a Python library for downloading, processing, and validating ERA5 reanalysis data from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/).

Kynera is designed for geospatial analysis workflows that require ERA5 as a ground-truth reference — for instance, validating weather forecasting models against reanalysis data. It provides a clean, function-based API built on top of `xarray`, `pandas`, and `cdsapi`, with native support for the **new CDS Beta format** (ZIP archive containing separate NetCDF files for instant and accumulated variables).

> Inspired by [era5cli](https://github.com/eWaterCycle/era5cli). Kynera focuses on post-download processing and in-situ validation rather than CLI-based retrieval.

Developed as a project for the course **Geospatial Processing 2025/2026** — Politecnico di Milano.

---

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Functions](#functions)
- [Testing](#testing)
- [Data](#data)
- [Contributing](#contributing)
- [License](#license)

---

## Features

- **Download** ERA5 data from CDS with a single function call — handles both legacy `.nc` and the new ZIP format automatically
- **Load** ERA5 datasets from `.nc` files or `.zip` archives, merging instant and accumulated variables into a single `xarray.Dataset`
- **Extract** ERA5 values at in-situ station coordinates via nearest-neighbour lookup
- **Convert** raw CDS units to human-readable values (K → °C, Pa → hPa, m → mm)
- **Compute** derived meteorological variables: wind speed, wind direction, relative humidity
- **Validate** ERA5 against in-situ observations with per-station error statistics (bias, MAE, RMSE, correlation)
- **Plot** 2D spatial maps of any ERA5 variable with Cartopy

---

## Repository Structure

```
Kynera/
├── kynera.py              ← main library (all functions)
├── __init__.py            ← package init and public API
├── environment.yml        ← conda environment
├── LICENSE
├── README.md
│
├── EXAMPLES/
│   ├── example.ipynb      ← end-to-end demonstration notebook
│   └── DATA/              ← place your ERA5 .zip file here
│       └── (download instructions below)
│
└── TESTS/
    ├── test_kynera.py     ← unit tests
    └── DATA_TEST/
        └── era5_adriatico_sample.zip  ← test data (download link below)
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Kynera.git
cd Kynera
```

### 2. Create and activate the conda environment

```bash
conda env create --file environment.yml
conda activate kynera
```

### 3. Verify the installation

```bash
python -c "import kynera; print(kynera.__version__)"
```

---

## Quick Start

```python
import kynera

# 1. Load an ERA5 dataset (ZIP or NetCDF)
ds = kynera.load_era5("era5_data.zip")

# 2. Define stations and extract ERA5 values
import pandas as pd
stations = pd.DataFrame({
    "station_id":   ["S001",     "S002"],
    "station_name": ["Trieste",  "Ancona"],
    "lat":          [45.65,      43.60],
    "lon":          [13.76,      13.50],
})
df = kynera.extract_at_stations(ds, stations)

# 3. Convert raw units
df = kynera.convert_units(df)

# 4. Compute derived variables (wind speed, RH, ...)
df = kynera.compute_derived(df)

# 5. Validate against observations
stats = kynera.validate_vs_observations(
    df_era5       = df,
    df_obs        = my_observations_df,
    variable_era5 = "t2m_celsius",
    variable_obs  = "temp_obs",
)
print(stats)

# 6. Plot a spatial map
fig = kynera.plot_field(ds, variable="t2m", time_index=0)
```

See `EXAMPLES/example.ipynb` for the full workflow including download, visualizations and error charts.

---

## Functions

### `download_era5()`
Download ERA5 reanalysis data from the Copernicus CDS. Skips the download if the output file already exists. Handles the new CDS Beta format which returns a ZIP archive.

```python
kynera.download_era5(
    variables = ['2m_temperature', 'total_precipitation'],
    year      = '2025',
    months    = '10',
    days      = ['25', '26', '27'],
    times     = ['00:00', '06:00', '12:00', '18:00'],
    area      = [46.5, 12.0, 39.0, 20.0],  # [N, W, S, E]
    output_path = 'era5_data.zip',
    cds_key   = 'your-api-key',
)
```

Requires a free CDS account — register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu).

---

### `load_era5()`
Load an ERA5 dataset from a `.nc` file or a `.zip` archive. Automatically detects the format and merges the `instant` and `accum` NetCDF files produced by the new CDS Beta into a single `xarray.Dataset`.

```python
ds = kynera.load_era5("era5_data.zip")
```

---

### `extract_at_stations()`
Extract ERA5 values at the coordinates of a list of in-situ stations using nearest-neighbour lookup on the ERA5 grid. Returns a long-format `pandas.DataFrame` with one row per (station × timestep).

```python
df = kynera.extract_at_stations(ds, stations)
# stations: DataFrame with columns 'lat', 'lon' (+ optional 'station_id', 'station_name')
```

---

### `convert_units()`
Convert ERA5 variables from raw CDS units to human-readable values. Adds new columns alongside the original raw ones.

| Variable | Raw unit | Converted column | Converted unit |
|----------|----------|-----------------|----------------|
| `t2m`    | K        | `t2m_celsius`   | °C             |
| `d2m`    | K        | `d2m_celsius`   | °C             |
| `msl`    | Pa       | `mslp_hpa`      | hPa            |
| `tp`     | m        | `tp_mm`         | mm             |

```python
df = kynera.convert_units(df)
```

---

### `compute_derived()`
Compute additional meteorological variables from ERA5 base fields.

| Derived variable  | Formula | Unit |
|-------------------|---------|------|
| `wind_speed_10m`  | `sqrt(u10² + v10²)` | m/s |
| `wind_dir_10m`    | `arctan2(-u10, -v10)` meteorological convention | degrees |
| `rh_2m`           | Magnus formula from `t2m` and `d2m` | % |

```python
df = kynera.compute_derived(df)
```

Requires `convert_units()` to have been applied first (needs `t2m_celsius`, `d2m_celsius`).

---

### `validate_vs_observations()`
Validate ERA5 reanalysis against in-situ observational data. Matches records on `(station_id, valid_time)` and computes per-station error metrics.

```python
stats = kynera.validate_vs_observations(
    df_era5       = df,
    df_obs        = df_observations,
    variable_era5 = "t2m_celsius",
    variable_obs  = "temp_obs",
)
```

**Output columns:**

| Metric | Description |
|--------|-------------|
| `bias` | mean(ERA5 − obs) — systematic offset |
| `mae`  | mean absolute error |
| `rmse` | root mean squared error |
| `r`    | Pearson correlation coefficient |
| `n`    | number of matched (station, time) pairs |

---

### `plot_field()`
Plot a 2D georeferenced map of any ERA5 variable at a given timestep using Cartopy and Matplotlib.

```python
fig = kynera.plot_field(
    ds          = ds,
    variable    = "t2m",
    time_index  = 0,
    cmap        = "RdYlBu_r",
    output_path = "t2m_map.png",   # optional — saves to file
)
```

---

## Testing

Test data is required. Download the sample file and place it in `TESTS/DATA_TEST/`:

📥 **[Download era5_adriatico_sample.zip](#)** ← replace with your Google Drive link

Then run the full test suite from the repository root:

```bash
# Run all tests
python -m pytest TESTS/test_kynera.py -v

# Run with coverage report
python -m pytest TESTS/test_kynera.py -v --cov=kynera --cov-report=term-missing
```

Tests are organised in 5 classes, one per function:

| Class | Tests |
|-------|-------|
| `TestLoadEra5` | ZIP detection, variable presence, time dimensions |
| `TestExtractAtStations` | Column names, coordinates, NaN checks |
| `TestConvertUnits` | Physical plausibility, exact conversion values |
| `TestComputeDerived` | Wind speed (3-4-5 triangle), RH=100% at saturation |
| `TestValidateVsObservations` | Bias correctness, RMSE sorting, empty overlap handling |

---

## Data

Kynera works with **ERA5 Single Levels** data from the Copernicus CDS.

To download data you need a free CDS account:
1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Retrieve your API key from your profile page
3. Pass it to `download_era5()` via the `cds_key` parameter, or save it in `~/.cdsapirc`:

```ini
url: https://cds.climate.copernicus.eu/api
key: your-api-key-here
```

The sample file used for tests (`era5_adriatico_sample.zip`) covers the Adriatic Sea area `[46.5°N, 12.0°E, 39.0°N, 20.0°E]` for 3 days in October 2025, with variables `t2m`, `msl`, and `tp`.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feat/your-feature`
3. Commit your changes: `git commit -m "feat: describe your change"`
4. Push and open a Pull Request

Please make sure all tests pass before submitting:
```bash
python -m pytest TESTS/test_kynera.py -v
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

- ERA5 data provided by the [Copernicus Climate Change Service](https://climate.copernicus.eu/)
- Download interface inspired by [era5cli](https://github.com/eWaterCycle/era5cli) (eWaterCycle, Netherlands eScience Center)
- Course: Geospatial Processing 2025/2026 — Politecnico di Milano
