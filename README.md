# Kynera

**Kynera** is a Python library for downloading and processing ERA5 reanalysis data from the [Copernicus Climate Data Store (CDS)](https://cds.climate.copernicus.eu/).

Kynera builds on the ideas of [era5cli](https://github.com/eWaterCycle/era5cli) and extends them with a pure Python API, native support for the new CDS Beta format, automatic unit conversion, derived meteorological variables, and georeferenced map visualisation — features that are not available in era5cli.

> Developed as a project for the course **Geospatial Processing 2025/2026** — Politecnico di Milano.

---

## Improvements over era5cli

| Feature | era5cli | Kynera |
|---|---|---|
| Interface | Command-line only | Pure Python API |
| New CDS Beta ZIP format | Partial support (open issue) | Full native support |
| Multi-year split download | `--splitmonths` flag | `split_by_year=True` parameter |
| Dry-run mode | `--dryrun` flag | `dry_run=True` parameter |
| Load data into memory | ✗ | `load_era5()` → `xarray.Dataset` |
| Unit conversion | ✗ | `convert_units()` K→°C, Pa→hPa, m→mm |
| Derived variables | ✗ | `compute_derived()` wind speed, RH |
| Variable catalogue | CLI text output only | Python dict, filterable by category |
| Map visualisation | ✗ | `plot_field()` with Cartopy |

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
│
└── TESTS/
    ├── test_kynera.py     ← unit tests
    └── DATA_TEST/
        └── era5_adriatico_sample.zip  ← test data (see Data section)
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

### 3. Verify

```bash
python -c "import kynera; print(kynera.__version__)"
```

---

## Quick Start

```python
import kynera

# 1. Browse available variables
kynera.list_variables()
kynera.list_variables(category='accumulated')

# 2. Preview a download request without submitting it
kynera.download_era5(
    variables=['2m_temperature', 'total_precipitation'],
    years=[2023, 2024],
    months=['10', '11'],
    days='all',
    times=['00:00', '06:00', '12:00', '18:00'],
    area=[46.5, 12.0, 39.0, 20.0],   # [N, W, S, E] — Adriatic Sea
    dry_run=True,
)

# 3. Download — one file per year, automatically
paths = kynera.download_era5(
    variables=['2m_temperature', 'total_precipitation',
               '10m_u_component_of_wind', '10m_v_component_of_wind',
               '2m_dewpoint_temperature'],
    years=[2023, 2024],
    months='10',
    days=['25', '26', '27'],
    times=['00:00', '06:00', '12:00', '18:00'],
    area=[46.5, 12.0, 39.0, 20.0],
    output_dir='data/',
    cds_key='your-api-key',
)

# 4. Load (handles ZIP and plain .nc automatically)
ds = kynera.load_era5('data/era5_2024.zip')

# 5. Convert raw CDS units to human-readable values
ds = kynera.convert_units(ds)
# t2m → t2m_c [°C],  msl → msl_hpa [hPa],  tp → tp_mm [mm]

# 6. Add derived meteorological variables
ds = kynera.compute_derived(ds)
# adds: wind_speed_10m [m/s], wind_dir_10m [°], rh_2m [%]

# 7. Plot a georeferenced map
fig = kynera.plot_field(ds, 't2m_c', time_index=0, output_path='t2m.png')
fig = kynera.plot_field(ds, 'wind_speed_10m', cmap='viridis')
```

See `EXAMPLES/example.ipynb` for the full workflow with visualisations.

---

## Functions

### `list_variables(category=None)`

Lists all ERA5 variables in the Kynera catalogue with short names, long names, raw CDS units, and converted units. Optionally filter by category: `'surface'`, `'accumulated'`, `'wave'`, or `'vertical'`.

```python
kynera.list_variables()                   # all 28 variables
kynera.list_variables(category='wave')    # wave parameters only
```

Returns a Python dictionary `{short_name: (long_name, raw_unit, converted_unit)}`, unlike the plain text output of `era5cli info`.

---

### `download_era5(...)`

Downloads ERA5 reanalysis data from CDS. Key improvements over era5cli:

- **Multi-year split**: pass `years=[2020, 2021, 2022]` and get one file per year automatically, avoiding CDS "Request Too Large" errors — no need to run multiple CLI commands.
- **Dry-run**: `dry_run=True` prints the exact CDS request without submitting it.
- **Skip existing**: files already on disk are skipped unless `overwrite=True`.
- **`days='all'`**: shortcut for all days in the selected months.

```python
paths = kynera.download_era5(
    variables=['2m_temperature', 'total_precipitation'],
    years=[2022, 2023, 2024],   # split into 3 files automatically
    months=['06', '07', '08'],
    days='all',
    times=['00:00', '06:00', '12:00', '18:00'],
    area=[46.5, 12.0, 39.0, 20.0],
    output_dir='data/',
    split_by_year=True,         # default
    dry_run=False,
    overwrite=False,
)
# → ['data/era5_2022.zip', 'data/era5_2023.zip', 'data/era5_2024.zip']
```

Requires a free CDS account at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu). Store credentials in `~/.cdsapirc` or pass them via `cds_key`.

---

### `load_era5(path, extract_dir=None)`

Loads an ERA5 dataset into an `xarray.Dataset`. Handles both plain `.nc` files and the new CDS Beta ZIP format, where the server returns a ZIP archive containing two separate NetCDF files — one for instantaneous variables (e.g. `t2m`, `msl`) and one for accumulated variables (e.g. `tp`). These are detected and merged automatically.

```python
ds = kynera.load_era5('data/era5_2024.zip')   # ZIP or .nc, both work
print(ds)
```

---

### `convert_units(ds)`

Converts raw CDS units to standard meteorological units, adding new variables alongside the originals in the `xarray.Dataset`.

| Raw variable | Raw unit | Converted variable | Converted unit |
|---|---|---|---|
| `t2m`, `d2m`, `sst`, `mx2t`, `mn2t` | K | `*_c` | °C |
| `msl`, `sp` | Pa | `*_hpa` | hPa |
| `tp`, `lsp`, `cp`, `sf` | m | `*_mm` | mm |

```python
ds = kynera.convert_units(ds)
print(ds['t2m_c'])    # 2m temperature in °C
print(ds['tp_mm'])    # total precipitation in mm
print(ds['msl_hpa'])  # sea level pressure in hPa
```

---

### `compute_derived(ds)`

Computes derived meteorological variables from ERA5 base fields, adding them to the dataset with proper units and `long_name` attributes. All spatial and temporal dimensions are preserved. Requires `convert_units()` to have been applied first (needs `t2m_c` and `d2m_c` for relative humidity).

| Derived variable | Formula | Unit |
|---|---|---|
| `wind_speed_10m` | `sqrt(u10² + v10²)` | m/s |
| `wind_dir_10m` | `arctan2(-u10, -v10)` meteorological convention | degrees |
| `rh_2m` | Magnus formula using `t2m_c` and `d2m_c` | % |

```python
ds = kynera.compute_derived(ds)
print(ds['wind_speed_10m'])
print(ds['wind_dir_10m'])
print(ds['rh_2m'])
```

---

### `plot_field(ds, variable, time_index=0, ...)`

Plots a georeferenced 2D map of any ERA5 variable at a selected timestep, using Cartopy with coastlines, country borders, and gridlines. Variable name and units are read automatically from the Dataset attributes set by `convert_units()` and `compute_derived()`.

```python
fig = kynera.plot_field(ds, 't2m_c',         time_index=0, cmap='RdYlBu_r')
fig = kynera.plot_field(ds, 'tp_mm',          time_index=2, cmap='Blues')
fig = kynera.plot_field(ds, 'wind_speed_10m', cmap='viridis', output_path='wind.png')
fig = kynera.plot_field(ds, 'rh_2m',          title='Relative Humidity — Oct 2024')
```

---

## Variable Catalogue

Kynera includes a built-in catalogue of 28 ERA5 variables organised in four categories.

**Surface — instantaneous** (18 variables):
`t2m`, `d2m`, `u10`, `v10`, `msl`, `sp`, `sst`, `tcc`, `lcc`, `mcc`, `hcc`, `blh`, `cape`, `cin`, `tcwv`, `i10fg`, `mx2t`, `mn2t`

**Surface — accumulated** (7 variables):
`tp`, `lsp`, `cp`, `sf`, `ssrd`, `sshf`, `slhf`

**Vertical integrals** (1 variable):
`viwvd`

**Wave** (4 variables):
`swh`, `pp1d`, `mwd`, `hmax`

Use `kynera.list_variables()` to see the full catalogue with units at runtime.

---

## Testing

Download the sample file and place it in `TESTS/DATA_TEST/`:

📥 **[Download era5_adriatico_sample.zip](#)** ← replace with your Google Drive link

The sample covers the Adriatic Sea area `[46.5°N, 12.0°E, 39.0°N, 20.0°E]` for 3 days in October 2025 with variables `t2m`, `msl`, and `tp`.

Run the full test suite from the repository root:

```bash
# Run all tests
python -m pytest TESTS/test_kynera.py -v

# With coverage report
python -m pytest TESTS/test_kynera.py -v --cov=kynera --cov-report=term-missing
```

---

## Data

Kynera works with **ERA5 Single Levels** data from the Copernicus CDS.

To access data you need a free account:

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Retrieve your API key from your profile page
3. Store credentials in `~/.cdsapirc`:

```ini
url: https://cds.climate.copernicus.eu/api
key: your-api-key-here
```

Or pass the key directly via the `cds_key` parameter in `download_era5()`.

---

## Contributing

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

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

- ERA5 data provided by the [Copernicus Climate Change Service](https://climate.copernicus.eu/), ECMWF
- Inspired by [era5cli](https://github.com/eWaterCycle/era5cli) — eWaterCycle, Netherlands eScience Center
- Hersbach et al. (2020): *The ERA5 global reanalysis*. Q J R Meteorol Soc. [doi:10.1002/qj.3803](https://doi.org/10.1002/qj.3803)
- Course: Geospatial Processing 2025/2026 — Politecnico di Milano
