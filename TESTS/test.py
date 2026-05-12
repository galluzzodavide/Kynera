"""
test.py
=======
Unit tests for the Kynera ERA5 processing library.

Run with:
    pytest test.py -v

Or for coverage:
    python coverage_report.py
"""

import os
import zipfile
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import xarray as xr
import pandas as pd

import kynera


# ==============================================================================
# Helpers
# ==============================================================================

def _make_ds(include_wind=True, include_temp=True, include_precip=True):
    """Build a minimal synthetic xarray.Dataset mimicking ERA5 output."""
    times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lats  = np.array([45.0, 46.0])
    lons  = np.array([9.0, 10.0])

    coords = {"valid_time": times, "latitude": lats, "longitude": lons}
    shape  = (len(times), len(lats), len(lons))
    dims   = ("valid_time", "latitude", "longitude")

    data_vars = {}
    if include_wind:
        data_vars["u10"]   = (dims, np.random.uniform(-10, 10, shape).astype("float32"))
        data_vars["v10"]   = (dims, np.random.uniform(-10, 10, shape).astype("float32"))
        data_vars["i10fg"] = (dims, np.random.uniform(0, 20, shape).astype("float32"))
    if include_temp:
        data_vars["t2m"] = (dims, np.random.uniform(270, 300, shape).astype("float32"))
        data_vars["d2m"] = (dims, np.random.uniform(260, 295, shape).astype("float32"))
        data_vars["msl"] = (dims, np.random.uniform(100000, 103500, shape).astype("float32"))
    if include_precip:
        data_vars["tp"]  = (dims, np.random.uniform(0, 0.01, shape).astype("float32"))

    return xr.Dataset(data_vars, coords=coords)


def _make_zip_with_nc(tmpdir, ds, filename="era5_2024.zip"):
    """Write ds to a NetCDF inside a ZIP, return the ZIP path."""
    nc_path  = os.path.join(tmpdir, "data.nc")
    zip_path = os.path.join(tmpdir, filename)
    ds.to_netcdf(nc_path)
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(nc_path, arcname="data.nc")
    return zip_path


# ==============================================================================
# 1. list_variables
# ==============================================================================
class TestListVariables(unittest.TestCase):

    def test_returns_full_catalogue_when_no_category(self):
        result = kynera.list_variables()
        self.assertIsInstance(result, dict)
        self.assertIn("t2m", result)
        self.assertIn("tp",  result)
        self.assertIn("swh", result)

    def test_filter_surface(self):
        result = kynera.list_variables(category="surface")
        self.assertIn("t2m",  result)
        self.assertNotIn("tp", result)   # accumulated, not surface

    def test_filter_accumulated(self):
        result = kynera.list_variables(category="accumulated")
        self.assertIn("tp",   result)
        self.assertNotIn("t2m", result)

    def test_filter_wave(self):
        result = kynera.list_variables(category="wave")
        self.assertIn("swh", result)
        self.assertNotIn("t2m", result)

    def test_filter_vertical(self):
        result = kynera.list_variables(category="vertical")
        self.assertIn("viwvd", result)

    def test_invalid_category_raises(self):
        with self.assertRaises(ValueError):
            kynera.list_variables(category="nonexistent")

    def test_each_entry_has_three_fields(self):
        result = kynera.list_variables()
        for short, info in result.items():
            self.assertEqual(len(info), 3, f"Entry for '{short}' should have 3 fields")


# ==============================================================================
# 2. download_era5 (dry_run only — no real HTTP calls)
# ==============================================================================
class TestDownloadEra5DryRun(unittest.TestCase):

    def test_dry_run_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kynera.download_era5(
                variables=["2m_temperature"],
                years=2024,
                months="01",
                days="all",
                times=["00:00"],
                area=[47, 6, 44, 14],
                output_dir=tmpdir,
                dry_run=True,
            )
        self.assertEqual(result, [])

    def test_dry_run_multi_year_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kynera.download_era5(
                variables=["2m_temperature", "total_precipitation"],
                years=[2022, 2023, 2024],
                months=["01", "02"],
                days="all",
                times=["00:00", "12:00"],
                area=[47, 6, 44, 14],
                output_dir=tmpdir,
                dry_run=True,
            )
        self.assertEqual(result, [])

    def test_skip_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create the output file
            existing = os.path.join(tmpdir, "era5_2024.zip")
            open(existing, "w").close()

            result = kynera.download_era5(
                variables=["2m_temperature"],
                years=2024,
                months="01",
                days=["01"],
                times=["00:00"],
                area=[47, 6, 44, 14],
                output_dir=tmpdir,
                overwrite=False,
            )
        self.assertIn(existing, result)

    def test_year_normalisation_int_and_str(self):
        """Single int year and list of strings should both work in dry_run."""
        with tempfile.TemporaryDirectory() as tmpdir:
            for y in [2023, "2023", [2023], ["2023"]]:
                result = kynera.download_era5(
                    variables=["2m_temperature"],
                    years=y,
                    months="06",
                    days="all",
                    times=["00:00"],
                    area=[47, 6, 44, 14],
                    output_dir=tmpdir,
                    dry_run=True,
                )
                self.assertEqual(result, [])

    def test_split_by_year_false_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kynera.download_era5(
                variables=["2m_temperature"],
                years=[2022, 2023],
                months="01",
                days=["01"],
                times=["00:00"],
                area=[47, 6, 44, 14],
                output_dir=tmpdir,
                split_by_year=False,
                dry_run=True,
            )
        self.assertEqual(result, [])


# ==============================================================================
# 3. load_era5
# ==============================================================================
class TestLoadEra5(unittest.TestCase):

    def test_load_netcdf_directly(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds = _make_ds()
            nc_path = os.path.join(tmpdir, "era5.nc")
            ds.to_netcdf(nc_path)

            loaded = kynera.load_era5(nc_path)
            self.assertIsInstance(loaded, xr.Dataset)
            self.assertIn("t2m", loaded)

    def test_load_zip_single_nc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds       = _make_ds()
            zip_path = _make_zip_with_nc(tmpdir, ds)

            loaded = kynera.load_era5(zip_path)
            self.assertIsInstance(loaded, xr.Dataset)
            self.assertIn("t2m", loaded)

    def test_load_zip_two_nc_files_merged(self):
        """ZIP with two .nc files (instant + accum) should merge into one dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_inst  = _make_ds(include_precip=False)
            ds_accum = _make_ds(include_wind=False, include_temp=False)

            nc1 = os.path.join(tmpdir, "instant.nc")
            nc2 = os.path.join(tmpdir, "accum.nc")
            ds_inst.to_netcdf(nc1)
            ds_accum.to_netcdf(nc2)

            zip_path = os.path.join(tmpdir, "era5_2024.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(nc1, arcname="instant.nc")
                zf.write(nc2, arcname="accum.nc")

            loaded = kynera.load_era5(zip_path)
            self.assertIsInstance(loaded, xr.Dataset)
            self.assertIn("t2m", loaded)
            self.assertIn("tp",  loaded)

    def test_file_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            kynera.load_era5("/nonexistent/path/era5.zip")

    def test_zip_without_nc_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "empty.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.writestr("readme.txt", "no netcdf here")

            with self.assertRaises(FileNotFoundError):
                kynera.load_era5(zip_path)

    def test_custom_extract_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds       = _make_ds()
            zip_path = _make_zip_with_nc(tmpdir, ds)
            extract  = os.path.join(tmpdir, "custom_extract")

            loaded = kynera.load_era5(zip_path, extract_dir=extract)
            self.assertTrue(os.path.isdir(extract))
            self.assertIsInstance(loaded, xr.Dataset)


# ==============================================================================
# 4. convert_units
# ==============================================================================
class TestConvertUnits(unittest.TestCase):

    def setUp(self):
        self.ds = _make_ds()

    def test_returns_dataset(self):
        result = kynera.convert_units(self.ds)
        self.assertIsInstance(result, xr.Dataset)

    def test_temperature_kelvin_to_celsius(self):
        result = kynera.convert_units(self.ds)
        self.assertIn("t2m_c", result)
        np.testing.assert_allclose(
            result["t2m_c"].values,
            self.ds["t2m"].values - 273.15,
            rtol=1e-5,
        )

    def test_dewpoint_kelvin_to_celsius(self):
        result = kynera.convert_units(self.ds)
        self.assertIn("d2m_c", result)
        np.testing.assert_allclose(
            result["d2m_c"].values,
            self.ds["d2m"].values - 273.15,
            rtol=1e-5,
        )

    def test_pressure_pa_to_hpa(self):
        result = kynera.convert_units(self.ds)
        self.assertIn("msl_hpa", result)
        np.testing.assert_allclose(
            result["msl_hpa"].values,
            self.ds["msl"].values / 100,
            rtol=1e-5,
        )

    def test_precipitation_m_to_mm(self):
        result = kynera.convert_units(self.ds)
        self.assertIn("tp_mm", result)
        np.testing.assert_allclose(
            result["tp_mm"].values,
            self.ds["tp"].values * 1000,
            rtol=1e-5,
        )

    def test_original_vars_preserved(self):
        result = kynera.convert_units(self.ds)
        for var in ["t2m", "d2m", "msl", "tp"]:
            self.assertIn(var, result)

    def test_missing_vars_skipped_gracefully(self):
        """Dataset without t2m should not raise — just skip that conversion."""
        ds_no_temp = _make_ds(include_temp=False)
        result = kynera.convert_units(ds_no_temp)
        self.assertNotIn("t2m_c", result)

    def test_units_attrs_set(self):
        result = kynera.convert_units(self.ds)
        self.assertEqual(result["t2m_c"].attrs.get("units"), "°C")
        self.assertEqual(result["msl_hpa"].attrs.get("units"), "hPa")
        self.assertEqual(result["tp_mm"].attrs.get("units"), "mm")

    def test_does_not_modify_original(self):
        original_vars = set(self.ds.data_vars)
        kynera.convert_units(self.ds)
        self.assertEqual(set(self.ds.data_vars), original_vars)


# ==============================================================================
# 5. compute_derived
# ==============================================================================
class TestComputeDerived(unittest.TestCase):

    def setUp(self):
        self.ds = kynera.convert_units(_make_ds())

    def test_returns_dataset(self):
        result = kynera.compute_derived(self.ds)
        self.assertIsInstance(result, xr.Dataset)

    def test_wind_speed_computed(self):
        result = kynera.compute_derived(self.ds)
        self.assertIn("wind_speed_10m", result)
        expected = np.sqrt(self.ds["u10"].values**2 + self.ds["v10"].values**2)
        np.testing.assert_allclose(result["wind_speed_10m"].values, expected, rtol=1e-5)

    def test_wind_speed_non_negative(self):
        result = kynera.compute_derived(self.ds)
        self.assertTrue((result["wind_speed_10m"].values >= 0).all())

    def test_wind_direction_computed(self):
        result = kynera.compute_derived(self.ds)
        self.assertIn("wind_dir_10m", result)

    def test_wind_direction_range(self):
        result = kynera.compute_derived(self.ds)
        vals = result["wind_dir_10m"].values
        self.assertTrue((vals >= 0).all())
        self.assertTrue((vals < 360).all())

    def test_relative_humidity_computed(self):
        result = kynera.compute_derived(self.ds)
        self.assertIn("rh_2m", result)

    def test_relative_humidity_range(self):
        result = kynera.compute_derived(self.ds)
        rh = result["rh_2m"].values
        # RH can exceed 100 when Td > T (supersaturation artefact), but should be > 0
        self.assertTrue((rh > 0).all())

    def test_no_wind_no_crash(self):
        """Dataset without u10/v10 should silently skip wind derivation."""
        ds_no_wind = kynera.convert_units(_make_ds(include_wind=False))
        result = kynera.compute_derived(ds_no_wind)
        self.assertNotIn("wind_speed_10m", result)
        self.assertNotIn("wind_dir_10m",   result)

    def test_no_temp_no_rh(self):
        """Dataset without t2m_c/d2m_c should silently skip RH derivation."""
        ds_no_temp = kynera.convert_units(_make_ds(include_temp=False))
        result = kynera.compute_derived(ds_no_temp)
        self.assertNotIn("rh_2m", result)

    def test_does_not_modify_original(self):
        original_vars = set(self.ds.data_vars)
        kynera.compute_derived(self.ds)
        self.assertEqual(set(self.ds.data_vars), original_vars)

    def test_attrs_set_on_derived(self):
        result = kynera.compute_derived(self.ds)
        self.assertEqual(result["wind_speed_10m"].attrs.get("units"), "m/s")
        self.assertEqual(result["rh_2m"].attrs.get("units"), "%")


# ==============================================================================
# 6. plot_field
# ==============================================================================
class TestPlotField(unittest.TestCase):

    def setUp(self):
        self.ds = kynera.compute_derived(kynera.convert_units(_make_ds()))

    def test_returns_figure(self):
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend for tests
        import matplotlib.pyplot as plt
        fig = kynera.plot_field(self.ds, "t2m_c", time_index=0)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_plot_derived_variable(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = kynera.plot_field(self.ds, "wind_speed_10m", time_index=0)
        self.assertIsNotNone(fig)
        plt.close("all")

    def test_invalid_variable_raises(self):
        with self.assertRaises(KeyError):
            kynera.plot_field(self.ds, "nonexistent_var")

    def test_saves_to_file(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "plot.png")
            kynera.plot_field(self.ds, "msl_hpa", time_index=0, output_path=out)
            self.assertTrue(os.path.exists(out))
        plt.close("all")

    def test_custom_title(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = kynera.plot_field(self.ds, "t2m_c", title="My Custom Title")
        ax  = fig.axes[0]
        self.assertIn("My Custom Title", ax.get_title())
        plt.close("all")

    def test_time_index_out_of_range_raises(self):
        import matplotlib
        matplotlib.use("Agg")
        with self.assertRaises(Exception):
            kynera.plot_field(self.ds, "t2m_c", time_index=9999)


# ==============================================================================
# 7. Version
# ==============================================================================
class TestVersion(unittest.TestCase):

    def test_version_attribute_exists(self):
        self.assertTrue(hasattr(kynera, "__version__"))

    def test_version_is_string(self):
        self.assertIsInstance(kynera.__version__, str)

    def test_version_non_empty(self):
        self.assertGreater(len(kynera.__version__), 0)


# ==============================================================================
# Entry point
# ==============================================================================
if __name__ == "__main__":
    unittest.main(verbosity=2)
