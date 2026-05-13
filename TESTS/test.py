"""
test.py
=======
Unit tests for the Kynera ERA5 processing library.

Uses the real sample file:
    Kynera/TESTS/DATA_TEST/era5_sample_adriatic.zip

Run with:
    pytest test.py -v

Or for coverage:
    python coverage_report.py
"""

import os
import zipfile
import tempfile
import unittest

import numpy as np
import xarray as xr
import pandas as pd

import kynera


# ==============================================================================
# Path to the real test data
# ==============================================================================
_HERE       = os.path.dirname(os.path.abspath(__file__))
_DATA_DIR   = os.path.join(_HERE, "DATA_TEST")
_SAMPLE_ZIP = os.path.join(_DATA_DIR, "era5_sample_adriatic.zip")
_SAMPLE_NC  = os.path.join(_DATA_DIR, "era5_sample_adriatic.nc")


def _get_sample_path():
    if os.path.exists(_SAMPLE_ZIP):
        return _SAMPLE_ZIP
    if os.path.exists(_SAMPLE_NC):
        return _SAMPLE_NC
    raise FileNotFoundError(
        f"[test.py] Real sample file not found.\n"
        f"  Expected: {_SAMPLE_ZIP}\n"
        f"  or:       {_SAMPLE_NC}\n"
        f"  Place era5_sample_adriatic.zip in Kynera/TESTS/DATA_TEST/"
    )


# ==============================================================================
# Module-level fixture: load once, reuse across all tests
# ==============================================================================
_SAMPLE_PATH = _get_sample_path()
_DS_RAW      = kynera.load_era5(_SAMPLE_PATH)
_DS_CONV     = kynera.convert_units(_DS_RAW)
_DS_FULL     = kynera.compute_derived(_DS_CONV)


# ==============================================================================
# Helpers
# ==============================================================================

def _make_minimal_ds(include_wind=True, include_temp=True, include_precip=True):
    """
    Synthetic dataset used only for edge-case tests that need controlled
    conditions (missing variables, empty ZIPs, etc.).
    All real-data tests use the Adriatic sample file.
    """
    times = pd.date_range("2024-01-01", periods=4, freq="6h")
    lats  = np.array([44.0, 45.0])
    lons  = np.array([13.0, 14.0])

    coords = {"valid_time": times, "latitude": lats, "longitude": lons}
    shape  = (len(times), len(lats), len(lons))
    dims   = ("valid_time", "latitude", "longitude")

    data_vars = {}
    if include_wind:
        data_vars["u10"]   = (dims, np.random.uniform(-10, 10, shape).astype("float32"))
        data_vars["v10"]   = (dims, np.random.uniform(-10, 10, shape).astype("float32"))
        data_vars["i10fg"] = (dims, np.random.uniform(0, 20,  shape).astype("float32"))
    if include_temp:
        data_vars["t2m"] = (dims, np.random.uniform(270, 300, shape).astype("float32"))
        data_vars["d2m"] = (dims, np.random.uniform(260, 295, shape).astype("float32"))
        data_vars["msl"] = (dims, np.random.uniform(100000, 103500, shape).astype("float32"))
    if include_precip:
        data_vars["tp"]  = (dims, np.random.uniform(0, 0.01, shape).astype("float32"))

    return xr.Dataset(data_vars, coords=coords)


def _make_zip_with_nc(tmpdir, ds, filename="era5_test.zip"):
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
        self.assertIn("t2m",   result)
        self.assertNotIn("tp", result)

    def test_filter_accumulated(self):
        result = kynera.list_variables(category="accumulated")
        self.assertIn("tp",     result)
        self.assertNotIn("t2m", result)

    def test_filter_wave(self):
        result = kynera.list_variables(category="wave")
        self.assertIn("swh",    result)
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
# 2. download_era5  (dry_run only — no real HTTP calls)
# ==============================================================================
class TestDownloadEra5DryRun(unittest.TestCase):

    def test_dry_run_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kynera.download_era5(
                variables=["2m_temperature"],
                years=2024, months="01", days="all",
                times=["00:00"], area=[47, 6, 44, 14],
                output_dir=tmpdir, dry_run=True,
            )
        self.assertEqual(result, [])

    def test_dry_run_multi_year_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kynera.download_era5(
                variables=["2m_temperature", "total_precipitation"],
                years=[2022, 2023, 2024], months=["01", "02"],
                days="all", times=["00:00", "12:00"],
                area=[47, 6, 44, 14], output_dir=tmpdir, dry_run=True,
            )
        self.assertEqual(result, [])

    def test_skip_existing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            existing = os.path.join(tmpdir, "era5_2024.zip")
            open(existing, "w").close()
            result = kynera.download_era5(
                variables=["2m_temperature"],
                years=2024, months="01", days=["01"],
                times=["00:00"], area=[47, 6, 44, 14],
                output_dir=tmpdir, overwrite=False,
            )
        self.assertIn(existing, result)

    def test_year_normalisation_int_and_str(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for y in [2023, "2023", [2023], ["2023"]]:
                result = kynera.download_era5(
                    variables=["2m_temperature"],
                    years=y, months="06", days="all",
                    times=["00:00"], area=[47, 6, 44, 14],
                    output_dir=tmpdir, dry_run=True,
                )
                self.assertEqual(result, [])

    def test_split_by_year_false_dry_run(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = kynera.download_era5(
                variables=["2m_temperature"],
                years=[2022, 2023], months="01", days=["01"],
                times=["00:00"], area=[47, 6, 44, 14],
                output_dir=tmpdir, split_by_year=False, dry_run=True,
            )
        self.assertEqual(result, [])


# ==============================================================================
# 3. load_era5  — real Adriatic sample file
# ==============================================================================
class TestLoadEra5(unittest.TestCase):

    def test_load_returns_dataset(self):
        self.assertIsInstance(_DS_RAW, xr.Dataset)

    def test_dataset_has_variables(self):
        self.assertGreater(len(_DS_RAW.data_vars), 0)

    def test_dataset_has_spatial_dims(self):
        dims = set(_DS_RAW.dims)
        self.assertTrue(any("lat" in d.lower() for d in dims),
                        f"Expected latitude dim, got: {dims}")
        self.assertTrue(any("lon" in d.lower() for d in dims),
                        f"Expected longitude dim, got: {dims}")

    def test_dataset_has_time_dim(self):
        dims = set(_DS_RAW.dims)
        self.assertTrue(any("time" in d.lower() for d in dims),
                        f"Expected time dim, got: {dims}")

    def test_variables_are_not_all_nan(self):
        for var in _DS_RAW.data_vars:
            vals = _DS_RAW[var].values
            self.assertFalse(np.all(np.isnan(vals)),
                             f"Variable '{var}' is entirely NaN")

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
        if not _SAMPLE_PATH.endswith(".zip"):
            self.skipTest("Sample is a .nc, not a ZIP")
        with tempfile.TemporaryDirectory() as tmpdir:
            extract = os.path.join(tmpdir, "custom_extract")
            loaded  = kynera.load_era5(_SAMPLE_PATH, extract_dir=extract)
            self.assertTrue(os.path.isdir(extract))
            self.assertIsInstance(loaded, xr.Dataset)

    def test_two_nc_zip_merged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ds_a = _make_minimal_ds(include_precip=False)
            ds_b = _make_minimal_ds(include_wind=False, include_temp=False)
            nc1  = os.path.join(tmpdir, "instant.nc")
            nc2  = os.path.join(tmpdir, "accum.nc")
            ds_a.to_netcdf(nc1)
            ds_b.to_netcdf(nc2)
            zip_path = os.path.join(tmpdir, "merged.zip")
            with zipfile.ZipFile(zip_path, "w") as zf:
                zf.write(nc1, arcname="instant.nc")
                zf.write(nc2, arcname="accum.nc")
            loaded = kynera.load_era5(zip_path)
            self.assertIn("t2m", loaded)
            self.assertIn("tp",  loaded)


# ==============================================================================
# 4. convert_units  — real Adriatic sample file
# ==============================================================================
class TestConvertUnits(unittest.TestCase):

    def test_returns_dataset(self):
        self.assertIsInstance(_DS_CONV, xr.Dataset)

    def test_original_vars_preserved(self):
        for var in _DS_RAW.data_vars:
            self.assertIn(var, _DS_CONV,
                          f"Original variable '{var}' should be preserved")

    def test_msl_converted_if_present(self):
        if "msl" not in _DS_RAW:
            self.skipTest("msl not in sample dataset")
        self.assertIn("msl_hpa", _DS_CONV)
        np.testing.assert_allclose(
            _DS_CONV["msl_hpa"].values,
            _DS_RAW["msl"].values / 100,
            rtol=1e-5,
        )

    def test_t2m_converted_if_present(self):
        if "t2m" not in _DS_RAW:
            self.skipTest("t2m not in sample dataset")
        self.assertIn("t2m_c", _DS_CONV)
        np.testing.assert_allclose(
            _DS_CONV["t2m_c"].values,
            _DS_RAW["t2m"].values - 273.15,
            rtol=1e-5,
        )

    def test_tp_converted_if_present(self):
        if "tp" not in _DS_RAW:
            self.skipTest("tp not in sample dataset")
        self.assertIn("tp_mm", _DS_CONV)
        np.testing.assert_allclose(
            _DS_CONV["tp_mm"].values,
            _DS_RAW["tp"].values * 1000,
            rtol=1e-5,
        )

    def test_converted_units_attrs(self):
        if "msl" in _DS_RAW:
            self.assertEqual(_DS_CONV["msl_hpa"].attrs.get("units"), "hPa")
        if "t2m" in _DS_RAW:
            self.assertEqual(_DS_CONV["t2m_c"].attrs.get("units"), "°C")
        if "tp" in _DS_RAW:
            self.assertEqual(_DS_CONV["tp_mm"].attrs.get("units"), "mm")

    def test_does_not_modify_original(self):
        original_vars = set(_DS_RAW.data_vars)
        kynera.convert_units(_DS_RAW)
        self.assertEqual(set(_DS_RAW.data_vars), original_vars)

    def test_missing_vars_skipped_gracefully(self):
        ds     = _make_minimal_ds(include_temp=False)
        result = kynera.convert_units(ds)
        self.assertNotIn("t2m_c", result)


# ==============================================================================
# 5. compute_derived  — real Adriatic sample file
# ==============================================================================
class TestComputeDerived(unittest.TestCase):

    def test_returns_dataset(self):
        self.assertIsInstance(_DS_FULL, xr.Dataset)

    def test_wind_speed_present_if_uv_available(self):
        if "u10" not in _DS_CONV or "v10" not in _DS_CONV:
            self.skipTest("u10/v10 not in sample dataset")
        self.assertIn("wind_speed_10m", _DS_FULL)

    def test_wind_speed_values(self):
        if "u10" not in _DS_CONV or "v10" not in _DS_CONV:
            self.skipTest("u10/v10 not in sample dataset")
        expected = np.sqrt(_DS_CONV["u10"].values**2 + _DS_CONV["v10"].values**2)
        np.testing.assert_allclose(_DS_FULL["wind_speed_10m"].values, expected, rtol=1e-5)

    def test_wind_speed_non_negative(self):
        if "wind_speed_10m" not in _DS_FULL:
            self.skipTest("wind_speed_10m not computed")
        self.assertTrue((_DS_FULL["wind_speed_10m"].values >= 0).all())

    def test_wind_direction_present_if_uv_available(self):
        if "u10" not in _DS_CONV or "v10" not in _DS_CONV:
            self.skipTest("u10/v10 not in sample dataset")
        self.assertIn("wind_dir_10m", _DS_FULL)

    def test_wind_direction_range(self):
        if "wind_dir_10m" not in _DS_FULL:
            self.skipTest("wind_dir_10m not computed")
        vals = _DS_FULL["wind_dir_10m"].values
        self.assertTrue((vals >= 0).all())
        self.assertTrue((vals < 360).all())

    def test_rh_present_if_temp_available(self):
        if "t2m_c" not in _DS_CONV or "d2m_c" not in _DS_CONV:
            self.skipTest("t2m_c/d2m_c not in converted dataset")
        self.assertIn("rh_2m", _DS_FULL)

    def test_rh_positive(self):
        if "rh_2m" not in _DS_FULL:
            self.skipTest("rh_2m not computed")
        self.assertTrue((_DS_FULL["rh_2m"].values > 0).all())

    def test_no_wind_no_crash(self):
        ds     = kynera.convert_units(_make_minimal_ds(include_wind=False))
        result = kynera.compute_derived(ds)
        self.assertNotIn("wind_speed_10m", result)
        self.assertNotIn("wind_dir_10m",   result)

    def test_no_temp_no_rh(self):
        ds     = kynera.convert_units(_make_minimal_ds(include_temp=False))
        result = kynera.compute_derived(ds)
        self.assertNotIn("rh_2m", result)

    def test_does_not_modify_input(self):
        original_vars = set(_DS_CONV.data_vars)
        kynera.compute_derived(_DS_CONV)
        self.assertEqual(set(_DS_CONV.data_vars), original_vars)

    def test_derived_attrs(self):
        if "wind_speed_10m" in _DS_FULL:
            self.assertEqual(_DS_FULL["wind_speed_10m"].attrs.get("units"), "m/s")
        if "rh_2m" in _DS_FULL:
            self.assertEqual(_DS_FULL["rh_2m"].attrs.get("units"), "%")


# ==============================================================================
# 6. plot_field  — real Adriatic sample file
# ==============================================================================
class TestPlotField(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")

    def tearDown(self):
        import matplotlib.pyplot as plt
        plt.close("all")

    def _first_plottable_var(self):
        for v in ["t2m_c", "msl_hpa", "wind_speed_10m", "u10", "msl"]:
            if v in _DS_FULL:
                return v
        return next(iter(_DS_FULL.data_vars))

    def test_returns_figure(self):
        import matplotlib.figure
        fig = kynera.plot_field(_DS_FULL, self._first_plottable_var(), time_index=0)
        self.assertIsInstance(fig, matplotlib.figure.Figure)

    def test_plot_wind_speed_if_available(self):
        if "wind_speed_10m" not in _DS_FULL:
            self.skipTest("wind_speed_10m not computed")
        fig = kynera.plot_field(_DS_FULL, "wind_speed_10m", time_index=0)
        self.assertIsNotNone(fig)

    def test_invalid_variable_raises(self):
        with self.assertRaises(KeyError):
            kynera.plot_field(_DS_FULL, "nonexistent_var_xyz")

    def test_saves_to_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = os.path.join(tmpdir, "plot.png")
            kynera.plot_field(_DS_FULL, self._first_plottable_var(),
                              time_index=0, output_path=out)
            self.assertTrue(os.path.exists(out))
            self.assertGreater(os.path.getsize(out), 0)

    def test_custom_title(self):
        fig = kynera.plot_field(_DS_FULL, self._first_plottable_var(),
                                title="Adriatic Test Plot")
        self.assertIn("Adriatic Test Plot", fig.axes[0].get_title())

    def test_time_index_out_of_range_raises(self):
        with self.assertRaises(Exception):
            kynera.plot_field(_DS_FULL, self._first_plottable_var(), time_index=99999)


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
