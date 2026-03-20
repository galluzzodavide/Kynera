"""
test_kynera.py
==============
Unit tests for the Kynera library.

Test data: TESTS/DATA_TEST/era5_adriatico_sample.zip
           (ERA5 reanalysis, Adriatic Sea area, Oct 2025, 3 days, 5 timesteps)

Run from the repository root:
    python -m pytest TESTS/test_kynera.py -v
    python -m pytest TESTS/test_kynera.py -v --cov=kynera --cov-report=term-missing
"""

import os
import pytest
import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kynera

# ==============================================================================
# PATHS
# ==============================================================================
TEST_DIR      = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_ZIP = os.path.join(TEST_DIR, "DATA_TEST", "era5_adriatico_sample.zip")

# Skip all tests that require the sample file if it is not present
requires_data = pytest.mark.skipif(
    not os.path.exists(TEST_DATA_ZIP),
    reason=f"Test data not found: {TEST_DATA_ZIP}"
)


# ==============================================================================
# FIXTURES
# ==============================================================================
@pytest.fixture(scope="module")
def ds():
    """Load the ERA5 sample dataset once for the entire test module."""
    return kynera.load_era5(TEST_DATA_ZIP)


@pytest.fixture(scope="module")
def df_extracted(ds):
    """Extract ERA5 values at two sample stations within the Adriatic domain."""
    stations = pd.DataFrame({
        "station_id":   ["S001", "S002"],
        "station_name": ["Trieste", "Ancona"],
        "lat":          [45.65,    43.60],
        "lon":          [13.76,    13.50],
    })
    return kynera.extract_at_stations(ds, stations)


@pytest.fixture(scope="module")
def df_converted(df_extracted):
    """Apply unit conversion to the extracted DataFrame."""
    return kynera.convert_units(df_extracted)


@pytest.fixture(scope="module")
def df_derived(df_converted):
    """Compute derived variables from the converted DataFrame."""
    return kynera.compute_derived(df_converted)


# ==============================================================================
# 1. LOAD
# ==============================================================================
class TestLoadEra5:

    @requires_data
    def test_returns_xarray_dataset(self, ds):
        assert isinstance(ds, xr.Dataset)

    @requires_data
    def test_has_latitude_longitude(self, ds):
        assert "latitude"  in ds.coords or "latitude"  in ds.dims
        assert "longitude" in ds.coords or "longitude" in ds.dims

    @requires_data
    def test_has_time_dimension(self, ds):
        time_dims = [d for d in ds.dims if "time" in d.lower()]
        assert len(time_dims) > 0, "Dataset should have at least one time dimension"

    @requires_data
    def test_has_expected_variables(self, ds):
        """Sample file contains t2m, msl (instant) and tp (accum)."""
        available = set(ds.data_vars)
        expected  = {"t2m", "msl", "tp"}
        assert expected.issubset(available), (
            f"Missing variables: {expected - available}. Found: {available}"
        )

    @requires_data
    def test_multiple_timesteps(self, ds):
        time_dim = [d for d in ds.dims if "time" in d.lower()][0]
        assert ds.dims[time_dim] > 1

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            kynera.load_era5("nonexistent_file.zip")


# ==============================================================================
# 2. EXTRACT AT STATIONS
# ==============================================================================
class TestExtractAtStations:

    @requires_data
    def test_returns_dataframe(self, df_extracted):
        assert isinstance(df_extracted, pd.DataFrame)

    @requires_data
    def test_has_station_columns(self, df_extracted):
        for col in ["station_id", "station_name", "lat_station", "lon_station"]:
            assert col in df_extracted.columns, f"Missing column: {col}"

    @requires_data
    def test_two_stations_present(self, df_extracted):
        assert df_extracted["station_id"].nunique() == 2

    @requires_data
    def test_no_missing_era5_values(self, df_extracted):
        for var in ["t2m", "msl"]:
            if var in df_extracted.columns:
                assert df_extracted[var].notna().all(), f"NaN found in {var}"

    @requires_data
    def test_coordinates_stored_correctly(self, df_extracted):
        trieste = df_extracted[df_extracted["station_id"] == "S001"]
        assert not trieste.empty
        assert trieste["lat_station"].iloc[0] == pytest.approx(45.65)
        assert trieste["lon_station"].iloc[0] == pytest.approx(13.76)

    def test_missing_lat_lon_raises(self, ds):
        bad_stations = pd.DataFrame({"name": ["A"], "x": [1.0], "y": [2.0]})
        with pytest.raises(ValueError, match="lat"):
            kynera.extract_at_stations(ds, bad_stations)


# ==============================================================================
# 3. CONVERT UNITS
# ==============================================================================
class TestConvertUnits:

    @requires_data
    def test_t2m_celsius_added(self, df_converted):
        assert "t2m_celsius" in df_converted.columns

    @requires_data
    def test_mslp_hpa_added(self, df_converted):
        assert "mslp_hpa" in df_converted.columns

    @requires_data
    def test_tp_mm_added(self, df_converted):
        assert "tp_mm" in df_converted.columns

    @requires_data
    def test_temperature_range_plausible(self, df_converted):
        """October Adriatic temperatures should be between 5°C and 35°C."""
        t = df_converted["t2m_celsius"]
        assert t.min() > 5,  f"Temperature too low: {t.min():.2f}°C"
        assert t.max() < 35, f"Temperature too high: {t.max():.2f}°C"

    @requires_data
    def test_pressure_range_plausible(self, df_converted):
        """Sea level pressure should be between 950 and 1060 hPa."""
        p = df_converted["mslp_hpa"]
        assert p.min() > 950,  f"Pressure too low: {p.min():.1f} hPa"
        assert p.max() < 1060, f"Pressure too high: {p.max():.1f} hPa"

    @requires_data
    def test_precipitation_non_negative(self, df_converted):
        assert (df_converted["tp_mm"] >= 0).all()

    @requires_data
    def test_raw_columns_preserved(self, df_converted):
        """Original raw columns must not be removed."""
        assert "t2m" in df_converted.columns
        assert "msl" in df_converted.columns

    def test_empty_dataframe_returns_empty(self):
        df_empty = pd.DataFrame(columns=["t2m", "msl", "tp"])
        result = kynera.convert_units(df_empty)
        assert result.empty
        assert "t2m_celsius" in result.columns

    def test_conversion_correctness(self):
        """Verify exact conversion values."""
        df = pd.DataFrame({"t2m": [273.15], "msl": [101325.0], "tp": [0.001]})
        result = kynera.convert_units(df)
        assert result["t2m_celsius"].iloc[0] == pytest.approx(0.0,   abs=1e-6)
        assert result["mslp_hpa"].iloc[0]    == pytest.approx(1013.25, abs=1e-3)
        assert result["tp_mm"].iloc[0]       == pytest.approx(1.0,   abs=1e-6)


# ==============================================================================
# 4. COMPUTE DERIVED
# ==============================================================================
class TestComputeDerived:

    def test_wind_speed_computed(self):
        df = pd.DataFrame({"u10": [3.0], "v10": [4.0]})
        result = kynera.compute_derived(df)
        assert "wind_speed_10m" in result.columns
        assert result["wind_speed_10m"].iloc[0] == pytest.approx(5.0, abs=1e-6)

    def test_wind_direction_range(self):
        """Wind direction must be in [0, 360)."""
        df = pd.DataFrame({
            "u10": [3.0, -3.0,  0.0,  3.0],
            "v10": [4.0,  4.0, -4.0, -4.0],
        })
        result = kynera.compute_derived(df)
        assert (result["wind_dir_10m"] >= 0).all()
        assert (result["wind_dir_10m"] < 360).all()

    def test_rh_computed(self):
        """Saturated air (T == Td) should give RH = 100%."""
        df = pd.DataFrame({"t2m_celsius": [20.0], "d2m_celsius": [20.0]})
        result = kynera.compute_derived(df)
        assert "rh_2m" in result.columns
        assert result["rh_2m"].iloc[0] == pytest.approx(100.0, abs=1e-3)

    def test_rh_range_plausible(self):
        """RH should be between 0% and 100% for realistic inputs."""
        df = pd.DataFrame({
            "t2m_celsius": [10.0, 20.0, 30.0],
            "d2m_celsius": [ 5.0, 15.0, 25.0],
        })
        result = kynera.compute_derived(df)
        assert (result["rh_2m"] >= 0).all()
        assert (result["rh_2m"] <= 100).all()

    def test_missing_wind_columns_skipped(self):
        """No wind columns → no wind output, no crash."""
        df = pd.DataFrame({"t2m_celsius": [15.0], "d2m_celsius": [10.0]})
        result = kynera.compute_derived(df)
        assert "wind_speed_10m" not in result.columns
        assert "rh_2m" in result.columns

    def test_input_not_mutated(self):
        df = pd.DataFrame({"u10": [1.0], "v10": [1.0]})
        original_cols = set(df.columns)
        kynera.compute_derived(df)
        assert set(df.columns) == original_cols


# ==============================================================================
# 5. VALIDATE VS OBSERVATIONS
# ==============================================================================
class TestValidateVsObservations:

    @pytest.fixture
    def era5_df(self):
        return pd.DataFrame({
            "station_id":  ["S001", "S001", "S002", "S002"],
            "valid_time":  pd.to_datetime(["2025-10-25 00:00", "2025-10-25 06:00",
                                           "2025-10-25 00:00", "2025-10-25 06:00"]),
            "t2m_celsius": [18.0, 19.0, 20.0, 21.0],
        })

    @pytest.fixture
    def obs_df(self):
        return pd.DataFrame({
            "station_id": ["S001", "S001", "S002", "S002"],
            "valid_time": pd.to_datetime(["2025-10-25 00:00", "2025-10-25 06:00",
                                          "2025-10-25 00:00", "2025-10-25 06:00"]),
            "temp_obs":   [17.5, 18.5, 21.0, 22.0],
        })

    def test_returns_dataframe(self, era5_df, obs_df):
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        assert isinstance(result, pd.DataFrame)

    def test_has_metric_columns(self, era5_df, obs_df):
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        for col in ["bias", "mae", "rmse", "r", "n"]:
            assert col in result.columns, f"Missing metric column: {col}"

    def test_two_stations_in_output(self, era5_df, obs_df):
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        assert len(result) == 2

    def test_bias_correctness(self, era5_df, obs_df):
        """S001: ERA5=[18,19], obs=[17.5,18.5] → bias = +0.5"""
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        s001 = result[result["station_id"] == "S001"]
        assert s001["bias"].iloc[0] == pytest.approx(0.5, abs=1e-6)

    def test_rmse_non_negative(self, era5_df, obs_df):
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        assert (result["rmse"] >= 0).all()

    def test_sorted_by_rmse(self, era5_df, obs_df):
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        assert result["rmse"].is_monotonic_increasing

    def test_n_counts_matched_rows(self, era5_df, obs_df):
        result = kynera.validate_vs_observations(era5_df, obs_df, "t2m_celsius", "temp_obs")
        assert (result["n"] == 2).all()

    def test_no_overlap_raises(self, era5_df):
        obs_no_match = pd.DataFrame({
            "station_id": ["S999"],
            "valid_time": pd.to_datetime(["2099-01-01"]),
            "temp_obs":   [0.0],
        })
        with pytest.raises(ValueError, match="Nessun dato in comune"):
            kynera.validate_vs_observations(era5_df, obs_no_match, "t2m_celsius", "temp_obs")
