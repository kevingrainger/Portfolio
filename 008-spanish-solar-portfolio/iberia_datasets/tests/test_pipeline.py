#-------- tests for the Iberian dataset pipeline --------------------------------
#-----------------------------------------------------------------------------
# End to end, with no downloads and no credentials:
#
#   synthetic ERA5-like GRIB  ->  anemoi-datasets create  ->  Zarr  ->  quality checks
#
# then every quality check is shown to CATCH the fault it exists for, by breaking
# a copy of the data on purpose. A check that only ever passes proves nothing.
#
#       python -m pytest tests

import datetime as dt
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from iberia import quality as q
from iberia import solar
from iberia.synthetic import write_synthetic_era5, grid_coordinates


#-------- build a real Anemoi dataset once, from synthetic GRIB ----------------------
@pytest.fixture(scope="session")
def dataset(tmp_path_factory):
    work = tmp_path_factory.mktemp("iberia")
    grib = work / "test-synthetic-2023-06.grib"
    write_synthetic_era5(grib, dt.datetime(2023, 6, 21, 0), hours=48)

    recipe_text = open(os.path.join(ROOT, "recipes", "iberia-era5.yaml"), encoding="utf8").read()
    #same recipe, pointed at 48 synthetic hours - dates replaced by pattern so a change
    #to the real recipe's period cannot break the test
    recipe_text = re.sub(r"start: \S+", "start: 2023-06-21T00:00:00", recipe_text, count=1)
    recipe_text = re.sub(r"end: \S+", "end: 2023-06-22T23:00:00", recipe_text, count=1)
    recipe_text = recipe_text.replace("data/era5/iberia-era5-{date:date(%Y-%m)}.grib",
                                      str(work / "test-synthetic-{date:date(%Y-%m)}.grib").replace("\\", "/"))
    recipe = work / "recipe.yaml"
    recipe.write_text(recipe_text, encoding="utf8")

    zarr_path = work / "test.zarr"
    exe = shutil.which("anemoi-datasets") or os.path.join(os.path.dirname(sys.executable), "anemoi-datasets")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    result = subprocess.run([exe, "create", str(recipe), str(zarr_path)],
                            capture_output=True, text=True, env=env, encoding="utf-8")
    assert result.returncode == 0, result.stderr[-3000:]
    return q.from_anemoi(str(zarr_path))


#-------- the dataset itself ---------------------------------------------------------
def test_dataset_has_the_expected_shape(dataset):
    data, variables, dates = dataset
    lats, lons = grid_coordinates()
    assert data.shape == (48, len(variables), len(lats) * len(lons))
    for name in ["2t", "tcc", "ssrd", "100u", "100v", "insolation", "cos_julian_day"]:
        assert name in variables


def test_every_quality_check_passes_on_good_data(dataset):
    results = q.run_all(*dataset)
    failed = {n: d for n, (ok, d) in results.items() if not ok}
    assert not failed, q.report(results)


#-------- every check catches its fault ---------------------------------------------
def broken(dataset):
    data, variables, dates = dataset
    return data.copy(), list(variables), np.asarray(dates).copy()


def test_catches_a_missing_time_step(dataset):
    data, variables, dates = broken(dataset)
    dates = np.delete(dates, 10)
    ok, detail = q.check_time_axis(dates)
    assert not ok and "irregular" in detail


def test_catches_missing_values(dataset):
    data, variables, dates = broken(dataset)
    data[5, variables.index("2t"), 100] = np.nan
    assert not q.check_missing(data, variables)[0]


def test_catches_temperature_in_celsius(dataset):
    data, variables, dates = broken(dataset)
    data[:, variables.index("2t"), :] -= 273.15         # someone converted to Celsius
    assert not q.check_ranges(data, variables)[0]


def test_catches_ssrd_already_divided_by_3600(dataset):
    data, variables, dates = broken(dataset)
    data[:, variables.index("ssrd"), :] /= 3600.0       # stored as W m^-2 by mistake
    assert not q.check_ssrd_units(data, variables)[0]


def test_catches_ssrd_shifted_by_an_hour(dataset):
    #label every accumulation one hour late: sunlight appears after sunset
    data, variables, dates = broken(dataset)
    i = variables.index("ssrd")
    data[:, i, :] = np.roll(data[:, i, :], 2, axis=0)
    night_ok = q.check_ssrd_dark_at_night(data, variables)[0]
    toa_ok = q.check_ssrd_below_top_of_atmosphere(data, variables)[0]
    assert not (night_ok and toa_ok)


def test_the_naive_timing_check_raises_false_alarms(dataset):
    #correct data, compared against the sun at the timestamp instead of over the
    #accumulation hour: fails every afternoon. This is the trap the real check avoids.
    data, variables, dates = dataset
    assert q.check_ssrd_below_top_of_atmosphere(data, variables)[0]
    assert not q.check_ssrd_below_top_of_atmosphere(data, variables, naive=True)[0]


#-------- solar conversions -----------------------------------------------------------
def test_ssrd_conversion_units_and_timing():
    times = np.array(["2023-06-21T12:00"], dtype="datetime64[s]")
    irradiance, mid = solar.ssrd_to_irradiance(np.array([3.6e6]), times)
    assert np.isclose(irradiance[0], 1000.0)
    assert mid[0] == np.datetime64("2023-06-21T11:30")


def test_solar_elevation_matches_pvlib():
    import pandas as pd
    import pvlib
    when = dt.datetime(2023, 12, 21, 9, 30)
    ref = pvlib.solarposition.get_solarposition(
        pd.DatetimeIndex([when]).tz_localize("UTC"), 37.4, -6.0)["elevation"].iloc[0]
    assert abs(solar._solar_elevation(37.4, -6.0, when)[0] - ref) < 0.3


def test_clear_sky_index_is_undefined_at_night_and_one_on_a_clear_day():
    k = solar.clear_sky_index(np.array([0.0, 500.0]), np.array([0.0, 500.0]))
    assert np.isnan(k[0]) and np.isclose(k[1], 1.0)
