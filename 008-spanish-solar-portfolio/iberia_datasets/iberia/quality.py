#-------- iberia/quality.py -----------------------------------------------------
#-----------------------------------------------------------------------------
# Data-quality checks for an Iberian ERA5/CERRA Anemoi dataset.
#
# Every check takes plain arrays - data[time, variable, gridpoint], the variable
# names and the dates - so the same checks run on a real dataset, on a synthetic
# test dataset, and on deliberately broken copies (which is how the tests prove
# each check actually catches the fault it is meant to catch).
#
# Each returns (passed, detail). run_all collects them into one report.

import numpy as np

SOLAR_CONSTANT = 1361.0           # W m^-2
SECONDS_PER_HOUR = 3600.0

# Physically possible ranges - generous on purpose: these catch broken data
# (wrong units, fill values, flipped signs), not unusual weather.
RANGES = {
    "2t": (200.0, 335.0),         # K: -73 C to +62 C
    "tcc": (0.0, 1.0),
    "100u": (-100.0, 100.0),      # m s^-1
    "100v": (-100.0, 100.0),
    "ssrd": (0.0, SOLAR_CONSTANT * SECONDS_PER_HOUR),   # J m^-2 in one hour
}


def _var(data, variables, name):
    return data[:, variables.index(name), :]


#-------- the time axis ----------------------------------------------------------
def check_time_axis(dates, frequency=np.timedelta64(1, "h")):
    dates = np.asarray(dates, dtype="datetime64[s]")
    steps = np.diff(dates)
    gaps = np.flatnonzero(steps != frequency)
    if len(gaps):
        return False, f"{len(gaps)} irregular steps, first after {dates[gaps[0]]} ({steps[gaps[0]]})"
    return True, f"{len(dates)} steps, regular at {frequency}"


#-------- missing values ---------------------------------------------------------
def check_missing(data, variables):
    bad = {v: int(np.isnan(_var(data, variables, v)).sum()) for v in variables}
    bad = {v: n for v, n in bad.items() if n}
    return (not bad), (f"missing values: {bad}" if bad else "no missing values")


#-------- physical ranges --------------------------------------------------------
def check_ranges(data, variables):
    problems = []
    for name, (lo, hi) in RANGES.items():
        if name not in variables:
            continue
        x = _var(data, variables, name)
        if np.nanmin(x) < lo or np.nanmax(x) > hi:
            problems.append(f"{name} in [{np.nanmin(x):.4g}, {np.nanmax(x):.4g}], allowed [{lo:g}, {hi:g}]")
    return (not problems), ("; ".join(problems) if problems else "all variables inside physical ranges")


#-------- solar radiation: units -------------------------------------------------
# An hour of midday sun is ~3 million J m^-2. If the largest value in a dataset
# is in the hundreds, someone has already divided by 3600 - or the field is a
# rate, not an accumulation - and every downstream conversion will be wrong by a
# factor of 3600.
def check_ssrd_units(data, variables):
    peak = np.nanmax(_var(data, variables, "ssrd"))
    if peak < 10_000:
        return False, f"ssrd peaks at {peak:.0f}: looks like W m^-2, expected J m^-2 accumulated over an hour"
    return True, f"ssrd peaks at {peak:.3g} J m^-2 ({peak / SECONDS_PER_HOUR:.0f} W m^-2 mean over the hour)"


#-------- solar radiation: timing ------------------------------------------------
# ssrd at time t is the energy received over the hour BEFORE t. Anemoi's insolation
# forcing is the cosine of the solar zenith AT t. So the sun's strength during the
# accumulation window is bounded by its values at t - 1h and at t; the window's
# mean can be compared against the larger of the two.
#
# Compare against insolation at t alone and every afternoon fails, because the
# sun was higher during the previous hour than it is at t. That is the classic
# half-hour/one-hour timing error, and `naive=True` reproduces it on purpose.
def _top_of_atmosphere_bound(data, variables, naive=False):
    ins = _var(data, variables, "insolation")
    if naive:
        return SOLAR_CONSTANT * ins
    previous = np.vstack([ins[:1], ins[:-1]])           # insolation one step earlier
    return SOLAR_CONSTANT * np.maximum(ins, previous)


def check_ssrd_below_top_of_atmosphere(data, variables, tolerance=1.02, naive=False):
    irradiance = _var(data, variables, "ssrd") / SECONDS_PER_HOUR
    bound = _top_of_atmosphere_bound(data, variables, naive) * tolerance + 1.0
    over = irradiance > bound
    if over.any():
        t, p = np.argwhere(over)[0]
        return False, (f"{over.sum()} values exceed the top-of-atmosphere bound, first at "
                       f"time index {t}, point {p}: {irradiance[t, p]:.0f} > {bound[t, p]:.0f} W m^-2")
    return True, "surface irradiance never exceeds the top-of-atmosphere bound"


def check_ssrd_dark_at_night(data, variables, threshold=1.0):
    irradiance = _var(data, variables, "ssrd") / SECONDS_PER_HOUR
    ins = _var(data, variables, "insolation")
    previous = np.vstack([ins[:1], ins[:-1]])
    night = (ins <= 0) & (previous <= 0)                # sun below the horizon all hour
    lit = night & (irradiance > threshold)
    if lit.any():
        return False, f"{lit.sum()} night-time values above {threshold} W m^-2"
    return True, f"zero irradiance at night ({night.sum()} night values checked)"


#-------- everything -------------------------------------------------------------
CHECKS = {
    "time axis": lambda d, v, t: check_time_axis(t),
    "missing values": lambda d, v, t: check_missing(d, v),
    "physical ranges": lambda d, v, t: check_ranges(d, v),
    "ssrd units": lambda d, v, t: check_ssrd_units(d, v),
    "ssrd below top of atmosphere": lambda d, v, t: check_ssrd_below_top_of_atmosphere(d, v),
    "ssrd dark at night": lambda d, v, t: check_ssrd_dark_at_night(d, v),
}


def run_all(data, variables, dates):
    return {name: check(data, variables, dates) for name, check in CHECKS.items()}


def from_anemoi(path):
    """Load an Anemoi dataset as (data[time, variable, gridpoint], variables, dates)."""
    from anemoi.datasets import open_dataset
    ds = open_dataset(path)
    data = np.asarray(ds[:])[:, :, 0, :]                # drop the ensemble axis
    return data, list(ds.variables), np.asarray(ds.dates)


def report(results):
    width = max(len(n) for n in results)
    lines = [f"{'PASS' if ok else 'FAIL'}  {name:<{width}}  {detail}" for name, (ok, detail) in results.items()]
    return "\n".join(lines)
