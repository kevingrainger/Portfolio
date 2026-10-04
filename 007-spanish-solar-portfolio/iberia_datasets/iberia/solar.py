#-------- iberia/solar.py -------------------------------------------------------
#-----------------------------------------------------------------------------
# Turning ERA5's solar radiation into something usable.
#
# ERA5 stores surface solar radiation downwards (ssrd) as energy ACCUMULATED over
# the hour ending at the timestamp, in J m^-2. Two consequences that silently
# break solar analyses when missed:
#
#   units    divide by 3600 to get the mean power over that hour, in W m^-2
#   timing   that mean belongs to the MIDDLE of the hour, 30 minutes before the
#            timestamp. Comparing it with a clear-sky model evaluated at the
#            timestamp itself puts the sun in the wrong place - worst at sunrise
#            and sunset, when half an hour is the difference between dark and day.
#
# The clear-sky index k = GHI / GHI_clear divides out the sun's daily and
# seasonal cycle, leaving the weather. It is what the correlation analysis uses.

import datetime as dt

import numpy as np

SECONDS_PER_HOUR = 3600.0
SOLAR_CONSTANT = 1361.0          # W m^-2 at the top of the atmosphere


def ssrd_to_irradiance(ssrd_joules, valid_times):
    """Hourly-accumulated ssrd (J m^-2) -> mean irradiance (W m^-2), timed at mid-hour.

    ssrd_joules : array (..., n_times)   valid_times : array of datetime64, n_times
    Returns (irradiance, mid_hour_times).
    """
    irradiance = np.asarray(ssrd_joules, dtype=float) / SECONDS_PER_HOUR
    mid = np.asarray(valid_times, dtype="datetime64[s]") - np.timedelta64(30, "m")
    return irradiance, mid


def _solar_elevation(lat, lon, when):
    """Solar elevation in degrees, for every grid point at once. when is a UTC datetime.

    pvlib's full position algorithm works one location at a time, which took minutes
    for a grid. Its analytical pieces - Spencer's declination and equation of time,
    the hour angle, the zenith - take arrays, and are accurate to a fraction of a
    degree: plenty for an hourly clear-sky reference.
    """
    import pandas as pd
    from pvlib import solarposition as sp

    t = pd.DatetimeIndex([pd.Timestamp(when)])
    t = t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    lon = np.atleast_1d(np.asarray(lon, dtype=float))

    day = t.dayofyear[0]
    declination = sp.declination_spencer71(day)                     # radians
    eot = sp.equation_of_time_spencer71(day)                        # minutes
    hours = t.hour[0] + t.minute[0] / 60 + t.second[0] / 3600
    hour_angle = np.radians(15.0 * (hours - 12.0) + lon + eot / 4.0)
    zenith = sp.solar_zenith_analytical(np.radians(lat), hour_angle, declination)
    return 90.0 - np.degrees(zenith)


def clear_sky_ghi(lat, lon, when):
    """Clear-sky global horizontal irradiance (W m^-2), Haurwitz model.

    Simple, needs only the sun's position, and is the standard choice when no
    aerosol or water-vapour data are available. lat/lon arrays; when a UTC datetime.
    """
    elevation = _solar_elevation(lat, lon, when)
    cos_zenith = np.sin(np.radians(elevation))
    ghi = np.where(cos_zenith > 0,
                   1098.0 * cos_zenith * np.exp(-0.059 / np.maximum(cos_zenith, 1e-3)), 0.0)
    return ghi


def top_of_atmosphere(lat, lon, when):
    """Irradiance on a horizontal surface at the top of the atmosphere (W m^-2)."""
    elevation = _solar_elevation(lat, lon, when)
    return np.maximum(0.0, SOLAR_CONSTANT * np.sin(np.radians(elevation)))


def clear_sky_index(ghi, ghi_clear, min_clear=50.0):
    """k = GHI / GHI_clear, only where the clear-sky value is large enough to divide by.

    Below min_clear (W m^-2) - night, and the low-sun hours where the ratio is
    dominated by geometry - the result is NaN, so those hours drop out of any
    correlation rather than adding fake ones.
    """
    ghi = np.asarray(ghi, dtype=float)
    ghi_clear = np.asarray(ghi_clear, dtype=float)
    return np.where(ghi_clear >= min_clear, ghi / np.where(ghi_clear > 0, ghi_clear, 1.0), np.nan)
