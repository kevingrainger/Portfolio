#-------- iberia/synthetic.py ---------------------------------------------------
#-----------------------------------------------------------------------------
# A small ERA5-lookalike GRIB file, for testing the pipeline without downloading
# anything.
#
# Same grid as the real dataset (0.25 degrees over Iberia), same parameters, and
# the same encoding traps as the real thing - most importantly, solar radiation
# stored as an ACCUMULATION over the previous hour in J m^-2, not as a rate.
# The values are physically shaped (clear-sky sunshine reduced by cloud, a daily
# temperature cycle) so the quality checks have something realistic to check.
#
# This is what lets the whole pipeline - GRIB in, anemoi-datasets create, Zarr
# out, quality checks - run in CI with no credentials and no network.

import datetime as dt

import numpy as np

from .solar import clear_sky_ghi

# ERA5 GRIB1 parameter ids (ECMWF local table 128 / 228)
PARAMS = {
    "2t": 167,      # 2 m temperature, K
    "tcc": 164,     # total cloud cover, 0-1
    "ssrd": 169,    # surface solar radiation downwards, J m^-2 accumulated over the hour
    "100u": 228246,  # 100 m wind, m s^-1
    "100v": 228247,
}

# The Iberian cut-out used throughout: north, west, south, east
AREA = (44.5, -10.0, 35.5, 4.5)
GRID = 0.25


def grid_coordinates(area=AREA, step=GRID):
    north, west, south, east = area
    lats = np.arange(north, south - step / 2, -step)
    lons = np.arange(west, east + step / 2, step)
    return lats, lons


#-------- planted weather: a cloud field with a known correlation length ----------
#-----------------------------------------------------------------------------
# Placeholder data should not be noise - anything analysed on noise produces
# confident nonsense. So the cloud field has the structure the real analysis is
# looking for, planted on purpose with known values:
#
#   correlation length L(t)  short in summer (convective cloud, ~80 km), long in
#                            winter (blocking highs, ~300 km), varying smoothly
#                            through the year
#   persistence              cloud evolves hour to hour (AR(1) in time), so the
#                            data are autocorrelated like real weather
#
# The correlation analysis run on this data must recover L(t). That is a real
# test of the method - and the results are labelled as such, never as findings.
KM_PER_DEGREE = 111.0
L_SUMMER_KM, L_WINTER_KM = 80.0, 300.0
HOURLY_PERSISTENCE = 0.97                              # AR(1) coefficient


def planted_length_km(when):
    """The correlation length planted for a given time: longest mid-January."""
    day = when.timetuple().tm_yday
    winterness = 0.5 * (1 + np.cos(2 * np.pi * (day - 15) / 365.25))
    return L_SUMMER_KM + (L_WINTER_KM - L_SUMMER_KM) * winterness


def _gaussian_field(rng, shape, length_km, dy_km, dx_km):
    """A unit-variance Gaussian random field with squared-exponential correlation of the given length."""
    pad = (int(shape[0] * 1.5), int(shape[1] * 1.5))  # avoid the FFT's wrap-around at the edges
    ky = np.fft.fftfreq(pad[0], d=dy_km)[:, None]
    kx = np.fft.fftfreq(pad[1], d=dx_km)[None, :]
    spectrum = np.exp(-(np.pi * length_km) ** 2 * (kx ** 2 + ky ** 2) / 2)
    noise = rng.standard_normal(pad)
    field = np.real(np.fft.ifft2(np.fft.fft2(noise) * np.sqrt(spectrum)))[:shape[0], :shape[1]]
    return field / field.std()


def write_synthetic_era5(path, start, hours, area=AREA, step=GRID, seed=0, state=None):
    """Write `hours` of hourly ERA5-lookalike fields to `path`. Returns the cloud state,
    so consecutive months can be written as one continuous stretch of weather."""
    import eccodes

    rng = np.random.default_rng(seed)
    lats, lons = grid_coordinates(area, step)
    lon2d, lat2d = np.meshgrid(lons, lats)
    shape = lat2d.shape
    dy_km = step * KM_PER_DEGREE
    dx_km = step * KM_PER_DEGREE * np.cos(np.radians(lats.mean()))

    latent = state if state is not None else _gaussian_field(rng, shape, planted_length_km(start), dy_km, dx_km)
    innovation_scale = np.sqrt(1 - HOURLY_PERSISTENCE ** 2)

    with open(path, "wb") as out:
        for h in range(hours):
            valid = start + dt.timedelta(hours=h)
            fresh = _gaussian_field(rng, shape, planted_length_km(valid), dy_km, dx_km)
            latent = HOURLY_PERSISTENCE * latent + innovation_scale * fresh
            cloud = 1 / (1 + np.exp(-1.6 * (latent + 0.3)))       # logistic: mostly clear, sometimes overcast

            #hourly mean irradiance over the hour ENDING at valid time, as ERA5 accumulates it.
            #Kasten & Czeplak (1980): GHI = clear-sky x (1 - 0.75 N^3.4)
            mid = valid - dt.timedelta(minutes=30)
            ghi = clear_sky_ghi(lat2d.ravel(), lon2d.ravel(), mid).reshape(shape) * (1 - 0.75 * cloud ** 3.4)
            ssrd = ghi * 3600.0                                  # W m^-2 -> J m^-2 over one hour

            hour_angle = 2 * np.pi * ((valid.hour + lon2d / 15.0) % 24 - 15) / 24
            season = np.cos(2 * np.pi * (valid.timetuple().tm_yday - 200) / 365.25)   # warmest in July
            t2m = (287 + 9 * season + 7 * np.cos(hour_angle) - 0.6 * (lat2d - 40)
                   - 3 * cloud + rng.normal(0, 0.3, shape))
            u = 4 + rng.normal(0, 1, shape)
            v = -2 + rng.normal(0, 1, shape)

            fields = {"2t": t2m, "tcc": cloud, "ssrd": ssrd, "100u": u, "100v": v}
            for name, values in fields.items():
                _write_field(eccodes, out, PARAMS[name], values, valid, area, step,
                             accumulated=(name == "ssrd"))
    return latent


# ERA5 accumulations come from forecasts started at 06 and 18 UTC, steps 1 to 12.
# A field valid at 07-18 UTC belongs to the 06 forecast; 19-06 UTC to the 18 one.
def _era5_forecast_base(valid):
    for hours_back in range(1, 13):
        base = valid - dt.timedelta(hours=hours_back)
        if base.hour in (6, 18):
            return base
    raise ValueError(valid)


def _write_field(eccodes, out, param_id, values, valid, area, step, accumulated):
    north, west, south, east = area
    lats, lons = grid_coordinates(area, step)
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib1")
    try:
        eccodes.codes_set(h, "centre", 98)
        eccodes.codes_set(h, "Ni", len(lons))
        eccodes.codes_set(h, "Nj", len(lats))
        eccodes.codes_set(h, "latitudeOfFirstGridPointInDegrees", north)
        eccodes.codes_set(h, "longitudeOfFirstGridPointInDegrees", west)
        eccodes.codes_set(h, "latitudeOfLastGridPointInDegrees", south)
        eccodes.codes_set(h, "longitudeOfLastGridPointInDegrees", east)
        eccodes.codes_set(h, "iDirectionIncrementInDegrees", step)
        eccodes.codes_set(h, "jDirectionIncrementInDegrees", step)
        eccodes.codes_set(h, "paramId", param_id)
        eccodes.codes_set(h, "class", "ea")             # ERA5

        if accumulated:
            #exactly as real ERA5 encodes it (checked against ECMWF's published Anemoi
            #ERA5 dataset): a short forecast from 06 or 18 UTC, the field accumulated over
            #the last hour of steps 1-12 - so 13 UTC is base 06, step 6-7
            base = _era5_forecast_base(valid)
            step = int((valid - base).total_seconds() // 3600)
            eccodes.codes_set(h, "type", "fc")
            eccodes.codes_set(h, "dataDate", int(base.strftime("%Y%m%d")))
            eccodes.codes_set(h, "dataTime", int(base.strftime("%H%M")))
            eccodes.codes_set(h, "stepType", "accum")
            eccodes.codes_set(h, "startStep", step - 1)
            eccodes.codes_set(h, "endStep", step)
        else:
            eccodes.codes_set(h, "type", "an")
            eccodes.codes_set(h, "dataDate", int(valid.strftime("%Y%m%d")))
            eccodes.codes_set(h, "dataTime", int(valid.strftime("%H%M")))
            eccodes.codes_set(h, "stepType", "instant")
            eccodes.codes_set(h, "step", 0)

        eccodes.codes_set(h, "bitsPerValue", 16)
        eccodes.codes_set_values(h, np.asarray(values, dtype=float).ravel())
        eccodes.codes_write(h, out)
    finally:
        eccodes.codes_release(h)
