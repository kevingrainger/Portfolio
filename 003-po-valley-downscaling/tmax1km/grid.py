#-------- grid.py -------------------------------------------------------------------
#-----------------------------------------------------------------------------
# The 1 km grid over the Po Valley and the static fields that live on it.
#
# Everything downstream is a (ny, nx) array on this grid: row 0 is the southern edge,
# column 0 the western edge, so array[iy, ix] sits at (x[ix], y[iy]) in UTM zone 32N.
# Lengths are in km and times in hours throughout, which keeps the PDE's matrix
# entries of order one.

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer
from scipy import ndimage

BBOX = dict(lat0=44.0, lat1=46.2, lon0=7.0, lon1=12.8)
EPSG = 32632
GAMMA = 6.5                                    # K per km: lapse rate used to reduce to sea level
CLASSES = ["built-up", "cropland", "tree cover", "grass/shrub", "water", "other"]

_to_xy = Transformer.from_crs(4326, EPSG, always_xy=True)
_to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)


@dataclass(frozen=True)
class Grid:
    x0: float = 340.0                          # km, western edge of the first cell
    y0: float = 4872.0                         # km, southern edge of the first cell
    nx: int = 460
    ny: int = 245
    dx: float = 1.0                            # km

    @property
    def shape(self):
        return (self.ny, self.nx)

    @property
    def n(self):
        return self.nx * self.ny

    @property
    def x(self):                               # cell centres, km
        return self.x0 + (np.arange(self.nx) + 0.5) * self.dx

    @property
    def y(self):
        return self.y0 + (np.arange(self.ny) + 0.5) * self.dx

    @property
    def extent(self):                          # for imshow(origin="lower")
        return (self.x0, self.x0 + self.nx * self.dx, self.y0, self.y0 + self.ny * self.dx)

    def lonlat(self):
        X, Y = np.meshgrid(self.x, self.y)
        lon, lat = _to_ll.transform(X * 1000, Y * 1000)
        return lon, lat

    def cell(self, lon, lat):
        """Row and column of the cell holding each point, and whether it is inside the grid."""
        px, py = _to_xy.transform(np.asarray(lon, float), np.asarray(lat, float))
        ix = np.floor((px / 1000 - self.x0) / self.dx).astype(int)
        iy = np.floor((py / 1000 - self.y0) / self.dx).astype(int)
        inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy, ix, inside

    def flat(self, iy, ix):
        return np.asarray(iy) * self.nx + np.asarray(ix)

    def interior(self):
        """True away from the outer ring of cells, where the PDE is solved; the ring is the boundary."""
        m = np.zeros(self.shape, bool)
        m[1:-1, 1:-1] = True
        return m

    def coarsen(self, k):
        """The same domain at k times the spacing (used to make the tests fast)."""
        return Grid(self.x0, self.y0, self.nx // k, self.ny // k, self.dx * k)


def lonlat_to_xy(lon, lat):
    px, py = _to_xy.transform(np.asarray(lon, float), np.asarray(lat, float))
    return px / 1000, py / 1000


def block_mean(a, k):
    """Average k x k blocks: a fine field onto a grid k times coarser."""
    ny, nx = a.shape[-2] // k, a.shape[-1] // k
    a = a[..., :ny * k, :nx * k]
    return a.reshape(*a.shape[:-2], ny, k, nx, k).mean(axis=(-3, -1))


#-------- terrain derivatives ------------------------------------------------------------
def slope(elev_km, dx):
    """Slope in degrees from central differences."""
    gy, gx = np.gradient(elev_km, dx)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def tpi(elev_km, dx, radius_km=5.0):
    """Topographic position index: elevation minus the mean of its surroundings, in km.
    Negative in valley floors, positive on ridges."""
    size = 2 * int(round(radius_km / dx)) + 1
    return elev_km - ndimage.uniform_filter(elev_km, size=size, mode="nearest")


def distance_to_coast(water_fraction, sea, dx):
    """Distance in km from each cell to the nearest sea cell (0 on the sea itself)."""
    if not sea.any():
        return np.full(sea.shape, 300.0)
    return ndimage.distance_transform_edt(~sea) * dx


def static_features(grid, elev_km, fractions, sea):
    """Everything about a cell that does not change from day to day."""
    return dict(elevation=elev_km, slope=slope(elev_km, grid.dx), tpi=tpi(elev_km, grid.dx),
                coast=distance_to_coast(fractions[CLASSES.index("water")], sea, grid.dx))
