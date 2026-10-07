#-------- ml.py ------------------------------------------------------------------------
#-----------------------------------------------------------------------------
# XGBoost, used in two ways.
#
#   M3  learns the missing-physics field q. The anchored solve yields, each day, the
#       heating q the equation needed in order to agree with the stations. If q is
#       systematic - more over towns, less under trees, different in the mountains -
#       a model can predict it from land cover and terrain where there are no stations.
#       That prediction q_hat becomes the prior for the next anchored solve, which still
#       returns every station exactly.
#
#   B3  the plain alternative: learn the station-minus-ERA5 difference directly from the
#       same covariates, with no equation in between.
#
# Chen & Guestrin (2016) is the XGBoost reference.

import numpy as np
from xgboost import XGBRegressor

from .grid import CLASSES

FEATURES = CLASSES + ["elevation", "slope", "tpi", "coast", "lst_excess", "wind", "day_of_year"]


def cell_features(ds, d, cells=None, fields=None):
    """Covariates for cells on day d: land-cover fractions, terrain, distance to coast,
    how much hotter the surface is than ERA5-Land's air, wind speed, day of year."""
    f = ds.day(d) if fields is None else fields
    pick = (lambda a: a.ravel()) if cells is None else (lambda a: a.ravel()[cells])
    cols = [pick(ds.frac[i]) for i in range(len(CLASSES))]
    cols += [pick(ds.features[k]) for k in ("elevation", "slope", "tpi", "coast")]
    cols += [pick(f["theta_s"] - f["theta_E"]), pick(np.hypot(f["u"], f["v"]))]
    cols.append(np.full(len(cols[0]), ds.days[d].dayofyear, float))
    return np.column_stack(cols)


def sample_by_footprint(weight, n, rng, allowed):
    """Draw cells with probability proportional to total footprint strength: q is only
    known where some station can see it, so that is where the training examples come from."""
    p = np.where(allowed, weight, 0.0)
    p = p / p.sum()
    return rng.choice(len(p), size=n, replace=True, p=p)


def new_model(n_estimators=300, max_depth=5, n_jobs=1):
    return XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, min_child_weight=5, n_jobs=n_jobs, random_state=0)


def predict_field(model, ds, d, fields=None):
    """q_hat on the whole grid for day d."""
    return model.predict(cell_features(ds, d, fields=fields)).reshape(ds.grid.shape)
