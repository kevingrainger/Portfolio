#-------- coral_crw.py ------------------------------------------------------------------
#-----------------------------------------------------------------------------
# The reef driven by the field's standard heat-stress measure, with corals that differ.
#
# Heat stress, as NOAA Coral Reef Watch defines it (Skirving et al. 2020):
#   HotSpot  HS(t)  = SST(t) - MMM            (MMM: the warmest month of the climatology)
#   DHW(t)          = (1/7) * sum over the last 84 days of HS, counting only HS >= 1 C
# in degree C-weeks. NOAA expects significant bleaching from about 4 DHW and severe
# bleaching with mortality from about 8.
#
# Corals differ. Each patch has its own tolerance D50 (the DHW at which it is as likely
# to bleach as not), drawn once from N(mu_D, sigma_D): within one reef population the
# most and least tolerant tenth of colonies differ by about 4-5 DHW (Humanes et al.
# 2022). In physics terms the uniform external field of the Potts model becomes a
# random field (quenched disorder). Tolerance is patchy: a smoothed random field, so
# neighbouring colonies are alike and whole sites can be tougher than others.
#
#   healthy patch:  energy of staying healthy rises by  s * (DHW - D50)  when positive
#   bleached patch: recovers once DHW has fallen below D50 and the water is no longer
#                   hot (HS < 1), helped by healthy neighbours
#   mortality:      bleached coral dies at a daily rate rising from zero at D50 + dD to
#                   m at D50 + dD + 4  (NOAA: mortality likely from about 8 DHW)
# Neighbour coupling J and the Metropolis noise beta are as in coral_model.Reef.
# Bleaching is reported as a share of LIVING coral, as surveys report it.

import numpy as np
from scipy.ndimage import gaussian_filter

from coral_model import ALGAE, BARE, BLEACHED, HEALTHY, Reef

MMM = 29.5454            # NOAA CRW, Scott to Ashmore Reefs virtual station


class CRWReef(Reef):
    def __init__(self, mu_D=6.0, sigma_D=2.0, s=0.5, dD=4.0, mmm=MMM, patch=3.0, **kw):
        kw.setdefault("grid", 60)
        super().__init__(**kw)
        self.mmm, self.s, self.dD = mmm, s, dD
        # tolerance varies colony to colony but is patchy: neighbouring colonies (same
        # species, same depth and flow) are alike. A smoothed Gaussian field, rescaled to
        # mean mu_D and standard deviation sigma_D.
        z = gaussian_filter(self.rng.normal(size=(self.n, self.n)), patch) if patch > 0 else self.rng.normal(size=(self.n, self.n))
        z = (z - z.mean()) / z.std()
        self.D50 = np.maximum(mu_D + sigma_D * z, 0.5)
        self.hs_hist = []                                  # last 84 days of counted HotSpots
        self.dhw = np.zeros((self.n, self.n))
        self.hs = np.zeros((self.n, self.n))

    #-------- heat: local SST, HotSpot, DHW on a rolling 84-day window -----------------
    def set_temperature(self, day_temp):
        noise = self.rng.normal(0, 0.15, (self.n, self.n))
        self.temp = gaussian_filter(day_temp + noise, sigma=1.5)
        self.hs = self.temp - self.mmm
        counted = np.where(self.hs >= 1.0, self.hs, 0.0)
        self.hs_hist.append(counted)
        if len(self.hs_hist) > 84:
            self.hs_hist.pop(0)
        self.dhw = np.sum(self.hs_hist, axis=0) / 7.0
        good = (self.dhw < self.D50) & (self.hs < 1.0)
        self.recovery_days = np.clip(np.where(good, self.recovery_days + 1, 0), 0, 100)

    #-------- energies ----------------------------------------------------------------------
    def _energy(self, state, f_healthy, f_bleached):
        excess = self.dhw - self.D50
        E = np.zeros(self.temp.shape)
        if state == HEALTHY:
            E -= 0.8 * self.J * 4 * f_healthy
            E -= 4.0
            E += np.where(excess > 0, self.s * excess * 4.0, 0.0)
        elif state == BLEACHED:
            E -= 0.3 * self.J * 4 * f_bleached
            E -= 1.5
            good = (excess < 0) & (self.hs < 1.0)
            R = self.recovery_days
            E += np.where(good, self.recovery * 1.5 + np.where(R > 5, self.recovery * np.minimum(R / 10.0, 2.0), 0)
                          + f_healthy * self.recovery, 0.0)
        else:
            E -= 0.5
            E -= np.where((f_healthy >= 0.5) & (excess < 0), 0.3 * f_healthy, 0)
        return E

    #-------- one Metropolis sweep: same move set, stress judged by DHW ----------------------
    def metropolis(self):
        c = self.coral
        fh, fb = self._fraction(HEALTHY), self._fraction(BLEACHED)
        r = self.rng.random
        tried = self.reef & (r(c.shape) < 1 - np.exp(-0.8))
        proposal = c.copy()
        stressed = self.dhw > self.D50 - 1.0                # bleaching needs accumulated heat, not one hot day
        h = c == HEALTHY
        proposal = np.where(h & stressed & (r(c.shape) < 0.5), BLEACHED, proposal)
        b = c == BLEACHED
        proposal = np.where(b & (r(c.shape) < 0.5), HEALTHY, proposal)
        k = c == BARE
        can_grow = (fh >= 0.5) & (self.dhw < self.D50) & (r(c.shape) < 0.1)
        proposal = np.where(k & can_grow, HEALTHY, proposal)
        change = tried & (proposal != c)
        if not change.any():
            return
        E = {st: self._energy(st, fh, fb) for st in (HEALTHY, BLEACHED, BARE)}
        e_old = np.select([c == HEALTHY, c == BLEACHED], [E[HEALTHY], E[BLEACHED]], E[BARE])
        e_new = np.select([proposal == HEALTHY, proposal == BLEACHED], [E[HEALTHY], E[BLEACHED]], E[BARE])
        dE = e_new - e_old + self.rng.normal(0, 0.02, c.shape)
        accept = change & ((dE <= 0) | (r(c.shape) < np.exp(-self.beta * np.maximum(dE, 0))))
        self.coral = np.where(accept, proposal, c)

    #-------- mortality above the tolerance, then algae and grazing as before ----------------
    def ecology(self):
        r = self.rng.random
        c = self.coral.copy()
        p_die = self.mortality * np.clip((self.dhw - self.D50 - self.dD) / 4.0, 0, 1)
        c = np.where((c == BLEACHED) & (r(c.shape) < p_die), BARE, c)
        if self.algae:
            f_algae = self._fraction(ALGAE)
            c = np.where((c == BARE) & self.reef & (r(c.shape) < 0.01 + 0.15 * f_algae), ALGAE, c)
            non_coral = np.mean(np.isin(c[self.reef], (BARE, ALGAE)))
            c = np.where((c == ALGAE) & (r(c.shape) < self.grazing / (1 + 4 * non_coral)), BARE, c)
        self.coral = c

    #-------- what a survey sees ------------------------------------------------------------
    def bleached_of_living(self):
        live = self.reef & np.isin(self.coral, (HEALTHY, BLEACHED))
        return np.mean(self.coral[live] == BLEACHED) if live.any() else np.nan

    def survey(self, n_sites, rng, size=6):
        """Bleached share of living coral in n_sites random size x size windows of reef.
        A window whose coral has all bleached or died since the start of the season is
        scored 100%; windows that never held coral are skipped."""
        out = []
        start = self.reef & self.coral0
        ii, jj = np.nonzero(start)
        for _ in range(50 * n_sites):
            if len(out) == n_sites:
                break
            k = rng.integers(len(ii))
            i0, j0 = max(ii[k] - size // 2, 0), max(jj[k] - size // 2, 0)
            win = self.coral[i0:i0 + size, j0:j0 + size]
            was = start[i0:i0 + size, j0:j0 + size]
            live = was & np.isin(win, (HEALTHY, BLEACHED))
            if live.sum() >= 5:
                out.append(np.mean(win[live] == BLEACHED))
            elif was.sum() >= 5:
                out.append(1.0)
        return np.array(out)

    def mark_start(self):
        """Remember which cells held living coral when the season began."""
        self.coral0 = np.isin(self.coral, (HEALTHY, BLEACHED))
