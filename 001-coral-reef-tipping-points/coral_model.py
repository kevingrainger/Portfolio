#!/usr/bin/env python3
#-------- coral_model.py ------------------------------------------------------------
#-----------------------------------------------------------------------------
# A coral reef as a Potts model - the fast version of Coral_Potts_Model_original.py.
#
# Each reef cell is healthy coral, bleached coral or bare rock. Heat stress builds
# up as degree heating days (slowly while it is hot, quickly forgotten once it
# cools); neighbours pull each other towards the same state; a Metropolis rule
# decides each change. The original applied every accepted change at the end of a
# step, from the state at the start of it - so all cells can be updated at once,
# exactly, with numpy. That is the speed-up; the rules are the original's.
#
# Changes, each a switch:
#   nudge          the original steers bleaching toward the observed record (the
#                  "target recovery push"). Fine for fitting, fatal for prediction:
#                  off unless asked for.
#   neighbourhood  'none' | 'nn1' (4 neighbours, the original) | 'nn2' (8) | 'nn3' (12)
#                  | 'meanfield' (every cell feels the whole reef)
#   coupling is divided by the number of neighbours, so a longer range changes the
#   REACH of the interaction, not its total strength - otherwise "more neighbours"
#   and "stronger coupling" could not be told apart. nn1 reproduces the original.
#   forcing        the daily record, or a constant temperature plus an anomaly, or any
#                  schedule - which is what the hysteresis and tipping runs need.
#
# Two additions (the proposal's section 2.3), because the original cannot tip:
#   mortality      in the original, coral can never die - its death rules need heat
#                  stress above 8 or 10, and heat stress is capped at 8. So bleached
#                  coral always recovers once it cools, and nothing can get stuck.
#                  Here prolonged heat stress kills bleached coral.
#   macroalgae     dead coral is colonised by algae, which coral larvae cannot settle
#                  on; grazers (fish, urchins) clear it. Grazing is DILUTED - the same
#                  grazers spread over more algae as coral dies (Mumby et al. 2007) -
#                  which is the feedback that makes real reefs bistable.
# The calibration record never reaches lethal heat stress, so neither addition
# changes the fit to the observed bleaching event.

import numpy as np
from scipy.ndimage import gaussian_filter, convolve

OCEAN, ROCK = 0, 1                      # substrate
BARE, HEALTHY, BLEACHED, ALGAE = 0, 2, 3, 4   # state on rock (original codes, plus macroalgae)

OFFSETS = {
    "nn1": [(-1, 0), (1, 0), (0, -1), (0, 1)],
    "nn2": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)],
    "nn3": [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1),
            (-2, 0), (2, 0), (0, -2), (0, 2)],
}


class Reef:
    def __init__(self, grid=100, bleaching_thresh=30.0, stress_multiplier=1.0, recovery_bonus=0.3,
                 coupling_strength=0.5, temp_sensitivity=0.8, beta=0.15, dhd_decay=0.85,
                 neighbourhood="nn1", nudge=False, target=None, seed=None,
                 mortality=0.03, lethal_dhd=3.0, algae=True, grazing=0.06):
        self.rng = np.random.default_rng(seed)
        self.n = grid
        self.thresh = bleaching_thresh
        self.stress = stress_multiplier
        self.recovery = recovery_bonus
        self.J = coupling_strength
        self.sens = temp_sensitivity
        self.beta = beta
        self.dhd_decay = dhd_decay
        self.neighbourhood = neighbourhood
        self.nudge = nudge
        self.target = target                    # observed bleached fraction by day, for the nudge
        self.day = 0
        self.mortality, self.lethal_dhd = mortality, lethal_dhd
        self.algae, self.grazing = algae, grazing

        self.substrate, self.coral = self._initial_reef()
        self.reef = self.substrate == ROCK
        self.temp = np.full((grid, grid), bleaching_thresh - 1.0)
        self.dhd = np.zeros((grid, grid))
        self.recovery_days = np.zeros((grid, grid))
        if neighbourhood in OFFSETS:
            kernel = np.zeros((5, 5))
            for dx, dy in OFFSETS[neighbourhood]:
                kernel[2 + dx, 2 + dy] = 1
            self.kernel = kernel
            self.z = len(OFFSETS[neighbourhood])

    #-------- the reef: a roughly circular outcrop, 75% healthy cover ----------------
    def _initial_reef(self):
        n = self.n
        i, j = np.indices((n, n))
        dist = np.hypot(i - n // 2, j - n // 2)
        rock = dist < (n // 3) * (0.8 + 0.3 * self.rng.random((n, n)))
        substrate = np.where(rock, ROCK, OCEAN)
        coral = np.where(rock & (self.rng.random((n, n)) < 0.75), HEALTHY, BARE)
        return substrate, coral

    #-------- fraction of each cell's neighbours in a given state -------------------
    def _fraction(self, state):
        same = (self.coral == state) & self.reef
        if self.neighbourhood == "none":
            return np.zeros(same.shape)
        if self.neighbourhood == "meanfield":
            return np.full(same.shape, same.sum() / self.reef.sum())
        return convolve(same.astype(float), self.kernel, mode="wrap") / self.z

    #-------- energies, all cells at once (the original's terms, per neighbour x4) -----
    def _energy(self, state, f_healthy, f_bleached):
        T, D, R = self.temp, self.dhd, self.recovery_days
        E = np.zeros(T.shape)
        if state == HEALTHY:
            E -= 0.8 * self.J * 4 * f_healthy
            E -= 4.0
            E += np.where(T > self.thresh, self.stress * (T - self.thresh) * self.sens, 0)
            E += np.where(D > 3.0, self.stress * np.log1p(np.maximum(D - 3.0, 0)) * 0.5, 0)
        elif state == BLEACHED:
            E -= 0.3 * self.J * 4 * f_bleached
            E -= 1.5
            good = (T <= self.thresh + 2.0) & (D <= 4.0)
            rec = np.where(good, self.recovery * 1.5
                           + np.where(R > 5, self.recovery * np.minimum(R / 10.0, 2.0), 0)
                           + np.where(f_healthy > 0, f_healthy * self.recovery, 0), 0)
            very = (T <= self.thresh + 0.5) & (D <= 1.0)
            rec += np.where(very, self.recovery * 2.0 + np.where(R >= 3, self.recovery * 1.5, 0), 0)
            if self.nudge and self._target_now() < 0.1:
                rec += self.recovery * 2.0 + np.where(T <= self.thresh + 1.0, self.recovery * 1.5, 0)
            E += rec
            E -= np.where((T > self.thresh + 6.0) & (D > 8.0), 0.5, 0)
        else:  # bare rock
            E -= 0.5
            E -= np.where((f_healthy >= 0.5) & (T <= self.thresh + 1.0), 0.3 * f_healthy, 0)
        return E

    def _target_now(self):
        if self.target is None or self.day == 0:
            return 1.0
        return self.target[min(self.day - 1, len(self.target) - 1)]

    #-------- temperature and heat stress --------------------------------------------
    def set_temperature(self, day_temp):
        noise = self.rng.normal(0, 0.15, (self.n, self.n))
        self.temp = gaussian_filter(day_temp + noise, sigma=1.5)
        excess = np.maximum(self.temp - self.thresh, 0)
        decay = np.where(excess > 0, self.dhd_decay, 0.35)      # remembered while hot, forgotten once cool
        self.dhd = np.clip(self.dhd * decay + excess * 0.2, 0, 8)
        good = (self.temp <= self.thresh + 1.0) & (self.dhd <= 2.0)
        self.recovery_days = np.clip(np.where(good, self.recovery_days + 1, 0), 0, 100)

    #-------- one Metropolis step for every cell at once --------------------------------
    def metropolis(self):
        c = self.coral
        fh, fb = self._fraction(HEALTHY), self._fraction(BLEACHED)
        T, D = self.temp, self.dhd
        r = self.rng.random

        #which cells are tried this step: the original made 0.8 N random draws with
        #replacement, which reaches about 55% of cells
        tried = self.reef & (r(c.shape) < 1 - np.exp(-0.8))

        #proposal: one of the allowed transitions, chosen uniformly (as the original)
        proposal = c.copy()
        h = (c == HEALTHY)
        stressed = (T > self.thresh) | (D > 2.0)
        extreme = (T > self.thresh + 8.0) & (D > 10.0)
        u = r(c.shape)
        n_opt = 1 + stressed.astype(int) + extreme.astype(int)
        choice = np.floor(u * n_opt).astype(int)                 # 0 = stay
        proposal = np.where(h & (choice == 1) & stressed, BLEACHED, proposal)
        proposal = np.where(h & (choice == 1) & ~stressed & extreme, BARE, proposal)
        proposal = np.where(h & (choice == 2), BARE, proposal)

        b = (c == BLEACHED)
        severe = (D > 8.0) & (T > self.thresh + 6.0)
        n_opt_b = 2 + severe.astype(int)
        choice_b = np.floor(r(c.shape) * n_opt_b).astype(int)
        proposal = np.where(b & (choice_b == 1), HEALTHY, proposal)
        proposal = np.where(b & (choice_b == 2), BARE, proposal)

        k = (c == BARE)
        can_grow = (fh >= 0.5) & (T <= self.thresh + 1.0) & (r(c.shape) < 0.1)
        choice_k = np.floor(r(c.shape) * (1 + can_grow)).astype(int)
        proposal = np.where(k & can_grow & (choice_k == 1), HEALTHY, proposal)

        change = tried & (proposal != c)
        if not change.any():
            return

        E = {s: self._energy(s, fh, fb) for s in (HEALTHY, BLEACHED, BARE)}
        e_old = np.select([c == HEALTHY, c == BLEACHED], [E[HEALTHY], E[BLEACHED]], E[BARE])
        e_new = np.select([proposal == HEALTHY, proposal == BLEACHED], [E[HEALTHY], E[BLEACHED]], E[BARE])
        dE = e_new - e_old

        #the original's general recovery bias for bleached -> healthy in good conditions
        recovering = (c == BLEACHED) & (proposal == HEALTHY)
        dE -= np.where(recovering & (T <= self.thresh + 2.0) & (D <= 4.0), 0.3, 0)
        if self.nudge and self._target_now() < 0.1:
            dE -= np.where(recovering, 0.8 + np.where((T <= self.thresh + 1.0) & (D <= 2.0), 0.5, 0), 0)
        dE += self.rng.normal(0, 0.02, c.shape)

        accept = change & ((dE <= 0) | (r(c.shape) < np.exp(-self.beta * np.maximum(dE, 0))))
        self.coral = np.where(accept, proposal, c)

    #-------- death, algae and grazing ---------------------------------------------------
    def ecology(self):
        r = self.rng.random
        c = self.coral.copy()
        #prolonged heat stress kills bleached coral, faster the longer it lasts
        p_die = self.mortality * np.clip((self.dhd - self.lethal_dhd) / (8 - self.lethal_dhd), 0, 1)
        c = np.where((c == BLEACHED) & (r(c.shape) < p_die), BARE, c)
        if self.algae:
            f_algae = self._fraction(ALGAE)
            #algae colonise bare rock, faster next to existing algae
            c = np.where((c == BARE) & self.reef & (r(c.shape) < 0.01 + 0.15 * f_algae), ALGAE, c)
            #grazers clear algae; the same grazers are spread over all non-coral area
            non_coral = np.mean(np.isin(c[self.reef], (BARE, ALGAE)))
            p_graze = self.grazing / (1 + 4 * non_coral)
            c = np.where((c == ALGAE) & (r(c.shape) < p_graze), BARE, c)
        self.coral = c

    def step(self, day_temp):
        self.set_temperature(day_temp)
        self.metropolis()
        self.ecology()
        self.day += 1

    #-------- what is measured ---------------------------------------------------------
    def cover(self):
        """Healthy coral as a fraction of the reef - the order parameter."""
        return np.mean(self.coral[self.reef] == HEALTHY)

    def bleached(self):
        return np.mean(self.coral[self.reef] == BLEACHED)

    def healthy_map(self):
        return np.where(self.reef, (self.coral == HEALTHY).astype(float), np.nan)

    def image(self):
        img = np.zeros((self.n, self.n, 3))
        img[self.substrate == OCEAN] = [0.08, 0.22, 0.45]
        img[self.reef & (self.coral == BARE)] = [0.55, 0.55, 0.55]
        img[self.coral == HEALTHY] = [0.95, 0.45, 0.35]           # living coral: coral-coloured
        img[self.coral == BLEACHED] = [0.97, 0.95, 0.88]          # bleached: white
        img[self.coral == ALGAE] = [0.30, 0.50, 0.20]             # macroalgae: dull green
        return img

    def algae_cover(self):
        return np.mean(self.coral[self.reef] == ALGAE)


#-------- experiments ----------------------------------------------------------------
def run_record(temps, seed=0, **kwargs):
    reef = Reef(seed=seed, **kwargs)
    healthy, bleached = [], []
    for T in temps:
        reef.step(T)
        healthy.append(reef.cover())
        bleached.append(reef.bleached())
    return np.array(healthy), np.array(bleached), reef


def hysteresis(anomalies, sweeps=60, base=None, seed=0, **kwargs):
    """Warm the reef step by step, then cool it back, holding each temperature for
    `sweeps` days. Returns cover on the way up and on the way down."""
    reef = Reef(seed=seed, **kwargs)
    base = reef.thresh - 1.0 if base is None else base
    up, down = [], []
    for a in anomalies:
        for _ in range(sweeps):
            reef.step(base + a)
        up.append(reef.cover())
    for a in anomalies[::-1]:
        for _ in range(sweeps):
            reef.step(base + a)
        down.append(reef.cover())
    return np.array(up), np.array(down[::-1])


def ramp(peak_anomaly, ramp_days, hold_days=120, recover_days=200, base=None, seed=0, record_maps=False, **kwargs):
    """Warm linearly to a peak over ramp_days, hold, then cool back. Returns cover through time."""
    reef = Reef(seed=seed, **kwargs)
    base = reef.thresh - 1.0 if base is None else base
    schedule = np.concatenate([np.linspace(0, peak_anomaly, ramp_days), np.full(hold_days, peak_anomaly),
                               np.zeros(recover_days)])
    cover, maps = [], []
    for a in schedule:
        reef.step(base + a)
        cover.append(reef.cover())
        if record_maps:
            maps.append(reef.healthy_map())
    return schedule, np.array(cover), (np.array(maps) if record_maps else None)
