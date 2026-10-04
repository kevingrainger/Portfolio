#!/usr/bin/env python3
#-------- olive_model.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Xylella fastidiosa in an olive grove - a stochastic lattice SIR model.
#
# A recreation of Fierro, Liccardo & Porcelli (2019), "A lattice model to manage
# the vector and the infection of the Xylella fastidiosa on olive trees",
# Scientific Reports 9. Meadow spittlebugs (Philaenus spumarius) random-walk over
# a grid of shrub and olive-tree cells, pick the bacterium up from infected
# trees, and pass it on to healthy ones.
#
# This is the fast version of OliveTree_SIR_model.py (kept alongside, unchanged,
# for reference). The rules are the same. What changed:
#
#   speed      the original loops over every spittlebug in Python, and looks up
#              which tree a cell belongs to by scanning all 256 trees each time.
#              Here a lookup table answers that in one step, and each rule is
#              applied to every spittlebug at once with numpy. Same rules, same
#              order within a step; about a hundred times faster.
#
#   fixes      vectors could get trapped on the centre cell of a symptomatic tree
#              (no shrub next to it, so the "repel" step did nothing and movement
#              was skipped forever); tree-to-tree moves now tell the same tree
#              from a different one, so p_bb is actually used; the sprout phase is
#              set correctly on day 0; the sampling "compensation" that always
#              returned 1.0 is gone - tree_susceptibility is now a parameter.
#
#   new        a transmission log (which tree infected which, and when), an
#              annual_reset switch, frames for animation, and a seed so every run
#              can be reproduced.
#
# Cells are 1 m. Trees are 3 x 3 cells on a 5 m grid: 256 trees on 90 x 90 m.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SHRUB, TREE = 0, 1
SUSCEPTIBLE, INFECTED, SYMPTOMATIC, FELLED = 0, 1, 2, 3  # the original calls SYMPTOMATIC 'removed'
TENDER, HARD = 0, 1

STEPS_4 = np.array([(0, 1), (0, -1), (1, 0), (-1, 0)])   # (dx, dy) for a random step
STEPS_8 = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)])


class OliveGrove:
    def __init__(self, width=90, height=90, tree_spacing=5, tree_size=3,
                 vectors_per_hectare=1_000_000, sampling_rate=0.001,
                 time_steps_per_day=30, tree_susceptibility=1.0, vector_susceptibility=1.0,
                 annual_reset=True, symptom_delay_mean=730, symptom_delay_std=100,
                 frame_every=None, seed=None, legacy_tender_start=False,
                 felling_radius=None, detection_delay=None):
        self.rng = np.random.default_rng(seed)
        self.width, self.height = width, height
        self.tree_spacing, self.tree_size = tree_spacing, tree_size
        self.time_steps_per_day = time_steps_per_day
        self.annual_reset = annual_reset

        #-------- the grove, and the lookup table that replaces the 256-tree scan
        self.cell_type = np.full((height, width), SHRUB, dtype=np.uint8)
        self.tree_of_cell = np.full((height, width), -1, dtype=np.int32)   # which tree a cell belongs to
        self._plant_trees()
        self._build_escape_table()

        #-------- trees
        n = len(self.tree_centres)
        self.tree_state = np.full(n, SUSCEPTIBLE, dtype=np.int8)
        self.infection_day = np.full(n, np.nan)
        self.symptom_day = np.full(n, np.inf)

        #-------- control: fell every tree within felling_radius (metres) of a detected one.
        #Detection happens when symptoms show, or - with detection_delay set - that many
        #days after infection, as with lab testing of trees that still look healthy.
        self.felling_radius = felling_radius
        self.detection_delay = detection_delay
        self.detected = np.zeros(n, dtype=bool)
        if felling_radius is not None:
            d = self.tree_centres[:, None, :] - self.tree_centres[None, :, :]
            self.tree_distance = np.sqrt((d ** 2).sum(axis=-1))

        #-------- vectors (spittlebugs). The paper uses a million per hectare; sampling
        #scales that down so it runs, and is the model's biggest approximation.
        area_hectares = width * height / 10_000
        self.actual_vectors = int(vectors_per_hectare * area_hectares)
        self.n_vectors = max(1, int(self.actual_vectors * sampling_rate))
        self.vector_x = np.zeros(self.n_vectors, dtype=np.int32)
        self.vector_y = np.zeros(self.n_vectors, dtype=np.int32)
        self.vector_infected = np.zeros(self.n_vectors, dtype=bool)
        self.vector_source = np.full(self.n_vectors, -1, dtype=np.int32)   # tree it caught it from

        #-------- parameters from the paper
        self.p_tt = 0.3                 # within a tree (twig to twig)
        self.p_bb = 0.15                # tree to a different tree (branch to branch)
        self.p_ss = 0.5                 # shrub to shrub
        self.tree_susceptibility = tree_susceptibility
        self.vector_susceptibility = vector_susceptibility
        self.xylem_propagation_rate = 0.167   # cm/day
        self.twig_length = 15                 # cm
        self.symptom_delay_mean = symptom_delay_mean
        self.symptom_delay_std = symptom_delay_std

        #-------- time
        self.time_step = 0
        self.day = 0
        self.update_growth_phase()      # the original started tender on day 0, which is hard season

        # legacy_tender_start reproduces that: tender-sprout movement for the first day.
        # Only for checking this version against the original - it turns out not to be
        # harmless, because the seed carrier can infect a tree on day 0, a full tender
        # season early.
        if legacy_tender_start:
            self.sprout_state = TENDER
            self.p_ts, self.p_st = 0.005, 1.0

        #-------- records
        self.history = []               # one dict per day
        self.transmission_log = []      # (day, source_tree, target_tree); source -1 = the seed
        self.frame_every = frame_every
        self.frames = []

    #-------- setup ---------------------------------------------------------------
    def _plant_trees(self):
        centres = []
        s, k = self.tree_spacing, self.tree_size
        for y in range(s, self.height - s, s):
            for x in range(s, self.width - s, s):
                if y + k < self.height and x + k < self.width:
                    self.cell_type[y:y + k, x:x + k] = TREE
                    self.tree_of_cell[y:y + k, x:x + k] = len(centres)
                    centres.append((y + k // 2, x + k // 2))
        self.tree_centres = np.array(centres)

    # Where a vector on a symptomatic tree is pushed to: the first shrub among its
    # eight neighbours, in the same order the original checks them. The centre
    # cell of a tree has no shrub neighbour - marked -1, handled in movement.
    def _build_escape_table(self):
        self.escape_y = np.full((self.height, self.width), -1, dtype=np.int32)
        self.escape_x = np.full((self.height, self.width), -1, dtype=np.int32)
        for y, x in np.argwhere(self.cell_type == TREE):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= nx < self.width and 0 <= ny < self.height and self.cell_type[ny, nx] == SHRUB:
                        self.escape_y[y, x], self.escape_x[y, x] = ny, nx
                        break
                else:
                    continue
                break

    def initialize_vectors(self, initial_infected=1):
        shrubs = np.argwhere(self.cell_type == SHRUB)
        pick = self.rng.integers(0, len(shrubs), size=self.n_vectors)   # several per shrub allowed
        self.vector_y[:], self.vector_x[:] = shrubs[pick, 0], shrubs[pick, 1]
        self.vector_infected[:] = False
        self.vector_source[:] = -1
        if initial_infected > 0:
            seeds = self.rng.choice(self.n_vectors, size=min(initial_infected, self.n_vectors), replace=False)
            self.vector_infected[seeds] = True

    #-------- seasons ----------------------------------------------------------------
    # Tender sprouts (days 90-185) pull vectors onto the trees; hard sprouts push
    # them off and keep them off.
    def update_growth_phase(self):
        if 90 <= self.day % 365 <= 185:
            self.sprout_state = TENDER
            self.p_ts, self.p_st = 0.005, 1.0
        else:
            self.sprout_state = HARD
            self.p_ts, self.p_st = 1.0, 0.0

    #-------- movement -------------------------------------------------------------
    def vector_movement(self):
        x, y = self.vector_x, self.vector_y
        tree_here = self.tree_of_cell[y, x]
        on_tree = tree_here >= 0
        on_symptomatic = on_tree & (self.tree_state[np.maximum(tree_here, 0)] >= SYMPTOMATIC)

        #symptomatic trees repel: push to the escape shrub; from the centre cell, step
        #to a random neighbouring cell instead (the original left these vectors stuck)
        repelled = np.where(on_symptomatic)[0]
        if len(repelled):
            ey, ex = self.escape_y[y[repelled], x[repelled]], self.escape_x[y[repelled], x[repelled]]
            has_exit = ey >= 0
            y[repelled[has_exit]], x[repelled[has_exit]] = ey[has_exit], ex[has_exit]
            stuck = repelled[~has_exit]
            if len(stuck):
                step = STEPS_8[self.rng.integers(0, 8, size=len(stuck))]
                x[stuck] += step[:, 0]
                y[stuck] += step[:, 1]

        #everyone else: try a step half the time
        movers = np.where(~on_symptomatic & (self.rng.random(self.n_vectors) < 0.5))[0]
        step = STEPS_4[self.rng.integers(0, 4, size=len(movers))]
        nx, ny = x[movers] + step[:, 0], y[movers] + step[:, 1]
        inside = (nx >= 0) & (nx < self.width) & (ny >= 0) & (ny < self.height)
        movers, nx, ny = movers[inside], nx[inside], ny[inside]

        from_tree = self.tree_of_cell[y[movers], x[movers]]
        to_tree = self.tree_of_cell[ny, nx]
        blocked = (to_tree >= 0) & (self.tree_state[np.maximum(to_tree, 0)] >= SYMPTOMATIC)

        p = np.empty(len(movers))
        shrub_from, shrub_to = from_tree < 0, to_tree < 0
        p[shrub_from & shrub_to] = self.p_ss
        p[shrub_from & ~shrub_to] = self.p_st
        p[~shrub_from & shrub_to] = self.p_ts
        tree_tree = ~shrub_from & ~shrub_to
        p[tree_tree] = np.where(from_tree[tree_tree] == to_tree[tree_tree], self.p_tt, self.p_bb)

        go = ~blocked & (self.rng.random(len(movers)) < p)
        x[movers[go]], y[movers[go]] = nx[go], ny[go]

    #-------- transmission ----------------------------------------------------------
    def transmission(self):
        tree_here = self.tree_of_cell[self.vector_y, self.vector_x]
        on_tree = tree_here >= 0
        state = np.where(on_tree, self.tree_state[np.maximum(tree_here, 0)], -1)

        #infected vector on a healthy tree -> the tree is infected
        hit = np.where(self.vector_infected & (state == SUSCEPTIBLE)
                       & (self.rng.random(self.n_vectors) < self.tree_susceptibility))[0]
        if len(hit):
            targets, first = np.unique(tree_here[hit], return_index=True)   # one infection per tree
            sources = self.vector_source[hit[first]]
            self.tree_state[targets] = INFECTED
            self.infection_day[targets] = self.day
            delay = np.maximum(0, self.rng.normal(self.symptom_delay_mean, self.symptom_delay_std, len(targets)))
            self.symptom_day[targets] = self.day + delay
            self.transmission_log.extend(zip([self.day] * len(targets), sources.tolist(), targets.tolist()))
            self.new_infections_today += len(targets)

        #healthy vector on an infected tree -> the vector picks it up, more likely
        #the longer the tree has been colonised (full after about 90 days)
        feeding = np.where(~self.vector_infected & (state == INFECTED))[0]
        if len(feeding):
            trees = tree_here[feeding]
            days_infected = self.day - self.infection_day[trees]
            colonisation = np.minimum(1.0, days_infected * self.xylem_propagation_rate / self.twig_length)
            prob = self.vector_susceptibility * np.where(days_infected < 90, colonisation, 1.0)
            caught = feeding[self.rng.random(len(feeding)) < prob]
            self.vector_infected[caught] = True
            self.vector_source[caught] = self.tree_of_cell[self.vector_y[caught], self.vector_x[caught]]

    #-------- daily bookkeeping -----------------------------------------------------
    def update_disease_state(self):
        turning = (self.tree_state == INFECTED) & (self.day >= self.symptom_day)
        self.tree_state[turning] = SYMPTOMATIC

        if self.felling_radius is not None:
            carrying = (self.tree_state == INFECTED) | (self.tree_state == SYMPTOMATIC)
            if self.detection_delay is None:
                found = carrying & (self.tree_state == SYMPTOMATIC) & ~self.detected
            else:
                found = carrying & (self.day >= self.infection_day + self.detection_delay) & ~self.detected
            for t in np.flatnonzero(found):
                self.detected[t] = True
                self.tree_state[self.tree_distance[t] <= self.felling_radius] = FELLED

    # Each spring a new generation of adults emerges, uninfected - nymphs do not
    # carry the bacterium through moulting. The original resets every vector on
    # day 120; the switch lets that assumption be tested.
    def _annual_vector_reset(self):
        self.initialize_vectors(initial_infected=0)

    def _record(self):
        n = len(self.tree_state)
        self.history.append(dict(
            day=self.day,
            S=np.sum(self.tree_state == SUSCEPTIBLE) / n,
            I=np.sum(self.tree_state == INFECTED) / n,
            R=np.sum(self.tree_state == SYMPTOMATIC) / n,
            F=np.sum(self.tree_state == FELLED) / n,
            new_infections=self.new_infections_today,
            vectors_infected=self.vector_infected.mean(),
        ))
        if self.frame_every and self.day % self.frame_every == 0:
            self.frames.append(dict(day=self.day, sprout=self.sprout_state,
                                    tree_state=self.tree_state.copy(),
                                    x=self.vector_x.copy(), y=self.vector_y.copy(),
                                    infected=self.vector_infected.copy()))

    #-------- one time step, in the same order as the original -----------------------
    def step(self):
        self.time_step += 1
        if self.time_step % self.time_steps_per_day == 0:
            self.day += 1
            self.update_growth_phase()
            self.update_disease_state()
            self._record()
            self.new_infections_today = 0
            if self.annual_reset and self.day % 365 == 120:
                self._annual_vector_reset()
        self.vector_movement()
        self.transmission()

    def run(self, days, initial_infected=1, verbose=False):
        self.new_infections_today = 0
        if self.time_step == 0:
            self.initialize_vectors(initial_infected)
            if self.frame_every:
                self._record()
                self.history.pop()      # frame of day 0 only
        for d in range(days):
            for _ in range(self.time_steps_per_day):
                self.step()
            if verbose and d % 90 == 0 and self.history:
                h = self.history[-1]
                print(f"day {self.day:4d}: S {h['S']:.1%}  I {h['I']:.1%}  symptomatic {h['R']:.1%}")
        return self

    #-------- results ---------------------------------------------------------------
    def series(self, key):
        return np.array([h[key] for h in self.history])

    def ever_infected(self):
        return np.mean(np.isfinite(self.infection_day))

    #-------- the grove, drawn -------------------------------------------------------
    def grove_image(self, tree_state=None):
        tree_state = self.tree_state if tree_state is None else tree_state
        img = np.zeros((self.height, self.width, 3))
        img[self.cell_type == SHRUB] = [0.85, 0.95, 0.85]
        colours = {SUSCEPTIBLE: [0.2, 0.6, 0.2], INFECTED: [0.9, 0.85, 0.2], SYMPTOMATIC: [0.8, 0.2, 0.2],
                   FELLED: [0.45, 0.33, 0.25]}
        for t, (cy, cx) in enumerate(self.tree_centres):
            img[cy - 1:cy + 2, cx - 1:cx + 2] = colours[int(tree_state[t])]
        return img

    def visualize(self, show_vectors=True):
        fig = plt.figure(figsize=(16, 8.5))
        grid = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
        ax_grove = fig.add_subplot(grid[:, :2])
        ax_sir = fig.add_subplot(grid[0, 2])
        ax_vec = fig.add_subplot(grid[1, 2])

        ax_grove.imshow(self.grove_image(), interpolation='nearest')
        if show_vectors:
            inf = self.vector_infected
            ax_grove.scatter(self.vector_x[~inf], self.vector_y[~inf], s=3, c='0.55', label='spittlebug')
            ax_grove.scatter(self.vector_x[inf], self.vector_y[inf], s=8, c='k', label='carrying Xylella')
        counts = np.bincount(self.tree_state, minlength=4)
        ax_grove.set_title(f"Day {self.day}:  {counts[0]} healthy, {counts[1]} infected, "
                           f"{counts[2]} symptomatic  (of {len(self.tree_state)} trees)", fontsize=11)
        handles = [patches.Patch(color=[0.2, 0.6, 0.2], label='healthy'),
                   patches.Patch(color=[0.9, 0.85, 0.2], label='infected, no symptoms yet'),
                   patches.Patch(color=[0.8, 0.2, 0.2], label='symptomatic')]
        ax_grove.legend(handles=handles + ax_grove.get_legend_handles_labels()[0], loc='upper center',
                        bbox_to_anchor=(0.5, -0.01), ncol=5, fontsize=8, frameon=False)
        ax_grove.set_xticks([])
        ax_grove.set_yticks([])

        days = self.series('day')
        ax_sir.plot(days, self.series('S'), 'g-', lw=2, label='healthy')
        ax_sir.plot(days, self.series('I'), color='goldenrod', lw=2, label='infected')
        ax_sir.plot(days, self.series('R'), 'r-', lw=2, label='symptomatic')
        _shade_seasons(ax_sir, days.max())
        ax_sir.set_xlabel('day')
        ax_sir.set_ylabel('fraction of trees')
        ax_sir.legend(fontsize=8)

        ax_vec.plot(days, self.series('vectors_infected'), 'k-', lw=1.5)
        _shade_seasons(ax_vec, days.max())
        ax_vec.set_xlabel('day')
        ax_vec.set_ylabel('fraction of spittlebugs infected')
        return fig


# Tender-sprout season shaded green, only over the days actually simulated
# (the original shaded a full extra year past the end of the data).
def _shade_seasons(ax, last_day):
    for year in range(int(last_day // 365) + 1):
        start, end = year * 365 + 90, min(year * 365 + 185, last_day)
        if start < last_day:
            ax.axvspan(start, end, color='green', alpha=0.1, lw=0)
    ax.set_xlim(0, last_day)


#-------- many runs at once ----------------------------------------------------------
# One run is one roll of the dice - the same grove and the same seed tree can give
# a contained outbreak or one that takes the whole grove. Every claim needs an
# ensemble, and seeds are independent, so they run in parallel across cores.
def _run_one(job):
    kwargs, days, seed = job
    g = OliveGrove(seed=seed, **kwargs).run(days)
    return dict(seed=seed, day=g.series('day'), S=g.series('S'), I=g.series('I'),
                F=g.series('F'), infection_day=g.infection_day,
                R=g.series('R'), vectors=g.series('vectors_infected'),
                ever_infected=g.ever_infected(), log=g.transmission_log,
                tree_centres=g.tree_centres)


def run_ensemble(days, seeds, processes=None, **kwargs):
    from multiprocessing import Pool
    jobs = [(kwargs, days, s) for s in seeds]
    with Pool(processes) as pool:
        return pool.map(_run_one, jobs)


#-------- animation ----------------------------------------------------------------
def animate(grove, path, fps=12):
    from matplotlib.animation import FuncAnimation, PillowWriter
    frames = grove.frames
    fig, ax = plt.subplots(figsize=(4.8, 5.2), dpi=72)
    im = ax.imshow(grove.grove_image(frames[0]['tree_state']), interpolation='nearest')
    healthy = ax.scatter([], [], s=3, c='0.45')
    carrying = ax.scatter([], [], s=8, c='k')
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title('')

    def draw(i):
        f = frames[i]
        img = grove.grove_image(f['tree_state'])
        if f['sprout'] == TENDER:                       # spring: tint the shrub layer
            shrub = grove.cell_type == SHRUB
            img[shrub] = [0.80, 0.92, 0.70]
        im.set_data(img)
        inf = f['infected']
        healthy.set_offsets(np.c_[f['x'][~inf], f['y'][~inf]])
        carrying.set_offsets(np.c_[f['x'][inf], f['y'][inf]])
        counts = np.bincount(f['tree_state'], minlength=4)
        season = 'tender sprouts - spittlebugs on the trees' if f['sprout'] == TENDER else 'hard sprouts'
        title.set_text(f"day {f['day']}  ({season})\n"
                       f"{counts[0]} healthy  {counts[1]} infected  {counts[2]} symptomatic")
        return im, healthy, carrying, title

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


if __name__ == "__main__":
    grove = OliveGrove(sampling_rate=0.001, time_steps_per_day=30, seed=1, frame_every=5)
    grove.run(365 * 3, verbose=True)
    grove.visualize()
    plt.show()
