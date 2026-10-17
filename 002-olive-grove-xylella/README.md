# The bug that kills olive trees, one spittlebug at a time

**Simulating *Xylella fastidiosa* in an olive grove shows the outbreak sitting on a knife-edge, and felling trees failing to push it off.**

![Unmanaged grove against vector control plus felling](figures/hero.png)

*Xylella* has killed millions of olive trees in southern Italy. It's carried by meadow spittlebugs, and symptoms take about two years to show. I rebuilt a published lattice model of the grove (Fierro et al., 2019), where every spittlebug walks, feeds and passes the bacterium on, then made it about 95× faster and asked what actually stops it.

![The grove over three years](figures/fig0_animation.gif)

- **Outbreaks are all-or-nothing.** 39 of 40 take the whole grove, with half the trees infected in about 2.3 years. One never starts.
- **A one-day slip moved a two-year answer by 38%.** The original code put the grove in the wrong season on day 0, which gave every outbreak a head start.
- **There's a sharp threshold.** A spittlebug stays on a tree for about 18 days per visit. Below about 16–29% chance of infection per visit, outbreaks die out; above it, they take the grove.
- **Felling never paid.** No felling radius or detection speed left more trees standing: too late above the threshold, unnecessary below it. Reducing spittlebug contact did the work.
- **Planting further apart works only with the ground cover managed.** Spittlebugs live in the grass between trees. With their number per tree held fixed, spreading a 16 x 16 grove from 4 to 10 cells cuts the share of trees infected in three years from 97% to 14%. The outbreak stops taking the grove between 6 and 8 cells apart: the share of groves more than half infected falls from 65% at 6 cells to 20% at 7 and none at 8. But with their density per hectare fixed, wider spacing adds as many spittlebugs as distance takes away, and 69–85% of trees are infected at every spacing (`spacing_threshold.py`).

| | |
|:-:|:-:|
| ![Threshold](figures/fig3_threshold.png) | ![Felling grid](figures/fig6_felling_grid.png) |
| ![Ensemble](figures/fig2_ensemble.png) | ![Vector control](figures/fig7_vector_control.png) |
| ![Planting spacing](figures/fig_spacing_threshold.png) | |

**Built:** a vectorised rewrite (`olive_model.py`), checked against my original model across 12 + 200 runs; four bug fixes; a transmission log; felling and detection policies; a planting-spacing sweep; and parallel ensembles.

`olive_grove_01_model.ipynb` · `olive_grove_02_felling.ipynb` · Python, NumPy, multiprocessing

*A side project, built for fun. These are model results, not policy advice.*
