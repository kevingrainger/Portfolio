# Coral reef lattice model
![The reef through a heatwave and back](figures/cover.gif)

## Overview

Lattice models (Ising and Potts models) were of big interest to me in college, as was the topic of mathematical ecology. This project is an attempt to represent a coral reef as a Potts model, a grid of cells, each pulled towards the state of its neighbours via nearest-neighbour interaction. A patch with healthy neighbours is more likely to stay healthy, and a patch with bleached neighbours is more likely to bleach. In the global bleaching event of 2023-25, about 84% of the world's reef area had heat stress high enough to bleach coral, and an earlier event at Scott Reef in Western Australia in 2016 gave me bleaching surveys to study. The best-known reef models, such as that of [Mumby, Hastings & Edwards (2007)](https://doi.org/10.1038/nature06252), are mean-field, where every patch of reef feels one uniform field, given by the health of the reef as a whole.

The aim was to rebuild my undergraduate Potts model in the form of a lattice reef, with patches of healthy coral, bleached coral, bare rock and seaweed, and to investigate how the reach of the nearest-neighbour interaction changes the way a reef recovers, compared with the mean-field models in standard use. From physics we know that mean-field models exaggerate tipping points, and I felt that nearest-neighbour coupling would better reflect how real reefs recover. The model was then fitted to the 2016 bleaching event at Scott Reef.

Via stochastic simulation, it was found that the reef has hysteresis (warmed and then cooled back, it returns along a lower path than it went out on) at every interaction range. Once heat kills enough coral, seaweed grows over the dead coral and blocks new coral from settling, so the coral only returns once the water is cooled well below the temperature where it was lost. In the mean-field reef the gap between the warming and cooling paths is about 60% wider than in the nearest-neighbour reef, where each patch feels only its four nearest neighbours. Cooled back to the starting temperature, the mean-field reef stays degraded at 0.49 coral cover, while the nearest-neighbour reef recovers to about 0.9.

Real corals differ in how much heat they can take, so each patch of the model reef has its own heat tolerance, drawn at random and smoothed so that neighbouring colonies are alike. The model was fitted to 65 surveys of Scott Reef from April 2016 by simulation-based inference. 8,000 simulated reefs were run through the 2015-16 heatwave and surveyed the same way as the real reef, and a neural network learned which parameters produce which survey. The fit estimates the corals' mean heat tolerance, its spread across the reef, how steeply corals respond once past their tolerance, and the noise. Two different reefs fit the surveys equally well. In the sensitive reef the corals start to feel heat stress at about 4.5 degree heating weeks (a measure of the heat built up over the last 12 weeks), and in the tolerant reef at about 12.7, but they respond more steeply once past it. The surveys were taken at 16.5 degree heating weeks, far beyond both, when both reefs had bleached about 85% of their coral, so they cannot tell the two apart. There seems to be an interplay between the stress threshold and the severity of the reaction past it, which would need further study.

This model was built for fun, and it could be improved a lot. Access to better coral bleaching data, better data assimilation, a more precise loss function for tuning the parameters and a closer match to the biological processes at play would make for a much more interesting simulation. Real reefs, for example, recover slowly from coral larvae carried in from other reefs, which this model leaves out.

## The model

Our reef is a 60 × 60 lattice. Each cell *x* is in one of four states: healthy coral, bleached coral, bare rock or macroalgae (seaweed). Each day every cell may propose a change of state, and the change is accepted with the Metropolis rule:

```
P(accept)  =  min(1, exp(−β ΔE))
```

where ΔE is the change in the cell's energy and β sets how noisy the reef is. Strictly, this is a Potts-inspired stochastic cellular automaton with Metropolis-type acceptance rather than a Potts model in the Gibbs sense: all cells update at once, the proposals depend on each cell's stress, a little noise is added to ΔE, and death and colonisation are one-way steps, so detailed balance fails and there is no equilibrium distribution. The interaction-range runs below use the original energy in `coral_model.py` (degree heating days capped at 8, with recovery terms), and the Scott Reef fit uses the energy written out here (`coral_crw.py`). A lower energy is a more likely state. For a healthy patch the energy can be written as such:

```
E_H(x)  =  −J f_H(x)  −  ε_H  +  s · max(DHW(x) − D₅₀(x), 0)
```

Here f_H(x) is the share of the cell's neighbours that are healthy, J is the neighbour coupling, ε_H is the baseline preference for healthy coral, and the last term is heat stress. In our case the coupling is divided by the number of neighbours. This means a longer range changes the *reach* of the interaction and not its total strength. Otherwise "more neighbours" and "stronger coupling" could not be told apart.

Heat stress is measured the way NOAA Coral Reef Watch measures it (Skirving et al. 2020), in degree heating weeks (DHW): the heat above the warmest-month mean (MMM), accumulated over the last 12 weeks.

```
HS(t)   =  SST(t) − MMM
DHW(t)  =  (1/7) Σ_{last 84 days} HS,   counting only HS ≥ 1 °C
```

NOAA expects significant bleaching from about 4 DHW and mortality from about 8. This is the measure every bleaching survey is compared against, so we use it too. My first version used a home-made heat accumulator instead, which I replaced once I read how the field actually measures it.

We needed two more pieces before our reef could tip at all. In my original code coral could never die: its death rule needed heat stress above a cap that the stress could never reach. This meant bleached coral always recovered, and nothing could get stuck. I added mortality, and macroalgae that colonise dead coral and are cleared by grazing fish. Grazing is diluted as coral dies, since the same fish are spread over more seaweed. This is the same feedback Mumby et al. use to make real reefs bistable.

## How far the interactions reach

To look for a tipping point we warm the water slowly, hold it, then cool it back down, and track the coral cover. If our reef comes back along the same path, there is no tipping point. If it comes back along a different, lower path, the reef has hysteresis, ie. it got stuck.

![Hysteresis loops for four interaction ranges](figures/hero.png)

We see the reef get stuck at every interaction range. Once heat kills enough coral, seaweed takes the space, and cooling the water does not bring the coral back. Our loop is about 60% wider when every cell feels the whole reef than when it feels its four nearest neighbours. This means the mean-field model does exaggerate collapse, as the physics suggested, and for the reason it suggested: with short-range coupling, surviving patches reseed the cells next to them.

I expected cleaner dynamics than this. We also found two negative results. How *fast* the water warms barely matters, but how hot it gets does. And none of the classic early-warning signals (rising variance and autocorrelation before a transition, Scheffer et al. 2009) fire before the collapse, because in our case the slide is gradual rather than sudden.

![Early-warning signals before collapse](figures/fig4_early_warning.png)

## Fitting it to a real reef

The first version was checked against a bleaching curve in `Coral temps.xlsx`. When I went back to it, I found the curve had been drawn by hand, and its temperatures do not match Scott Reef in 2016 (a correlation of 0.12 with NOAA's record). Those fits have been removed. This was a lesson in checking the data before the model. Reef data are hard to find, as most surveys are sparse, a few sites at a few dates, and say nothing about where on the reef each patch is.

Our target is now 65 site surveys by the Australian Institute of Marine Science (AIMS) at Scott Reef on 13–20 April 2016, from the Global Coral Bleaching Database (van Woesik & Kratochwill 2022). Each site reports bleaching as a category: 6 sites at 30–60%, 24 at 60–90% and 35 above 90%. The heat forcing is NOAA's daily DHW for the Scott–Ashmore station.

The other change came from the literature. Corals in one reef differ: the most and least heat-tolerant tenth of colonies differ by 4–5 DHW (Humanes et al. 2022). This is why we give each cell its own tolerance D₅₀(x) ~ N(μ_D, σ_D²), drawn once and smoothed so neighbouring colonies are alike. In physics terms the uniform field of the Potts model becomes a random field.

Fitting a lattice model is not easy, because there is no likelihood to write down, and we can only simulate it. I first tried tuning it with Optuna, and it was hard to get anything that behaved like an actual reef rather than collapsing completely. Instead I used simulation-based inference (Cranmer, Brehmer & Louppe 2020):

1. We simulate 8,000 reefs through the 2015–16 heat record, drawing four parameters from wide priors: mean tolerance μ_D (2–16 DHW), its spread σ_D, the stress sensitivity s and the noise β.
2. We "survey" each simulated reef exactly as AIMS did: 65 random sites on 16 April, binned into the same four categories.
3. We train a mixture density network (Bishop 1994, Papamakarios & Murray 2016) to map a survey to a distribution over the parameters, q(θ | survey) = Σ_k π_k 𝒩(θ; μ_k, diag σ_k²). I wrote it in NumPy with the gradients derived by hand, so as to not treat it as a black box.

## How I checked it

Before using the real survey, we check the inference on 800 simulated reefs it has never seen. For every parameter, the 90% credible interval contains the true value 90% of the time (89.5–90.4%). This means the uncertainty it reports can be trusted. The fitted model then predicts 0.5, 6.7, 20 and 37 sites in the four categories, against 0, 6, 24 and 35 observed.

## What the survey says, and what it can't

We see a bimodal posterior: two reefs fit the survey, a sensitive reef with mean tolerance 4.5 DHW (37% of the posterior), and a tolerant reef at 12.7 DHW with a sharper response to stress (63%). By mid-April both have bleached about 85% of their living coral. This is an identifiability problem. One snapshot, taken at 16.5 DHW, far beyond both thresholds, cannot separate how tolerant the corals are from how steeply they respond once past it.

![Heat stress and the two reefs that fit the April surveys](figures/fig8_scott_reef_2016.png)

The two reefs disagree earlier in the summer. By 1 February the sensitive reef has 59% of its coral bleached and the tolerant reef 26%. A single survey in February would decide between them, but the database has none for Scott Reef in 2016. They also disagree about dead coral. By 5 April the sensitive reef has lost about half of its living coral (16–66% across draws), while the tolerant one has lost none. AIMS's own update from Scott Reef on 5 April 2016 reported that "wide-spread mortality is already evident". That is a field report, not a survey statistic, so I have not refitted to it, but it leans towards the sensitive reef (`inference/check_mortality.py`).

![Posterior over tolerance and sensitivity](figures/fig9_tolerance.png)

## Where Scott Reef tips

With our fitted parameters we can now ask the tipping question again, with error bars. For 40 parameter sets from the posterior, 20 from each mode, we hold the water at a fixed temperature above the warmest-month mean for 120 days, step it up 0.1 °C at a time, then step it back down.

![The tipping point with error bars](figures/fig10_tipping_crw.png)

We see the sensitive reef tip at +1.1 to +1.2 °C (about 30.6–30.7 °C) and the tolerant reef at +1.4 to +1.7 °C. Neither comes back on cooling, in any draw. This is because our model has no coral larvae arriving from outside the lattice, so a reef that has died has nothing to regrow from. Real reefs do recover, slowly, from larvae carried in from other reefs, so the hysteresis here is an upper bound.

## What went wrong, and what I learned

The data were the hardest part of this project. Most of the time went on the reef collapsing totally and refusing to behave like a real one, and on finding out that the curve I had calibrated to was not data at all. The nearest-neighbour analogy also turned out to be weaker than I hoped. Corals do help their neighbours (shade, shelter, larvae settling nearby), but it is only loosely true, and the energies in the model are chosen rather than derived. The hysteresis results should be read as qualitative.

## What I'd do next

Our model needs far more data than one survey: repeated, spatial surveys of a reef through a whole season, so that a cellular automaton or Potts model could be tuned with machine learning rather than by hand. Adding larval supply from neighbouring reefs is the obvious missing biology. I would also like to work on the pure mathematics of these systems (when a lattice model has a sharp transition, and what sets its order) as well as the applied side.

## Code

`coral_model.py` (the fast Potts reef, about 9× faster than `Coral_Potts_Model_original.py`) · `coral_crw.py` (DHW forcing and random tolerances) · `run_tipping_crw.py` · `inference/` (`simulate_bank_crw.py`, `mdn.py`, `infer_crw.py`, `check_mortality.py`) · `coral_potts.ipynb` · `data/` · Python, NumPy, SciPy

## Credits

Reef model and grazing feedback: Mumby, Hastings & Edwards (2007), Nature 450:98. Heat stress: NOAA Coral Reef Watch daily 5 km product, Scott–Ashmore virtual station (Skirving et al. 2020, Remote Sensing 12(23):3856, [doi:10.3390/rs12233856](https://doi.org/10.3390/rs12233856)). Surveys: Global Coral Bleaching Database, NOAA NCEI accession 0228498 (van Woesik & Kratochwill 2022, Scientific Data 9:20, [doi:10.1038/s41597-022-01121-y](https://doi.org/10.1038/s41597-022-01121-y)), with Scott Reef surveys by the Australian Institute of Marine Science. Biology: Humanes et al. (2022), Proc. R. Soc. B 289:20220872; Ainsworth et al. (2016), Science 352:338. Early-warning signals: Scheffer et al. (2009), Nature 461:53. Inference: Cranmer, Brehmer & Louppe (2020), PNAS 117:30055; Papamakarios & Murray (2016), NeurIPS; Bishop (1994), mixture density networks.
