# Hedging a wheat harvest through Russia's 2010 export ban

![CBOT wheat through 2010, with the hedge window, the heatwave, the export ban and every strike](figures/fig5_wheat_2010.png)

## Why I did this

In commodities the hedger who matters most is often the producer: a farm that knows roughly how much it will harvest but not what it will be paid. The standard tools are futures, puts and a handful of option structures, and a producer has to choose between them months before the price is known. I wanted to test those choices against one real shock, set up the way a producer would have set them up beforehand, and then see in hindsight what each would have paid. The options are priced with the same Black (1976) model used in the Asian option pricer (014).

## The shock

Russia had its worst heatwave on record through July 2010. On 5 August it banned grain exports. CBOT wheat went from $4.90 a bushel on 3 May to $6.53 on 31 August, up 33%. Seen from May, with the volatility the market had shown, a price that high had about a 5% chance.

## The setup

A US winter-wheat producer hedges on 3 May 2010 the crop it will sell on 31 August. CBOT front-month futures stand in for the cash price (basis is ignored) and everything is per bushel. Options are European, expire on the sale date and are priced by Black (1976) at the 60-day volatility before 3 May (31% a year), since free option quotes for 2010 don't exist.

The eight hedges are the ones producers use day to day:

| Hedge | What it is |
|---|---|
| No hedge | Sell at harvest at whatever the price is. |
| Short futures | Sell the whole crop forward on 3 May at $4.90. |
| 50% futures | Sell half forward, leave half open. |
| Long put | Buy an at-the-money put ($0.35): a floor at $4.90 less the premium, upside kept. |
| Zero-cost collar | Buy a 95% put and pay for it by selling a call at $5.19: a floor and a cap. |
| Put spread | Buy the at-the-money put and sell an 85% put: a cheaper floor that only covers a fall to $4.17. |
| Three-way (fence) | A 95% / 85% put spread paid for by selling a call at $5.46. |
| Layered | Sell a quarter of the crop forward at the start of May, June, July and August. |

![Revenue at sale against the wheat price, for each hedge](figures/fig7_wheat_payoffs.png)

## What happened

![Revenue by hedge: 2010 against the range expected on 3 May](figures/fig6_wheat_revenue.png)

| Hedge | Revenue in 2010 ($/bu) | Average of the worst 5% expected on 3 May ($/bu) |
|---|:-:|:-:|
| No hedge | 6.53 | 3.35 |
| Put spread | 6.26 | 3.82 |
| Long put | 6.18 | 4.55 |
| 50% futures | 5.71 | 4.13 |
| Three-way | 5.46 | 3.84 |
| Layered | 5.30 | 4.15 |
| Zero-cost collar | 5.19 | 4.66 |
| Short futures | 4.90 | 4.90 |

- **In hindsight the producer did best unhedged.** A price spike is good news for a seller, and every hedge gave some of it away.
- **Of the hedges, the ones that kept the upside won.** The put spread and the long put lost only their premium. Short futures gave up the whole rally, and the collar and the three-way were capped at their call strikes.
- **Before the event, the ranking reverses.** Judged on 3 May by the worst 5% of outcomes, short futures and the collar protected the producer best, and no hedge was the most dangerous. The put sits in between: a decent floor at a known cost, with the upside kept.
- **So the right hedge depends on what the producer can't afford.** A farm that must cover its costs whatever happens should lock in a price; one that can stand a bad year but hates missing a rally should buy puts. In 2010 the second kind did well, but that was a 1-in-20 outcome.

## Before this: the option seller's side

The first version of this project took the other side: sell a 3-month at-the-money call on copper, gold, wheat, corn, coffee or cocoa every month since 2002, and delta-hedge it eight ways (static, weekly, on 2% moves, daily, daily with GARCH volatility, Leland's cost adjustment and the Whalley–Wilmott band). In calm markets hedging daily tracks best, and the band comes within a point at three-quarters of the cost. In a shock every dynamic hedge lands within two points of the others, because the loss is set by the volatility the option was sold at: error / premium ≈ −½ × (σ²_realised − σ²_sold) / σ²_sold, a well-known result that the 1,738 trades follow closely.

| | |
|:-:|:-:|
| ![Error against cost](figures/fig1_frontier.png) | ![The variance surprise](figures/fig3_variance_surprise.png) |

## Limits

One shock, one producer and one sale date. Basis between the local cash price and CBOT is ignored, and so is yield risk: a Russian producer that lost its crop in 2010 faced a very different problem, because its quantity fell as the price rose. Options are priced at realised volatility, not market quotes. The continuous front-month series jumps slightly at each roll.

## What I'd do next

Add yield risk, so the producer's revenue is price times an uncertain harvest, which changes the best hedge ratio. Repeat the test across several shocks (the 2012 US drought, the 2022 invasion) to see whether the ranking holds. And price the options at market implied volatility where it can be found.

## Code

`wheat_2010.py` (the producer hedge) · `make_wheat_figures.py` · `hedging.py` (Black-76 and the delta-hedging rules) · `backtest.py` · `monte_carlo.py` · `make_figures.py` · Python, NumPy, SciPy, pandas

*A side project, built for fun. Credits: Black (1976), J. Financ. Econ. 3, 167; Leland (1985), J. Finance 40, 1283; Whalley & Wilmott (1997), Math. Finance 7, 307; Hull & White (2017), J. Bank. Finance 82, 180. Prices: CBOT, COMEX and ICE futures via Yahoo Finance.*
