#!/usr/bin/env python3
#-------- spx_chain.py ------------------------------------------------------
#-----------------------------------------------------------------------------
# Reading a full SPX option chain snapshot.
#
# The file is one row per strike per expiry, with calls on the left and puts on
# the right - the same butterfly layout as the TFEX page, just already in csv.
# Columns that matter:
#
#   QUOTE_DATE, S        the day, and the index level that day
#   EXPIRE_DATE, DTE     the expiry, and days until it
#   K                    strike
#   C_BID C_ASK C_IV     call quote and the provider's implied vol
#   P_BID P_ASK P_IV     put quote and the same
#   C_VOLUME P_VOLUME    how much actually traded
#
# Everything below is the same logic as tfex_options.py, pointed at this format,
# because the questions are identical: which rows can be trusted, where is the
# forward, and what does the volatility term structure look like.

import csv
import numpy as np
import matplotlib.pyplot as plt


def _number(token):
    token = (token or '').strip().replace(',', '')
    if token in ('', '-'):
        return np.nan
    try:
        return float(token)
    except ValueError:
        return np.nan


#-------- Load ---------------------------------------------------------------
#-----------------------------------------------------------------------------
def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({
                'quote_date': r['QUOTE_DATE'].strip(),
                'expiry': r['EXPIRE_DATE'].strip(),
                'dte': _number(r['DTE']),
                'underlying': _number(r['S']),
                'strike': _number(r['K']),
                'call_bid': _number(r['C_BID']), 'call_ask': _number(r['C_ASK']),
                'call_iv': _number(r['C_IV']), 'call_volume': _number(r['C_VOLUME']),
                'put_bid': _number(r['P_BID']), 'put_ask': _number(r['P_ASK']),
                'put_iv': _number(r['P_IV']), 'put_volume': _number(r['P_VOLUME']),
            })
    return rows


#-------- Which rows can be trusted ------------------------------------------
#-----------------------------------------------------------------------------
# Same test as the Thai chain. A quote with nothing on one side, or no trades
# at all, is the provider's model rather than the market's opinion.
#
# The extra test here is the relative spread. Far out of the money, an option
# worth 0.05 can be quoted 0.05 bid at 0.30 offer - a 140% spread - and any
# volatility fitted to the middle of that is meaningless.
def usable(rows, max_relative_spread=0.5, require_volume=True):
    out = []
    for r in rows:
        for side in ('call', 'put'):
            bid, ask = r[f'{side}_bid'], r[f'{side}_ask']
            if np.isnan(bid) or np.isnan(ask) or ask <= 0 or ask <= bid:
                continue
            mid = 0.5 * (bid + ask)
            if (ask - bid) / mid > max_relative_spread:
                continue
            if require_volume and (np.isnan(r[f'{side}_volume']) or r[f'{side}_volume'] <= 0):
                continue
            break                                       # at least one side is usable
        else:
            continue
        out.append(r)
    return out


#-------- The forward, from put-call parity ----------------------------------
#-----------------------------------------------------------------------------
# call - put = discounted (forward - strike)
#
# So plotting (call mid - put mid) against strike gives a straight line, and the
# line tells you two things at once: its slope is the discount factor, and where
# it crosses zero is the forward price.
#
# This is worth doing rather than assuming a rate and a dividend yield, because
# the market has already priced both and this reads them straight back out.
def forward_from_parity(rows_for_one_expiry):
    strikes = []
    diffs = []
    for r in rows_for_one_expiry:
        call_mid = 0.5 * (r['call_bid'] + r['call_ask'])
        put_mid = 0.5 * (r['put_bid'] + r['put_ask'])
        if np.isnan(call_mid) or np.isnan(put_mid):
            continue
        strikes.append(r['strike'])
        diffs.append(call_mid - put_mid)

    if len(strikes) < 3:
        return np.nan, np.nan

    slope, intercept = np.polyfit(strikes, diffs, 1)    # straight line fit
    discount = -slope                                   # the line falls at the discount rate
    forward = intercept / discount if discount != 0 else np.nan
    return forward, discount


#-------- One volatility per expiry ------------------------------------------
#-----------------------------------------------------------------------------
# At the money, take the average of the call and put implied vol. Away from the
# money the two disagree because of the skew, but at the forward they should
# agree closely - and if they do not, something is wrong with the data.
def atm_volatility(rows_for_one_expiry, forward):
    best = None
    for r in rows_for_one_expiry:
        distance = abs(r['strike'] - forward)
        if best is None or distance < best[0]:
            best = (distance, r)
    if best is None:
        return np.nan

    r = best[1]
    ivs = [v for v in (r['call_iv'], r['put_iv']) if not np.isnan(v) and v > 0]
    return float(np.mean(ivs)) if ivs else np.nan


#-------- The term structure -------------------------------------------------
#-----------------------------------------------------------------------------
# Walk every expiry in the file and pull out one number each. Stacked together
# these are sigma(t) - the curve the pricer is calibrated to.
def term_structure(rows, min_strikes=5):
    by_expiry = {}
    for r in rows:
        by_expiry.setdefault(r['expiry'], []).append(r)

    out = []
    for expiry, group in sorted(by_expiry.items()):
        if len(group) < min_strikes:
            continue
        forward, discount = forward_from_parity(group)
        if np.isnan(forward):
            continue
        out.append({'expiry': expiry,
                    'dte': group[0]['dte'],
                    'years': group[0]['dte'] / 365.0,
                    'forward': forward,
                    'discount': discount,
                    'atm_vol': atm_volatility(group, forward),
                    'n_strikes': len(group)})
    return out


#-------- The smile, one expiry ----------------------------------------------
#-----------------------------------------------------------------------------
# Out of the money on each side: puts below the forward, calls above. Those are
# the liquid ones, and it avoids deep in-the-money quotes where the spread is
# wider than the option's whole time value.
def smile(rows_for_one_expiry, forward):
    strikes, vols = [], []
    for r in sorted(rows_for_one_expiry, key=lambda x: x['strike']):
        iv = r['put_iv'] if r['strike'] < forward else r['call_iv']
        if np.isnan(iv) or iv <= 0:
            continue
        strikes.append(r['strike'])
        vols.append(iv)
    return np.array(strikes), np.array(vols)


#-------- Plot : the volatility term structure -------------------------------
#-----------------------------------------------------------------------------
# The curve the pricer needs, straight off the market.
def plot_term_structure(term):
    years = np.array([t['years'] for t in term])
    vols = np.array([t['atm_vol'] for t in term])
    ok = ~np.isnan(vols)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(years[ok], 100*vols[ok], 'o-', color='k', ms=4, lw=1.2)

    #half the expiries sit inside the first month, so a straight time axis
    #squashes them into the left edge. log spacing spreads them out.
    ax.set_xscale('log')
    ax.set_xticks([1/365, 7/365, 30/365, 90/365, 1, 2])
    ax.set_xticklabels(['1 day', '1 week', '1 month', '3 months', '1 year', '2 years'])
    ax.set_xlabel('time to expiry (log spacing)')
    ax.set_ylabel('at-the-money implied volatility (%)')
    ax.set_title('The volatility term structure - this is sigma(t)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


#-------- Plot : the smile, a few expiries -----------------------------------
#-----------------------------------------------------------------------------
# Short expiries have a steep, narrow smile; long ones are flatter and wider.
# Plotting against moneyness rather than strike puts them on the same axis.
def plot_smiles(rows, expiries):
    by_expiry = {}
    for r in rows:
        by_expiry.setdefault(r['expiry'], []).append(r)

    fig, ax = plt.subplots(figsize=(10, 5))
    for expiry in expiries:
        group = by_expiry.get(expiry, [])
        if not group:
            continue
        forward, _ = forward_from_parity(group)
        k, v = smile(group, forward)
        if len(k) == 0:
            continue
        ax.plot(k/forward, 100*v, 'o-', ms=3, lw=1,
                label=f"{expiry}  ({group[0]['dte']:.0f} days)")

    ax.axvline(1.0, color='0.6', ls='--', lw=1)
    ax.set_xlabel('strike / forward   (1.0 is at the money)')
    ax.set_ylabel('implied volatility (%)')
    ax.set_title('The smile: steep and narrow when close, flat and wide when far')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


#-------- Main ---------------------------------------------------------------
#-----------------------------------------------------------------------------
# Run this file directly and it loads the chain, fits the forward at every
# expiry, prints the term structure and draws the two figures.
#
# The check worth watching is the discount column. Nothing in the code is told
# what interest rates were - the discount factor falls out of put-call parity,
# and the rate it implies should match what rates actually were on the quote
# date. If it does, the parse and the fit are both right.
def main(path='data/spx_chain_2023-01-04.csv'):
    rows = load(path)
    print(f"loaded {len(rows)} rows")
    print(f"quote date {rows[0]['quote_date']},  index level {rows[0]['underlying']}")

    good = usable(rows)
    print(f"{len(good)} rows survive the liquidity filter "
          f"({100*len(good)/len(rows):.0f}% - the rest are one-sided or never traded)")

    term = term_structure(good)
    print(f"\n{len(term)} expiries with a usable fit\n")
    print(f"{'expiry':12s} {'days':>6s} {'forward':>9s} {'discount':>9s} {'ATM vol':>9s}")
    for t in term:
        vol = 100*t['atm_vol'] if not np.isnan(t['atm_vol']) else float('nan')
        print(f"{t['expiry']:12s} {t['dte']:6.0f} {t['forward']:9.1f} "
              f"{t['discount']:9.4f} {vol:8.1f}%")

    #implied rate from the longest expiry, as a sanity check on the whole pipeline
    far = term[-1]
    implied_rate = -np.log(far['discount']) / far['years']
    print(f"\nimplied interest rate from the {far['dte']:.0f}-day expiry: "
          f"{100*implied_rate:.2f}% a year")

    plot_term_structure(term)
    plt.savefig('term_structure.png', dpi=150)

    #three expiries spread across the curve - near, middle, far
    picks = [term[2]['expiry'], term[len(term)//2]['expiry'], term[-1]['expiry']]
    plot_smiles(good, picks)
    plt.savefig('smiles.png', dpi=150)
    print("saved term_structure.png and smiles.png")

    return rows, term


if __name__ == "__main__":
    rows, term = main()
