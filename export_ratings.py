"""Fit team ratings (goals, corners, cards) on football-data.co.uk E0 CSVs and write them into the app.
Usage: python export_ratings.py football-predictor.html E0_2122.csv E0_2223.csv ...
Rewrites the text between /*RATINGS*/ and /*END*/ in the HTML file. Run again after each new matchday.
Config (from the backtest): half-life 180 days; ridge goals 2, corners 30, cards 8.
"""
import sys, json, re
import numpy as np, pandas as pd
from scipy.optimize import minimize

html_path, csvs = sys.argv[1], sys.argv[2:]
df = pd.concat([pd.read_csv(p, encoding="utf-8-sig") for p in csvs], ignore_index=True).copy()
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df = df.dropna(subset=["FTHG", "FTAG", "HC", "AC", "HY", "AY"]).sort_values("Date").reset_index(drop=True)
df["HCd"] = df.HY + 2 * df.HR; df["ACd"] = df.AY + 2 * df.AR
season = np.where(df.Date.dt.month >= 7, df.Date.dt.year, df.Date.dt.year - 1)
current = set(df.HomeTeam[season == season.max()]) | set(df.AwayTeam[season == season.max()])
teams = sorted(set(df.HomeTeam) | set(df.AwayTeam)); T = len(teams); tid = {t: i for i, t in enumerate(teams)}
h = df.HomeTeam.map(tid).values; a = df.AwayTeam.map(tid).values
days = (df.Date.max() - df.Date).dt.days.values.astype(float) + 1
HALF = 180; w = 0.5 ** (days / HALF)

def fit(yh, ya, ridge):
    def f(p):
        mu, ha = p[0], p[1]; att = p[2:2 + T]; dfn = p[2 + T:]
        lh = np.exp(mu + ha + att[h] - dfn[a]); la = np.exp(mu + att[a] - dfn[h])
        ll = (w * (yh * np.log(lh) - lh + ya * np.log(la) - la)).sum() - ridge * (att @ att + dfn @ dfn)
        rh = w * (yh - lh); ra = w * (ya - la); g = np.zeros_like(p)
        g[0] = rh.sum() + ra.sum(); g[1] = rh.sum()
        g[2:2 + T] = np.bincount(h, rh, T) + np.bincount(a, ra, T) - 2 * ridge * att
        g[2 + T:] = -np.bincount(a, rh, T) - np.bincount(h, ra, T) - 2 * ridge * dfn
        return -ll, -g
    p0 = np.zeros(2 + 2 * T); p0[0] = np.log(yh.mean())
    return minimize(f, p0, jac=True, method="L-BFGS-B").x

specs = {"g": (df.FTHG.values.astype(float), df.FTAG.values.astype(float), 2),
         "c": (df.HC.values.astype(float), df.AC.values.astype(float), 30),
         "k": (df.HCd.values.astype(float), df.ACd.values.astype(float), 8)}
out = {"asof": df.Date.max().strftime("%d %b %Y"), "teams": {}}
fits = {m: fit(*s) for m, s in specs.items()}
for m, p in fits.items(): out[m] = {"mu": round(float(p[0]), 4), "ha": round(float(p[1]), 4)}
for t in sorted(current):
    i = tid[t]
    out["teams"][t] = {f"{m}{s}": round(float(fits[m][2 + i + (T if s == "d" else 0)]), 3) for m in "gck" for s in "ad"}
blob = json.dumps(out, separators=(",", ":"))
src = open(html_path, encoding="utf-8").read()
new, n = re.subn(r"/\*RATINGS\*/.*?/\*END\*/", lambda m: "/*RATINGS*/" + blob + "/*END*/", src, flags=re.S)
if n != 1: sys.exit("Markers /*RATINGS*/ ... /*END*/ not found exactly once in the HTML file")
open(html_path, "w", encoding="utf-8").write(new)
print(f"Wrote ratings for {len(out['teams'])} teams, data to {out['asof']}")
