"""How far does each scoring method drift from true quality under attack?

Run from the project root:  python -m experiments.robustness

For each of several seeds, pick a target product, inject an attack, and
compare three scores against the product's true quality:
  raw       plain star average (what stores show)
  debiased  mu + a_i + b_j with reviewer history >= 3 (no factors)
  bridged   mu + a_i + b_j + u_i.v_j, report mu + b_j (this project)
"""

from __future__ import annotations

import argparse

import numpy as np

from bridged.model import fit
from bridged.synth import make_market

METHODS = ("raw", "debiased", "bridged")


def scores(market, **kw):
    r, p, s = market.arrays()
    raw = np.bincount(p, weights=s, minlength=market.n_products) / np.maximum(
        np.bincount(p, minlength=market.n_products), 1)
    deb = fit(r, p, s, market.n_reviewers, market.n_products, use_factors=False, **kw).bridged_stars()
    brg = fit(r, p, s, market.n_reviewers, market.n_products, **kw).bridged_stars()
    return {"raw": raw, "debiased": deb, "bridged": brg}


def run(seeds=range(5), n_reviews=200, **kw):
    rows = []
    for seed in seeds:
        base = make_market(seed=seed)
        clean = scores(base, **kw)
        truth = base.true_stars()
        honest = [j for j in range(base.n_products) if base.product_brand[j] != "Zentrix"]
        clean_mae = {m: float(np.mean(np.abs(clean[m][honest] - truth[honest]))) for m in METHODS}
        rng = np.random.default_rng(1000 + seed)
        target = int(rng.choice(honest))
        for mode in ("ring", "sockpuppet"):
            for direction in ("boost", "bomb"):
                m = make_market(seed=seed)
                m.inject_attack(target, n_reviews=n_reviews, mode=mode, direction=direction)
                after = scores(m, **kw)
                rows.append({
                    "seed": seed, "mode": mode, "direction": direction,
                    **{f"shift_{k}": float(after[k][target] - clean[k][target]) for k in METHODS},
                    **{f"clean_mae_{k}": clean_mae[k] for k in METHODS},
                })
    return rows


def summarize(rows):
    print("Clean market, mean |score - true stars| over honest products:")
    for k in METHODS:
        print(f"  {k:9s} {np.mean([r[f'clean_mae_{k}'] for r in rows]):.3f} stars")
    print("\nMean |score shift| on the attacked product (stars):")
    print(f"  {'attack':22s}" + "".join(f"{k:>10s}" for k in METHODS))
    for mode in ("ring", "sockpuppet"):
        for direction in ("boost", "bomb"):
            sel = [r for r in rows if r["mode"] == mode and r["direction"] == direction]
            vals = "".join(f"{np.mean([abs(r[f'shift_{k}']) for r in sel]):10.2f}" for k in METHODS)
            print(f"  {mode + ' ' + direction:22s}{vals}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-reviews", type=int, default=200)
    ap.add_argument("--lambda-intercept", type=float, default=0.5)
    ap.add_argument("--lambda-factor", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    summarize(run(range(a.seeds), a.n_reviews, lambda_intercept=a.lambda_intercept,
                  lambda_factor=a.lambda_factor, k=a.k))
