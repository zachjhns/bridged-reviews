"""Demo server: a simulated store whose ratings you can attack live.

Run:  uvicorn server:app --port 8000   then open http://localhost:8000
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from bridged.model import FitResult, fit, stars_to_centered
from bridged.synth import Market, make_market

MIN_HISTORY = 3
SUSPICIOUS_GAP = 0.3  # stars between raw and bridged


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.market: Market = make_market(seed=7)
        self.refit()
        self.baseline_raw = self.raw.copy()
        self.baseline_bridged = self.bridged.copy()

    def refit(self):
        t = time.perf_counter()
        r, p, s = self.market.arrays()
        self.r, self.p, self.s = r, p, s
        self.res: FitResult = fit(r, p, s, self.market.n_reviewers, self.market.n_products,
                                  min_history=MIN_HISTORY)
        self.fit_ms = (time.perf_counter() - t) * 1000
        n = np.bincount(p, minlength=self.market.n_products)
        self.n_reviews = n
        self.raw = np.bincount(p, weights=s, minlength=self.market.n_products) / np.maximum(n, 1)
        self.bridged = self.res.bridged_stars()
        self.history = np.bincount(r, minlength=self.market.n_reviewers)


state = State()
app = FastAPI(title="Bridged Reviews")


def r2(x) -> float:
    return round(float(x), 2)


def product_summary(j: int) -> dict:
    m = state.market
    return {
        "id": j,
        "name": m.product_names[j],
        "brand": m.product_brand[j],
        "n_reviews": int(state.n_reviews[j]),
        "raw": r2(state.raw[j]),
        "bridged": r2(state.bridged[j]),
        "true": r2(m.true_stars()[j]),
        "baseline_raw": r2(state.baseline_raw[j]),
        "baseline_bridged": r2(state.baseline_bridged[j]),
        "suspicious": bool(abs(state.raw[j] - state.bridged[j]) > SUSPICIOUS_GAP),
    }


def product_detail(j: int) -> dict:
    m, res = state.market, state.res
    sel = np.flatnonzero(state.p == j)
    ri, stars = state.r[sel], state.s[sel]
    est = state.history[ri] >= MIN_HISTORY

    # Exact decomposition of the raw average (in stars) into bridged + effects.
    y = stars_to_centered(stars)
    a, U, b, v = res.reviewer_intercept, res.reviewer_factor, res.product_intercept[j], res.product_factor[j]
    camp = U[ri] @ v  # how much each rating is explained by the reviewer's "camp"
    raw_c = y.mean()
    if est.any():
        inc_mean = y[est].mean()
        lean = a[ri[est]].mean()
        view = camp[est].mean()
    else:
        inc_mean = lean = view = 0.0
    bridged_c = res.mu + b
    waterfall = {
        "bridged": r2(3 + 2 * bridged_c),
        "leniency": r2(2 * lean),
        "viewpoint": r2(2 * view),
        "noise": r2(2 * (inc_mean - bridged_c - lean - view)),
        "new_accounts": r2(2 * (raw_c - inc_mean)),
        "raw": r2(3 + 2 * raw_c),
    }

    # Reviewer map: project this product's established reviewers' factors to 2D (PCA).
    pts = []
    if est.sum() >= 3:
        F = U[ri[est]]
        Fc = F - F.mean(0)
        _, _, Vt = np.linalg.svd(Fc, full_matrices=False)
        xy = F @ Vt[:2].T
        # Orient axis 1 so reviewers whose camp pushes this product up sit to the right.
        if np.corrcoef(xy[:, 0], camp[est])[0, 1] < 0:
            xy[:, 0] *= -1
        if xy.shape[1] < 2:
            xy = np.column_stack([xy[:, 0], np.zeros(len(xy))])
        for (x, yy), rid, st, c in zip(xy, ri[est], stars[est], camp[est]):
            pts.append({"x": round(float(x), 3), "y": round(float(yy), 3), "stars": int(st),
                        "fake": bool(m.reviewer_fake[rid]), "camp": r2(2 * c)})

    hist = np.bincount(stars.astype(int), minlength=6)[1:].tolist()

    days = np.array([m.reviews[i].day for i in sel])
    week = np.minimum(days // 7, (m.n_days - 1) // 7)
    n_weeks = int((m.n_days - 1) // 7 + 1)
    timeline = [[int(np.sum((week == w) & (stars == s))) for s in range(1, 6)] for w in range(n_weeks)]

    order = sel[np.argsort(-days)][:40]
    reviews = []
    for i in order:
        rv = m.reviews[i]
        h = int(state.history[rv.reviewer])
        reviews.append({
            "reviewer": f"user{rv.reviewer:05d}",
            "stars": rv.stars, "day": rv.day, "text": rv.text,
            "history": h,
            "excluded": h < MIN_HISTORY,
            "camp": r2(2 * float(U[rv.reviewer] @ v)) if h >= MIN_HISTORY else None,
            "fake": rv.fake,
        })

    return {
        **product_summary(j),
        "n_established": int(est.sum()),
        "waterfall": waterfall,
        "histogram": hist,
        "timeline": timeline,
        "reviewer_map": pts,
        "reviews": reviews,
        "fit_ms": round(state.fit_ms),
    }


@app.get("/api/products")
def products():
    with state.lock:
        return {"products": [product_summary(j) for j in range(state.market.n_products)],
                "n_reviews": len(state.market.reviews), "n_reviewers": state.market.n_reviewers,
                "campaigns": state.market.campaigns}


@app.get("/api/products/{j}")
def product(j: int):
    with state.lock:
        if not 0 <= j < state.market.n_products:
            raise HTTPException(404, "no such product")
        return product_detail(j)


class Attack(BaseModel):
    product: int
    n_reviews: int = Field(200, ge=1, le=2000)
    mode: Literal["ring", "sockpuppet"] = "ring"
    direction: Literal["boost", "bomb"] = "boost"


@app.post("/api/attack")
def attack(body: Attack):
    with state.lock:
        if not 0 <= body.product < state.market.n_products:
            raise HTTPException(404, "no such product")
        before = product_summary(body.product)
        state.market.inject_attack(body.product, body.n_reviews, body.mode, body.direction)
        state.refit()
        return {"before": before, "after": product_detail(body.product)}


@app.post("/api/reset")
def reset():
    with state.lock:
        state.reset()
        return {"ok": True}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "web", html=True), name="web")
