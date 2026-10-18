"""Bridging matrix factorization for product reviews.

Adapted from the idea behind X's Community Notes ranking: every rating is
explained as

    r_ij ~ mu + a_i + b_j + u_i . v_j

where i is a reviewer and j a product. The factor term u_i . v_j soaks up
*viewpoint-dependent* agreement: reviewers who systematically love one group of
products and hate another (a taste, or a paid review ring) share a direction in
factor space. The product intercept b_j is what is left over: the part of a
product's rating that reviewers agree on regardless of which camp they are in.

Intercepts are regularized more heavily than factors, so the model prefers to
explain a coordinated block of ratings with the factor term. That is what keeps
a brigade from moving b_j, which we report as the "bridged" score.

Ratings are stored on a centered scale: stars 1..5 -> (stars - 3) / 2 in [-1, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stars_to_centered(stars: np.ndarray) -> np.ndarray:
    return (np.asarray(stars, dtype=float) - 3.0) / 2.0


def centered_to_stars(x: np.ndarray | float) -> np.ndarray | float:
    return np.clip(3.0 + 2.0 * np.asarray(x, dtype=float), 1.0, 5.0)


@dataclass
class FitResult:
    mu: float
    reviewer_intercept: np.ndarray  # a_i, shape (n_reviewers,)
    reviewer_factor: np.ndarray  # u_i, shape (n_reviewers, k)
    product_intercept: np.ndarray  # b_j, shape (n_products,)
    product_factor: np.ndarray  # v_j, shape (n_products, k)
    included: np.ndarray  # bool mask over ratings that entered the fit
    rmse: float

    def bridged_stars(self) -> np.ndarray:
        """Per-product score on the 1-5 star scale, from mu + b_j only."""
        return centered_to_stars(self.mu + self.product_intercept)


def fit(
    reviewer_idx: np.ndarray,
    product_idx: np.ndarray,
    stars: np.ndarray,
    n_reviewers: int,
    n_products: int,
    k: int = 3,
    lambda_intercept: float = 0.5,
    lambda_factor: float = 0.1,
    min_history: int = 3,
    n_iter: int = 60,
    use_factors: bool = True,
) -> FitResult:
    """Fit the model with alternating least squares.

    Each half-step is an exact ridge regression per reviewer (or per product),
    solved for all entities at once with batched (k+1)x(k+1) systems.

    min_history drops reviewers with fewer than this many ratings, the same
    way Community Notes only counts raters with a track record. One-shot
    accounts carry no information about which camp they belong to.

    use_factors=False gives the "debiased mean" baseline: mu + a_i + b_j.
    """
    reviewer_idx = np.asarray(reviewer_idx, dtype=np.int64)
    product_idx = np.asarray(product_idx, dtype=np.int64)
    y_all = stars_to_centered(stars)

    history = np.bincount(reviewer_idx, minlength=n_reviewers)
    included = history[reviewer_idx] >= min_history
    ri, pj, y = reviewer_idx[included], product_idx[included], y_all[included]

    kk = k if use_factors else 0
    reg = np.diag([lambda_intercept] + [lambda_factor] * kk)
    counts = np.bincount(ri, minlength=n_reviewers).astype(float)

    def solve_side(idx, other_factor, target, n):
        # Features per rating: [1, other_factor_j]; ridge per entity.
        # Normal equations are accumulated with bincount (much faster than np.add.at).
        X = np.concatenate([np.ones((len(idx), 1)), other_factor], axis=1)
        d = X.shape[1]
        A = np.empty((n, d, d))
        c = np.empty((n, d))
        for p in range(d):
            c[:, p] = np.bincount(idx, weights=X[:, p] * target, minlength=n)
            for q in range(p, d):
                A[:, p, q] = A[:, q, p] = np.bincount(idx, weights=X[:, p] * X[:, q], minlength=n)
        A += reg[:d, :d]
        sol = np.linalg.solve(A, c[:, :, None])[:, :, 0]
        return sol[:, 0], sol[:, 1:]

    def center(U):
        # Gauge fix: a shared offset in reviewer factors is indistinguishable
        # from a per-product intercept (u_bar . v_j), which makes b_j drift
        # with initialization. Pin the count-weighted mean reviewer to the
        # origin so "factor 0" means "the typical reviewer".
        if U.shape[1] == 0 or counts.sum() == 0:
            return U
        u_bar = (counts[:, None] * U).sum(0) / counts.sum()
        return U - u_bar * (counts[:, None] > 0)

    mu = float(y.mean()) if len(y) else 0.0
    a = np.zeros(n_reviewers)
    b = np.zeros(n_products)
    empty = np.zeros((0, 0))
    # Warm start: intercept-only model.
    for _ in range(10):
        a, _ = solve_side(ri, empty.reshape(len(ri), 0), y - mu - b[pj], n_reviewers)
        b, _ = solve_side(pj, empty.reshape(len(pj), 0), y - mu - a[ri], n_products)
        mu = float(np.mean(y - a[ri] - b[pj])) if len(y) else 0.0

    # Deterministic factor init from an SVD of the residual matrix, so the same
    # data always gives the same answer (random init lets scores wobble).
    U = np.zeros((n_reviewers, kk))
    V = np.zeros((n_products, kk))
    if kk and len(y):
        M = np.zeros((n_reviewers, n_products))
        M[ri, pj] = y - mu - a[ri] - b[pj]
        Us, S, Vt = np.linalg.svd(M, full_matrices=False)
        root = np.sqrt(S[:kk])
        U = Us[:, :kk] * root
        V = Vt[:kk].T * root

    for _ in range(n_iter if kk else 0):
        a, U = solve_side(ri, V[pj], y - mu - b[pj], n_reviewers)
        U = center(U)
        b, V = solve_side(pj, U[ri], y - mu - a[ri], n_products)
        mu = float(np.mean(y - a[ri] - b[pj] - np.sum(U[ri] * V[pj], axis=1)))

    pred = mu + a[ri] + b[pj] + np.sum(U[ri] * V[pj], axis=1)
    rmse = float(np.sqrt(np.mean((y - pred) ** 2))) if len(y) else 0.0

    # Reviewers excluded by min_history get zero parameters (they never entered the fit).
    excluded = history < min_history
    a[excluded] = 0.0
    U[excluded] = 0.0

    if not use_factors:
        U = np.zeros((n_reviewers, k))
        V = np.zeros((n_products, k))

    return FitResult(mu, a, U, b, V, included, rmse)
