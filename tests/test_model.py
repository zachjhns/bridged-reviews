import numpy as np
import pytest

from bridged.model import centered_to_stars, fit, stars_to_centered
from bridged.synth import make_market


def _fit(market, **kw):
    r, p, s = market.arrays()
    return fit(r, p, s, market.n_reviewers, market.n_products, **kw)


def _raw(market):
    r, p, s = market.arrays()
    return np.bincount(p, weights=s, minlength=market.n_products) / np.bincount(p, minlength=market.n_products)


@pytest.fixture(scope="module")
def clean():
    return make_market(seed=3, seed_ring=False)


def test_star_scale_round_trip():
    stars = np.array([1, 2, 3, 4, 5])
    assert np.allclose(centered_to_stars(stars_to_centered(stars)), stars)


def test_unanimous_reviewers_match_raw_average():
    # 30 reviewers who all give product 0 four stars and product 1 two stars.
    r = np.repeat(np.arange(30), 3)
    p = np.tile([0, 1, 2], 30)
    s = np.tile([4.0, 2.0, 3.0], 30)
    res = fit(r, p, s, 30, 3)
    assert np.allclose(res.bridged_stars(), [4, 2, 3], atol=0.05)


def test_fit_is_deterministic(clean):
    a = _fit(clean).bridged_stars()
    b = _fit(clean).bridged_stars()
    assert np.array_equal(a, b)


def test_clean_market_tracks_true_quality(clean):
    scores = _fit(clean).bridged_stars()
    truth = clean.true_stars()
    assert np.mean(np.abs(scores - truth)) < 0.12
    assert np.corrcoef(scores, truth)[0, 1] > 0.97


def test_sockpuppets_are_ignored():
    m = make_market(seed=4)
    before = _fit(m).bridged_stars()
    m.inject_attack(5, n_reviews=300, mode="sockpuppet")
    after = _fit(m).bridged_stars()
    assert abs(after[5] - before[5]) < 0.01


@pytest.mark.parametrize("direction", ["boost", "bomb"])
def test_review_ring_moves_bridged_score_far_less_than_raw(direction):
    m = make_market(seed=5)
    target = m.brand_products("Hollis")[0]
    raw0, br0 = _raw(m)[target], _fit(m).bridged_stars()[target]
    m.inject_attack(target, n_reviews=200, mode="ring", direction=direction)
    raw_shift = abs(_raw(m)[target] - raw0)
    br_shift = abs(_fit(m).bridged_stars()[target] - br0)
    assert raw_shift > 0.2
    assert br_shift < 0.5 * raw_shift


def test_ring_members_separate_in_factor_space():
    m = make_market(seed=6)
    res = _fit(m)
    fake = np.array(m.reviewer_fake)
    active = np.bincount(m.arrays()[0], minlength=m.n_reviewers) >= 3
    norms = np.linalg.norm(res.reviewer_factor, axis=1)
    assert np.median(norms[fake & active]) > 3 * np.median(norms[~fake & active])
