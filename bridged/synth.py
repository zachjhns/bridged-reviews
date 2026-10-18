"""Synthetic marketplace with known ground truth, plus review-attack injectors.

Real review datasets don't label which reviews are fake. A simulator does, so
we can measure how far each scoring method drifts from a product's *true*
quality when it is attacked.

Genuine reviewers: rating = true quality + personal leniency
                   + taste x product style + noise, rounded to 1..5 stars.
Review rings:      accounts that 5-star one brand's products, 1-star its
                   competitors, and sprinkle a few honest-looking "camouflage"
                   reviews to look normal. This matches how paid review groups
                   are usually described: reused accounts working for a seller.
Sockpuppets:       brand-new accounts that leave a single review each.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BRANDS = {
    "Auralis": ["Auralis One", "Auralis Pro", "Auralis Lite"],
    "Kestrel": ["Kestrel K2", "Kestrel K3 ANC", "Kestrel Sport"],
    "NovaPod": ["NovaPod Air", "NovaPod Max", "NovaPod Mini", "NovaPod Studio"],
    "Sonique": ["Sonique S1", "Sonique S2", "Sonique Bass+"],
    "Driftwave": ["Driftwave Flow", "Driftwave Surf", "Driftwave Calm"],
    "Quillo": ["Quillo Q", "Quillo Q Pro", "Quillo Kids"],
    "Hollis": ["Hollis Classic", "Hollis Tour", "Hollis Wireless"],
    "Tempo": ["Tempo Run", "Tempo Gym", "Tempo Daily"],
    "Lumen": ["Lumen Halo", "Lumen Edge", "Lumen Glow"],
    "Brisk": ["Brisk Go", "Brisk Go 2", "Brisk Buds"],
    "Zentrix": ["Zentrix Z1", "Zentrix Z1 Plus", "Zentrix ProMax"],
    "Ferro": ["Ferro Mono", "Ferro Duo", "Ferro Open"],
}

GENUINE_TEXT = {
    1: ["Stopped charging after two weeks.", "Left bud cuts out constantly.", "Returned it. Tinny and uncomfortable."],
    2: ["Fit is awkward and the case feels cheap.", "Sound is muddy, battery is fine.", "Not worth the price."],
    3: ["Decent for the price, nothing special.", "Good sound, mediocre mic.", "Fine for the gym, I wouldn't use them for music."],
    4: ["Solid sound and battery, the app is clunky.", "Comfortable for hours. Noise canceling is OK.", "Great value, minor connection hiccups."],
    5: ["Best earbuds I've owned, the battery lasts forever.", "Crisp sound, great fit, flawless pairing.", "Noise canceling is excellent on flights."],
}
FAKE_PRAISE = ["Amazing!!! Best purchase ever, highly recommend!!!", "Perfect product. 5 stars. Must buy!",
               "Incredible quality, better than all the expensive brands!!", "Love love love these. Buy now!"]
FAKE_BOMB = ["Terrible, broke immediately. Avoid!!!", "Worst product ever. Scam. Do not buy.",
             "Total garbage, returned it the same day!!"]


@dataclass
class Review:
    reviewer: int
    product: int
    stars: int
    day: int
    text: str
    fake: bool
    campaign: int  # -1 for genuine, otherwise id of the attack that created it


@dataclass
class Market:
    product_names: list[str]
    product_brand: list[str]
    true_quality: np.ndarray  # centered scale, see model.stars_to_centered
    style: np.ndarray
    reviewer_fake: list[bool]
    reviews: list[Review]
    rng: np.random.Generator
    n_days: int = 365
    campaigns: list[dict] = field(default_factory=list)

    @property
    def n_products(self) -> int:
        return len(self.product_names)

    @property
    def n_reviewers(self) -> int:
        return len(self.reviewer_fake)

    def true_stars(self) -> np.ndarray:
        return np.clip(3.0 + 2.0 * self.true_quality, 1.0, 5.0)

    def arrays(self):
        r = np.array([x.reviewer for x in self.reviews], dtype=np.int64)
        p = np.array([x.product for x in self.reviews], dtype=np.int64)
        s = np.array([x.stars for x in self.reviews], dtype=float)
        return r, p, s

    def _new_reviewer(self, fake: bool) -> int:
        self.reviewer_fake.append(fake)
        return len(self.reviewer_fake) - 1

    def _honest_stars(self, product: int, leniency: float = 0.0, taste: float = 0.0) -> int:
        latent = (self.true_quality[product] + leniency + 0.3 * taste * self.style[product]
                  + self.rng.normal(0, 0.3))
        return int(np.clip(np.rint(3 + 2 * latent), 1, 5))

    def brand_products(self, brand: str) -> list[int]:
        return [j for j, b in enumerate(self.product_brand) if b == brand]

    def inject_attack(self, target: int, n_reviews: int = 200, mode: str = "ring",
                      direction: str = "boost", start_day: int | None = None,
                      burst_days: int = 7) -> dict:
        """Add n_reviews fake reviews on `target`, one per new account.

        mode="ring": each account also promotes the target's sibling products,
        attacks competitors, and leaves 1-2 camouflage reviews.
        mode="sockpuppet": each account leaves only the one review.
        direction="boost" gives 5 stars, "bomb" gives 1 star.
        """
        rng = self.rng
        cid = len(self.campaigns)
        start = self.n_days - burst_days if start_day is None else start_day
        hi, lo = (5, 1) if direction == "boost" else (1, 5)
        texts = FAKE_PRAISE if direction == "boost" else FAKE_BOMB
        brand = self.product_brand[target]
        siblings = [j for j in self.brand_products(brand) if j != target]
        rivals = [j for j in range(self.n_products) if self.product_brand[j] != brand]
        # Rings tend to hit the competitors with the best ratings.
        rivals = sorted(rivals, key=lambda j: -self.true_quality[j])[:6]

        def day():
            return int(start + rng.integers(0, burst_days))

        for _ in range(n_reviews):
            acct = self._new_reviewer(fake=True)
            self.reviews.append(Review(acct, target, hi, day(), str(rng.choice(texts)), True, cid))
            if mode != "ring":
                continue
            for j in rng.choice(siblings, size=min(len(siblings), int(rng.integers(1, 3))), replace=False):
                self.reviews.append(Review(acct, int(j), hi, day(), str(rng.choice(texts)), True, cid))
            for j in rng.choice(rivals, size=int(rng.integers(1, 3)), replace=False):
                other = FAKE_BOMB if direction == "boost" else FAKE_PRAISE
                self.reviews.append(Review(acct, int(j), lo, day(), str(rng.choice(other)), True, cid))
            # Camouflage: honest-looking ratings on unrelated products.
            pool = [j for j in range(self.n_products) if self.product_brand[j] != brand and j not in rivals]
            for j in rng.choice(pool, size=int(rng.integers(1, 3)), replace=False):
                s = self._honest_stars(int(j))
                self.reviews.append(Review(acct, int(j), s, day(), str(rng.choice(GENUINE_TEXT[s])), True, cid))

        info = {"id": cid, "target": target, "n_reviews": n_reviews, "mode": mode,
                "direction": direction, "start_day": start}
        self.campaigns.append(info)
        return info


def make_market(seed: int = 7, n_reviewers: int = 2500, seed_ring: bool = True) -> Market:
    rng = np.random.default_rng(seed)
    names, brands = [], []
    for brand, products in BRANDS.items():
        for name in products:
            names.append(name)
            brands.append(brand)
    n_p = len(names)
    quality = rng.normal(0.15, 0.35, size=n_p)
    # The ring's own brand makes mediocre products; that's why it pays for reviews.
    for j, b in enumerate(brands):
        if b == "Zentrix":
            quality[j] = rng.normal(-0.35, 0.08)
    quality = np.clip(quality, -0.9, 0.9)
    style = rng.normal(0, 1, size=n_p)
    popularity = rng.dirichlet(np.full(n_p, 1.5))

    market = Market(names, brands, quality, style, [], [], rng)

    for _ in range(n_reviewers):
        acct = market._new_reviewer(fake=False)
        leniency = rng.normal(0, 0.2)
        taste = rng.normal(0, 1)
        n = min(n_p, 1 + rng.geometric(0.15))
        for j in rng.choice(n_p, size=n, replace=False, p=popularity):
            s = market._honest_stars(int(j), leniency, taste)
            market.reviews.append(Review(acct, int(j), s, int(rng.integers(0, market.n_days - 14)),
                                         str(rng.choice(GENUINE_TEXT[s])), False, -1))

    if seed_ring:
        # A long-running ring for Zentrix, already baked into the history.
        z1 = market.brand_products("Zentrix")[0]
        market.inject_attack(z1, n_reviews=90, mode="ring", start_day=120, burst_days=200)

    return market
