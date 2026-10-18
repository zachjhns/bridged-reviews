# Bridged Reviews

**A star rating that only counts what reviewers in different camps agree on, so paid review rings stop moving the score.**

Bridged Reviews takes the *bridging* algorithm behind X's Community Notes and applies it to product reviews. In Community Notes, a note is shown only when raters who usually disagree both find it helpful. Here, a product's score keeps only the part of its ratings that every camp of reviewers agrees on. A coordinated campaign, a 5★ boost or a 1★ review-bomb, gets attributed to its own camp instead of to the product.

## The problem

- Fake and coordinated reviews distort the star ratings people use to decide what to buy.
- Mozilla shut down **Fakespot** on July 1, 2025, and users have complained there's no replacement ([1](https://mjtsai.com/blog/2025/07/02/fakespot-shuts-down), [2](https://connect.mozilla.org/t5/discussions/longtime-firefox-user-deeply-disappointed-by-fakespot-removal/m-p/107890)).
- A filter for spam and fake reviews comes up in Hacker News' "What do you wish existed?" thread ([3](https://hn.svelte.dev/item/45500937)).
- Most fake-review detectors classify each review on its own, by its text or account metadata. Rings get around this by reusing aged accounts and posting honest-looking "camouflage" reviews.

Bridged Reviews doesn't try to judge each review. It looks for **who agrees with whom across the whole store**, and that pattern is hard for a ring to hide.

## How it works

Every rating is modelled as

```
rating_ij ≈ μ + a_i + b_j + u_i · v_j
```

| term | meaning |
|---|---|
| `a_i` | reviewer *i*'s general leniency (some people give everything 5★) |
| `u_i · v_j` | agreement that depends on the reviewer's **camp**. Taste groups and review rings each form their own direction in factor space |
| `b_j` | what's left: the part of product *j*'s rating that every camp agrees on |

The **bridged rating** is `μ + b_j`, converted back to stars.

Three details make it work:

1. **Intercepts are penalized more heavily than factors** (λ = 0.5 vs 0.1). A coordinated block of ratings is cheaper to explain as "a camp likes this" than as "this is good", the same trick Community Notes uses.
2. **Gauge fixing and deterministic init.** A shared offset in reviewer factors can't be told apart from a product intercept, so scores drifted by about 0.2★ depending on the random seed. Centering the reviewer factors on the typical reviewer and initializing from an SVD of the residual makes the fit converge in about 60 ALS sweeps and give the same answer every time.
3. **Track-record gate.** Accounts with fewer than 3 reviews don't count toward the bridged score, just as Community Notes only counts raters with history. A one-shot account gives no evidence about which camp it's in.

The fit is exact alternating least squares in NumPy, with one small ridge regression per reviewer and per product, batched. It refits the demo store (about 20k ratings) in about 0.4 s, which is fast enough to rerun live in the demo.

The UI also breaks the store rating down **exactly**:

```
store average = bridged + leniency + camp effect + new-account effect + noise
```

so you can see *why* the two ratings differ.

## Results

Real review datasets don't label which reviews are fake, so the evaluation uses a simulated marketplace with known ground truth (`bridged/synth.py`):

- 37 earbuds, 2,500 genuine reviewers with their own leniency and taste, and about 20k reviews.
- A pre-existing review ring for the brand "Zentrix".
- Injected attacks of **200 fake reviews**. Ring accounts also boost the brand's other products, 1★ its top competitors, and post camouflage reviews.

Output of `python -m experiments.robustness`, averaged over 5 seeds:

| | plain average | debiased (no camps) | **bridged** |
|---|---|---|---|
| error vs. true quality, no attack | 0.093★ | 0.101★ | **0.082★** |
| shift from a ring 5★ boost | 0.30★ | 0.40★ | **0.07★** |
| shift from a ring 1★ review-bomb | 0.83★ | 0.74★ | **0.13★** |
| shift from one-shot sockpuppets | 0.30 / 0.83★ | 0.00★ | **0.00★** |

The bridged score is **more accurate on a clean store** and moves **4–6× less** under a ring attack. The debiased baseline handles sockpuppets just as well, but it does worse than the plain average against rings. That shows the gain comes from the camp factors, not just from the history gate.

`tests/test_model.py` checks these properties:

- the fit is deterministic
- it tracks true quality
- sockpuppets are ignored
- a ring moves the bridged score less than half as much as the raw score
- ring members separate in factor space

## Run it

```bash
pip install -r requirements.txt
uvicorn server:app --port 8000      # open http://localhost:8000
python -m pytest -q                 # 8 tests
python -m experiments.robustness    # reproduces the table above
```

## Demo script (2–3 min video)

1. **Hook.** "Fakespot shut down in 2025. Star ratings are still easy to buy." Open **Zentrix Z1**, which shows 3.20★ in the store but 2.53★ bridged. The sidebar's red dots flag products where the two ratings disagree.
2. **Why.** The waterfall shows most of the gap is **camp effect**. The reviewer map shows a separate cluster of 5★ reviewers.
3. **Live attack.** Open an honest product such as **Kestrel Sport**. Inject 200 review-ring 5★ reviews. The store rating jumps +0.84★ while the bridged rating moves +0.16★, and a new cluster appears on the reviewer map.
4. **Collateral damage.** The ring also 1★'d competitors. **Sonique S2** drops to about 2.0★ in the store but holds about 3.1★ bridged.
5. **Ground truth.** Tick "Show simulator ground truth" to outline the seeded fakes and show each product's true quality.
6. **Close.** Show the results table: it's more accurate with no attack and moves 4–6× less under one.

## Limitations and next steps

- **The data is simulated.** The next step is real data, such as Amazon Reviews 2023 or the Yelp Open Dataset. Real data is much sparser, so the dense SVD init would need to become a sparse `svds`, and the factor count and regularization would need retuning.
- **Patient attackers.** A ring that builds long, honest histories and spreads out its targets looks more like a taste group. Bridging raises the cost of an attack but doesn't make one impossible.
- **New genuine reviewers are excluded too**, which is the price of the track-record gate. The UI shows how many reviews were counted.
- **Planned: a browser extension** that overlays the bridged score on store pages.

## Project layout

```
bridged/model.py         bridging matrix factorization (ALS, NumPy)
bridged/synth.py         simulated marketplace, review rings, sockpuppets
experiments/robustness.py  attack experiment behind the results table
server.py                FastAPI demo API (attack, refit, explain)
web/                     demo UI (vanilla JS + hand-drawn SVG charts)
tests/                   pytest suite
```
