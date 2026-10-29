"""
The shortlist-disagreement acquisition criterion.

This file is the invention, stated independently of how it is evaluated. It
takes posterior samples and stakeholder weights and returns which candidate to
measure next. It knows nothing about Gaussian processes, variograms, the Moon,
or any particular sensor -- evaluate.py supplies a spatial GP over lunar
patches, but any sampler producing draws of the same shape works unchanged.

That independence is deliberate and load-bearing. A criterion tied to one model
claims narrowly; one that consumes any posterior covers the family.

THE PROBLEM

A set of candidate items must be narrowed to a shortlist of k. Each item has
several attributes that matter. Several stakeholders care about those
attributes in different proportions, and nobody knows the exact proportions --
the weights are uncertain, and no amount of measuring the items will resolve
them, because they are preferences rather than facts.

The attributes themselves are unknown until an item is measured, and measuring
is expensive. Which item should be measured next?

WHAT IS CLAIMED

The usual answers target the wrong quantity:

  uncertainty sampling  measures where the posterior is widest, whether or not
                        that item could ever make the shortlist
  value of information  measures where the expected score moves most, which
                        rewards resolving items deep inside or far outside the
                        shortlist where the decision is not in doubt

Neither asks which measurement would settle MEMBERSHIP. An item whose
shortlist inclusion is already certain -- clearly in, clearly out -- teaches
nothing about the decision however uncertain its attributes are. An item sitting
on the boundary, included under some stakeholder weightings and excluded under
others, is where a measurement changes what gets chosen.

So the criterion scores each candidate by the DISAGREEMENT its measurement
would resolve: the entropy of its shortlist membership, pooled over both
posterior draws (attribute uncertainty) and stakeholder weightings (preference
uncertainty), weighted by how much observing it also resolves the membership of
items correlated with it, and divided by what the measurement costs.

Pooling over stakeholders is the part with no counterpart in the prior art
found to date. Existing boundary-targeted acquisition rules model one scalar
quality per item and ask whether THAT puts the item in the top k. Here an item
can be securely inside one stakeholder's shortlist and securely outside
another's, and that disagreement is precisely what a measurement can resolve --
not by changing anyone's preferences, but by establishing the attribute values
all of them are arguing over.

SHAPES, used throughout

    samples   (n_samp, n_attr, n_item)   posterior draws of attribute values,
                                         oriented so larger is always better
    weights   (n_samp, n_stake, n_attr)  draws of each stakeholder's weights,
                                         non-negative, summing to 1 per
                                         stakeholder
    coupling  (n_item, n_item)           how much observing i informs j, in
                                         [0, 1], 1 on the diagonal. Identity
                                         means independent items.
    cost      (n_item,)                  measurement cost per item

The synthetic fixtures under __main__ are a smoke test of the arithmetic.
They are NOT validation: the empirical results come from evaluate.py over
Chandrayaan-2 OHRC scenes.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Shortlist:
    """A top-k membership pattern and the disagreement it carries."""
    inclusion: np.ndarray      # (n_item,) P(item in shortlist)
    disagreement: float        # mean binary entropy of inclusion, in bits


def utilities(samples: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Utility of every item under every (draw, stakeholder) pair.

    Returns (n_samp, n_stake, n_item).
    """
    if samples.ndim != 3 or weights.ndim != 3:
        raise ValueError("samples must be (n_samp, n_attr, n_item) and "
                         "weights (n_samp, n_stake, n_attr)")
    if samples.shape[0] != weights.shape[0]:
        raise ValueError("samples and weights must share the sample axis")
    if samples.shape[1] != weights.shape[2]:
        raise ValueError("attribute count differs between samples and weights")
    return np.einsum('sta,sap->stp', weights, samples)


def shortlist(samples: np.ndarray, weights: np.ndarray, k: int) -> Shortlist:
    """Membership probability and disagreement of the top-k shortlist.

    Disagreement is the mean binary entropy of membership across items. It is
    zero when every item is decisively in or out under every draw and every
    stakeholder, and maximal when membership is a coin flip. It is the quantity
    the criterion drives down, and it is defined on the DECISION rather than on
    the attribute values -- two posteriors with identical variance can carry
    very different disagreement.
    """
    util = utilities(samples, weights)
    n_p = util.shape[2]
    kk = int(min(k, n_p))
    if kk < 1:
        raise ValueError("k must be at least 1")
    top = np.argpartition(-util, kk - 1, axis=2)[:, :, :kk]
    mem = np.zeros(util.shape, dtype=bool)
    np.put_along_axis(mem, top, True, axis=2)
    p = mem.reshape(-1, n_p).mean(0)
    return Shortlist(inclusion=p, disagreement=_entropy(p))


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p))))


def boundary_uncertainty(inclusion: np.ndarray) -> np.ndarray:
    """How undecided each item's membership is: 1 at p=0.5, 0 at p=0 or 1.

    The triangular form 1 - |2p - 1| is used rather than binary entropy. Both
    peak at p = 0.5 and vanish at the ends; the triangle is linear in p, so an
    item at p = 0.1 scores a tenth of one at p = 0.5 rather than the ~0.47 that
    entropy assigns. That keeps the criterion from spending budget on items
    that are nearly decided, which is the behaviour the whole thing exists to
    avoid.
    """
    return 1.0 - np.abs(2.0 * inclusion - 1.0)


def score(samples: np.ndarray,
          weights: np.ndarray,
          k: int,
          spread: np.ndarray,
          coupling: Optional[np.ndarray] = None,
          cost: Optional[np.ndarray] = None,
          observed: Optional[np.ndarray] = None) -> np.ndarray:
    """Score every candidate; the next measurement is the argmax.

    score_i = boundary_uncertainty_i * spread_i * reach_i / cost_i

      boundary_uncertainty  how undecided item i's shortlist membership is
      spread                how much is left to learn about i (e.g. posterior
                            standard deviation of its utility)
      reach                 sum_j coupling[i, j] * spread_j -- how much
                            observing i also resolves items correlated with it
      cost                  what the measurement costs

    The three factors are necessary together. Drop boundary_uncertainty and the
    rule degenerates into value-of-information, which is the ablation in
    evaluate.py and which it beats on every scene tested. Drop reach and it
    cannot exploit correlation between items. Drop cost and it ignores that
    measurements differ in price.

    `observed` marks items already measured; they score -inf so they are never
    chosen again.
    """
    n_item = samples.shape[2]
    sl = shortlist(samples, weights, k)
    unc = boundary_uncertainty(sl.inclusion)

    spread = np.asarray(spread, dtype=float)
    if spread.shape != (n_item,):
        raise ValueError(f"spread must have shape ({n_item},)")

    if coupling is None:
        reach = spread
    else:
        if coupling.shape != (n_item, n_item):
            raise ValueError(f"coupling must be ({n_item}, {n_item})")
        reach = coupling @ spread

    s = unc * spread * reach
    if cost is not None:
        cost = np.asarray(cost, dtype=float)
        if np.any(cost <= 0):
            raise ValueError("costs must be positive")
        s = s / cost
    if observed is not None:
        s = np.where(np.asarray(observed, dtype=bool), -np.inf, s)
    return s


def next_measurement(samples, weights, k, spread, coupling=None, cost=None,
                     observed=None, rng=None) -> int:
    """Index of the item to measure next, ties broken at random.

    Random tie-breaking is not cosmetic. With a deterministic argmax, an
    all-equal score vector -- which is exactly the state before any measurement
    has been taken -- always returns index 0, so the entire trajectory follows
    from array order rather than from the data.
    """
    s = score(samples, weights, k, spread, coupling, cost, observed)
    best = np.flatnonzero(s >= s.max() - 1e-12)
    if rng is None:
        return int(best[0])
    return int(rng.choice(best))


if __name__ == '__main__':
    # Smoke test of the arithmetic only. NOT validation -- see evaluate.py.
    rng = np.random.default_rng(0)
    n_samp, n_attr, n_item, n_stake, k = 200, 3, 60, 3, 10

    truth = rng.normal(size=(n_attr, n_item))
    samples = truth[None] + rng.normal(scale=0.6, size=(n_samp, n_attr, n_item))
    W = rng.normal([[0.15, 0.20, 0.65], [0.70, 0.20, 0.10], [0.20, 0.60, 0.20]],
                   0.07, (n_samp, n_stake, n_attr))
    W = np.clip(W, 0, None)
    W /= W.sum(2, keepdims=True)

    sl = shortlist(samples, W, k)
    print(f"disagreement           {sl.disagreement:.4f} bits")
    print(f"items with p in .2-.8  {int(((sl.inclusion > .2) & (sl.inclusion < .8)).sum())}"
          f" of {n_item}  (the contested boundary)")

    spread = samples.std(0).mean(0)
    pick = next_measurement(samples, W, k, spread, rng=rng)
    print(f"next measurement       item {pick}  "
          f"(inclusion p = {sl.inclusion[pick]:.2f})")

    # A posterior that is certain about membership must carry no disagreement,
    # however wide its attribute uncertainty: the decision, not the variance,
    # is what the criterion measures.
    decided = np.tile(np.linspace(5, -5, n_item), (n_samp, n_attr, 1))
    decided += rng.normal(scale=0.01, size=decided.shape)
    print(f"\ndecided-shortlist case {shortlist(decided, W, k).disagreement:.4f} bits"
          f"  (near zero as required)")
