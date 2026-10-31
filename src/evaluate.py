"""
LunaRec evaluation: does the shortlist-disagreement criterion beat the baselines?

MULTI-ATTRIBUTE VERSION. Earlier revisions used one unknown attribute (the OHRC
hazard label) against two known DEM attributes (slope, roughness). That was
abandoned because the DEM could not be registered to the OHRC grid: every
DEM-derived quantity was uncorrelated with every image-derived quantity on all
four scenes (|r| < 0.14), while two OHRC-derived columns correlated with each
other perfectly sensibly. The DEM features described different ground than the
labels, and no translation fixed it.

So all three attributes now come from OHRC itself, co-registered by
construction. That changes the problem for the better: every attribute is
unknown until a patch is imaged, and imaging reveals all three at once. Each
stakeholder weights them differently, so the shortlist is uncertain in all
three dimensions at once -- which is what multi-stakeholder recommendation
under acquisition actually looks like.

  safety      <- hazard       (shadow_local, minimised)  boulder/shadow density
  operations  <- illumination (dn_mean, maximised)       solar power, thermal
  science     <- texture      (dn_std, maximised)        surface diversity

Each attribute gets its OWN Gaussian process with its own fitted variogram
(ranges differ: ~1.1-2.0 km for illumination, ~2.5-3.5 km for hazard). The
attributes are modelled as independent given position, so the consensus utility
u = sum_a w_a z_a is itself a GP with covariance sum_a w_a^2 K_a -- exact, not
an approximation, and what the criterion propagates through.

Policies compared, all under the same acquisition budget:
  random          : pick uniformly
  coverage        : pick the patch furthest from everything already imaged
  max-variance    : pick the highest posterior utility variance
  voi-value       : ABLATION -- same variance and spatial reach, blind to
                    shortlist structure
  disagreement    : the criterion -- expected reduction in shortlist
                    disagreement across stakeholders

Metrics, in three families:
  shortlist disagreement  -- the contribution
  Precision@K / NDCG@K    -- standard recommender metrics vs the
  / Kendall tau              full-information oracle shortlist
  acquisitions to P@K     -- the same result counted in imaging operations,
                             which is a technical effect rather than a statistic

Usage:
    python evaluate.py <patches.csv> [n_rep] [budget]
"""

import os
import sys

import numpy as np
import pandas as pd

K = 20                 # shortlist size
N_SAMP = 240           # posterior samples
N_STAKE = 3

# (column, sign, label). Sign is +1 when more is better, -1 when less is.
ATTRS = [('shadow_local', -1.0, 'hazard'),
         ('dn_mean', +1.0, 'illumination'),
         ('dn_std', +1.0, 'texture')]

# Per-attribute GP priors, (range_m, nugget_fraction). Defaults are placeholders
# only -- main() fits them from each scene and overwrites this. They are not
# constants to trust: an earlier revision carried a hardcoded 1100 m / 0.10 that
# was wrong by ~3x and ~5x against every scene later measured.
PRIOR = [(2500.0, 0.45), (1600.0, 0.08), (2300.0, 0.35)]

# Stakeholder weight centroids: rows are stakeholders, columns are ATTRS.
W_CENTROID = np.array([[0.15, 0.20, 0.65],     # science     -> texture
                       [0.70, 0.20, 0.10],     # safety      -> hazard
                       [0.20, 0.60, 0.20]])    # operations  -> illumination
W_CONSENSUS = W_CENTROID.mean(0)

# How uncertain the stakeholder weights are. A MODELLING CHOICE, not a
# measurement -- no elicited preference data exists to fit it from -- and it
# sets the size of the disagreement the criterion acts on. sensitivity.py
# sweeps it; a result that holds only at one value is a result about the value.
W_SPREAD = 0.07


def spherical_cov(D, rng_, nugget):
    """Spherical covariance: 1 at zero lag, 0 beyond the range."""
    c = np.where(D < rng_, 1.0 - 1.5 * D / rng_ + 0.5 * (D / rng_) ** 3, 0.0)
    return (1 - nugget) * c + nugget * (D == 0)


def posterior(obs_idx, obs_val, K_full, noise=1e-4):
    """GP posterior mean and covariance given observations (zero prior mean)."""
    n = K_full.shape[0]
    if len(obs_idx) == 0:
        return np.zeros(n), K_full.copy()
    Koo = K_full[np.ix_(obs_idx, obs_idx)] + noise * np.eye(len(obs_idx))
    Kso = K_full[:, obs_idx]
    mu = Kso @ np.linalg.solve(Koo, obs_val)
    cov = K_full - Kso @ np.linalg.solve(Koo, Kso.T)
    return mu, cov


def shortlist_entropy(util, k=None):
    """Disagreement = mean inclusion entropy over (sample, stakeholder) shortlists.

    util : (n_samp, n_stake, n_patch) utility draws
    """
    k = K if k is None else k
    n_s, n_t, n_p = util.shape
    kk = min(k, n_p)
    top = np.argpartition(-util, kk - 1, axis=2)[:, :, :kk]
    mem = np.zeros(util.shape, bool)
    np.put_along_axis(mem, top, True, axis=2)
    p = mem.reshape(-1, n_p).mean(0)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))), p


def precision_at_k(pred_top, oracle_top, k=None):
    # k=None, resolved at CALL time. Written as k=K this bound 20 permanently
    # at import, so a sweep over shortlist length scored 5-item shortlists
    # against a 20-item metric.
    k = K if k is None else k
    return len(set(pred_top[:k]) & set(oracle_top[:k])) / float(k)


def ndcg_at_k(pred_top, util_true, oracle_top, k=None):
    """Graded-relevance NDCG, relevance = oracle utility shifted non-negative.

    Shifting by the global minimum rather than clipping at zero preserves every
    item's ordering; clipping would collapse all below-zero utilities to equal
    relevance and flatter any ranking that gets the bottom of the list wrong.
    """
    k = K if k is None else k
    k = min(k, len(pred_top), len(oracle_top))
    rel = util_true - util_true.min()
    disc = 1.0 / np.log2(np.arange(k) + 2)
    dcg = float((rel[pred_top[:k]] * disc).sum())
    idcg = float((rel[oracle_top[:k]] * disc).sum())
    return dcg / idcg if idcg > 0 else np.nan


def kendall_tau(a, b):
    try:
        from scipy.stats import kendalltau
        return float(kendalltau(a, b).statistic)
    except Exception:
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])


def pick_best(score, free, rng, tol=1e-12):
    """argmax over `free`, breaking ties at random.

    GP posterior variance depends only on WHERE observations are, never on
    their values, so max-variance is deterministic: 30 repetitions gave 30
    identical runs. At step 0 every patch has equal prior variance, so a plain
    np.argmax returned index 0 every time and the whole trajectory followed
    from that arbitrary start -- understating the baseline and inflating the
    reported margin.
    """
    s = score[free]
    best = np.flatnonzero(s >= s.max() - tol)
    return int(free[rng.choice(best)])


def run(df, policy, budget, rng):
    n = len(df)
    xy = df[['patch_x', 'patch_y']].values
    D = np.hypot(xy[:, 0][:, None] - xy[:, 0], xy[:, 1][:, None] - xy[:, 1])

    # Standardise each attribute and orient it so larger is always better. All
    # three are on the same footing: an earlier revision left the hazard in raw
    # units (std ~0.03 against standardised attributes at 1.0), giving it ~3% of
    # its intended weight, and every policy then tied regardless of what it
    # imaged.
    Z = np.empty((len(ATTRS), n))
    for a, (col, sgn, _) in enumerate(ATTRS):
        raw = df[col].values.astype(float)
        Z[a] = sgn * (raw - raw.mean()) / raw.std()

    Ks = [spherical_cov(D, PRIOR[a][0], PRIOR[a][1]) for a in range(len(ATTRS))]

    W = rng.normal(W_CENTROID, W_SPREAD, (N_SAMP, N_STAKE, len(ATTRS)))
    W = np.clip(W, 0, None)
    # At large W_SPREAD the clip can zero an entire stakeholder's row, and
    # normalising then divides by zero. A stakeholder who cares about nothing is
    # better modelled as caring about everything equally than as NaN.
    den = W.sum(2, keepdims=True)
    W = np.where(den > 1e-12, W / np.maximum(den, 1e-12), 1.0 / len(ATTRS))

    # Full-information reference: the shortlist perfect knowledge would choose.
    util_true = W_CONSENSUS @ Z
    oracle_top = np.argsort(-util_true)[:K]

    # Prior covariance of the consensus utility. Attributes are independent
    # given position, so Var(sum_a w_a z_a) = sum_a w_a^2 K_a exactly.
    w2 = W_CONSENSUS ** 2
    Ku_prior = sum(w2[a] * Ks[a] for a in range(len(ATTRS)))

    obs_idx, history = [], []

    for step in range(budget + 1):
        oi = np.array(obs_idx, int)
        mus, covs = [], []
        for a in range(len(ATTRS)):
            mu, cov = posterior(oi, Z[a][oi] if len(oi) else np.array([]), Ks[a])
            mus.append(mu)
            covs.append(cov)

        # Samples per attribute, then utility per (sample, stakeholder, patch).
        S = np.stack([rng.normal(mus[a], np.sqrt(np.clip(np.diag(covs[a]), 1e-12, None)),
                                 (N_SAMP, n)) for a in range(len(ATTRS))], axis=1)
        util = np.einsum('sta,sap->stp', W, S)

        ent, incl = shortlist_entropy(util)
        util_hat = W_CONSENSUS @ np.array(mus)
        pred_top = np.argsort(-util_hat)[:K]
        history.append({
            'ent': ent,
            'prec': precision_at_k(pred_top, oracle_top),
            'ndcg': ndcg_at_k(pred_top, util_true, oracle_top),
            'tau': kendall_tau(util_hat, util_true) if step == budget else np.nan,
        })

        if step == budget:
            break

        cov_u = sum(w2[a] * covs[a] for a in range(len(ATTRS)))
        var_u = np.clip(np.diag(cov_u), 1e-12, None)

        free = np.setdiff1d(np.arange(n), obs_idx)
        if policy == 'random':
            pick = int(rng.choice(free))
        elif policy == 'coverage':
            if not obs_idx:
                pick = int(rng.choice(free))
            else:
                pick = int(free[np.argmax(D[np.ix_(free, obs_idx)].min(1))])
        elif policy == 'max-variance':
            pick = pick_best(var_u, free, rng)
        elif policy == 'voi-value':
            # ABLATION: identical to 'disagreement' except blind to shortlist
            # structure. Same variance, same spatial propagation -- only the
            # inclusion-uncertainty term is removed. The gap between this and
            # 'disagreement' is exactly what the shortlist framing buys.
            reach = (Ku_prior * np.sqrt(var_u)[None, :]).sum(1)
            pick = pick_best(np.sqrt(var_u) * reach, free, rng)
        elif policy == 'disagreement':
            unc = 1.0 - np.abs(2 * incl - 1)            # 1 when 50/50
            reach = (Ku_prior * np.sqrt(var_u)[None, :]).sum(1)
            pick = pick_best(unc * np.sqrt(var_u) * reach, free, rng)
        else:
            raise ValueError(policy)

        obs_idx.append(pick)

    return history


def stability(history, tol=0.05):
    """Acquisitions needed to get, and stay, within `tol` of the final disagreement.

    An earlier definition (90% overlap with the final top-k) fired spuriously:
    a shortlist can brush past its final membership early while still being
    highly uncertain, so one policy reported 2.2 on one scene and 32.2 on
    another.
    """
    ent = np.array([h['ent'] for h in history])
    start, final = ent[0], ent[-1]
    if start <= final:
        return len(ent)
    target = final + tol * (start - final)
    below = ent <= target
    for i in range(len(ent)):
        if below[i:].all():
            return i
    return len(ent)


def acq_to_precision(prec_curve, target=0.90):
    """Acquisitions needed to reach, and hold, Precision@K >= target.

    This is the figure to quote as a technical effect: it counts operations of
    a scarce instrument, not a change in a statistic.
    """
    p = np.asarray(prec_curve)
    for i in range(len(p)):
        if (p[i:] >= target).all():
            return i
    return len(p)


def acq_to_match(prec_curve, target):
    """Acquisitions for this curve to reach, and hold, `target`.

    Differs from acq_to_precision only in that the target is supplied rather
    than fixed at 0.90. That matters: no baseline reaches 0.90 within 150
    acquisitions on any scene, so acq_to_precision cannot express a saving --
    there is nothing to compare against. Setting the target to what the best
    baseline achieves with its ENTIRE budget asks the operationally meaningful
    question instead: how many acquisitions does the criterion need to match
    the best result anything else can produce?

    Computed PER REPETITION so the saving carries a confidence interval and a
    paired test, like every other claim in the evaluation.
    """
    p = np.asarray(prec_curve)
    for i in range(len(p)):
        if (p[i:] >= target).all():
            return i
    return len(p)


def fit_priors(full):
    """Fit range and nugget per attribute from this scene."""
    global PRIOR
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    from fit_variogram import empirical_variogram, fit as fit_vgm
    sub = full.sample(min(1400, len(full)), random_state=0)
    xy = sub[['patch_x', 'patch_y']].values
    out = []
    for col, _, label in ATTRS:
        raw = sub[col].values.astype(float)
        z = (raw - raw.mean()) / raw.std()
        c, g, k = empirical_variogram(xy, z)
        (nug, sill, rng_), r2 = fit_vgm(c, g, k)
        out.append((float(rng_), float(np.clip(nug / sill, 0.0, 0.95)), r2, label))
    PRIOR = [(r, n) for r, n, _, _ in out]
    return out


def main(path, n_rep=5, budget=60):
    n_rep, budget = int(n_rep), int(budget)
    cols = [c for c, _, _ in ATTRS] + ['patch_x', 'patch_y']
    full = pd.read_csv(path).dropna(subset=cols)
    n_sub = min(400, len(full))
    print(f"{path}: {len(full)} patches, {n_sub} per rep, K={K}, "
          f"budget={budget}, {n_rep} reps")

    try:
        fits = fit_priors(full)
        print("fitted GP priors (per attribute, from this scene):")
        for rng_, nug, r2, label in fits:
            print(f"  {label:<13} range {rng_:6.0f} m   nugget {nug:.3f}   "
                  f"variogram R2 {r2:.3f}")
        print()
    except Exception as e:
        print(f"variogram fit unavailable ({e}); using placeholder priors "
              f"{PRIOR} -- these are ASSUMPTIONS, not measurements\n")

    pols = ['random', 'coverage', 'max-variance', 'voi-value', 'disagreement']
    keys = ('ent', 'stab', 'prec', 'ndcg', 'tau', 'acq90')
    res = {p: {k: [] for k in keys} for p in pols}
    for rep in range(n_rep):
        # A DIFFERENT subsample each rep. With a fixed seed, deterministic
        # policies saw the same problem every time and reported sd = 0, leaving
        # the ablation -- the comparison the claim rests on -- with no error
        # bars. All policies share the rep's subsample, so tests are PAIRED.
        df = (full.sample(n_sub, random_state=rep).reset_index(drop=True)
              if len(full) > n_sub else full)
        for p in pols:
            h = run(df, p, budget, np.random.default_rng(100 + rep))
            prec = [s['prec'] for s in h]
            res[p]['ent'].append([s['ent'] for s in h])
            res[p]['prec'].append(prec)
            res[p]['ndcg'].append([s['ndcg'] for s in h])
            res[p]['tau'].append(h[-1]['tau'])
            res[p]['stab'].append(stability(h))
            res[p]['acq90'].append(acq_to_precision(prec))

    print("the contribution -- shortlist disagreement")
    print('%-14s %12s %12s %16s' % ('policy', 'final dis.', 'dis. @20',
                                    'acq to within 5%'))
    for p in pols:
        E = np.array(res[p]['ent'])
        print('%-14s %12.4f %12.4f %16.1f'
              % (p, E[:, -1].mean(), E[:, min(20, budget)].mean(),
                 np.mean(res[p]['stab'])))

    print("\nstandard recommender metrics -- vs the full-information shortlist")
    print('%-14s %12s %9s %12s %12s' % ('policy', 'P@K final', 'sd', 'NDCG@K',
                                        'Kendall tau'))
    for p in pols:
        P = np.array(res[p]['prec'])
        N = np.array(res[p]['ndcg'])
        sd = P[:, -1].std()
        flag = '  <- deterministic' if sd < 1e-12 and n_rep > 1 else ''
        print('%-14s %12.3f %9.3f %12.3f %12.3f%s'
              % (p, P[:, -1].mean(), sd, N[:, -1].mean(),
                 np.nanmean(res[p]['tau']), flag))

    # The operationally meaningful regime. Imaging 150 of 400 patches at
    # 0.3 m/px is not a realistic tasking budget -- Chandrayaan-3 screened 20
    # candidate sites and 8 received stereo coverage.
    checks = [c for c in (5, 10, 20, 30, 50, 75, 100, 150) if c <= budget]
    print("\nPrecision@K against acquisition budget -- the realistic regime")
    print('%-14s' % 'policy' + ''.join('%9d' % c for c in checks))
    for p in pols:
        P = np.array(res[p]['prec'])
        print('%-14s' % p + ''.join('%9.3f' % P[:, c].mean() for c in checks))

    curve = pd.DataFrame({p: np.array(res[p]['prec']).mean(0) for p in pols})
    curve.index.name = 'acquisitions'
    out = path.replace('.csv', '_precision_curve.csv')
    curve.to_csv(out)

    # Per-rep results, not just the mean curve. The curve CSV averages over
    # reps, which loses exactly what the paired tests need: the per-instance
    # differences. Writing this means a notebook can recompute margins, wins
    # and t-statistics from data rather than by parsing printed tables.
    # Matched-performance saving, per rep and PAIRED: each repetition's target
    # is the best baseline's own final score in THAT repetition, so the two
    # policies are compared on the same problem instance.
    base_pols = [p for p in pols if p != 'disagreement']
    best_base = max(base_pols, key=lambda p: np.array(res[p]['prec'])[:, -1].mean())
    match = {}
    for p in pols:
        match[p] = [acq_to_match(res[p]['prec'][i],
                                 res[best_base]['prec'][i][-1])
                    for i in range(n_rep)]

    per_rep = pd.DataFrame([
        {"scene": os.path.basename(path).replace(".csv", ""),
         "policy": p, "rep": i,
         "acq_to_match_best_base": match[p][i],
         "best_baseline": best_base,
         "target": res[best_base]["prec"][i][-1],
         "prec_final": res[p]["prec"][i][-1],
         "prec_at_20": res[p]["prec"][i][min(20, budget)],
         "ndcg_final": res[p]["ndcg"][i][-1],
         "ent_final": res[p]["ent"][i][-1],
         "tau": res[p]["tau"][i],
         "stability": res[p]["stab"][i],
         "acq_to_p90": res[p]["acq90"][i]}
        for p in pols for i in range(n_rep)])
    per_rep["K"] = K
    per_rep["budget"] = budget
    per_rep["prior"] = str([(round(r), round(n, 3)) for r, n in PRIOR])
    rep_out = path.replace(".csv", "_per_rep.csv")
    per_rep.to_csv(rep_out, index=False)
    print(f"per-rep results -> {rep_out}")

    # Per-rep CURVES. Without these the matched-performance saving can only be
    # computed on the averaged curve, which leaves the headline technical-effect
    # figure as the one claim in the evaluation with no error bar.
    rec = [(p, i, b, v)
           for p in pols for i in range(n_rep)
           for b, v in enumerate(res[p]['prec'][i])]
    pc_out = path.replace('.csv', '_per_rep_curve.csv')
    pd.DataFrame(rec, columns=['policy', 'rep', 'acquisitions', 'prec']) \
      .to_csv(pc_out, index=False)
    print(f"per-rep curves  -> {pc_out}  ({len(rec)} rows)")
    print(f"\nfull curve -> {out}  (for the figure)")

    print("\ntechnical effect -- imaging operations consumed")
    print("  (a) acquisitions to a FIXED target of P@K >= 0.90")
    print('%-14s %22s' % ('policy', 'acq to P@K>=0.90'))
    reached = 0
    for p in pols:
        a = np.mean(res[p]['acq90'])
        note = '' if a < budget else '   (not reached within budget)'
        reached += int(a < budget)
        print('%-14s %22.1f%s' % (p, a, note))
    if reached < 2:
        print("  -> fewer than two policies reach the fixed target, so no saving")
        print("     can be quoted from it. Use (b).")

    print("\n  (b) acquisitions to MATCH the best baseline's full-budget score")
    tgt = np.array([res[best_base]['prec'][i][-1] for i in range(n_rep)])
    ours_m = np.array(match['disagreement'], float)
    saved = budget - ours_m
    sd = saved.std(ddof=1) if n_rep > 1 else 0.0
    ci = 1.96 * sd / np.sqrt(n_rep) if n_rep > 1 else 0.0
    print(f"      best baseline   : {best_base}  (target {tgt.mean():.3f})")
    print(f"      criterion reaches it at {ours_m.mean():.1f} of {budget} acquisitions")
    print(f"      saved           : {saved.mean():.1f} +/- {ci:.1f} (95% CI), "
          f"{100*saved.mean()/budget:.0f}%")
    print(f"      reps where the criterion got there early: "
          f"{int((saved > 0).sum())}/{n_rep}")
    print("      -> this is the sentence for the specification: a measured")
    print("         reduction in use of a scarce imaging instrument.")

    print("\npaired comparison on Precision@K -- criterion vs each baseline")
    print('%-14s %10s %10s %12s %10s' % ('baseline', 'mean diff', 'wins',
                                         't (paired)', 'p'))
    ours_p = np.array(res['disagreement']['prec'])[:, -1]
    for p in pols:
        if p == 'disagreement':
            continue
        base_p = np.array(res[p]['prec'])[:, -1]
        d = ours_p - base_p
        wins = int((d > 0).sum())
        if d.std(ddof=1) > 0:
            try:
                from scipy.stats import ttest_rel
                t = ttest_rel(ours_p, base_p)
                tval, pval = float(t.statistic), float(t.pvalue)
            except Exception:
                tval = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                pval = np.nan
        else:
            tval, pval = (np.inf if d.mean() > 0 else np.nan), 0.0
        tag = '  <- ABLATION' if p == 'voi-value' else ''
        print('%-14s %10.3f %10s %12.2f %10.2g%s'
              % (p, d.mean(), f"{wins}/{len(d)}", tval, pval, tag))

    base = np.array(res['max-variance']['ent'])[:, -1].mean()
    ours = np.array(res['disagreement']['ent'])[:, -1].mean()
    print(f"\ndisagreement vs max-variance: {100*(base-ours)/base:+.1f}% final disagreement")
    print("(negative = worse; the criterion must beat max-variance to matter)")

    a_base = np.mean(res['max-variance']['acq90'])
    a_ours = np.mean(res['disagreement']['acq90'])
    if a_base < budget and a_ours < budget:
        print(f"acquisitions saved vs max-variance: {a_base - a_ours:+.1f} "
              f"of {a_base:.1f}  ({100*(a_base-a_ours)/a_base:+.1f}%)")
        print("  -> this is the sentence for the specification: a measured")
        print("     reduction in use of a scarce imaging instrument.")
    else:
        print("acquisitions-saved comparison needs both policies to reach "
              "P@K>=0.90 within the budget; raise the budget to quote it.")


if __name__ == '__main__':
    main(*sys.argv[1:4])