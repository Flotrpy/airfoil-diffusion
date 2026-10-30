"""Metrics from the papers.

Geng et al.: tolerances of 1% on CL and 5% on CD, at most 100 design trials per
target, N95 = trials needed for a 95% chance of at least one design inside the
tolerances, with q = 1 - (1 - p)^N; designs the solver cannot evaluate count as
infeasible.
Graves et al.: slope of achieved vs requested value (1.0 is ideal).
Diniz: maximum mean discrepancy between generated and real shape distributions.
"""
import numpy as np

TOL_CL, TOL_CD, MAX_TRIALS = 0.01, 0.05, 100
LOOSE = [(0.02, 0.05), (0.05, 0.10)]   # looser (CL, CD) tolerances, reported alongside Geng's


def n95(p):
    if p <= 0:
        return float("inf")
    if p >= 1:
        return 1.0
    return float(np.log(0.05) / np.log(1.0 - p))


def score(evaluator, params, targets):
    """Solver-check generated params (n, 11) against targets (n, 4: Re, alpha, CL, CD).
    Returns arrays CL, CD, ok."""
    n = len(params)
    cl, cd, ok = np.full(n, np.nan), np.full(n, np.nan), np.zeros(n, bool)
    for i in range(n):
        r = evaluator.polar(params[i], targets[i, 0], [targets[i, 1]])
        cl[i], cd[i], ok[i] = r["CL"][0], r["CD"][0], bool(r["ok"][0])
    return cl, cd, ok


def trial_protocol(generate, evaluator, targets, max_trials=MAX_TRIALS):
    """generate(cond (m,4)) -> params (m,11). Runs Geng's protocol for every target.
    Returns a dict of summary numbers."""
    trials_used, best_err_cl, best_err_cd, fail, hit_rate = [], [], [], 0, []
    within10, loose = [], {tl: [] for tl in LOOSE}
    for tg in targets:
        cond = np.repeat(tg[None, :], max_trials, axis=0)
        x = generate(cond)
        cl, cd, ok = score(evaluator, x, cond)
        e_cl = np.abs(cl - tg[2]) / max(abs(tg[2]), 1e-9)
        e_cd = np.abs(cd - tg[3]) / max(abs(tg[3]), 1e-9)
        hit = ok & (e_cl <= TOL_CL) & (e_cd <= TOL_CD)
        hit_rate.append(hit.mean())
        within10.append(bool(hit[:10].any()))
        for tl in LOOSE:
            loose[tl].append(float((ok & (e_cl <= tl[0]) & (e_cd <= tl[1])).mean()))
        trials_used.append(int(np.argmax(hit)) + 1 if hit.any() else max_trials)
        if ok.any():
            j = np.flatnonzero(ok)[np.argmin(np.hypot(e_cl[ok], e_cd[ok]))]
            best_err_cl.append(e_cl[j])
            best_err_cd.append(e_cd[j])
        else:
            fail += 1
    p = float(np.mean(hit_rate))
    return dict(
        targets=len(targets),
        p_single_trial=p,
        N95=n95(p),
        mean_trials_to_tolerance=float(np.mean(trials_used)),
        share_with_hit=float(np.mean([h > 0 for h in hit_rate])),
        share_hit_within_10=float(np.mean(within10)),
        **{f"p_single_trial_tol_{a:g}_{b:g}": float(np.mean(v)) for (a, b), v in loose.items()},
        failure_rate=fail / len(targets),
        best_rel_rmse_CL=float(np.sqrt(np.mean(np.square(best_err_cl)))) if best_err_cl else float("nan"),
        best_rel_rmse_CD=float(np.sqrt(np.mean(np.square(best_err_cd)))) if best_err_cd else float("nan"),
    )


def single_shot(predict, evaluator, targets):
    """One design per target (baselines). Same tolerances."""
    x = predict(targets)
    cl, cd, ok = score(evaluator, x, targets)
    e_cl = np.abs(cl - targets[:, 2]) / np.maximum(np.abs(targets[:, 2]), 1e-9)
    e_cd = np.abs(cd - targets[:, 3]) / np.maximum(np.abs(targets[:, 3]), 1e-9)
    hit = ok & (e_cl <= TOL_CL) & (e_cd <= TOL_CD)
    return dict(
        targets=len(targets),
        p_single_trial=float(hit.mean()),
        **{f"p_single_trial_tol_{a:g}_{b:g}": float((ok & (e_cl <= a) & (e_cd <= b)).mean()) for a, b in LOOSE},
        infeasible_share=float(1 - ok.mean()),
        rel_rmse_CL=float(np.sqrt(np.nanmean(np.where(ok, e_cl, np.nan) ** 2))),
        rel_rmse_CD=float(np.sqrt(np.nanmean(np.where(ok, e_cd, np.nan) ** 2))),
        slope_CL=conditioning_slope(targets[ok, 2], cl[ok]),
        slope_CD=conditioning_slope(targets[ok, 3], cd[ok]),
    )


def conditioning_slope(requested, achieved):
    if len(requested) < 3:
        return float("nan")
    return float(np.polyfit(requested, achieved, 1)[0])


def mmd(x, y, max_n=1500, seed=0):
    """Gaussian-kernel MMD^2 with the median heuristic bandwidth."""
    rng = np.random.default_rng(seed)
    x = x[rng.permutation(len(x))[:max_n]]
    y = y[rng.permutation(len(y))[:max_n]]
    z = np.concatenate([x, y])
    d2 = ((z[:, None, :] - z[None, :, :]) ** 2).sum(-1)
    h = np.median(d2[d2 > 0])
    k = np.exp(-d2 / h)
    n, m = len(x), len(y)
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    return float((kxx.sum() - np.trace(kxx)) / (n * (n - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
