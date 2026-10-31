"""Off-grid stand-in (follow-up to 08): for targets between the training grid points,
estimate a retrieved airfoil's solver result by bilinear interpolation of its stored
NeuralFoil results at the 4 surrounding (Re, alpha) grid points. Used for the gate and the
free bias correction. The gate threshold (fraction of the tolerance) is chosen on
VALIDATION off-grid targets, then reported on test off-grid targets.
No diffusion (t = 0, the best setting in 08). Scoring as before: NeuralFoil, CL 1%, CD 5%.
"""
import argparse
import json

import numpy as np

import common
from afd import data, evaluate, experiment, metrics, refine

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--surrogate", default=str(common.MODELS / "surrogate.npz"))
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
ap.add_argument("--targets", type=int, default=100)
ap.add_argument("--budget", type=int, default=10)
ap.add_argument("--sur-steps", type=int, default=60000)
a = ap.parse_args()

d = experiment.load_train_test(a.dataset)
ev = evaluate.make(a.evaluator)
E, P, tr = d["entries"], d["params"], d["train_idx"]
try:
    sur = refine.Surrogate.load(a.surrogate)
except FileNotFoundError:
    sur = refine.Surrogate().fit(P[E["airfoil"][tr]], E["Re"][tr], E["alpha"][tr], E["CL"][tr], E["CD"][tr], steps=a.sur_steps)
    sur.save(a.surrogate)


class Norm:          # refine() only needs the parameter scale when diffusion is off
    x_sd = P[E["airfoil"][tr]].std(0) + 1e-9


mu, sd = d["c_train"].mean(0), d["c_train"].std(0)
C = (d["c_train"] - mu) / sd
ids = E["airfoil"][tr]
label = {(int(E["airfoil"][j]), float(E["Re"][j]), float(E["alpha"][j])): (E["CL"][j], E["CD"][j]) for j in tr}
RES, ALS = np.array(data.RES), np.array(data.ALPHAS)


def neighbours(Tg, k):
    out = []
    for t in Tg:
        seen = []
        for j in np.argsort(((C - (t - mu) / sd) ** 2).sum(1)):
            if ids[j] not in seen:
                seen.append(ids[j])
            if len(seen) == k:
                break
        out.append(seen)
    return np.array(out)


def interp(aids, tg):
    """Bilinear interpolation of stored (CL, CD) in (Re, alpha); nan if a corner is missing."""
    out = np.full((len(aids), 2), np.nan)
    for n, (i, t) in enumerate(zip(aids, tg)):
        r1 = int(np.clip(np.searchsorted(RES, t[0]) - 1, 0, len(RES) - 2))
        a1 = int(np.clip(np.searchsorted(ALS, t[1]) - 1, 0, len(ALS) - 2))
        fr = (t[0] - RES[r1]) / (RES[r1 + 1] - RES[r1])
        fa = (t[1] - ALS[a1]) / (ALS[a1 + 1] - ALS[a1])
        v = [label.get((int(i), float(RES[r1 + x]), float(ALS[a1 + y]))) for x in (0, 1) for y in (0, 1)]
        if any(c is None for c in v):
            continue
        v = np.array(v)
        out[n] = (1 - fr) * (1 - fa) * v[0] + (1 - fr) * fa * v[1] + fr * (1 - fa) * v[2] + fr * fa * v[3]
    return out


def ratio(y, tg):
    return np.maximum(abs(y[:, 0] - tg[:, 2]) / (0.01 * abs(tg[:, 2])), abs(y[:, 1] - tg[:, 3]) / (0.05 * tg[:, 3]))


def hits(x, tg):
    cl, cd, ok = metrics.score(ev, x, tg)
    y = np.column_stack([cl, cd])
    r = ratio(np.nan_to_num(y, nan=1e9), tg)
    return ok & (r <= 1), np.where(ok, r, np.inf), y


def offgrid(pool, seed):
    rng, out = np.random.default_rng(seed), []
    while len(out) < a.targets:
        re, al = rng.uniform(2.0e5, 3.5e5), rng.uniform(0.2, 9.8)
        if min(abs(re - g) for g in data.RES) < 1.0e4 or abs(al - round(al)) < 0.2:
            continue
        r = ev.polar(P[rng.choice(pool)], re, [al])
        if r["ok"][0] and 0 <= r["CL"][0] <= 2 and 0 < r["CD"][0] <= 0.05:
            out.append([re, al, r["CL"][0], r["CD"][0]])
    return np.array(out)


def rrv(tg, nb, th, use_interp=True):
    """Retrieve-refine-verify without diffusion; gate keeps the retrieved airfoil when its
    interpolated result is within th x tolerance (th = 0: never keep)."""
    n, K, rng = len(tg), nb.shape[1], np.random.default_rng(0)
    got = np.zeros((n, K), bool)
    best_x, best_e, bias = P[nb[:, 0]].copy(), np.full(n, np.inf), np.zeros((n, 2))
    for k in range(K):
        if k < 3:
            start = P[nb[:, k]]
            yi = interp(nb[:, k], tg) if use_interp else np.full((n, 2), np.nan)
            b0 = np.where(np.isfinite(yi), yi - sur.predict(start, tg[:, 0], tg[:, 1]), 0.0)
            x = refine.refine(Norm, sur, start, tg, b0, rng, use_diffusion=False)
            keep = np.isfinite(yi[:, 0]) & (ratio(np.nan_to_num(yi, nan=1e9), tg) <= th)
            x[keep] = start[keep]
        else:
            x = refine.refine(Norm, sur, best_x, tg, bias, rng, use_diffusion=False)
        h, e, y = hits(x, tg)
        got[:, k] = h
        b = e < best_e
        if b.any():
            best_x[b], best_e[b] = x[b], e[b]
            bias[b] = y[b] - sur.predict(x[b], tg[b, 0], tg[b, 1])
    return [float(got[:, 0].mean()), float(got.any(1).mean())]


def lookup(tg, nb):
    got = np.column_stack([hits(P[nb[:, k]], tg)[0] for k in range(nb.shape[1])])
    return [float(got[:, 0].mean()), float(got.any(1).mean())]


va_air = np.unique(E["airfoil"][data.split_idx(E, d["split"], "val")])
te_air = np.unique(E["airfoil"][d["test_idx"]])
TV, TT = offgrid(va_air, 11), offgrid(te_air, 7)          # seed 7 = same test targets as 08
nbV, nbT = neighbours(TV, a.budget), neighbours(TT, a.budget)
yi = interp(nbV[:, 0], TV)
yt = np.column_stack(metrics.score(ev, P[nbV[:, 0]], TV)[:2])
okm = np.isfinite(yi[:, 0]) & np.isfinite(yt[:, 0])
OUT = {"interp_error_val": dict(
    median_abs_CL=float(np.median(abs(yi[okm, 0] - yt[okm, 0]))),
    median_rel_CD=float(np.median(abs(yi[okm, 1] - yt[okm, 1]) / yt[okm, 1])),
    coverage=float(okm.mean()))}
print("interpolation vs solver on val retrieved airfoils:", OUT["interp_error_val"])
OUT["val"] = {"lookup": lookup(TV, nbV), "no interpolation (08 method)": rrv(TV, nbV, 0, use_interp=False)}
for th in (0, 0.5, 0.8, 1.0):
    OUT["val"][f"interp, gate {th}"] = rrv(TV, nbV, th)
    print("val", th, OUT["val"][f"interp, gate {th}"])
best = max((0, 0.5, 0.8, 1.0), key=lambda th: (OUT["val"][f"interp, gate {th}"][0], OUT["val"][f"interp, gate {th}"][1], -th))
OUT["chosen_gate"] = best
OUT["test"] = {"lookup": lookup(TT, nbT), "no interpolation (08 method)": rrv(TT, nbT, 0, use_interp=False),
               f"interp, gate {best}": rrv(TT, nbT, best)}
print("test", OUT["test"])
(common.RESULTS / "offgrid.json").write_text(json.dumps(OUT, indent=2))
print(json.dumps(OUT, indent=2))
print("EXIT_OK")
