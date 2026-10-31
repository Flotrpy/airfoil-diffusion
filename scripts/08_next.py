"""Next steps after the Phase 1 pass (approved: A then B).

A1. Diffusion strength sweep for retrieve-refine-verify: warm-start depth t in {0, 10, 25, 50}
    (t = 0 means no diffusion), chosen on VALIDATION targets, then reported on the test targets.
A2. Off-grid targets: Re and alpha between the training grid points (like live telemetry),
    where the stored-result gate has nothing to read and every candidate is refined.
B.  Phase 2: PIDM-style physics-informed fine-tuning with ConFIG gradient combining
    (src/afd/pidm.py), compared with the Phase 1 model on the same targets.
Scoring as in Phase 1: NeuralFoil, CL within 1%, CD within 5%, one solver call per candidate.
"""
import argparse
import json
import time

import numpy as np

import common
from afd import data, evaluate, experiment, metrics, pidm, refine
from afd.diffusion import CondDDPM

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--model", default=str(common.MODELS / "ddpm.npz"))
ap.add_argument("--surrogate", default=str(common.MODELS / "surrogate.npz"))
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
ap.add_argument("--targets", type=int, default=100)
ap.add_argument("--budget", type=int, default=10)
ap.add_argument("--sur-steps", type=int, default=60000)
ap.add_argument("--ft-steps", type=int, default=8000)
ap.add_argument("--pidm-steps", type=int, default=30000)
ap.add_argument("--samples", type=int, default=10, help="cold samples per target")
a = ap.parse_args()
OUT = {}


def save():
    (common.RESULTS / "next_steps.json").write_text(json.dumps(OUT, indent=2, default=float))


T0 = time.time()
d = experiment.load_train_test(a.dataset)
ev = evaluate.make(a.evaluator)
model = CondDDPM.load(a.model)
E, P, tr = d["entries"], d["params"], d["train_idx"]
try:
    sur = refine.Surrogate.load(a.surrogate)
except FileNotFoundError:
    print("training surrogate on training-split labels ...")
    sur = refine.Surrogate().fit(P[E["airfoil"][tr]], E["Re"][tr], E["alpha"][tr], E["CL"][tr], E["CD"][tr], steps=a.sur_steps)
    sur.save(a.surrogate)

mu, sd = d["c_train"].mean(0), d["c_train"].std(0)
C = (d["c_train"] - mu) / sd
ids = E["airfoil"][tr]
label = {(int(E["airfoil"][j]), float(E["Re"][j]), float(E["alpha"][j])): (E["CL"][j], E["CD"][j]) for j in tr}
K = a.budget


def neighbours(Tg, k):
    out = []
    for t in Tg:
        order = np.argsort(((C - (t - mu) / sd) ** 2).sum(1))
        seen = []
        for j in order:
            if ids[j] not in seen:
                seen.append(ids[j])
            if len(seen) == k:
                break
        out.append(seen)
    return np.array(out)


def stored(aids, tg):
    return np.array([label.get((int(i), float(t[0]), float(t[1])), (np.nan, np.nan)) for i, t in zip(aids, tg)])


def within_tol(y, tg):
    return (abs(y[:, 0] - tg[:, 2]) <= 0.01 * abs(tg[:, 2])) & (abs(y[:, 1] - tg[:, 3]) <= 0.05 * tg[:, 3])


def hits(x, tg):
    cl, cd, ok = metrics.score(ev, x, tg)
    y = np.column_stack([cl, cd])
    e = np.hypot((cl - tg[:, 2]) / (0.01 * abs(tg[:, 2])), (cd - tg[:, 3]) / (0.05 * tg[:, 3]))
    return ok & within_tol(np.nan_to_num(y, nan=1e9), tg), np.where(ok, e, np.inf), y


def lookup(tg, nb):
    got = np.column_stack([hits(P[nb[:, k]], tg)[0] for k in range(nb.shape[1])])
    return [float(got[:, :b].any(1).mean()) for b in (1, nb.shape[1])]


def rrv(mdl, tg, nb, t_warm):
    """Approved Phase 1 method: stored-result gate, free bias, surrogate polish, diffusion
    warm start of depth t_warm (0 = no diffusion). Returns hit rate at budget 1 and K."""
    n, rng = len(tg), np.random.default_rng(0)
    use = t_warm > 0
    kw = dict(use_diffusion=use, t_warm=max(t_warm, 1), t_proj=min(5, max(t_warm, 1)))
    got = np.zeros((n, nb.shape[1]), bool)
    best_x, best_e, bias = P[nb[:, 0]].copy(), np.full(n, np.inf), np.zeros((n, 2))
    for k in range(nb.shape[1]):
        if k < 3:
            start = P[nb[:, k]]
            ylab = stored(nb[:, k], tg)
            ysur = sur.predict(start, tg[:, 0], tg[:, 1])
            x = refine.refine(mdl, sur, start, tg, np.where(np.isfinite(ylab), ylab - ysur, 0.0), rng, **kw)
            keep = np.isfinite(ylab[:, 0]) & within_tol(np.nan_to_num(ylab, nan=1e9), tg)
            x[keep] = start[keep]
        else:
            x = refine.refine(mdl, sur, best_x, tg, bias, rng, **kw)
        h, e, y = hits(x, tg)
        got[:, k] = h
        b = e < best_e
        best_x[b], best_e[b] = x[b], e[b]
        bias[b] = y[b] - sur.predict(x[b], tg[b, 0], tg[b, 1]) if b.any() else bias[b]
    return [float(got[:, 0].mean()), float(got.any(1).mean())]


def cold(mdl, tg, m):
    """Diffusion alone: m cold samples per target. Share hit with 1 sample and within m."""
    rng = np.random.default_rng(0)
    got = np.zeros((len(tg), m), bool)
    for j in range(m):
        got[:, j] = hits(mdl.sample(tg, rng, ddim_steps=50), tg)[0]
    return [float(got[:, 0].mean()), float(got.mean()), float(got.any(1).mean())]


def stamp(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}")


# targets
TT = experiment.pick_targets(d["c_test"], a.targets)
va = data.split_idx(E, d["split"], "val")
TV = experiment.pick_targets(data.conditions(E, va), a.targets, seed=1)
te_air = np.unique(E["airfoil"][d["test_idx"]])
rng = np.random.default_rng(7)
off = []
while len(off) < a.targets:
    re = rng.uniform(2.0e5, 3.5e5)
    al = rng.uniform(0.2, 9.8)
    if min(abs(re - g) for g in data.RES) < 1.0e4 or abs(al - round(al)) < 0.2:
        continue
    p = P[rng.choice(te_air)]
    r = ev.polar(p, re, [al])
    if r["ok"][0] and 0 <= r["CL"][0] <= 2 and 0 < r["CD"][0] <= 0.05:
        off.append([re, al, r["CL"][0], r["CD"][0]])
TO = np.array(off)
nbT, nbV, nbO = neighbours(TT, K), neighbours(TV, K), neighbours(TO, K)

# local surrogate fine-tune (as in the approved run), on training airfoils near all requests
near = set(np.concatenate([nbT[:, :5], nbV[:, :5], nbO[:, :5]]).ravel().tolist()) | set(neighbours(TT, 50).ravel().tolist())
loc = tr[np.isin(E["airfoil"][tr], list(near))]
mix = np.concatenate([loc, np.random.default_rng(0).choice(tr, len(loc), replace=False)])
sur.finetune(P[E["airfoil"][mix]], E["Re"][mix], E["alpha"][mix], E["CL"][mix], E["CD"][mix], steps=a.ft_steps)
stamp("targets ready, surrogate fine-tuned")

# ---- A1: strength sweep on validation, report on test --------------------------------
OUT["A1_val"] = {"lookup": lookup(TV, nbV)}
for tw in (0, 10, 25, 50):
    OUT["A1_val"][f"t={tw}"] = rrv(model, TV, nbV, tw)
    stamp(f"A1 val t={tw}: {OUT['A1_val'][f't={tw}']}  (budget 1, {K})")
    save()
best_t = max((0, 10, 25, 50), key=lambda tw: (OUT["A1_val"][f"t={tw}"][0], OUT["A1_val"][f"t={tw}"][1], -tw))
best_diff_t = max((10, 25, 50), key=lambda tw: (OUT["A1_val"][f"t={tw}"][0], OUT["A1_val"][f"t={tw}"][1], -tw))
OUT["A1_choice"] = dict(best_t=best_t, best_t_with_diffusion=best_diff_t)
OUT["A1_test"] = {"lookup": lookup(TT, nbT), f"t={best_t}": rrv(model, TT, nbT, best_t)}
if best_t != best_diff_t:
    OUT["A1_test"][f"t={best_diff_t}"] = rrv(model, TT, nbT, best_diff_t)
stamp(f"A1 test: {OUT['A1_test']}")
save()

# ---- A2: off-grid targets ----------------------------------------------------------
OUT["A2_offgrid"] = {"lookup": lookup(TO, nbO), f"rrv t={best_t}": rrv(model, TO, nbO, best_t),
                     "diffusion alone, Phase 1 model": cold(model, TO, a.samples)}
stamp(f"A2 off-grid: {OUT['A2_offgrid']}")
save()

# ---- B: Phase 2 PIDM fine-tune -----------------------------------------------------
m2 = CondDDPM.load(a.model)
t1 = time.time()
ld, lp = pidm.finetune(m2, sur, P[E["airfoil"][tr]], d["c_train"], steps=a.pidm_steps)
m2.save(common.MODELS / "pidm.npz")
OUT["B_train"] = dict(diffusion_loss=ld, physics_loss=lp, seconds=time.time() - t1)
stamp(f"PIDM fine-tune done: {OUT['B_train']}")
OUT["B_test"] = {
    "diffusion alone, Phase 1 model": cold(model, TT, a.samples),
    "diffusion alone, Phase 2 PIDM model": cold(m2, TT, a.samples),
    f"rrv t={best_diff_t}, Phase 2 model": rrv(m2, TT, nbT, best_diff_t),
}
stamp(f"B test: {OUT['B_test']}")
save()
OUT["B_offgrid"] = {"diffusion alone, Phase 2 PIDM model": cold(m2, TO, a.samples)}
stamp(f"B off-grid: {OUT['B_offgrid']}")
save()

print("\nRESULTS (hit rates; rrv/lookup = [budget 1, budget %d]; cold = [1 sample, mean per sample, within %d])" % (K, a.samples))
for k, v in OUT.items():
    print(f"{k}: {v}")
print("EXIT_OK")
