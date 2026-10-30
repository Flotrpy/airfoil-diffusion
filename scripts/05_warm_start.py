"""Phase 3 (simulated): warm start from the current shape and the update-time check.

Scenario: the vehicle flies airfoil A at a flow condition; the target changes to
what a different held-out airfoil B achieves at the same condition. Compare a cold
start (pure noise) with warm starts from A at several T_start values (Diniz used 50
of 1,000). The 'wind tunnel' here is the solver, as the plan says; no real telemetry.

Reports shape change (how far the new design moves from A, in normalized parameter
units), the share of samples inside the tolerance, and time per update =
sampling time + solver time for N candidates, against your update period."""
import argparse
import json
import time

import numpy as np

import common
from afd import data, evaluate, experiment, metrics
from afd.diffusion import CondDDPM

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--model", default=str(common.MODELS / "ddpm.npz"))
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
ap.add_argument("--scenarios", type=int, default=40)
ap.add_argument("--candidates", type=int, default=20, help="N designs sampled per update")
ap.add_argument("--ddim-steps", type=int, default=50)
ap.add_argument("--t-starts", type=int, nargs="+", default=[10, 25, 50, 100, 200])
ap.add_argument("--period", type=float, default=30.0, help="your update period in seconds")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = experiment.load_train_test(a.dataset)
ev = evaluate.make(a.evaluator)
model = CondDDPM.load(a.model)
rng = np.random.default_rng(a.seed)
E, params, te = d["entries"], d["params"], d["test_idx"]

# build scenarios: two held-out entries at the same (Re, alpha) from different airfoils
key = {}
for i in te:
    key.setdefault((E["Re"][i], E["alpha"][i]), []).append(i)
scen = []
for k, idx in key.items():
    for _ in range(max(1, a.scenarios // len(key) + 1)):
        if len(idx) >= 2:
            i, j = rng.choice(idx, 2, replace=False)
            if E["airfoil"][i] != E["airfoil"][j]:
                scen.append((i, j))
scen = scen[: a.scenarios]
print(f"{len(scen)} scenarios, {a.candidates} candidates per update, update period {a.period}s")

results = {}
for label, ts in [("cold start", None)] + [(f"warm T_start={t}", t) for t in a.t_starts]:
    hits, moves, upd_times, any_hit = [], [], [], []
    for i, j in scen:
        cur = params[E["airfoil"][i]]
        tgt = data.conditions(E, [j])[0]
        cond = np.repeat(tgt[None, :], a.candidates, axis=0)
        t0 = time.time()
        if ts is None:
            x = model.sample(cond, rng, ddim_steps=a.ddim_steps)
        else:
            x = model.sample(cond, rng, ddim_steps=max(2, int(a.ddim_steps * ts / 1000) + 2),
                             x_init=np.repeat(cur[None, :], a.candidates, axis=0), t_start=ts)
        cl, cd, ok = metrics.score(ev, x, cond)
        upd_times.append(time.time() - t0)
        e_cl, e_cd = np.abs(cl - tgt[2]) / abs(tgt[2]), np.abs(cd - tgt[3]) / abs(tgt[3])
        hit = ok & (e_cl <= metrics.TOL_CL) & (e_cd <= metrics.TOL_CD)
        hits.append(hit.mean())
        any_hit.append(hit.any())
        move = np.linalg.norm(model.nx(x) - model.nx(cur[None, :]), axis=1)
        moves.append(move[hit].mean() if hit.any() else np.nan)
    results[label] = dict(
        hit_rate=float(np.mean(hits)), share_with_a_hit=float(np.mean(any_hit)),
        shape_change_of_hits=float(np.nanmean(moves)) if not np.all(np.isnan(moves)) else float("nan"),
        seconds_per_update=float(np.mean(upd_times)), fits_period=bool(np.mean(upd_times) <= a.period),
    )
    r = results[label]
    print(f"{label:>18}: hit rate {r['hit_rate']:.3f}, update has a hit {r['share_with_a_hit']:.2f}, "
          f"shape change {r['shape_change_of_hits']:.2f}, {r['seconds_per_update']:.2f}s per update "
          f"({'fits' if r['fits_period'] else 'misses'} {a.period:.0f}s)")
(common.RESULTS / "phase3.json").write_text(json.dumps(dict(evaluator=ev.name, results=results), indent=2))
print("saved results/phase3.json")
print("Phase 3 gate: pick the T_start with a good hit rate and a small shape change, then confirm it fits your period.")
if a.evaluator == "mock":
    print("NOTE: MOCK evaluator. Plumbing test only.")
