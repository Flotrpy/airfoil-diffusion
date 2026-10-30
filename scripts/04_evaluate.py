"""Phase 1 gate: diffusion vs baselines, checked by the solver, on held-out targets.

Metrics follow Geng et al. (1% CL and 5% CD tolerance, up to 100 trials, N95),
Graves et al. (conditioning slope) and Diniz (MMD)."""
import argparse
import json
import time

import numpy as np

import common
from afd import evaluate, experiment, metrics
from afd.diffusion import CondDDPM

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--model", default=str(common.MODELS / "ddpm.npz"))
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
ap.add_argument("--targets", type=int, default=100, help="each target costs up to 100 solver calls")
ap.add_argument("--ddim-steps", type=int, default=50)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

d = experiment.load_train_test(a.dataset)
ev = evaluate.make(a.evaluator)
model = CondDDPM.load(a.model)
rng = np.random.default_rng(a.seed)
targets = experiment.pick_targets(d["c_test"], a.targets)

print("baselines ...")
base = experiment.baseline_table(d, ev, targets)

print("diffusion, one sample per target ...")
t0 = time.time()
one = metrics.single_shot(lambda c: model.sample(c, rng, ddim_steps=a.ddim_steps), ev, targets)
print("diffusion, Geng trial protocol (up to 100 trials) ...")
proto = metrics.trial_protocol(lambda c: model.sample(c, rng, ddim_steps=a.ddim_steps), ev, targets)
gen = model.sample(targets, rng, ddim_steps=a.ddim_steps)
one["MMD_vs_train_shapes"] = metrics.mmd(gen, d["x_train"])

rows = dict(base)
rows["diffusion (1 sample)"] = one
keys = ["p_single_trial", "p_single_trial_tol_0.02_0.05", "p_single_trial_tol_0.05_0.1", "infeasible_share", "rel_rmse_CL", "rel_rmse_CD", "slope_CL", "slope_CD"]
print("\n" + experiment.fmt_rows(rows, keys))
print("\ndiffusion with up to 100 trials per target (Geng protocol):")
for k, v in proto.items():
    print(f"  {k}: {v:.4g}")
print(f"\nMMD between generated and training shapes (lower = closer): {one['MMD_vs_train_shapes']:.4g}")

best_base = max(b["p_single_trial"] for b in base.values())
print("\nPhase 1 gate (my thresholds, see the plan):")
lookup = base["nearest-neighbour"]["p_single_trial"]
passed = one["p_single_trial"] >= 0.8 * lookup or proto["share_hit_within_10"] >= lookup
print(f"  lookup hit rate {lookup:.3f}; diffusion 1 sample {one['p_single_trial']:.3f}; diffusion hit within 10 samples {proto['share_hit_within_10']:.3f}")
print(f"  GATE {'PASSED' if passed else 'FAILED'} (pass = 1 sample within 80% of lookup, or 10 samples beat lookup)")
print(f"  best baseline hit rate {best_base:.3f}  vs diffusion single-sample hit rate {one['p_single_trial']:.3f}")
print(f"  diffusion conditioning slopes CL {one['slope_CL']:.2f}, CD {one['slope_CD']:.2f}  (Graves saw 0.25 and 0.07)")
print(f"  diffusion needs about {proto['N95']:.1f} trials for a 95% chance of an in-tolerance design")

out = common.RESULTS / "phase1.json"
out.write_text(json.dumps(dict(gate_passed=bool(passed), evaluator=ev.name, baselines=base, diffusion_single=one, diffusion_protocol=proto), indent=2))
print("saved", out)
if a.evaluator == "mock":
    print("NOTE: MOCK evaluator. Plumbing test only.")
