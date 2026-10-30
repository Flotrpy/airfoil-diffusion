"""Phase 0 gate: baseline errors on held-out airfoils, and the solver's failure rate."""
import argparse
import json

import common
from afd import evaluate, experiment

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
ap.add_argument("--targets", type=int, default=300)
a = ap.parse_args()

d = experiment.load_train_test(a.dataset)
print("dataset:", d["stats"])
ev = evaluate.make(a.evaluator)
targets = experiment.pick_targets(d["c_test"], a.targets)
rows = experiment.baseline_table(d, ev, targets)
keys = ["p_single_trial", "infeasible_share", "rel_rmse_CL", "rel_rmse_CD", "slope_CL", "slope_CD"]
print(experiment.fmt_rows(rows, keys))
out = common.RESULTS / "baselines.json"
out.write_text(json.dumps(dict(evaluator=ev.name, rows=rows), indent=2))
print("\nsaved", out)
print("Phase 0 gate: these errors are written down, and the failure rate of the solver is known.")
if a.evaluator == "mock":
    print("NOTE: MOCK evaluator. Plumbing test only.")
