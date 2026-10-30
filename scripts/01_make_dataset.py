"""Phase 0: build the dataset (Bernstein airfoils x flow conditions -> solver labels)."""
import argparse

import common
from afd import data, evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--n-airfoils", type=int, default=1000, help="Airbus used 1,000; its Table 4 used 6,000")
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(common.DATA / "dataset.npz"))
a = ap.parse_args()

ev = evaluate.make(a.evaluator)
print(f"Building dataset with evaluator '{ev.name}' ...")
params, split, entries, stats = data.build(a.n_airfoils, ev, seed=a.seed)
data.save(a.out, params, split, entries, stats)
print("done:", stats)
print("train/val/test airfoils:", [(s, int((split == s).sum())) for s in ("train", "val", "test")])
if a.evaluator == "mock":
    print("NOTE: labels came from the MOCK evaluator. Plumbing test only.")
