"""Dataset: airfoil x (Reynolds number, angle of attack) -> (CL, CD, CM).

Follows Geng et al.: flow conditions on a grid, non-converged cases dropped,
and entries outside CL in [0, 2] or CD in [0, 0.05] dropped. The Mach axis of
Geng's grid is left out (see README). The split is BY AIRFOIL, 60/20/20
(Airbus used 600/200/200), so one airfoil never sits in two splits; Geng's
split was by entry.
"""
import time

import numpy as np

from . import shapes

RES = (2.0e5, 2.5e5, 3.0e5, 3.5e5)          # Geng: 2e5 to 3.5e5 in steps of 5e4
ALPHAS = tuple(float(a) for a in range(0, 11))  # Geng: 0 to 10 degrees in 1 degree steps
CL_RANGE = (0.0, 2.0)
CD_RANGE = (0.0, 0.05)


def build(n_airfoils, evaluator, seed=0, res=RES, alphas=ALPHAS, log=print):
    rng = np.random.default_rng(seed)
    params = shapes.sample_airfoils(n_airfoils, rng)
    split = np.empty(n_airfoils, dtype="U5")
    order = rng.permutation(n_airfoils)
    n_tr, n_va = int(0.6 * n_airfoils), int(0.2 * n_airfoils)
    split[order[:n_tr]] = "train"
    split[order[n_tr : n_tr + n_va]] = "val"
    split[order[n_tr + n_va :]] = "test"

    rows = {k: [] for k in ("airfoil", "Re", "alpha", "CL", "CD", "CM")}
    n_try = n_fail = 0
    t0 = time.time()
    for i, p in enumerate(params):
        for re in res:
            r = evaluator.polar(p, re, alphas)
            n_try += len(alphas)
            good = r["ok"].copy()
            good &= (r["CL"] >= CL_RANGE[0]) & (r["CL"] <= CL_RANGE[1])
            good &= (r["CD"] >= CD_RANGE[0]) & (r["CD"] <= CD_RANGE[1])
            n_fail += int((~good).sum())
            for j in np.flatnonzero(good):
                rows["airfoil"].append(i)
                rows["Re"].append(re)
                rows["alpha"].append(alphas[j])
                rows["CL"].append(r["CL"][j])
                rows["CD"].append(r["CD"][j])
                rows["CM"].append(r["CM"][j])
        if i == 19 and not rows["CL"]:
            raise RuntimeError(
                "The first 20 airfoils produced no valid evaluations. Last evaluator error: "
                f"{getattr(evaluator, 'last_error', None)}. Run scripts/00_check_env.py."
            )
        if (i + 1) % 100 == 0:
            log(f"  {i + 1}/{n_airfoils} airfoils, {len(rows['CL'])} entries, {time.time() - t0:.0f}s")
    entries = {k: np.asarray(v) for k, v in rows.items()}
    entries["airfoil"] = entries["airfoil"].astype(int)
    stats = dict(evaluator=evaluator.name, n_airfoils=n_airfoils, n_entries=len(entries["CL"]),
                 dropped_share=n_fail / max(n_try, 1), seconds=time.time() - t0)
    return params, split, entries, stats


def save(path, params, split, entries, stats):
    np.savez(path, params=params, split=split, stats=np.array([str(stats)]), **entries)


def load(path):
    d = np.load(path, allow_pickle=False)
    entries = {k: d[k] for k in ("airfoil", "Re", "alpha", "CL", "CD", "CM")}
    return d["params"], d["split"], entries, str(d["stats"][0])


def conditions(entries, idx=None):
    """Condition vector per entry: (Re, alpha, CL, CD)."""
    c = np.stack([entries["Re"], entries["alpha"], entries["CL"], entries["CD"]], axis=1)
    return c if idx is None else c[idx]


def split_idx(entries, split, name):
    return np.flatnonzero(split[entries["airfoil"]] == name)
