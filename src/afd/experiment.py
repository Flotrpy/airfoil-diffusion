"""Shared helpers for the evaluation scripts."""
import numpy as np

from . import baselines, data, metrics


def load_train_test(path):
    params, split, entries, stats = data.load(path)
    tr = data.split_idx(entries, split, "train")
    te = data.split_idx(entries, split, "test")
    return dict(
        params=params, split=split, entries=entries, stats=stats,
        x_train=params[entries["airfoil"][tr]], c_train=data.conditions(entries, tr),
        c_test=data.conditions(entries, te), test_idx=te, train_idx=tr,
    )


def pick_targets(c_test, n, seed=0):
    """Targets come from held-out airfoils, so each one is achievable."""
    rng = np.random.default_rng(seed)
    return c_test[rng.permutation(len(c_test))[:n]]


def fit_baselines(d):
    return [
        baselines.NearestNeighbour().fit(d["c_train"], d["x_train"]),
        baselines.LinearModes().fit(d["c_train"], d["x_train"]),
        baselines.MLPRegression().fit(d["c_train"], d["x_train"]),
    ]


def baseline_table(d, evaluator, targets):
    rows = {}
    for b in fit_baselines(d):
        rows[b.name] = metrics.single_shot(b.predict, evaluator, targets)
    return rows


def fmt_rows(rows, keys):
    head = "| method | " + " | ".join(keys) + " |\n|---|" + "---|" * len(keys)
    lines = [head]
    for name, r in rows.items():
        lines.append("| " + name + " | " + " | ".join(f"{r.get(k, float('nan')):.4g}" for k in keys) + " |")
    return "\n".join(lines)
