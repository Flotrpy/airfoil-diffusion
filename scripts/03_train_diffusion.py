"""Phase 1: train the conditional diffusion model (Airbus-style 4-layer tanh MLP)."""
import argparse
import time

import common
from afd import data
from afd.diffusion import CondDDPM

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--steps", type=int, default=40000)
ap.add_argument("--width", type=int, default=32, help="Airbus: 32. Their Table 4 used 64 with 6,000 airfoils.")
ap.add_argument("--batch", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=str(common.MODELS / "ddpm.npz"))
a = ap.parse_args()

params, split, entries, stats = data.load(a.dataset)
tr = data.split_idx(entries, split, "train")
x = params[entries["airfoil"][tr]]
c = data.conditions(entries, tr)
print(f"training on {len(tr)} entries from {int((split == 'train').sum())} airfoils; dim {x.shape[1]}, cond {c.shape[1]}")
m = CondDDPM(x.shape[1], c.shape[1], width=a.width, seed=a.seed)
t0 = time.time()
loss = m.train(x, c, steps=a.steps, batch=a.batch, lr=a.lr, seed=a.seed)
m.save(a.out)
print(f"final smoothed loss {loss:.4f}; {time.time() - t0:.0f}s; saved {a.out}")
