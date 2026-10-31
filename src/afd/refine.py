"""Retrieve -> refine -> verify (approved method; fixes the Phase 1 failure).

1. Retrieve: nearest training airfoils to the target conditions (the lookup).
2. Refine: diffusion warm start (partial noising, Diniz / SDEdit) conditioned on the
   exact target, then gradient steps on a small differentiable surrogate of the solver
   (Dflow-SUR / Chen et al. idea), re-noising lightly between rounds so the shape stays
   on the learned airfoil manifold (PIDM-style surrogate constraint).
3. Verify: every candidate is checked by NeuralFoil and counts against the budget.
   Later candidates start from the best checked shape and correct the surrogate's
   bias using the solver results already paid for (output space mapping).
"""
import numpy as np

from .diffusion import MLP, Adam


class Surrogate:
    """MLP: (11 params, Re, alpha) -> (CL, CD). numpy, with input gradients."""

    def __init__(self, width=128, depth=3, seed=0):
        self.net = MLP([13] + [width] * depth + [2], np.random.default_rng(seed))

    def fit(self, params, re, alpha, cl, cd, steps=60000, batch=512, lr=1e-3, seed=0, log=print):
        x = np.column_stack([params, re, alpha])
        y = np.column_stack([cl, cd])
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-9
        self.ym, self.ys = y.mean(0), y.std(0) + 1e-9
        xn, yn = (x - self.xm) / self.xs, (y - self.ym) / self.ys
        rng, opt = np.random.default_rng(seed), Adam(self.net, lr)
        for s in range(steps):
            i = rng.integers(0, len(xn), batch)
            out, acts = self.net.forward(xn[i], keep=True)
            err = out - yn[i]
            gw, gb = self.net.backward(acts, 2 * err / err.size)
            opt.step(gw, gb, 1e-5 + 0.5 * (lr - 1e-5) * (1 + np.cos(np.pi * s / steps)))
            if log and (s + 1) % max(steps // 5, 1) == 0:
                log(f"  surrogate step {s + 1}/{steps} mse {float((err ** 2).mean()):.5f}")
        return self

    def finetune(self, params, re, alpha, cl, cd, steps=8000, batch=512, lr=2e-4, seed=1, log=print):
        """Continue training with the existing normalization (local refinement of the fit)."""
        xn = (np.column_stack([params, re, alpha]) - self.xm) / self.xs
        yn = (np.column_stack([cl, cd]) - self.ym) / self.ys
        rng, opt = np.random.default_rng(seed), Adam(self.net, lr)
        for s in range(steps):
            i = rng.integers(0, len(xn), batch)
            out, acts = self.net.forward(xn[i], keep=True)
            err = out - yn[i]
            gw, gb = self.net.backward(acts, 2 * err / err.size)
            opt.step(gw, gb, 1e-6 + 0.5 * (lr - 1e-6) * (1 + np.cos(np.pi * s / steps)))
        if log:
            log(f"  surrogate fine-tune {steps} steps, final batch mse {float((err ** 2).mean()):.5f}")
        return self

    def predict(self, params, re, alpha):
        x = np.column_stack([params, np.broadcast_to(re, len(params)), np.broadcast_to(alpha, len(params))])
        return self.net.forward((x - self.xm) / self.xs) * self.ys + self.ym

    def grad(self, params, re, alpha, dl_dy):
        """Gradient of a loss w.r.t. the 11 raw params, given dL/d(CL, CD)."""
        x = np.column_stack([params, np.broadcast_to(re, len(params)), np.broadcast_to(alpha, len(params))])
        out, acts = self.net.forward((x - self.xm) / self.xs, keep=True)
        d = dl_dy * self.ys
        for i in reversed(range(len(self.net.w))):
            d = d @ self.net.w[i].T
            if i > 0:
                d = d * (1.0 - acts[i] ** 2)
        return d[:, :11] / self.xs[:11]

    def save(self, path):
        d = dict(xm=self.xm, xs=self.xs, ym=self.ym, ys=self.ys, n=len(self.net.w))
        for i, (w, b) in enumerate(zip(self.net.w, self.net.b)):
            d[f"w{i}"], d[f"b{i}"] = w, b
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        s = cls.__new__(cls)
        s.net = MLP([1, 1], np.random.default_rng(0))
        s.net.w = [d[f"w{i}"] for i in range(int(d["n"]))]
        s.net.b = [d[f"b{i}"] for i in range(int(d["n"]))]
        s.xm, s.xs, s.ym, s.ys = d["xm"], d["xs"], d["ym"], d["ys"]
        return s


def polish(sur, p, tg, bias, xs, u0=None, steps=60, lr=0.02, prox=0.01):
    """Adam on normalized params to hit (CL, CD) target per Geng tolerances; tg rows
    are (Re, alpha, CL, CD); bias = solver minus surrogate, subtracted from targets."""
    u = p / xs
    u0 = u.copy() if u0 is None else u0
    m = np.zeros_like(u)
    v = np.zeros_like(u)
    cl_t, cd_t = tg[:, 2] - bias[:, 0], tg[:, 3] - bias[:, 1]
    s_cl, s_cd = 0.01 * np.abs(tg[:, 2]) + 1e-3, 0.05 * tg[:, 3]
    for k in range(1, steps + 1):
        pr = u * xs
        y = np.vstack([sur.predict(pr[i:i + 1], tg[i, 0], tg[i, 1]) for i in range(len(pr))])
        dl = np.column_stack([2 * (y[:, 0] - cl_t) / s_cl**2, 2 * (y[:, 1] - cd_t) / s_cd**2])
        g = np.vstack([sur.grad(pr[i:i + 1], tg[i, 0], tg[i, 1], dl[i:i + 1]) for i in range(len(pr))]) * xs
        g += 2 * prox * (u - u0)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        u -= lr * (m / (1 - 0.9**k)) / (np.sqrt(v / (1 - 0.999**k)) + 1e-8)
    return u * xs


def refine(model, sur, start, tg, bias, rng, use_diffusion=True, t_warm=50, rounds=3, t_proj=5):
    """One refined candidate per row of start (raw params)."""
    xs = model.x_sd
    p = model.sample(tg, rng, ddim_steps=10, x_init=start, t_start=t_warm) if use_diffusion else start.copy()
    u0 = p / xs
    for _ in range(rounds):
        p = polish(sur, p, tg, bias, xs, u0=u0)
        if use_diffusion:
            p = model.sample(tg, rng, ddim_steps=3, x_init=p, t_start=t_proj)
    return polish(sur, p, tg, bias, xs, u0=u0, steps=40)
