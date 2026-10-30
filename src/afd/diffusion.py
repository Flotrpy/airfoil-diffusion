"""Conditional DDPM / DDIM in plain numpy (no PyTorch needed for Phases 0 to 1).

Follows Airbus (Wagenaar et al.): noise estimator eps(x_t, t/T, f) is an MLP of
4 hidden layers with tanh, output size = number of airfoil parameters; samples
and features are z-score normalized with training-set statistics; beta is
linear from 1e-3 to 0.2 over T = 1000 steps.

Sampling is deterministic DDIM (sigma = 0, Chen et al. / Diniz). Passing
`x_init` and `t_start` starts from a noised known shape instead of pure noise:
the warm start of Diniz (T_start = 50 of 1000) and Shu et al. (partial chain).

Phase 2 (PIDM-style loss through a differentiable surrogate) needs autodiff,
so that phase moves to PyTorch; this module is the Phase 1 baseline.
"""
import numpy as np

T = 1000
BETA = np.linspace(1e-3, 0.2, T)
ALPHA_BAR = np.cumprod(1.0 - BETA)


class MLP:
    def __init__(self, sizes, rng):
        self.w, self.b = [], []
        for a, b in zip(sizes[:-1], sizes[1:]):
            self.w.append(rng.normal(0, np.sqrt(1.0 / a), size=(a, b)))
            self.b.append(np.zeros(b))

    def forward(self, x, keep=False):
        acts = [x]
        for i, (w, b) in enumerate(zip(self.w, self.b)):
            x = x @ w + b
            if i < len(self.w) - 1:
                x = np.tanh(x)
            acts.append(x)
        return (x, acts) if keep else x

    def backward(self, acts, dout):
        gw, gb = [None] * len(self.w), [None] * len(self.w)
        d = dout
        for i in reversed(range(len(self.w))):
            gw[i] = acts[i].T @ d
            gb[i] = d.sum(axis=0)
            if i > 0:
                d = (d @ self.w[i].T) * (1.0 - acts[i] ** 2)
        return gw, gb


class Adam:
    def __init__(self, net, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.net, self.lr, self.b1, self.b2, self.eps, self.t = net, lr, b1, b2, eps, 0
        self.m = [np.zeros_like(p) for p in net.w + net.b]
        self.v = [np.zeros_like(p) for p in net.w + net.b]

    def step(self, gw, gb, lr):
        self.t += 1
        params = self.net.w + self.net.b
        for k, (p, g) in enumerate(zip(params, gw + gb)):
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1**self.t)
            vh = self.v[k] / (1 - self.b2**self.t)
            p -= lr * mh / (np.sqrt(vh) + self.eps)


class CondDDPM:
    def __init__(self, dim, cond_dim, width=32, depth=4, seed=0):
        self.dim, self.cond_dim, self.width, self.depth = dim, cond_dim, width, depth
        rng = np.random.default_rng(seed)
        self.net = MLP([dim + 1 + cond_dim] + [width] * depth + [dim], rng)
        self.x_mu = self.x_sd = self.c_mu = self.c_sd = None
        self.x_lim = 5.0   # clip on the predicted clean sample (normalized units); set from data in train()

    # ---- normalization -------------------------------------------------
    def fit_norm(self, x, c):
        self.x_mu, self.x_sd = x.mean(0), x.std(0) + 1e-9
        self.c_mu, self.c_sd = c.mean(0), c.std(0) + 1e-9

    def nx(self, x):
        return (x - self.x_mu) / self.x_sd

    def dx(self, z):
        return z * self.x_sd + self.x_mu

    def nc(self, c):
        return (c - self.c_mu) / self.c_sd

    def _eps(self, xt, t, cn):
        inp = np.concatenate([xt, (t / T)[:, None], cn], axis=1)
        return self.net.forward(inp)

    # ---- training ------------------------------------------------------
    def train(self, x, c, steps=30000, batch=256, lr=1e-3, seed=0, log=print):
        rng = np.random.default_rng(seed)
        self.fit_norm(x, c)
        xn, cn = self.nx(x), self.nc(c)
        self.x_lim = float(np.abs(xn).max() * 1.1)
        opt = Adam(self.net, lr)
        n = len(xn)
        ema = None
        for s in range(steps):
            i = rng.integers(0, n, batch)
            t = rng.integers(0, T, batch)
            eps = rng.standard_normal((batch, self.dim))
            ab = ALPHA_BAR[t][:, None]
            xt = np.sqrt(ab) * xn[i] + np.sqrt(1 - ab) * eps
            inp = np.concatenate([xt, (t / T)[:, None], cn[i]], axis=1)
            pred, acts = self.net.forward(inp, keep=True)
            err = pred - eps
            loss = float((err**2).mean())
            gw, gb = self.net.backward(acts, 2.0 * err / err.size)
            cur = 1e-5 + 0.5 * (lr - 1e-5) * (1 + np.cos(np.pi * s / steps))  # cosine decay
            opt.step(gw, gb, cur)
            ema = loss if ema is None else 0.99 * ema + 0.01 * loss
            if log and (s + 1) % max(steps // 10, 1) == 0:
                log(f"  step {s + 1}/{steps}  loss {ema:.4f}")
        return ema

    # ---- sampling ------------------------------------------------------
    def _denoise(self, xt, t, cn):
        """Predict eps, form the clean-sample estimate, clip it, and re-derive eps from the
        clipped estimate. Clipping matters here: with beta up to 0.2, ALPHA_BAR is about
        1e-44 at t = T, so dividing by sqrt(alpha_bar) amplifies any noise-prediction error
        enormously, and 50-step DDIM diverged to ~1e20 without it (found in testing)."""
        n = len(xt)
        eps = self._eps(xt, np.full(n, float(t)), cn)
        ab = ALPHA_BAR[t]
        x0 = np.clip((xt - np.sqrt(1 - ab) * eps) / np.sqrt(ab), -self.x_lim, self.x_lim)
        eps = (xt - np.sqrt(ab) * x0) / np.sqrt(1 - ab)
        return x0, eps

    def sample(self, c, rng, ddim_steps=50, x_init=None, t_start=None):
        """Deterministic DDIM. c: (n, cond_dim) raw conditions. Returns raw params (n, dim).
        With x_init (n, dim, raw) and t_start (1..T-1), start from x_init noised to t_start."""
        n = len(c)
        cn = self.nc(c)
        if x_init is None:
            t0 = T - 1
            xt = rng.standard_normal((n, self.dim))
        else:
            t0 = int(t_start)
            ab = ALPHA_BAR[t0]
            xt = np.sqrt(ab) * self.nx(x_init) + np.sqrt(1 - ab) * rng.standard_normal((n, self.dim))
        ts = np.unique(np.linspace(t0, 0, ddim_steps + 1).round().astype(int))[::-1]
        for a, b in zip(ts[:-1], ts[1:]):
            x0, eps = self._denoise(xt, a, cn)
            xt = np.sqrt(ALPHA_BAR[b]) * x0 + np.sqrt(1 - ALPHA_BAR[b]) * eps
        x0, _ = self._denoise(xt, 0, cn)
        return self.dx(x0)

    # ---- io ------------------------------------------------------------
    def save(self, path):
        d = dict(dim=self.dim, cond_dim=self.cond_dim, width=self.width, depth=self.depth,
                 x_mu=self.x_mu, x_sd=self.x_sd, c_mu=self.c_mu, c_sd=self.c_sd, x_lim=self.x_lim)
        for i, (w, b) in enumerate(zip(self.net.w, self.net.b)):
            d[f"w{i}"], d[f"b{i}"] = w, b
        np.savez(path, **d)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(int(d["dim"]), int(d["cond_dim"]), int(d["width"]), int(d["depth"]))
        m.x_mu, m.x_sd, m.c_mu, m.c_sd = d["x_mu"], d["x_sd"], d["c_mu"], d["c_sd"]
        m.x_lim = float(d["x_lim"]) if "x_lim" in d.files else 5.0
        for i in range(len(m.net.w)):
            m.net.w[i], m.net.b[i] = d[f"w{i}"], d[f"b{i}"]
        return m
