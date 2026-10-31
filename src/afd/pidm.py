"""Phase 2: physics-informed fine-tuning of the diffusion model (PIDM-style), in numpy.

Extra loss: the surrogate's (CL, CD) of the denoised estimate x0_hat must match the
requested (CL, CD) in the condition vector, in units of Geng's tolerances (1% CL, 5% CD).
Only applied for t < t_phys, where x0_hat is meaningful (PIDM applies the residual to the
predicted clean sample; at large t the 1/sqrt(alpha_bar) factor makes it pure noise).

The diffusion gradient and the physics gradient are combined with ConFIG (Liu et al.):
the update direction has an equal positive projection on both unit gradients, so neither
loss is increased, and its length is the sum of both projections.
"""
import numpy as np

from .diffusion import ALPHA_BAR, T, Adam


def _flat(gw, gb):
    return np.concatenate([g.ravel() for g in gw + gb])


def _unflat(v, like_w, like_b):
    out, k = [], 0
    for p in like_w + like_b:
        out.append(v[k:k + p.size].reshape(p.shape))
        k += p.size
    return out[:len(like_w)], out[len(like_w):]


def config_combine(g1, g2):
    """ConFIG for two gradients. Returns the combined gradient."""
    n1, n2 = np.linalg.norm(g1), np.linalg.norm(g2)
    if n1 < 1e-12 or n2 < 1e-12:
        return g1 + g2
    u = np.stack([g1 / n1, g2 / n2])                 # (2, P)
    gu = np.linalg.lstsq(u, np.ones(2), rcond=None)[0]   # u @ gu = 1 (equal projections)
    gu /= np.linalg.norm(gu) + 1e-12
    return (g1 @ gu + g2 @ gu) * gu


def finetune(model, sur, x, c, steps=30000, batch=256, lr=2e-4, t_phys=100, seed=0, log=print):
    """x: raw params (n, dim); c: raw conditions (n, 4) = (Re, alpha, CL, CD). Uses the
    model's existing normalization. Returns (diffusion loss EMA, physics loss EMA)."""
    rng = np.random.default_rng(seed)
    xn, cn = model.nx(x), model.nc(c)
    opt = Adam(model.net, lr)
    net = model.net
    ema_d = ema_p = None
    for s in range(steps):
        i = rng.integers(0, len(xn), batch)
        t = rng.integers(0, T, batch)
        t[: batch // 4] = rng.integers(0, t_phys, batch // 4)   # a quarter of each batch at small t
        eps = rng.standard_normal((batch, model.dim))
        ab = ALPHA_BAR[t][:, None]
        xt = np.sqrt(ab) * xn[i] + np.sqrt(1 - ab) * eps
        inp = np.concatenate([xt, (t / T)[:, None], cn[i]], axis=1)
        pred, acts = net.forward(inp, keep=True)
        err = pred - eps
        gd = _flat(*net.backward(acts, 2.0 * err / err.size))
        ld = float((err ** 2).mean())

        m = t < t_phys
        x0n = (xt[m] - np.sqrt(1 - ab[m]) * pred[m]) / np.sqrt(ab[m])
        p = model.dx(x0n)
        ci = c[i][m]
        y = sur.predict(p, ci[:, 0], ci[:, 1])
        s_cl = 0.01 * np.abs(ci[:, 2]) + 1e-3
        s_cd = 0.05 * ci[:, 3]
        r = np.column_stack([(y[:, 0] - ci[:, 2]) / s_cl, (y[:, 1] - ci[:, 3]) / s_cd])
        lp = float((r ** 2).sum(1).mean())
        dl_dy = np.column_stack([2 * r[:, 0] / s_cl, 2 * r[:, 1] / s_cd]) / m.sum()
        dl_dx0n = sur.grad(p, ci[:, 0], ci[:, 1], dl_dy) * model.x_sd
        dout = np.zeros_like(pred)
        dout[m] = dl_dx0n * (-np.sqrt(1 - ab[m]) / np.sqrt(ab[m]))
        gp = _flat(*net.backward(acts, dout))

        gw, gb = _unflat(config_combine(gd, gp), net.w, net.b)
        opt.step(gw, gb, 1e-6 + 0.5 * (lr - 1e-6) * (1 + np.cos(np.pi * s / steps)))
        ema_d = ld if ema_d is None else 0.99 * ema_d + 0.01 * ld
        ema_p = lp if ema_p is None else 0.99 * ema_p + 0.01 * lp
        if log and (s + 1) % max(steps // 5, 1) == 0:
            log(f"  PIDM step {s + 1}/{steps}: diffusion loss {ema_d:.4f}, physics loss {ema_p:.2f} (tolerance units^2)")
    return ema_d, ema_p
