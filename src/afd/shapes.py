"""Airfoil geometry: Bernstein parameterization (Airbus, Wagenaar et al.).

y/c = (x/c)^N1 * (1 - x/c)^N2 * sum_i A_i * C(n,i) (x/c)^i (1 - x/c)^(n-i)

Airbus uses N1 = 0.5 and N2 = 1.0 (rounded leading edge, sharp trailing edge),
"order 6" on both surfaces, and sets the first lower and upper coefficient
equal, giving 11 parameters in total. Reading used here: 6 coefficients per
surface (polynomial degree 5), because that is what yields 11 parameters.

Parameter vector p (length 11):
    p[0]      shared first coefficient A0 (upper = lower)
    p[1:6]    upper coefficients A_u1 .. A_u5
    p[6:11]   lower coefficients A_l1 .. A_l5
The lower surface is y_l = -class(x) * sum(A_l B), so lower coefficients are
positive for a normal airfoil.
"""
from math import comb

import numpy as np
from scipy.stats import chi2

N1, N2 = 0.5, 1.0
K = 6                      # coefficients per surface
N_PARAMS = 2 * K - 1       # 11

# Airbus samples all parameters uniformly in [-0.5, 1.5] and the shared A0 in
# [0.3, 1.0]. With the class function as written here, that box gives very thick
# shapes (median max thickness 38% of chord), which NeuralFoil scores with low
# confidence: in testing the first 20 airfoils yielded no usable entry at all. So
# the box is narrowed for a drone wing (my choice): A0 in [0.08, 0.25], the rest in
# [-0.05, 0.35]. That gives 8% to 17% thickness and up to about 3% camber (5th to
# 95th percentile in testing).
LO = np.full(N_PARAMS, -0.05)
HI = np.full(N_PARAMS, 0.35)
LO[0], HI[0] = 0.08, 0.25
AIRBUS_LO = np.full(N_PARAMS, -0.5)
AIRBUS_HI = np.full(N_PARAMS, 1.5)
AIRBUS_LO[0], AIRBUS_HI[0] = 0.3, 1.0

# Geometry limits for a drone wing section (my choice): max thickness 6% to 20%
# of chord, max camber at most 8%.
T_RANGE = (0.06, 0.20)
CAMBER_MAX = 0.08

N_POINTS = 101             # points per surface, cosine spaced (Airbus: 100 elements)


def cosine_x(n=N_POINTS):
    theta = np.linspace(0.0, np.pi, n)
    return 0.5 * (1.0 - np.cos(theta))


def _basis(x):
    n = K - 1
    return np.stack([comb(n, i) * x**i * (1.0 - x) ** (n - i) for i in range(K)], axis=-1)


def surfaces(p, x=None):
    """Return (x, y_upper, y_lower) for parameter vector p."""
    p = np.asarray(p, dtype=float)
    x = cosine_x() if x is None else np.asarray(x, dtype=float)
    a_u = p[:K]
    a_l = np.concatenate([[p[0]], p[K:]])
    cls = x**N1 * (1.0 - x) ** N2
    b = _basis(x)
    return x, cls * (b @ a_u), -cls * (b @ a_l)


def coordinates(p):
    """Selig-ordered coordinates: upper surface trailing edge -> leading edge,
    then lower surface leading edge -> trailing edge. Shape (2*N_POINTS-1, 2)."""
    x, yu, yl = surfaces(p)
    upper = np.stack([x[::-1], yu[::-1]], axis=1)
    lower = np.stack([x[1:], yl[1:]], axis=1)
    return np.concatenate([upper, lower], axis=0)


def features(p):
    """Geometric features used for outlier filtering: max thickness, max camber,
    chordwise position of max thickness."""
    x, yu, yl = surfaces(p)
    t = yu - yl
    camber = 0.5 * (yu + yl)
    return np.array([t.max(), np.abs(camber).max(), x[np.argmax(t)]])


def is_valid(p, min_thickness=0.0, max_thickness=0.5):
    """Reject self-intersecting surfaces (upper must stay above lower)."""
    p = np.asarray(p, dtype=float)
    if p.shape != (N_PARAMS,) or not np.all(np.isfinite(p)):
        return False
    x, yu, yl = surfaces(p)
    t = (yu - yl)[1:-1]
    return bool(np.all(t > min_thickness) and t.max() < max_thickness)


def sample_airfoils(n, rng, mahalanobis_q=0.99, batch=4000):
    """Sample n valid airfoils uniformly in the box above, drop self-intersecting
    ones and shapes outside the thickness/camber limits, then drop geometric
    outliers by Mahalanobis distance on `features` (Airbus filters outliers too).
    mahalanobis_q is my choice; the Airbus paper filters outliers but its exact
    threshold is not stated in my notes."""
    kept = []
    total = 0
    while total < n:
        cand = rng.uniform(LO, HI, size=(batch, N_PARAMS))
        cand = cand[[is_valid(c) for c in cand]]
        if len(cand) < 10:
            continue
        f = np.array([features(c) for c in cand])
        geo = (f[:, 0] >= T_RANGE[0]) & (f[:, 0] <= T_RANGE[1]) & (f[:, 1] <= CAMBER_MAX)
        cand, f = cand[geo], f[geo]
        if len(cand) < 10:
            continue
        mu = f.mean(axis=0)
        cov = np.cov(f.T) + 1e-9 * np.eye(f.shape[1])
        d2 = np.einsum("ij,jk,ik->i", f - mu, np.linalg.inv(cov), f - mu)
        cand = cand[d2 <= chi2.ppf(mahalanobis_q, df=f.shape[1])]
        kept.append(cand)
        total += len(cand)
    return np.concatenate(kept)[:n]
