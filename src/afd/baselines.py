"""Baselines every diffusion result must be compared against (Phase 0).

NearestNeighbour   Airbus: take the training airfoil whose features are closest
                   to the target.
LinearModes        Airbus applied POD and linearly combined modes to hit target
                   features; with 11 modes of an 11-parameter space POD is just
                   a rotation, so this is a ridge regression from conditions to
                   parameters. This is my reading of their description.
MLPRegression      Plain deterministic network, conditions -> parameters. My
                   addition, not from a paper: it shows what a non-generative
                   network can do.
All three return ONE airfoil per target, which is the limit Airbus points out.
"""
import numpy as np


class _Norm:
    def fit(self, c):
        self.mu, self.sd = c.mean(0), c.std(0) + 1e-9
        return self

    def __call__(self, c):
        return (c - self.mu) / self.sd


class NearestNeighbour:
    name = "nearest-neighbour"

    def fit(self, c, x):
        self.n, self.x = _Norm().fit(c), x
        self.c = self.n(c)
        return self

    def predict(self, c):
        q = self.n(c)
        d = ((q[:, None, :] - self.c[None, :, :]) ** 2).sum(-1)
        return self.x[d.argmin(1)]


class LinearModes:
    name = "linear (POD-style)"

    def __init__(self, ridge=1e-3):
        self.ridge = ridge

    def fit(self, c, x):
        self.n = _Norm().fit(c)
        a = np.c_[self.n(c), np.ones(len(c))]
        self.w = np.linalg.solve(a.T @ a + self.ridge * np.eye(a.shape[1]), a.T @ x)
        return self

    def predict(self, c):
        return np.c_[self.n(c), np.ones(len(c))] @ self.w


class MLPRegression:
    name = "MLP regression"

    def __init__(self, hidden=(64, 64), seed=0):
        from sklearn.neural_network import MLPRegressor

        self.m = MLPRegressor(hidden_layer_sizes=hidden, activation="tanh", max_iter=300, random_state=seed)

    def fit(self, c, x):
        self.n = _Norm().fit(c)
        self.xm, self.xs = x.mean(0), x.std(0) + 1e-9
        self.m.fit(self.n(c), (x - self.xm) / self.xs)
        return self

    def predict(self, c):
        return self.m.predict(self.n(c)) * self.xs + self.xm
