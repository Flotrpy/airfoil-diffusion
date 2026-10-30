"""Evaluators that score an airfoil (parameter vector) at a Reynolds number and
angles of attack. Two backends:

NeuralFoilEvaluator  The evaluator Graves et al. used (a neural network trained
                     on XFOIL runs, so a stand-in for XFOIL, not independent
                     ground truth).
MockEvaluator        A smooth made-up function, ONLY for testing that the code
                     runs end to end. Never report its numbers as findings.

Both return dict(CL, CD, CM, ok), arrays with one entry per angle of attack.
`ok` is False when the shape is invalid or the solver fails (Geng et al. treat
non-converged cases as infeasible and drop them from the dataset).
"""
import numpy as np

from . import shapes

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _empty(n):
    return dict(CL=np.full(n, np.nan), CD=np.full(n, np.nan), CM=np.full(n, np.nan), ok=np.zeros(n, bool))


class NeuralFoilEvaluator:
    name = "neuralfoil"

    def __init__(self, model_size="large", min_confidence=0.9):
        import neuralfoil as nf  # pip install neuralfoil

        self.nf = nf
        self.model_size = model_size
        self.min_confidence = min_confidence
        self.last_error = None

    def polar(self, params, Re, alphas):
        alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
        n = len(alphas)
        out = _empty(n)
        if not shapes.is_valid(params):
            return out
        try:
            r = self.nf.get_aero_from_coordinates(
                coordinates=shapes.coordinates(params),
                alpha=alphas,
                Re=float(Re),
                model_size=self.model_size,
            )
            for k in ("CL", "CD", "CM"):
                out[k] = np.broadcast_to(np.asarray(r[k], dtype=float), (n,)).copy()
            conf = np.broadcast_to(np.asarray(r.get("analysis_confidence", 1.0), dtype=float), (n,))
            out["ok"] = np.isfinite(out["CL"]) & np.isfinite(out["CD"]) & (conf >= self.min_confidence)
        except Exception as e:  # keep going, but remember why
            self.last_error = repr(e)
        return out


class MockEvaluator:
    """Stand-in with thin-airfoil-like behaviour. For plumbing tests only."""

    name = "MOCK"
    last_error = None

    def polar(self, params, Re, alphas):
        alphas = np.atleast_1d(np.asarray(alphas, dtype=float))
        n = len(alphas)
        out = _empty(n)
        if not shapes.is_valid(params):
            return out
        x, yu, yl = shapes.surfaces(params)
        camber = 0.5 * (yu + yl)
        tmax = (yu - yl).max()
        a0 = -2.0 * _trapz(camber * (1.0 - x), x) * 4.0       # zero-lift angle, rad
        cl = 2 * np.pi * (np.deg2rad(alphas) - a0) * 0.92
        cl = 1.7 * np.tanh(cl / 1.7)
        cd = 0.008 * (3e5 / Re) ** 0.2 + 0.03 * tmax + 0.011 * cl**2
        cm = -0.04 - 0.8 * np.abs(camber).max() * (1 + 0.1 * cl)
        out["CL"], out["CD"], out["CM"] = cl, cd, cm
        out["ok"] = np.isfinite(cl) & (cl < 1.6) & (cd < 0.08)
        return out


def make(name):
    if name == "neuralfoil":
        return NeuralFoilEvaluator()
    if name == "mock":
        return MockEvaluator()
    raise ValueError(f"unknown evaluator {name!r}")
