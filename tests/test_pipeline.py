"""Quick checks that the pieces are wired correctly. Run:  python tests/test_pipeline.py
Uses the MOCK evaluator, so it needs no NeuralFoil and takes seconds."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from afd import data, evaluate, metrics, shapes  # noqa: E402
from afd.diffusion import ALPHA_BAR, CondDDPM, MLP  # noqa: E402


def test_parameter_count():
    assert shapes.N_PARAMS == 11


def test_shape_closed_and_valid():
    p = np.array([0.17, 0.15, 0.16, 0.15, 0.14, 0.15, 0.14, 0.13, 0.13, 0.13, 0.13])
    c = shapes.coordinates(p)
    assert c.shape == (2 * shapes.N_POINTS - 1, 2)
    assert abs(c[0, 1]) < 1e-12 and abs(c[-1, 1]) < 1e-12      # sharp trailing edge
    assert abs(c[shapes.N_POINTS - 1, 0]) < 1e-12              # leading edge in the middle
    assert shapes.is_valid(p)
    assert not shapes.is_valid(np.array([0.17] + [0.15] * 5 + [-0.5] * 5))   # lower above upper


def test_sampler_respects_bounds():
    x = shapes.sample_airfoils(50, np.random.default_rng(0))
    assert x.shape == (50, 11) and all(shapes.is_valid(p) for p in x)
    assert x[:, 0].min() >= shapes.LO[0] and x[:, 0].max() <= shapes.HI[0]
    t = np.array([shapes.features(p)[0] for p in x])
    assert t.min() >= shapes.T_RANGE[0] and t.max() <= shapes.T_RANGE[1]


def test_backprop_matches_finite_differences():
    rng = np.random.default_rng(1)
    net = MLP([6, 8, 8, 3], rng)
    x, y = rng.standard_normal((5, 6)), rng.standard_normal((5, 3))

    def loss():
        return float(((net.forward(x) - y) ** 2).mean())

    out, acts = net.forward(x, keep=True)
    gw, gb = net.backward(acts, 2 * (out - y) / out.size)
    for arr, g in [(net.w[1], gw[1]), (net.b[0], gb[0]), (net.w[2], gw[2])]:
        idx = (0,) * arr.ndim
        old = arr[idx]
        arr[idx] = old + 1e-6
        up = loss()
        arr[idx] = old - 1e-6
        dn = loss()
        arr[idx] = old
        assert abs((up - dn) / 2e-6 - g[idx]) < 1e-6, (up, dn, g[idx])


def test_ddim_returns_clean_sample_with_an_exact_noise_oracle():
    # If the network predicted the noise exactly, deterministic DDIM must return the
    # clean sample for any number of steps, cold or warm. This tests the sampler loop.
    m = CondDDPM(3, 2, width=4)
    m.fit_norm(np.array([[0.0, 1.0, 2.0], [1.0, 3.0, 2.5]]), np.zeros((2, 2)))
    x0n = np.array([[0.3, -0.2, 0.9]] * 4)
    m._eps = lambda xt, t, cn: (xt - np.sqrt(ALPHA_BAR[int(t[0])]) * x0n) / np.sqrt(1 - ALPHA_BAR[int(t[0])])
    want = m.dx(x0n)
    c = np.zeros((4, 2))
    for steps in (5, 50):
        got = m.sample(c, np.random.default_rng(0), ddim_steps=steps)
        assert np.allclose(got, want, atol=1e-5), (steps, got, want)
    got = m.sample(c, np.random.default_rng(0), ddim_steps=5, x_init=want * 0.5, t_start=200)
    assert np.allclose(got, want, atol=1e-5)


def test_n95_matches_formula():
    assert metrics.n95(0.0) == float("inf") and metrics.n95(1.0) == 1.0
    assert abs(metrics.n95(0.1) - np.log(0.05) / np.log(0.9)) < 1e-12


def test_end_to_end_small():
    ev = evaluate.MockEvaluator()
    params, split, entries, _ = data.build(120, ev, seed=0, log=lambda *_: None)
    assert set(split) == {"train", "val", "test"}
    # no airfoil appears in two splits
    assert len({(a, s) for a, s in zip(entries["airfoil"], split[entries["airfoil"]])}) == len(set(entries["airfoil"]))
    tr = data.split_idx(entries, split, "train")
    x, c = params[entries["airfoil"][tr]], data.conditions(entries, tr)
    m = CondDDPM(11, 4, seed=0)
    m.train(x, c, steps=300, log=None)
    out = m.sample(c[:7], np.random.default_rng(0), ddim_steps=10)
    assert out.shape == (7, 11) and np.all(np.isfinite(out))
    warm = m.sample(c[:7], np.random.default_rng(0), ddim_steps=5, x_init=x[:7], t_start=50)
    assert warm.shape == (7, 11) and np.all(np.isfinite(warm))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
    print("all checks passed")
