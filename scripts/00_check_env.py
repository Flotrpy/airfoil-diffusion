"""Run this first. It checks the packages and that the NeuralFoil call works."""
import inspect
import sys

import common  # noqa: F401
import numpy as np

print("python", sys.version.split()[0])
for name in ("numpy", "scipy", "sklearn"):
    m = __import__(name)
    print(name, m.__version__)

try:
    import neuralfoil as nf
except Exception as e:
    print("\nNeuralFoil is NOT importable:", repr(e))
    print("Install it with:  pip install neuralfoil")
    print("(Graves et al. cite github.com/peterdsharpe/NeuralFoil, so  pip install git+https://github.com/peterdsharpe/NeuralFoil  is the fallback.)")
    sys.exit(1)

print("neuralfoil", getattr(nf, "__version__", "version not exposed"))
print("signature:", inspect.signature(nf.get_aero_from_coordinates))
print("(If the signature has no 'mach' argument, the scripts correctly leave Mach out.)")

from afd import evaluate, shapes  # noqa: E402

# NACA 0012 from its thickness formula (sharp trailing edge), Selig ordering.
x = shapes.cosine_x()
yt = 5 * 0.12 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4)
naca = np.concatenate([np.stack([x[::-1], yt[::-1]], 1), np.stack([x[1:], -yt[1:]], 1)])
r = nf.get_aero_from_coordinates(coordinates=naca, alpha=np.array([0.0, 5.0]), Re=3.0e5, model_size="large")
print("\nNACA 0012, Re 3e5, alpha 0 and 5 degrees")
for k in ("CL", "CD", "CM", "analysis_confidence"):
    print(f"  {k}:", np.asarray(r.get(k, 'missing')))
print("  Sanity: CL should be near 0 at 0 degrees and roughly 0.5 to 0.7 at 5 degrees (NeuralFoil gave 0.65 in testing) for a symmetric airfoil.")

ev = evaluate.NeuralFoilEvaluator()
rng = np.random.default_rng(0)
p = shapes.sample_airfoils(1, rng)[0]
out = ev.polar(p, 3.0e5, [0.0, 5.0, 10.0])
print("\nRandom Bernstein airfoil through the project wrapper:")
print("  CL", out["CL"], "\n  CD", out["CD"], "\n  ok", out["ok"], "\n  last_error", ev.last_error)
print("\nIf everything above printed numbers, you are ready for scripts/01_make_dataset.py")
