# airfoil-diffusion: Phases 0, 1 and 3 of the game plan

Conditional diffusion model that proposes airfoil shapes for a small delivery-drone wing.
NeuralFoil checks every proposed shape. Everything follows the uploaded papers (Airbus,
Geng, Graves, Diniz). Lines marked *my choice* are mine.

## Your settings (from our chat)
- Regime: Reynolds number 2e5 to 3.5e5 (steps of 5e4), angle of attack 0 to 10 degrees in
  1-degree steps (Geng et al.). Mach is left out: Geng varied it, but NeuralFoil's call as I
  wrote it has no Mach input. `00_check_env.py` prints the real signature so you can confirm.
- Evaluator: NeuralFoil. Graves et al. used it. It is a network trained on XFOIL runs, so it
  stands in for XFOIL and is not independent ground truth.
- Update period: 30 seconds (the Phase 3 gate).

## Setup on Windows (PowerShell, in this folder)
```
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python tests\test_pipeline.py        # wiring checks, about 10 s, no NeuralFoil needed
python scripts\00_check_env.py       # confirms NeuralFoil works
```
If `pip install neuralfoil` fails, Graves et al. cite github.com/peterdsharpe/NeuralFoil:
`pip install git+https://github.com/peterdsharpe/NeuralFoil`.

## Run
| Step | Command | What it does | Gate |
|---|---|---|---|
| Phase 0 | `python scripts\01_make_dataset.py --n-airfoils 1000` | Samples 1,000 Bernstein airfoils (Airbus: 11 parameters, N1 = 0.5, N2 = 1.0, box [-0.5, 1.5], A0 in [0.3, 1.0]). Removes self-intersecting shapes and Mahalanobis outliers. Scores 44 flow conditions per airfoil with NeuralFoil, drops CL outside [0, 2] and CD outside [0, 0.05] (Geng). Splits 60/20/20 **by airfoil**. | none |
| Phase 0 | `python scripts\02_baselines.py` | Fits the nearest-neighbour lookup, a linear POD-style model and an MLP regression, then checks one design per target with the solver. | Errors and solver failure rate recorded |
| Phase 1 | `python scripts\03_train_diffusion.py` | Trains a DDPM with Airbus's network: 4 tanh layers of 32 units, inputs (x_t, t/T, conditions), beta from 1e-3 to 0.2 over 1,000 steps. Conditions are (Re, alpha, CL, CD). | none |
| Phase 1 | `python scripts\04_evaluate.py` | Runs Geng's protocol: 1% CL and 5% CD tolerance, up to 100 trials, N95. Also reports the conditioning slope (Graves) and MMD (Diniz), all against the baselines. | Slopes far above Graves's 0.25 and 0.07; close to the lookup; beats POD on drag |
| Phase 3 | `python scripts\05_warm_start.py --period 30` | Starts from the current shape with T_start of 10, 25, 50, 100 or 200 (Diniz used 50) and compares against a cold start. Reports hit rate, shape change and seconds per update. | Fits 30 s |

A full run takes minutes on a laptop. Training itself took about 30 s here for 40,000 steps
on 2 CPU cores. Dataset time depends on NeuralFoil speed; `01_make_dataset.py` prints progress.

**To scale up:** Airbus's Table 4 used 6,000 airfoils and 64 units per layer, and says diffusion
pulled ahead of POD with more data. Try
`--n-airfoils 6000` and `03_train_diffusion.py --width 64`.

## What was tested and what wasn't
- Tested here, with a **mock** evaluator (a made-up smooth function, `--evaluator mock`):
  every script runs end to end, backprop matches finite differences, and DDIM returns the
  exact clean sample when given a perfect noise prediction. Mock numbers mean nothing
  physically, so don't quote them.
- **Not** tested here: the NeuralFoil call itself. This workspace blocks the package
  sites, so NeuralFoil couldn't be installed. `00_check_env.py` is the first real test on your laptop.
- One fix made during testing: Airbus's schedule pushes alpha-bar to about 1e-44 at t = T,
  so 50-step DDIM diverged (values near 1e20). The sampler now clips the predicted clean
  sample to the training range before each step, which is the standard fix. See `_denoise`
  in `src/afd/diffusion.py`.

## Choices that are mine, not the papers'
- "Order 6" is read as 6 coefficients per surface, because that gives Airbus's 11 parameters.
- Mahalanobis threshold at the 99% chi-square quantile, on max thickness, max camber and
  position of max thickness.
- The "POD" baseline is a ridge regression. With 11 modes of 11 parameters, POD is only a rotation.
- The MLP regression baseline.
- Learning-rate schedule (cosine, 1e-3), batch 256, 40,000 steps, 50 DDIM steps.

## Next (Phase 2)
Phase 2 adds a PIDM-style loss through a differentiable surrogate, plus ConFIGu and
PG-Diff-style corrections. That needs autodiff, so it moves to PyTorch (`pip install torch`).
