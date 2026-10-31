# airfoil-diffusion

Conditional diffusion model and a retrieve-refine-verify pipeline that propose airfoil
shapes for a small delivery-drone wing from requested flight conditions and lift/drag
targets (the inverse-design core of the "aerodynamic shapes from live telemetry" plan).
NeuralFoil checks every proposed shape. Methods follow the papers in the game plan
(Airbus/Wagenaar, Geng, Graves, Diniz, PIDM, ConFIG); lines marked *my choice* are mine.

Everything is plain numpy (no PyTorch), so it runs on a laptop CPU.

## Settings
- Regime: Reynolds number 2e5 to 3.5e5 (steps of 5e4), angle of attack 0 to 10 degrees in
  1-degree steps (Geng et al.). No Mach input (NeuralFoil's call has none).
- Evaluator: NeuralFoil (Graves et al.). It is a network trained on XFOIL runs, so it stands
  in for XFOIL and is not independent ground truth. **No Navier-Stokes solver is used.**
- Shapes: 11-parameter Bernstein/CST airfoil (Airbus). *My choice:* the sampling box is
  narrowed for a drone wing (A0 in [0.08, 0.25], others in [-0.05, 0.35]) because Airbus's
  box gave ~38%-thick sections here.
- Success = CL within 1% and CD within 5% of the request (Geng), one solver call per candidate.
- Update period target: 30 seconds.

## Setup
```
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
python tests/test_pipeline.py        # wiring checks with a mock evaluator, ~10 s
python scripts/00_check_env.py       # confirms NeuralFoil works
```

## Scripts
| Step | Script | What it does |
|---|---|---|
| Phase 0 | `01_make_dataset.py --n-airfoils 12000` | Samples airfoils, removes invalid shapes and Mahalanobis outliers, scores 44 conditions each with NeuralFoil, splits 60/20/20 by airfoil. |
| Phase 0 | `02_baselines.py` | Nearest-neighbour lookup, POD-style ridge, MLP regression. |
| Phase 1 | `03_train_diffusion.py --steps 200000 --width 128` | Conditional DDPM, Airbus network (4 tanh layers), conditions (Re, alpha, CL, CD). |
| Phase 1 | `04_evaluate.py` | Geng protocol (up to 100 trials, N95), conditioning slope, MMD, against baselines. |
| Phase 3 | `05_warm_start.py --period 30` | Warm-started updates from the current shape (Diniz T_start) vs cold start, timing per update. |
| Charts | `06_plots.py` | Matplotlib figures in `results/figures`. |
| Phase 1 re-test | `07_refine_eval.py` | Retrieve-refine-verify vs the lookup at budgets 1/10/100. |
| Next steps | `08_next.py` | Diffusion-strength sweep, off-grid targets, Phase 2 PIDM + ConFIG fine-tuning. |
| Off-grid | `09_offgrid.py` | Interpolated stored-result stand-in for targets between grid points. |

Retrieve-refine-verify (`src/afd/refine.py`): retrieve the nearest training airfoils;
check their stored NeuralFoil result at the target condition and submit unchanged if it
already hits; otherwise polish with gradients of a small surrogate MLP (optionally after a
diffusion warm start), correcting the surrogate's bias with solver results already paid for.

Phase 2 (`src/afd/pidm.py`): fine-tunes the diffusion model with an extra loss that the
surrogate's lift/drag of the denoised shape match the request, combined with the
denoising loss by ConFIG gradient combining.

## Results (sandbox runs, 12,000 airfoils, NeuralFoil-scored, 100 targets per set)
Share of targets hit within the solver budget.

**Phase 1 gate (on-grid test targets):** passed.

| Method | 1 try | 10 tries | 100 tries |
|---|---|---|---|
| Nearest-neighbour lookup | 75% | 91% | 98% |
| Diffusion alone (1 sample / 10 / 100) | 10% | 66% | 96% |
| Retrieve-refine-verify, diffusion warm start | 83% | 97% | 99% |
| Retrieve-refine-verify, no diffusion | 93% | 98% | 98% |

**Diffusion strength** (chosen on validation, reported on test): no diffusion 90% / 97%
at 1 / 10 tries; best diffusion setting (t = 25) 84% / 94%.

**Off-grid targets** (Re and alpha between grid points, like live telemetry):

| Method | 1 try | 10 tries |
|---|---|---|
| Lookup | 1% | 16% |
| Diffusion alone, Phase 1 / Phase 2 model | 10% / 18% | 58% / 70% |
| Retrieve-refine-verify, no stored-result gate | 47% | 100% |
| Retrieve-refine-verify, interpolated gate | OFFGRID_1 | OFFGRID_10 |

**Phase 2 (PIDM + ConFIG):** surrogate lift/drag error of denoised shapes fell ~30% during
fine-tuning; diffusion-alone hit rates rose slightly (test: 10%→11% at 1 sample,
64%→69% within 10). Diffusion still does not beat direct surrogate polishing inside the
pipeline. Raw numbers: `results/*.json`.

## Limits
- NeuralFoil is both the training labeller and the judge; results should be re-checked with
  XFOIL or a RANS/Navier-Stokes solver before trusting them for a real wing.
- 2-D airfoil sections only; no 3-D wing, no structural or manufacturing constraints.
- 100 targets per set: differences of a few points are within noise.

## Choices that are mine, not the papers'
- "Order 6" read as 6 coefficients per surface (gives Airbus's 11 parameters).
- Mahalanobis threshold at the 99% chi-square quantile on thickness/camber features.
- The "POD" baseline is a ridge regression; the MLP regression baseline.
- Cosine learning-rate schedule; 50 DDIM steps; x0 clipping in `_denoise` (Airbus's schedule
  makes alpha-bar ~1e-44 at t = T, and unclipped DDIM diverged).
- Surrogate (13→2 MLP), gate, bias correction, local surrogate fine-tune, interpolated gate.
