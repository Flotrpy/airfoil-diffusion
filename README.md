# airfoil-diffusion

![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy only](https://img.shields.io/badge/NumPy-only%2C%20no%20PyTorch-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-baselines-F7931E?logo=scikitlearn&logoColor=white)
![Evaluator](https://img.shields.io/badge/evaluator-NeuralFoil-0A7BBB)
![Runs on CPU](https://img.shields.io/badge/runs%20on-laptop%20CPU-2ea44f)
![Cost](https://img.shields.io/badge/cost-%240-brightgreen)
![Phase 1 gate](https://img.shields.io/badge/Phase%201%20gate-passed-success)
![Status](https://img.shields.io/badge/status-research%20prototype-orange)
![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)

## The problem (as originally stated)
> **The Problem:** Designing aerodynamic profiles for electric vehicles (EVs) or delivery
> drones requires intensive Computational Fluid Dynamics (CFD) simulations that take
> supercomputers days or weeks to run for a single shape modification.
>
> **The Solution:** Train a diffusion model constrained by the Navier-Stokes fluid equations
> to instantly generate and optimize surface geometries in real-time based on live
> wind-tunnel or flight telemetry.

The brief also asked for a game plan built only from 18 uploaded research papers, and for
a solution that costs nothing and can be built right away.

## How the solution changed with each iteration
| # | What we tried | What happened | What changed next |
|---|---|---|---|
| 0 | **Game plan from the papers.** Full Navier-Stokes CFD is out of reach at zero cost, so: 2-D drone-wing airfoils (11-parameter CST shapes, Airbus), NeuralFoil as the stand-in solver (Graves), a DDPM conditioned on (Re, angle, CL, CD), a physics loss later (PIDM), and warm starts for live updates (Diniz). | Plan written; delivery-drone regime, NeuralFoil, and a 30 s update period chosen. | Build Phase 0 (data + baselines), Phase 1 (diffusion), Phase 3 (warm start) first. |
| 1 | **Diffusion vs. simple baselines** at 1k, 6k, then 12k airfoils. | Diffusion lost to a nearest-neighbour lookup on one try: 2% vs 30%, 7% vs 56%, 10% vs 75%. **Phase 1 gate failed.** Warm-start updates took ~0.03 s, far inside 30 s. | Keep the problem, change the method: use diffusion to *refine* retrieved shapes instead of generating from scratch. |
| 2 | **Retrieve → refine → verify:** start from the nearest airfoils, diffusion warm start, then polish with gradients of a small surrogate. | Won at 10 and 100 tries (96%, 99%) but lost at one try (57% vs 75%): it was "fixing" airfoils that were already right. | Add a gate that leaves good airfoils alone. |
| 3 | **Surrogate gate**, then a gate threshold tuned on validation data. | 65%, then only a tie (75%). The surrogate's error is about as big as the 1% lift tolerance, so it can't tell hits from misses. | Make the stand-in solver more accurate. |
| 4 | **Stored-result gate:** read each retrieved airfoil's already-computed NeuralFoil result (exact on the training grid), use it for a free bias correction, fine-tune the surrogate locally. | **Phase 1 gate passed:** 83% / 97% / 99% vs lookup 75% / 91% / 98%. Surprise: with diffusion switched off it scored 93% on one try. | Measure how much diffusion actually helps, and test realistic off-grid conditions. |
| 5 | **Diffusion strength sweep** and **off-grid targets** (flight conditions between grid points, like real telemetry). | No diffusion was best (90% vs 84% on test). Off-grid, the lookup collapsed to 1%; the pipeline still reached 47% on one try and 100% within ten. | Try to make diffusion pull its weight with the physics loss from the plan. |
| 6 | **Phase 2: physics-informed diffusion** (PIDM-style loss through the surrogate, balanced with ConFIG). | Physics error of generated shapes fell ~30%; diffusion alone improved modestly (off-grid one-sample hits 10% → 18%), but still trails direct polishing. | Improve the off-grid first try instead. |
| 7 | **Interpolated stored results** for off-grid targets (from the 4 surrounding grid points). | ~3.5× more accurate than the surrogate; off-grid first-try hits 50% → **58%** (97% within ten). | Current state. |

**Where it ended up vs. the original idea:** the working system is a fast
retrieve-refine-verify loop over a learned airfoil space, with diffusion as an optional,
physics-informed generator rather than the core engine. Updates take seconds on a laptop
instead of days of CFD. The "Navier-Stokes" constraint is currently NeuralFoil (a learned
XFOIL stand-in) through a surrogate, not the flow equations themselves, and the shapes are
2-D sections, not full EV or drone bodies. Those are the next gaps to close (see *Limits*).

## What this repo is

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
| Validation | `10_validate.py` | XFOIL re-check, UIUC wind-tunnel targets, 3-D wing with a spar constraint (needs `xfoil`, `aerosandbox`, and a display or `Xvfb :99`). |

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
| Retrieve-refine-verify, no stored results (`08`/`09` runs) | 47–50% | 98–100% |
| Retrieve-refine-verify, interpolated stored results (`09`) | **58%** | 97% |

Interpolating a retrieved airfoil's stored NeuralFoil results between the 4 surrounding
grid points is ~3.5x more accurate than the surrogate (median CL error 0.0013 vs 0.0044).
Its gain comes from the bias correction; the keep-unchanged gate made no difference on
validation, so it was left off (threshold chosen on validation, 53%→63% there).

**Phase 2 (PIDM + ConFIG):** surrogate lift/drag error of denoised shapes fell ~30% during
fine-tuning; diffusion-alone hit rates rose slightly (test: 10%→11% at 1 sample,
64%→69% within 10). Diffusion still does not beat direct surrogate polishing inside the
pipeline. Raw numbers: `results/*.json`.

## Independent checks (XFOIL, wind-tunnel data, 3-D wing) — `scripts/10_validate.py`
All with the final pipeline (no diffusion, interpolated stored results, 10 NeuralFoil checks per target).

**1. XFOIL re-check.** XFOIL 6.99 played no part in training or selection.
NeuralFoil and XFOIL disagree on the final designs by a median of 0.014 in CL (about 2–3%)
and 2–3% in CD, which is *larger than the 1% CL tolerance*, so the headline hit rates do not
survive an independent solver at that tolerance:

| Targets | Method | NeuralFoil, 1% / 5% | XFOIL, 1% / 5% | XFOIL, 5% / 10% |
|---|---|---|---|---|
| On-grid | Lookup, 1st try | 75% | 15% | 62% |
| On-grid | Pipeline, best of 10 | 96% | 13% | 65% |
| Off-grid | Lookup, 1st try | 1% | 2% | 25% |
| Off-grid | Pipeline, 1st try | 59% | 17% | 72% |
| Off-grid | Pipeline, best of 10 | 97% | 22% | 73% |

On-grid, the pipeline's edge over the lookup disappears under XFOIL. Off-grid (the realistic
case) it holds: roughly 3× the lookup's hit rate at the looser tolerance.

**2. Real wind-tunnel targets.** 100 measured (Re, angle, CL, CD) points from the UIUC
Low-Speed Airfoil Tests (85 airfoils, Re 1.8–3.6×10⁵), scored with XFOIL:

| | CL 1% / CD 5% | CL 5% / CD 10% | median CL error | median CD error |
|---|---|---|---|---|
| The actual tested airfoil, XFOIL | 1% | 5% | 0.076 | 11% |
| Lookup, 1st try | 12% | 55% | 0.031 | 4.2% |
| Pipeline, 1st try | 23% | 83% | 0.017 | 2.8% |
| Pipeline, best of 10 | 29% | 75% | 0.013 | 2.2% |

XFOIL itself misses the measurements of the real airfoils by ~0.08 in CL (low-Re
laminar-bubble effects and tunnel uncertainty), so 1% agreement with a wind tunnel is not a
meaningful target at this Reynolds number. The pipeline produces shapes whose XFOIL
performance is much closer to the requested measured values than the lookup's.

**3. 3-D wing.** Tapered semi-wing (0.6 m, chords 0.25/0.20/0.15 m, 20 m/s, Re 2–3.4×10⁵),
three sections designed for CL 0.6 at 4° with a spar constraint (≥ 10% thick at quarter
chord), compared with the best stored "lookup" sections under the same constraint. Section
drag fell 6–10% (XFOIL confirms 9–10% at root and tip, with CL 2–6% low); no improvement was
found mid-span. AeroSandbox AeroBuildup: max L/D **20.0 vs 19.5** (+2.5%), L/D at CL 0.5
19.6 vs 19.2. AeroBuildup uses NeuralFoil for section data, so this is not independent.

## Limits
- NeuralFoil labels the data and selects the designs. Against XFOIL the 1% CL tolerance is not
  met (see *Independent checks*); a RANS/Navier-Stokes check is still missing.
- Mostly 2-D sections; the 3-D wing test uses a simple build-up model and a single spar-depth
  constraint, not structural analysis.
- 100 targets per set: differences of a few points are within noise.

## Choices that are mine, not the papers'
- "Order 6" read as 6 coefficients per surface (gives Airbus's 11 parameters).
- Mahalanobis threshold at the 99% chi-square quantile on thickness/camber features.
- The "POD" baseline is a ridge regression; the MLP regression baseline.
- Cosine learning-rate schedule; 50 DDIM steps; x0 clipping in `_denoise` (Airbus's schedule
  makes alpha-bar ~1e-44 at t = T, and unclipped DDIM diverged).
- Surrogate (13→2 MLP), gate, bias correction, local surrogate fine-tune, interpolated gate.

## License
MIT, see [LICENSE](LICENSE).
