# X thread (each post is under 280 characters; attach the image named in brackets)

**1/9**
I tried to replace days of CFD with a diffusion model that designs drone-wing airfoils instantly.

A lookup table beat it.

Here's how that loss turned into a pipeline that hits 97% of lift/drag targets within 10 tries, built for $0 in pure NumPy 🧵

**2/9**
Setup: 12,000 random airfoils × 44 flight conditions, scored with NeuralFoil.

Task: given a flight condition and target lift + drag, propose a shape.
Hit = lift within 1%, drag within 5%, confirmed by the solver.

**3/9**
Iteration 1: a conditional diffusion model vs "return the nearest airfoil in the dataset."

First-try hits:
1k airfoils: 2% vs 30%
6k: 7% vs 56%
12k: 10% vs 75%

The dumb baseline won every time.

**4/9**
So I stopped generating from scratch: retrieve → refine → verify.

Take the nearest airfoils, polish them with gradients of a small surrogate, and check every candidate with the solver.

Won at 10 and 100 tries, lost on the 1st: it kept "fixing" shapes that were already right. [1_iterations.png]

**5/9**
The fix: the solver had already scored every training airfoil at those exact conditions.

Read the stored result instead of guessing:
✅ exact keep/refine gate
✅ free bias correction
✅ zero extra solver calls

83% / 97% / 99% vs lookup's 75% / 91% / 98% [2_budgets.png]

**6/9**
Plot twist: with diffusion switched OFF, the same pipeline hit 93% on the first try.

A sweep on validation data agreed. Diffusion was optional, and reporting that is part of the result.

**7/9**
Real telemetry lands between grid points. There, the lookup collapses to 1%.

The pipeline still gets ~50% on the first try and 98% within 10. Interpolating stored results from neighbouring grid points pushes the first try to 58%. [3_offgrid.png]

**8/9**
Then I made the diffusion model physics-aware: a PIDM-style loss through the surrogate, balanced with ConFIG gradient combining. Hand-written backprop, no PyTorch.

The physics error fell ~30%. Off-grid one-sample hits went 10% → 18%. Helpful, not magic. [4_phase2.png]

**9/9**
Lessons:
• run the dumb baseline first
• your stand-in solver must beat your tolerance
• reuse what you've already computed
• test the realistic case

Caveats: 2-D sections, NeuralFoil not Navier-Stokes.

Code + data: github.com/Flotrpy/airfoil-diffusion
