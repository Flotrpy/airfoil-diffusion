"""Charts of the sandbox runs, made with matplotlib.

Input: results/small.json (exported from the run: run A = 1,000 airfoils, width 32;
run B = 6,000 airfoils, width 64; scored with NeuralFoil). Output: results/figures/*.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "results" / "small.json"
out = src.parent / "figures"
out.mkdir(exist_ok=True)
d = json.loads(src.read_text())

# reference palette (dataviz skill): categorical slots in fixed order, text inks, surface
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b2"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
    "legend.frameon": False, "figure.dpi": 150,
})


def save(fig, name, source):
    fig.text(0.01, 0.005, source, fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out / name)
    plt.close(fig)


SRC = "Source: sandbox run, NeuralFoil-scored, 100 held-out targets per run. Tolerance per Geng et al.: CL within 1%, CD within 5%."

# 1. hit rate per method, run A vs run B
methods = ["nearest-neighbour", "linear (POD-style)", "MLP regression", "diffusion"]
labels = ["Nearest-neighbour lookup", "Linear (POD-style)", "MLP regression", "Diffusion (1 sample)"]
vals = {r: [d[r]["phase1"]["baselines"][m]["p_single_trial"] if m != "diffusion"
            else d[r]["phase1"]["diffusion_single"]["p_single_trial"] for m in methods] for r in "AB"}
fig, ax = plt.subplots(figsize=(8, 4.2))
y = np.arange(len(methods))
h = 0.36
for k, (r, c, lab) in enumerate([("A", BLUE, "Run A: 1,000 airfoils, 4x32 net"), ("B", ORANGE, "Run B: 6,000 airfoils, 4x64 net")]):
    bars = ax.barh(y + (k - 0.5) * (h + 0.02), vals[r], height=h, color=c, label=lab)
    for b, v in zip(bars, vals[r]):
        ax.text(v + 0.008, b.get_y() + b.get_height() / 2, f"{v:.0%}", va="center", fontsize=9, color=INK)
ax.set_yticks(y, labels)
ax.invert_yaxis()
ax.set_xlim(0, 0.7)
ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.grid(axis="y", visible=False)
ax.set_xlabel("Share of targets where one design lands inside tolerance")
ax.set_title("The lookup baseline beats one diffusion sample in both runs")
ax.legend(loc="lower right")
save(fig, "1_hit_rate_by_method.png", SRC)

# 2. diffusion with up to 100 trials (Geng protocol)
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, key, title, fmt in [(axs[0], "share_with_hit", "Targets reached within 100 samples", "{:.0%}"),
                            (axs[1], "N95", "Samples for a 95% chance of a hit (N95)", "{:.0f}")]:
    v = [d[r]["phase1"]["diffusion_protocol"][key] for r in "AB"]
    bars = ax.bar(["Run A\n1,000 airfoils", "Run B\n6,000 airfoils"], v, color=[BLUE, ORANGE], width=0.55)
    for b, x in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, x, fmt.format(x), ha="center", va="bottom", fontsize=10)
    ax.set_title(title, fontsize=10.5)
    ax.grid(axis="x", visible=False)
    if key == "share_with_hit":
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    else:
        ax.set_ylim(0, 280)
fig.suptitle("More data helps: 6x the airfoils took target coverage from 49% to 79%", x=0.01, ha="left", fontweight="bold", fontsize=12)
save(fig, "2_trial_protocol.png", "Source: sandbox run, Geng et al.'s protocol (up to 100 samples per target, solver-checked).")

# 3. requested vs achieved, run B
fig, axs = plt.subplots(1, 2, figsize=(8.5, 4))
for ax, q, lim, title in [(axs[0], "CL", (0, 1.6), "Lift coefficient"), (axs[1], "CD", (0, 0.05), "Drag coefficient")]:
    ax.plot(lim, lim, color=MUTED, lw=1, ls="--", zorder=1)
    ax.text(lim[1] * 0.97, lim[1] * 0.9, "achieved = requested", ha="right", fontsize=8, color=INK2)
    for key, c, lab in [("req_vs_ach_nn", BLUE, "Nearest-neighbour"), ("req_vs_ach_diff", ORANGE, "Diffusion (1 sample)")]:
        r = d[key]
        ok = np.array(r["ok"], bool)
        ax.scatter(np.array(r["req" + q])[ok], np.array(r[q])[ok], s=16, color=c, alpha=0.75, label=lab,
                   edgecolors=SURF, linewidths=0.5, zorder=2)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel(f"Requested {q}")
    ax.set_ylabel(f"Achieved {q} (NeuralFoil)")
    ax.set_title(title, fontsize=10.5)
axs[0].legend(loc="upper left")
fig.suptitle("Run B: diffusion follows the request on average but scatters more", x=0.01, ha="left", fontweight="bold", fontsize=12)
save(fig, "3_requested_vs_achieved.png", "Source: sandbox run B, 200 held-out targets; feasible designs only.")

# 4. warm start trade-off
ts = [10, 25, 50, 100, 200]
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
for r, c, lab in [("A", BLUE, "Run A"), ("B", ORANGE, "Run B")]:
    res = d[r]["phase3"]["results"]
    axs[0].plot(ts, [res[f"warm T_start={t}"]["share_with_a_hit"] for t in ts], "-o", color=c, lw=2, ms=6, label=lab)
    axs[0].axhline(res["cold start"]["share_with_a_hit"], color=c, ls=":", lw=1.2)
    axs[1].plot(ts, [res[f"warm T_start={t}"]["shape_change_of_hits"] for t in ts], "-o", color=c, lw=2, ms=6, label=lab)
    axs[1].axhline(res["cold start"]["shape_change_of_hits"], color=c, ls=":", lw=1.2)
axs[0].set_title("Updates with at least one hit (of 20 candidates)", fontsize=10.5)
axs[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
axs[0].set_ylim(0, 0.6)
axs[1].set_title("Shape change of successful designs", fontsize=10.5)
axs[1].set_ylabel("Distance from current shape\n(normalized parameters)")
for ax in axs:
    ax.set_xscale("log")
    ax.set_xticks(ts, [str(t) for t in ts])
    ax.set_xlabel("T_start (of 1,000 steps; Diniz used 50)")
axs[0].legend(loc="upper left")
axs[1].text(10, 4.0, "dotted lines = cold start\n(pure noise)", fontsize=8, color=INK2, ha="left", va="top")
axs[0].text(10, 0.57, "dotted lines = cold start", fontsize=8, color=INK2, ha="left")
axs[0].legend(loc="upper left", bbox_to_anchor=(0, 0.92))
fig.suptitle("Warm start trades success for smaller shape changes", x=0.01, ha="left", fontweight="bold", fontsize=12)
save(fig, "4_warm_start_tradeoff.png", "Source: sandbox runs, 40 simulated target changes each. Every update took about 0.03 s, far inside the 30 s period.")

# 5. dataset
s = d["dataset_sample"]
alpha, cl, cd, re = (np.array(s[k]) for k in ("alpha", "CL", "CD", "Re"))
fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.8))
shades = {2.0e5: "#9ec5f4", 2.5e5: "#5598e7", 3.0e5: "#2a78d6", 3.5e5: "#184f95"}
for rv, c in shades.items():
    m = np.isclose(re, rv)
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, m.sum())
    axs[0].scatter(alpha[m] + jitter, cl[m], s=8, color=c, alpha=0.7, label=f"Re {rv:.1e}".replace("e+05", "e5"))
    axs[1].scatter(cd[m], cl[m], s=8, color=c, alpha=0.7)
axs[0].set_xlabel("Angle of attack (deg)")
axs[0].set_ylabel("CL")
axs[0].set_title("Lift vs angle of attack", fontsize=10.5)
axs[1].set_xlabel("CD")
axs[1].set_title("Drag polar", fontsize=10.5)
axs[0].legend(loc="upper left", fontsize=8, markerscale=1.5)
fig.suptitle("Training data: 6,000 drone-wing airfoils, 174,732 labelled conditions", x=0.01, ha="left", fontweight="bold", fontsize=12)
save(fig, "5_dataset.png", "Source: run B dataset (random sample of 700 entries), labels from NeuralFoil.")

# 6. shapes for one target
sd = d["shape_demo"]
tgt = sd["target"]
fig, ax = plt.subplots(figsize=(8.5, 3.3))
nhit = 0
for i, c in enumerate(sd["cands"]):
    if c is None:
        continue
    c = np.array(c)
    nhit += int(sd["ok"][i] and abs(sd["CL"][i] - tgt[2]) / tgt[2] <= 0.01 and abs(sd["CD"][i] - tgt[3]) / tgt[3] <= 0.05)
    ax.plot(c[:, 0], c[:, 1], color=ORANGE, lw=1.1, alpha=0.8, label="Diffusion samples (8)" if i == 0 else None)
nn = np.array(sd["nn"])
ax.plot(nn[:, 0], nn[:, 1], color=BLUE, lw=2.2, ls="--", label="Nearest-neighbour lookup (1)", zorder=4)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, fontsize=9)
ax.set_aspect("equal")
ax.set_xlabel("x / chord")
ax.set_ylabel("y / chord")
ax.set_title(f"One target, eight different shapes: Re {tgt[0]:.0e}, {tgt[1]:.0f} deg, CL {tgt[2]:.2f}, CD {tgt[3]:.4f}".replace("e+05", "e5"), fontsize=11)
ax.text(1.0, 0.085, f"{nhit} of 8 inside tolerance", ha="right", fontsize=8.5, color=INK2)
save(fig, "6_candidate_shapes.png", "Source: sandbox run B. Diversity is the diffusion model's advantage over a lookup, which returns one existing airfoil.")
print("wrote", sorted(p.name for p in out.glob("*.png")))
