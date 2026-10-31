"""Charts and data for the write-up. All numbers are from the sandbox runs (NeuralFoil-scored,
100 targets per set; hit = CL within 1% and CD within 5%). Run: python docs/blog/make_charts.py"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, SURF, MUTED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#b9b8b2"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 15, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "font.size": 12, "legend.frameon": False, "figure.dpi": 100,
})
PCT = matplotlib.ticker.PercentFormatter(100)
ROWS = []


def rec(chart, group, series, value):
    ROWS.append(dict(chart=chart, group=group, series=series, hit_rate_pct=value))


def finish(fig, name, note):
    fig.text(0.01, 0.012, note + "  ·  github.com/Flotrpy/airfoil-diffusion", fontsize=9.5, color=INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)


def grouped(ax, groups, series, colors, fmt="{:.0f}%"):
    n, w = len(series), 0.8 / len(series)
    for k, ((label, vals), c) in enumerate(zip(series.items(), colors)):
        xs = [i + (k - (n - 1) / 2) * (w + 0.01) for i in range(len(groups))]
        bars = ax.bar(xs, vals, width=w, color=c, label=label, edgecolor=SURF, linewidth=2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, fmt.format(v), ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(len(groups)), groups)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(PCT)


# 1. the journey: first-try hit rate per iteration
steps = [("Diffusion alone", 10), ("Retrieve-refine-verify", 57), ("+ surrogate gate", 65),
         ("+ gate tuned on validation", 75), ("+ stored-result gate (passed)", 83), ("Same, diffusion switched off", 93)]
fig, ax = plt.subplots(figsize=(12, 6.75))
y = range(len(steps))
bars = ax.barh(list(y), [v for _, v in steps], color=[ORANGE] + [BLUE] * 5, height=0.62, edgecolor=SURF, linewidth=2)
for b, (_, v) in zip(bars, steps):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v}%", va="center", fontsize=12)
ax.axvline(75, color=INK2, ls="--", lw=1.5)
ax.text(75.8, -0.55, "nearest-neighbour lookup: 75%", fontsize=10.5, color=INK2)
ax.set_yticks(list(y), [s for s, _ in steps])
ax.invert_yaxis()
ax.set_ylim(5.5, -0.9)
ax.set_xlim(0, 105)
ax.xaxis.set_major_formatter(PCT)
ax.grid(axis="y", visible=False)
ax.set_xlabel("Targets hit on the first try")
ax.set_title("How the pipeline went from losing to a lookup table to beating it")
for s, v in steps:
    rec("1_iterations", "first try", s, v)
rec("1_iterations", "first try", "nearest-neighbour lookup", 75)
finish(fig, "1_iterations.png", "12,000 airfoils, 100 held-out targets, NeuralFoil-checked. Hit = CL within 1%, CD within 5%.")

# 2. on-grid result at 1 / 10 / 100 tries
s2 = {"Nearest-neighbour lookup": [75, 91, 98], "Retrieve-refine-verify": [83, 97, 99], "Same, no diffusion": [93, 98, 98]}
fig, ax = plt.subplots(figsize=(12, 6.75))
grouped(ax, ["1 try", "10 tries", "100 tries"], s2, [MUTED, BLUE, AQUA])
ax.set_ylim(0, 110)
ax.set_ylabel("Targets hit within the solver budget")
ax.set_title("Final pipeline vs. the lookup, by number of solver checks")
ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.0))
for k, vals in s2.items():
    for g, v in zip(["1", "10", "100"], vals):
        rec("2_budgets", f"{g} tries", k, v)
finish(fig, "2_budgets.png", "Targets on the training grid. Each candidate shape costs one NeuralFoil call.")

# 3. off-grid (live-telemetry-like) targets
g3 = ["Lookup", "Diffusion alone\n(Phase 1)", "Diffusion alone\n(Phase 2, physics loss)", "Retrieve-refine-\nverify", "+ interpolated\nstored results"]
s3 = {"1 try": [1, 10, 18, 50, 58], "within 10 tries": [16, 58, 70, 98, 97]}
fig, ax = plt.subplots(figsize=(12, 6.75))
grouped(ax, g3, s3, [BLUE, ORANGE])
ax.set_ylim(0, 110)
ax.set_ylabel("Targets hit")
ax.set_title("Between grid points (like real telemetry), the lookup collapses")
ax.legend(loc="upper left")
for k, vals in s3.items():
    for g, v in zip(g3, vals):
        rec("3_offgrid", k, g.replace("\n", " "), v)
finish(fig, "3_offgrid.png", "Re and angle of attack chosen between the training grid points; 100 test targets.")

# 4. Phase 2 effect on diffusion alone
g4 = ["On-grid, 1 sample", "On-grid, within 10", "Off-grid, 1 sample", "Off-grid, within 10"]
s4 = {"Phase 1 diffusion": [10, 64, 10, 58], "Phase 2 (PIDM loss + ConFIG)": [11, 69, 18, 70]}
fig, ax = plt.subplots(figsize=(12, 6.75))
grouped(ax, g4, s4, [MUTED, ORANGE])
ax.set_ylim(0, 85)
ax.set_ylabel("Targets hit by diffusion alone")
ax.set_title("A physics-informed loss helps diffusion a little, most off-grid")
ax.legend(loc="upper left")
for k, vals in s4.items():
    for g, v in zip(g4, vals):
        rec("4_phase2", g, k, v)
finish(fig, "4_phase2.png", "Fine-tuned 30k steps; surrogate lift/drag error of generated shapes fell ~30%. 100 targets: small gaps are noise.")

with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["chart", "group", "series", "hit_rate_pct"])
    w.writeheader()
    w.writerows(ROWS)
print("wrote", sorted(p.name for p in OUT.glob("*.png")), len(ROWS), "rows")
