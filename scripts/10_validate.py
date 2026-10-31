"""Approved next steps:
(1) XFOIL check: re-score final designs (on-grid and off-grid targets) with XFOIL 6.99,
    which played no part in training or selection.
(2) Real data: UIUC Low-Speed Airfoil Tests wind-tunnel measurements (Re ~2e5-3.5e5) as
    targets; designs and the real tested airfoils are scored with XFOIL against them.
(3) 3-D wing: tapered drone wing from 3 sections designed by the pipeline under a spar
    constraint (>= 10% thick at quarter chord), evaluated with AeroSandbox vs a lookup wing.
Pipeline = retrieve-refine-verify without diffusion (the best setting), interpolated stored
results for the gate/bias, budget 10 NeuralFoil checks per target.
Needs the `xfoil` binary on PATH (apt install xfoil) and `aerosandbox` for part 3.
On a headless machine run a virtual display first (Ubuntu xfoil crashes with graphics off):
    Xvfb :99 -screen 0 800x600x16 &   (or set XFOIL_DISPLAY to a real display)
"""
import argparse
import json
import os
import re as regex
import subprocess
import tempfile
import time
import urllib.request
from multiprocessing import get_context

import numpy as np

import common
from afd import data, evaluate, experiment, metrics, refine, shapes

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default=str(common.DATA / "dataset.npz"))
ap.add_argument("--surrogate", default=str(common.MODELS / "surrogate.npz"))
ap.add_argument("--targets", type=int, default=100)
ap.add_argument("--budget", type=int, default=10)
ap.add_argument("--parts", default="123")
ap.add_argument("--evaluator", default="neuralfoil", choices=["neuralfoil", "mock"])
a = ap.parse_args()
T0 = time.time()
OUT = {}


def stamp(msg):
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def save():
    (common.RESULTS / "validation.json").write_text(json.dumps(OUT, indent=2, default=float))


# ---------------------------------------------------------------- XFOIL ----
def _xfoil_one(job):
    coords, re, alpha = job
    with tempfile.TemporaryDirectory() as d:
        f, pol = os.path.join(d, "af.dat"), os.path.join(d, "pol.txt")
        with open(f, "w") as fh:
            fh.write("af\n" + "\n".join(f"{x:.6f} {y:.6f}" for x, y in coords) + "\n")
        cmd = (f"LOAD {f}\nPANE\nOPER\nVISC {re:.0f}\nITER 200\n"
               f"PACC\n{pol}\n\nALFA {alpha:.3f}\n\nQUIT\n")
        try:
            subprocess.run(["xfoil"], input=cmd, text=True, capture_output=True, timeout=25,
                           env={**os.environ, "DISPLAY": os.environ.get("XFOIL_DISPLAY", ":99")})
            rows = [l.split() for l in open(pol) if regex.match(r"^\s*-?\d+\.\d+\s+-?\d", l)]
            if rows:
                return float(rows[-1][1]), float(rows[-1][2])
        except Exception:
            pass
    return np.nan, np.nan


def xfoil(coord_list, conds):
    with get_context("fork").Pool(2) as pool:   # fork: Python 3.14 defaults to forkserver
        return np.array(pool.map(_xfoil_one, [(c, t[0], t[1]) for c, t in zip(coord_list, conds)]))


def xf_params(xs, tg):
    return xfoil([shapes.coordinates(x) for x in xs], tg)


# -------------------------------------------------------------- pipeline ----
d = experiment.load_train_test(a.dataset)
ev = evaluate.make(a.evaluator)
E, P, tr = d["entries"], d["params"], d["train_idx"]
try:
    sur = refine.Surrogate.load(a.surrogate)
except FileNotFoundError:
    sur = refine.Surrogate().fit(P[E["airfoil"][tr]], E["Re"][tr], E["alpha"][tr], E["CL"][tr], E["CD"][tr], steps=60000)
    sur.save(a.surrogate)


class Norm:
    x_sd = P[E["airfoil"][tr]].std(0) + 1e-9


mu, sd = d["c_train"].mean(0), d["c_train"].std(0)
C = (d["c_train"] - mu) / sd
ids = E["airfoil"][tr]
label = {(int(E["airfoil"][j]), float(E["Re"][j]), float(E["alpha"][j])): (E["CL"][j], E["CD"][j]) for j in tr}
RES, ALS = np.array(data.RES), np.array(data.ALPHAS)


def neighbours(Tg, k, allowed=None):
    out = []
    for t in Tg:
        seen = []
        for j in np.argsort(((C - (t - mu) / sd) ** 2).sum(1)):
            if ids[j] not in seen and (allowed is None or allowed[ids[j]]):
                seen.append(ids[j])
            if len(seen) == k:
                break
        out.append(seen)
    return np.array(out)


def interp(aids, tg):
    out = np.full((len(aids), 2), np.nan)
    for n, (i, t) in enumerate(zip(aids, tg)):
        re_, al = float(np.clip(t[0], RES[0], RES[-1])), float(np.clip(t[1], ALS[0], ALS[-1]))
        r1 = int(np.clip(np.searchsorted(RES, re_) - 1, 0, len(RES) - 2))
        a1 = int(np.clip(np.searchsorted(ALS, al) - 1, 0, len(ALS) - 2))
        fr = (re_ - RES[r1]) / (RES[r1 + 1] - RES[r1])
        fa = (al - ALS[a1]) / (ALS[a1 + 1] - ALS[a1])
        v = [label.get((int(i), float(RES[r1 + x]), float(ALS[a1 + y]))) for x in (0, 1) for y in (0, 1)]
        if all(c is not None for c in v):
            v = np.array(v)
            out[n] = (1 - fr) * (1 - fa) * v[0] + (1 - fr) * fa * v[1] + fr * (1 - fa) * v[2] + fr * fa * v[3]
    return out


def ratio(y, tg):
    return np.maximum(abs(y[:, 0] - tg[:, 2]) / (0.01 * abs(tg[:, 2])), abs(y[:, 1] - tg[:, 3]) / (0.05 * tg[:, 3]))


def rrv(tg, nb, th=1.0):
    """Returns first-try design, best design (by NeuralFoil), all candidates and their NF scores."""
    n, K, rng = len(tg), nb.shape[1], np.random.default_rng(0)
    cands, ys = [], []
    best_x, best_e, bias = P[nb[:, 0]].copy(), np.full(n, np.inf), np.zeros((n, 2))
    for k in range(K):
        if k < 3:
            start = P[nb[:, k]]
            yi = interp(nb[:, k], tg)
            b0 = np.where(np.isfinite(yi), yi - sur.predict(start, tg[:, 0], tg[:, 1]), 0.0)
            x = refine.refine(Norm, sur, start, tg, b0, rng, use_diffusion=False)
            keep = np.isfinite(yi[:, 0]) & (ratio(np.nan_to_num(yi, nan=1e9), tg) <= th)
            x[keep] = start[keep]
        else:
            x = refine.refine(Norm, sur, best_x, tg, bias, rng, use_diffusion=False)
        cl, cd, ok = metrics.score(ev, x, tg)
        y = np.column_stack([cl, cd])
        e = np.where(ok, ratio(np.nan_to_num(y, nan=1e9), tg), np.inf)
        cands.append(x.copy())
        ys.append(np.where(ok[:, None], y, np.nan))
        b = e < best_e
        if b.any():
            best_x[b], best_e[b] = x[b], e[b]
            bias[b] = y[b] - sur.predict(x[b], tg[b, 0], tg[b, 1])
    return cands[0], best_x, np.array(cands), np.array(ys)


def rates(y, tg, tols=((0.01, 0.05), (0.02, 0.10), (0.05, 0.10))):
    ok = np.isfinite(y[:, 0])
    out = {"converged": float(ok.mean())}
    for tcl, tcd in tols:
        h = ok & (abs(y[:, 0] - tg[:, 2]) <= tcl * abs(tg[:, 2])) & (abs(y[:, 1] - tg[:, 3]) <= tcd * tg[:, 3])
        out[f"hit_CL{tcl:g}_CD{tcd:g}"] = float(h.mean())
    out["median_abs_CL_err"] = float(np.nanmedian(abs(y[:, 0] - tg[:, 2])))
    out["median_rel_CD_err"] = float(np.nanmedian(abs(y[:, 1] - tg[:, 3]) / tg[:, 3]))
    return out


def offgrid(pool, seed, n):
    rng, out = np.random.default_rng(seed), []
    while len(out) < n:
        re_, al = rng.uniform(2.0e5, 3.5e5), rng.uniform(0.2, 9.8)
        if min(abs(re_ - g) for g in data.RES) < 1.0e4 or abs(al - round(al)) < 0.2:
            continue
        r = ev.polar(P[rng.choice(pool)], re_, [al])
        if r["ok"][0] and 0 <= r["CL"][0] <= 2 and 0 < r["CD"][0] <= 0.05:
            out.append([re_, al, r["CL"][0], r["CD"][0]])
    return np.array(out)


# ------------------------------------------------------- (1) XFOIL check ----
if "1" in a.parts:
    OUT["xfoil_selftest"] = list(_xfoil_one((shapes.coordinates(P[0]), 3e5, 4.0)))
    stamp(f"XFOIL self-test on one airfoil: CL, CD = {OUT['xfoil_selftest']}")
    te_air = np.unique(E["airfoil"][d["test_idx"]])
    sets = {"on-grid": experiment.pick_targets(d["c_test"], a.targets), "off-grid": offgrid(te_air, 7, a.targets)}
    OUT["xfoil_check"] = {}
    for name, tg in sets.items():
        nb = neighbours(tg, a.budget)
        x1, xb, cands, ys = rrv(tg, nb)
        nf1, nfb = ys[0], ys[np.argmin(np.nan_to_num(np.stack([ratio(np.nan_to_num(y, nan=1e9), tg) for y in ys]), nan=1e9), axis=0), np.arange(len(tg))]
        xf_l, xf_1, xf_b = xf_params(P[nb[:, 0]], tg), xf_params(x1, tg), xf_params(xb, tg)
        both = np.isfinite(nf1[:, 0]) & np.isfinite(xf_1[:, 0])
        OUT["xfoil_check"][name] = {
            "NeuralFoil vs XFOIL on final designs": dict(
                median_abs_CL_diff=float(np.median(abs(nf1[both, 0] - xf_1[both, 0]))),
                median_rel_CD_diff=float(np.median(abs(nf1[both, 1] - xf_1[both, 1]) / xf_1[both, 1]))),
            "lookup, 1st try (judged by NeuralFoil)": rates(np.column_stack(metrics.score(ev, P[nb[:, 0]], tg)[:2]), tg),
            "pipeline, 1st try (judged by NeuralFoil)": rates(nf1, tg),
            "pipeline, best of 10 (judged by NeuralFoil)": rates(nfb, tg),
            "lookup, 1st try (judged by XFOIL)": rates(xf_l, tg),
            "pipeline, 1st try (judged by XFOIL)": rates(xf_1, tg),
            "pipeline, best of 10 (judged by XFOIL)": rates(xf_b, tg),
        }
        stamp(f"(1) {name}: " + json.dumps({k: (v.get('hit_CL0.01_CD0.05', v)) for k, v in OUT['xfoil_check'][name].items()}, default=float))
        save()


# ------------------------------------------------- (2) UIUC wind tunnel ----
def uiuc_targets():
    rows = []
    for v in range(1, 6):
        url = f"https://m-selig.ae.illinois.edu/pd/pub/lsat/volume{v:02d}/DRAG{v:02d}.TXT"
        try:
            txt = urllib.request.urlopen(url, timeout=30).read().decode("latin-1")
        except Exception as ex:
            print("  skip", url, ex)
            continue
        name, lines, i = None, txt.splitlines(), 0
        while i < len(lines):
            l = lines[i].strip()
            if l.startswith("Airfoil:"):
                name = l.split(":", 1)[1].strip()
            elif l.startswith("Average Reynolds"):
                re_ = float(lines[i + 1].split()[0])
            elif l.startswith("Number of angles"):
                n = int(lines[i + 1].split()[0])
                j = i + 2
                while j < len(lines) and not regex.match(r"^\s*-?\d", lines[j]):
                    j += 1
                for r in lines[j:j + n]:
                    t = r.split()
                    try:
                        rows.append((name, re_, float(t[0]), float(t[1]), float(t[2]), v))
                    except (ValueError, IndexError):
                        pass
                i = j + n
                continue
            i += 1
    return rows


def uiuc_coords(name, cache={}):
    key = regex.split(r"[\s(]", name.strip())[0].lower()
    if key in cache:
        return cache[key]
    xy = None
    try:
        txt = urllib.request.urlopen(f"https://m-selig.ae.illinois.edu/ads/coord/{key}.dat", timeout=30).read().decode("latin-1")
        pts = []
        for l in txt.splitlines()[1:]:
            t = l.split()
            try:
                pts.append((float(t[0]), float(t[1])))
            except (ValueError, IndexError):
                pass
        pts = np.array(pts)
        if len(pts) and pts[0, 0] > 1.5:                       # Lednicer: counts, upper LE->TE, lower LE->TE
            nu, nl = int(pts[0, 0]), int(pts[0, 1])
            up, lo = pts[1:1 + nu], pts[1 + nu:1 + nu + nl]
            pts = np.concatenate([up[::-1], lo[1:]])
        xy = pts if len(pts) > 20 else None
    except Exception:
        pass
    cache[key] = xy
    return xy


if "2" in a.parts:
    rows = uiuc_targets()
    sel = [r for r in rows if 1.8e5 <= r[1] <= 3.6e5 and 0 <= r[2] <= 10 and 0.1 <= r[3] <= 1.6 and 0 < r[4] <= 0.05]
    OUT["uiuc_pool"] = dict(rows_parsed=len(rows), rows_in_regime=len(sel), airfoils=len({r[0] for r in sel}))
    stamp(f"(2) UIUC rows parsed {len(rows)}, in regime {len(sel)}, airfoils {OUT['uiuc_pool']['airfoils']}")
    if not sel:
        raise SystemExit("no UIUC rows in regime (download blocked?)")
    rng = np.random.default_rng(3)
    pick = [sel[i] for i in rng.permutation(len(sel))[:a.targets]]
    tg = np.array([[r[1], r[2], r[3], r[4]] for r in pick])
    nb = neighbours(tg, a.budget)
    x1, xb, cands, ys = rrv(tg, nb)
    real = [uiuc_coords(r[0]) for r in pick]
    has = np.array([c is not None for c in real])
    xf_real = np.full((len(tg), 2), np.nan)
    if has.any():
        xf_real[has] = xfoil([c for c in real if c is not None], tg[has])
    OUT["uiuc"] = {
        "targets": int(len(tg)), "real_airfoil_coords_found": float(has.mean()),
        "tested airfoil itself, XFOIL vs wind tunnel": rates(xf_real, tg),
        "lookup, 1st try, XFOIL vs wind tunnel": rates(xf_params(P[nb[:, 0]], tg), tg),
        "pipeline, 1st try, XFOIL vs wind tunnel": rates(xf_params(x1, tg), tg),
        "pipeline, best of 10, XFOIL vs wind tunnel": rates(xf_params(xb, tg), tg),
        "pipeline, best of 10, NeuralFoil vs wind tunnel": rates(
            ys[np.argmin(np.nan_to_num(np.stack([ratio(np.nan_to_num(y, nan=1e9), tg) for y in ys]), nan=1e9), axis=0), np.arange(len(tg))], tg),
    }
    stamp("(2) " + json.dumps(OUT["uiuc"], default=float))
    save()


# ----------------------------------------------------------- (3) 3-D wing ----
def spar_depth(p):
    x, yu, yl = shapes.surfaces(p, x=np.array([0.25]))
    return float(yu[0] - yl[0])


if "3" in a.parts:
    V, NU, ALPHA, CL_T, SPAR = 20.0, 1.46e-5, 4.0, 0.6, 0.10
    secs = [(0.0, 0.25), (0.3, 0.20), (0.6, 0.15)]          # (span station y [m], chord [m]); semispan 0.6 m
    tr_air = np.unique(ids)
    depth = np.zeros(len(P))
    depth[tr_air] = [spar_depth(P[i]) for i in tr_air]
    allowed = depth >= SPAR
    wing = {"base": [], "design": []}
    OUT["wing_sections"] = []
    for y, c in secs:
        re_ = V * c / NU
        cand = tr_air[allowed[tr_air]]
        yi = interp(cand, np.tile([re_, ALPHA, 0, 0], (len(cand), 1)))
        okc = np.isfinite(yi[:, 0])
        tol = 0.01
        m = okc & (abs(yi[:, 0] - CL_T) <= tol * CL_T)
        if m.sum() < 3:
            tol = 0.03
            m = okc & (abs(yi[:, 0] - CL_T) <= tol * CL_T)
        base_i = cand[m][np.argmin(yi[m, 1])]
        cd_base = float(yi[m, 1].min())
        tg = np.array([[re_, ALPHA, CL_T, 0.9 * cd_base]])           # ask for 10% less drag
        nb = neighbours(tg, a.budget, allowed=allowed)
        _, _, cands, ys = rrv(tg, nb)
        feas = [k for k in range(len(cands)) if np.isfinite(ys[k, 0, 0]) and spar_depth(cands[k, 0]) >= SPAR
                and abs(ys[k, 0, 0] - CL_T) <= tol * CL_T]
        best = min(feas, key=lambda k: ys[k, 0, 1]) if feas else None
        des = cands[best, 0] if best is not None else P[base_i]
        xf = xf_params(np.array([P[base_i], des]), np.repeat(tg, 2, 0))
        row = dict(y=y, chord=c, Re=re_, CL_tolerance=tol, base_CD_NeuralFoil=cd_base,
                   design_CD_NeuralFoil=float(ys[best, 0, 1]) if best is not None else cd_base,
                   design_CL_NeuralFoil=float(ys[best, 0, 0]) if best is not None else float("nan"),
                   base_XFOIL=list(map(float, xf[0])), design_XFOIL=list(map(float, xf[1])),
                   spar_depth_base=spar_depth(P[base_i]), spar_depth_design=spar_depth(des))
        OUT["wing_sections"].append(row)
        wing["base"].append(P[base_i])
        wing["design"].append(des)
        stamp(f"(3) section y={y}: {row}")
    try:
        import aerosandbox as asb

        def evaluate_wing(params):
            xsecs = [asb.WingXSec(xyz_le=[0.25 * (0.25 - c), y, 0], chord=c,
                                  airfoil=asb.Airfoil(name=f"s{k}", coordinates=shapes.coordinates(p)))
                     for k, ((y, c), p) in enumerate(zip(secs, params))]
            plane = asb.Airplane(wings=[asb.Wing(name="w", symmetric=True, xsecs=xsecs)])
            res = []
            for al in np.arange(0, 9, 1.0):
                r = asb.AeroBuildup(airplane=plane, op_point=asb.OperatingPoint(velocity=V, alpha=al)).run()
                res.append((al, float(np.ravel(r["CL"])[0]), float(np.ravel(r["CD"])[0])))
            res = np.array(res)
            ld = res[:, 1] / res[:, 2]
            return dict(polar=res.tolist(), max_LD=float(ld.max()),
                        CD_at_CL_0p5=float(np.interp(0.5, res[:, 1], res[:, 2])),
                        LD_at_CL_0p5=float(0.5 / np.interp(0.5, res[:, 1], res[:, 2])))

        OUT["wing_3d"] = {"lookup wing": evaluate_wing(wing["base"]), "designed wing": evaluate_wing(wing["design"])}
        stamp("(3) wing: " + json.dumps({k: {kk: v[kk] for kk in ("max_LD", "LD_at_CL_0p5")} for k, v in OUT["wing_3d"].items()}))
    except Exception as ex:
        OUT["wing_3d_error"] = repr(ex)
        stamp(f"(3) AeroSandbox failed: {ex!r}")
    save()

print(json.dumps(OUT, indent=1, default=float)[:6000])
print("EXIT_OK")
