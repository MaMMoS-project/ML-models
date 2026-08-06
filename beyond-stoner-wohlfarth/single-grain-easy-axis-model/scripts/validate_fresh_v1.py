#!/usr/bin/env python3
"""
Validate the forward single-grain-easy-axis BEST model on FRESH data.

The latest models are trained on the V2 dataset
(data/single_grain_cube_50nm_aligned.csv).  This script uses the older V1 file
data/magnetic_materials.csv (~1,497 points) as an external validation set: the two
share 0 % of their (Ms, A, K1) points, so V1 is genuinely held out.  V1 is aligned
(H-K1 relative angle a constant ~1 deg), matching the aligned model.

It runs the deployed prediction pipeline from scripts/load_onnx_models.py
    validate-inputs classifier -> soft/hard classifier -> per-class RF regressor
    (all ONNX; features log1p-transformed, targets expm1-inverted)
then reports per-target statistics and writes a parity plot.

Outputs (in ./validation_v1/):
    parity.png   predicted vs. true Hc / Mr / (BH)max, colored by soft/hard class
    stats.csv    per-target R2 (linear & log), MAE, RMSE, median relative error
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))          # so `import load_onnx_models` works either way
import load_onnx_models as lom          # reuse the real deployed pipeline
ROOT = HERE.parent
OUT = ROOT / "validation_v1"
OUT.mkdir(exist_ok=True)

TARGETS = ["Hc", "Mr", "BHmax"]
UNITS = {"Hc": "A/m", "Mr": "A/m", "BHmax": "J/m³"}
# Okabe-Ito colourblind-safe pair; shape is a redundant (secondary) encoding.
CLASS_STYLE = {"soft": ("#E69F00", "^"), "hard": ("#0072B2", "o")}
INK, MUTED = "#222222", "#888888"
LABEL_REL_ERR = 0.01   # ~1% simulation error on the target values (noise floor on the truth)


def load_mammos_csv(path):
    """Read a MaMMoS-format CSV (skip the '#'-prefixed metadata header)."""
    lines = open(path).readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Ms,"))
    return pd.read_csv(path, skiprows=hdr)


def predict_batch(Ms, A, K):
    """Vectorised form of load_onnx_models.calculate_extrinsic_properties.
    Returns pred[N,3] (Hc, Mr, BHmax; NaN where invalid), valid_mask[N], classes[N]."""
    Ms, A, K = (np.asarray(v, float) for v in (Ms, A, K))
    X = np.column_stack([Ms, A, K]).astype(np.float32)
    valid = np.asarray(lom.validate_input(Ms, A, K)).ravel()
    cls = np.asarray(lom.classify_magnetic_material(Ms, A, K)).ravel()
    pred = np.full((len(Ms), 3), np.nan, np.float32)
    Xlog = np.log1p(X)                                     # same preprocessing as the pipeline
    vmask = valid == "valid"
    for c in ("soft", "hard"):
        m = vmask & (cls == c)
        if m.any():
            sess = ort.InferenceSession(str(lom.MODELS[c]), lom._SESSION_OPTIONS)
            ylog = sess.run(None, {sess.get_inputs()[0].name: Xlog[m]})[0]
            pred[m] = np.expm1(ylog)                       # inverse log transform
    return pred, vmask, cls


def target_stats(y_true, y_pred):
    m = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    yt, yp = y_true[m], y_pred[m]
    return dict(
        N=int(m.sum()),
        R2=r2_score(yt, yp),
        R2_log=r2_score(np.log1p(yt), np.log1p(yp)),
        MAE=mean_absolute_error(yt, yp),
        RMSE=np.sqrt(mean_squared_error(yt, yp)),
        MedRelErr=float(np.median(np.abs(yp - yt) / np.abs(yt))),
    )


def parity_plot(df, pred, vmask, cls, misclass, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        yt_all, yp_all = df[t].values, pred[:, i]
        base = vmask & np.isfinite(yp_all) & (yt_all > 0) & (yp_all > 0)
        # robust axis range (1st percentile .. max of true+pred), so a handful of
        # near-zero points don't stretch the view; all points still count in the stats.
        allv = np.concatenate([yt_all[base], yp_all[base]]); allv = allv[allv > 0]
        lo, hi = np.percentile(allv, 1) / 1.5, allv.max() * 1.5
        ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1.2, zorder=1)
        # ±1% simulation-noise floor on the (target) truth: no model can validate below it.
        xs = np.array([lo, hi])
        ax.fill_between(xs, xs * (1 - LABEL_REL_ERR), xs * (1 + LABEL_REL_ERR),
                        color=MUTED, alpha=0.25, lw=0, zorder=1,
                        label="±1% sim. floor" if i == 0 else None)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        for c in ("hard", "soft"):
            m = base & (cls == c)
            color, marker = CLASS_STYLE[c]
            ax.scatter(yt_all[m], yp_all[m], s=14, c=color, marker=marker,
                       alpha=0.5, linewidths=0, label=c, zorder=2)
        # highlight points routed through the WRONG regressor (classifier disagrees
        # with the true Mr/Ms>0.4 class) — open red rings on top.
        mm = base & misclass
        ax.scatter(yt_all[mm], yp_all[mm], s=46, facecolors="none",
                   edgecolors="#d62728", linewidths=1.3, marker="o", zorder=3,
                   label="wrong model (classifier error)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel(f"true {t}  [{UNITS[t]}]", color=INK)
        ax.set_ylabel(f"predicted {t}  [{UNITS[t]}]", color=INK)
        s = target_stats(yt_all[vmask], yp_all[vmask])
        ax.set_title(f"{t}", color=INK, fontsize=12)
        ax.text(0.04, 0.96, f"$R^2$={s['R2']:.3f}\n$R^2_{{log}}$={s['R2_log']:.3f}\n"
                f"med|rel|={100*s['MedRelErr']:.1f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, alpha=0.85))
        ax.grid(True, which="major", color="#e6e6e6", lw=0.6, zorder=0)
        ax.tick_params(colors=INK)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Forward model (V2-trained) validated on fresh V1 data "
                 "(magnetic_materials.csv, 0 % overlap)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    df = load_mammos_csv(ROOT / "data" / "magnetic_materials.csv")
    Ms, A, K = df["Ms"].values, df["A"].values, df["K1"].values
    pred, vmask, cls = predict_batch(Ms, A, K)
    # exclude points outside the training volume (extrapolation) and inputs the
    # validate-inputs classifier flags as invalid
    in_vol = np.asarray(lom.check_in_training_volume(Ms, A, K, warn=False)).ravel()
    use = vmask & in_vol

    # Ground-truth hard/soft class from the TRUE Mr/Ms (threshold_clustering: >0.4 = hard),
    # vs the classifier's routing decision (cls) -> which points used the WRONG regressor.
    ratio = np.divide(df["Mr"].values, df["Ms"].values,
                      out=np.zeros(len(df)), where=df["Ms"].values != 0)
    true_cls = np.where(ratio > 0.4, "hard", "soft")
    misclass = cls != true_cls

    n = len(df)
    print(f"Fresh V1 points (magnetic_materials.csv): {n}")
    print(f"  flagged invalid by validate-inputs : {int((~vmask).sum())}")
    print(f"  outside V2 training volume (excluded): {int((~in_vol).sum())}")
    print(f"  used for validation                 : {int(use.sum())}")
    print(f"  class split (used): soft={int((use & (cls=='soft')).sum())}  "
          f"hard={int((use & (cls=='hard')).sum())}")

    rows = []
    for i, t in enumerate(TARGETS):
        s = {"target": f"{t} ({UNITS[t]})", **target_stats(df[t].values[use], pred[use, i])}
        rows.append(s)
    tbl = pd.DataFrame(rows)[["target", "N", "R2", "R2_log", "MAE", "RMSE", "MedRelErr"]]
    tbl.to_csv(OUT / "stats.csv", index=False)
    print("\n=== per-target validation statistics (fresh, in-volume V1 data) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
        print(tbl.to_string(index=False))
    floor = 100 * LABEL_REL_ERR
    print(f"\nThe targets carry a ~{floor:.0f}% simulation error, so a median relative error at or "
          f"below ~{floor:.0f}% is at the noise floor (cannot be meaningfully improved):")
    for r in rows:
        med = 100 * r["MedRelErr"]
        print(f"    {r['target']:<16} med|rel|={med:5.1f}%   "
              f"[{'NOISE-LIMITED' if med <= floor else 'above floor'}]")

    # --- hard/soft classifier routing check ---
    n_used = int(use.sum())
    mis = use & misclass
    n_mis = int(mis.sum())
    soft_as_hard = int((use & (true_cls == "soft") & (cls == "hard")).sum())
    hard_as_soft = int((use & (true_cls == "hard") & (cls == "soft")).sum())
    acc = 100.0 * (1 - n_mis / n_used)
    pd.DataFrame([{"used": n_used, "misclassified": n_mis, "accuracy_pct": round(acc, 2),
                   "soft_routed_to_hard": soft_as_hard, "hard_routed_to_soft": hard_as_soft}]
                 ).to_csv(OUT / "classifier_routing.csv", index=False)
    print("\n=== hard/soft classifier routing check (truth: Mr/Ms > 0.4 = hard) ===")
    print(f"  used points                         : {n_used}")
    print(f"  classifier accuracy on fresh data   : {acc:.1f}%")
    print(f"  misrouted (wrong regressor used)    : {n_mis} ({100*n_mis/n_used:.1f}%)")
    print(f"    soft (Mr/Ms<=0.4) -> HARD model   : {soft_as_hard}")
    print(f"    hard (Mr/Ms>0.4)  -> SOFT model   : {hard_as_soft}")
    # isolate regressor quality from classifier error, for Mr (the affected target)
    ok = use & ~misclass
    mr = TARGETS.index("Mr")
    r2_all = target_stats(df["Mr"].values[use], pred[use, mr])["R2"]
    r2_ok = target_stats(df["Mr"].values[ok], pred[ok, mr])["R2"]
    print(f"  Mr R^2: all used = {r2_all:.3f}   |   correctly-routed only = {r2_ok:.3f}  "
          f"(the gap is classifier-error, not regressor-error)")

    parity_plot(df, pred, use, cls, misclass, OUT / "parity.png")


if __name__ == "__main__":
    main()
