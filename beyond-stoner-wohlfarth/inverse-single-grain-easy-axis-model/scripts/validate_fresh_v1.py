#!/usr/bin/env python3
"""
Validate the INVERSE single-grain-easy-axis BEST model on FRESH data.

The inverse model predicts the intrinsic properties (Ms, A, K1) from the extrinsic
properties (Hc, Mr, (BH)max) -- the mirror of the forward single-grain-easy-axis model.

The latest models are trained on the V2 dataset
(data/single_grain_cube_50nm_aligned.csv).  This script uses the older V1 file
data/magnetic_materials.csv (~1,497 points) as an external validation set: the two
share 0 % of their (Ms, A, K1) points, so V1 is genuinely held out.

It runs the deployed inverse pipeline from scripts/load_onnx_models.py
    soft/hard classifier (raw Hc, Mr, BHmax) -> per-class RF regressor
    (regressor inputs log1p-transformed, targets expm1-inverted)
then reports per-target statistics and writes a parity plot.

Outputs (in ./validation_v1/):
    parity.png             predicted vs. true Ms / A / K1, colored by soft/hard class
    stats.csv              per-target R2 (linear & log), MAE, RMSE, median relative error
    classifier_routing.csv hard/soft routing accuracy on the fresh data
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
import predict as lom          # reuse the real deployed inverse pipeline
ROOT = HERE.parent
OUT = ROOT / "validation_v1"
OUT.mkdir(exist_ok=True)

TRAINING_DATA = ROOT / "data" / "single_grain_cube_50nm_aligned.csv"

# Predicted-target order matches load_onnx_models: (Ms, A, K). The true columns in the
# fresh CSV are named Ms, A, K1.
TARGETS = ["Ms", "A", "K"]
TRUE_COL = {"Ms": "Ms", "A": "A", "K": "K1"}
UNITS = {"Ms": "A/m", "A": "J/m", "K": "J/m³"}
# The three extrinsic inputs, as named in both CSVs.
INPUTS = ["Hc", "Mr", "BHmax"]
# Okabe-Ito colourblind-safe pair; shape is a redundant (secondary) encoding.
CLASS_STYLE = {"soft": ("#E69F00", "^"), "hard": ("#0072B2", "o")}
INK, MUTED = "#222222", "#888888"
LABEL_REL_ERR = 0.01   # ~1% sim. error — here on the extrinsic INPUTS, not the (exact) targets


def load_mammos_csv(path):
    """Read a MaMMoS-format CSV (skip the '#'-prefixed metadata header)."""
    lines = open(path).readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Ms"))
    return pd.read_csv(path, skiprows=hdr)


def predict_batch(Hc, Mr, BHmax):
    """Vectorised form of load_onnx_models.calculate_intrinsic_properties.
    Returns pred[N,3] (Ms, A, K; NaN where a class has no session) and classes[N]."""
    Hc, Mr, BHmax = (np.asarray(v, float) for v in (Hc, Mr, BHmax))
    X = np.column_stack([Hc, Mr, BHmax]).astype(np.float32)
    cls = np.asarray(lom.classify_magnetic_material(Hc, Mr, BHmax)).ravel()
    pred = np.full((len(Hc), 3), np.nan, np.float32)
    Xlog = np.log1p(X)                                     # same preprocessing as the pipeline
    for c in ("soft", "hard"):
        m = cls == c
        if m.any():
            sess = ort.InferenceSession(str(lom.MODELS[c]), lom._SESSION_OPTIONS)
            ylog = sess.run(None, {sess.get_inputs()[0].name: Xlog[m]})[0]
            pred[m] = np.expm1(ylog)                       # inverse log transform
    return pred, cls


def training_bounds():
    """Per-input (min, max) of Hc, Mr, BHmax over the V2 training data."""
    df = load_mammos_csv(TRAINING_DATA)
    return {c: (float(df[c].min()), float(df[c].max())) for c in INPUTS}


def in_training_volume(df, bounds):
    """True where all three extrinsic inputs lie within the training min/max box."""
    ok = np.ones(len(df), dtype=bool)
    for c in INPUTS:
        lo, hi = bounds[c]
        ok &= (df[c].values >= lo) & (df[c].values <= hi)
    return ok


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


def parity_plot(df, pred, use, cls, misclass, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        yt_all, yp_all = df[TRUE_COL[t]].values, pred[:, i]
        base = use & np.isfinite(yp_all) & (yt_all > 0) & (yp_all > 0)
        # robust axis range (1st percentile .. max of true+pred), so a handful of
        # near-zero points don't stretch the view; all points still count in the stats.
        allv = np.concatenate([yt_all[base], yp_all[base]]); allv = allv[allv > 0]
        lo, hi = np.percentile(allv, 1) / 1.5, allv.max() * 1.5
        ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1.2, zorder=1)
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
        s = target_stats(yt_all[use], yp_all[use])
        ax.set_title(f"{t}", color=INK, fontsize=12)
        ax.text(0.04, 0.96, f"$R^2$={s['R2']:.3f}\n$R^2_{{log}}$={s['R2_log']:.3f}\n"
                f"med|rel|={100*s['MedRelErr']:.1f}%",
                transform=ax.transAxes, va="top", ha="left", fontsize=9, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=MUTED, alpha=0.85))
        ax.grid(True, which="major", color="#e6e6e6", lw=0.6, zorder=0)
        ax.tick_params(colors=INK)
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Inverse model (V2-trained) validated on fresh V1 data "
                 "(magnetic_materials.csv, 0 % overlap): Ms, A, K1 from Hc, Mr, (BH)max",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    df = load_mammos_csv(ROOT / "data" / "magnetic_materials.csv")
    Hc, Mr, BHmax = df["Hc"].values, df["Mr"].values, df["BHmax"].values
    pred, cls = predict_batch(Hc, Mr, BHmax)

    # exclude points whose extrinsic inputs fall outside the training volume (extrapolation)
    in_vol = in_training_volume(df, training_bounds())
    use = in_vol

    # Ground-truth hard/soft class from the TRUE Mr/Ms (threshold_clustering: >0.4 = hard),
    # vs the classifier's routing decision (cls) -> which points used the WRONG regressor.
    ratio = np.divide(df["Mr"].values, df["Ms"].values,
                      out=np.zeros(len(df)), where=df["Ms"].values != 0)
    true_cls = np.where(ratio > 0.4, "hard", "soft")
    misclass = cls != true_cls

    n = len(df)
    print(f"Fresh V1 points (magnetic_materials.csv): {n}")
    print(f"  outside V2 training volume (excluded): {int((~in_vol).sum())}")
    print(f"  used for validation                 : {int(use.sum())}")
    print(f"  class split (used): soft={int((use & (cls=='soft')).sum())}  "
          f"hard={int((use & (cls=='hard')).sum())}")

    rows = []
    for i, t in enumerate(TARGETS):
        s = {"target": f"{t} ({UNITS[t]})", **target_stats(df[TRUE_COL[t]].values[use], pred[use, i])}
        rows.append(s)
    tbl = pd.DataFrame(rows)[["target", "N", "R2", "R2_log", "MAE", "RMSE", "MedRelErr"]]
    tbl.to_csv(OUT / "stats.csv", index=False)
    print("\n=== per-target validation statistics (fresh, in-volume V1 data) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
        print(tbl.to_string(index=False))
    p = 100 * LABEL_REL_ERR
    print(f"\nNote: for the INVERSE model the ~{p:.0f}% simulation error is on the extrinsic INPUTS "
          f"(Hc, Mr, BHmax); the intrinsic targets (Ms, A, K1) are exact. It is therefore INPUT "
          f"noise (propagated by Monte-Carlo in load_onnx_models.calculate_intrinsic_properties), "
          f"not a floor on the truth. The errors above — especially A — are model/identifiability "
          f"error and dominate the ~{p:.0f}% input-noise contribution.")

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
    # isolate regressor quality from classifier error, for every target
    ok = use & ~misclass
    print("  per-target R^2: all used | correctly-routed only  (gap = classifier error):")
    for i, t in enumerate(TARGETS):
        r2_all = target_stats(df[TRUE_COL[t]].values[use], pred[use, i])["R2"]
        r2_ok = target_stats(df[TRUE_COL[t]].values[ok], pred[ok, i])["R2"]
        print(f"    {t:<3}: {r2_all:6.3f}  |  {r2_ok:6.3f}")

    parity_plot(df, pred, use, cls, misclass, OUT / "parity.png")


if __name__ == "__main__":
    main()
