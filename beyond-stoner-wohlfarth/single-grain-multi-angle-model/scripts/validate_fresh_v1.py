#!/usr/bin/env python3
"""
Validate the multi-angle (angle-dependent) forward BEST model on FRESH data.

The model is trained on the V2 angle-dependent dataset
(data/processed/micromagnetics_angle_dependent_symmetries.csv). This script uses the
older V1 file data/magnetic_materials.csv (~1,497 points) as an external validation
set: the two share 0 % of their (Ms, A, K1) points, so V1 is genuinely held out.

IMPORTANT — V1 covers only the (nearly) aligned geometry: the field is a constant
~1.025 deg (0.0179 rad) off the easy axis, so this validates ONLY the small-angle
slice of the multi-angle model, not its full angular range. The relative angle is
derived exactly from the data (arccos(|u_K . u_H|), the same definition used in
training) rather than hard-coded to 0.

Unlike the easy-axis / inverse models there is no hard/soft classifier (single unified
regressor, clustering.method: none), so there is no routing analysis — just per-target
statistics and a parity plot.

Outputs (in ./validation_v1/):
    parity.png   predicted vs. true Hc / Mr / (BH)max
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
# Descriptive colour by the true hard/soft class (Mr/Ms>0.4); the model does NOT route,
# this is only to see whether one regime predicts worse. Okabe-Ito colourblind-safe.
CLASS_STYLE = {"soft": ("#E69F00", "^"), "hard": ("#0072B2", "o")}
INK, MUTED = "#222222", "#888888"


def load_mammos_csv(path):
    """Read a MaMMoS-format CSV (skip the '#'-prefixed metadata header)."""
    lines = open(path).readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Ms"))
    return pd.read_csv(path, skiprows=hdr)


def _sph_to_cart(theta, phi):
    st, ct = np.sin(theta), np.cos(theta)
    return np.array([st * np.cos(phi), st * np.sin(phi), ct])


def relative_angle(df):
    """Relative angle (rad) between the field H and the K1 axis, exactly as computed in
    training (make_reduced_dataset): arccos(|u_K . u_H|), unsigned, in [0, pi/2].
    V1 stores k1/h directions as spherical angles in radians."""
    uk = _sph_to_cart(df["k1_theta"].to_numpy(float), df["k1_phi"].to_numpy(float))
    uh = _sph_to_cart(df["h_theta"].to_numpy(float), df["h_phi"].to_numpy(float))
    dot = (uk * uh).sum(0)
    return np.arccos(np.clip(np.abs(dot), 0.0, 1.0))


def predict_batch(Ms, A, K, angle):
    """Vectorised form of load_onnx_models.calculate_extrinsic_properties.
    Returns pred[N,3] (Hc, Mr, BHmax)."""
    Ms, A, K, angle = (np.asarray(v, float) for v in (Ms, A, K, angle))
    X = np.column_stack([Ms, A, K, angle]).astype(np.float32)
    Xp = X.copy()
    Xp[:, lom._LOG_MASK] = np.log1p(Xp[:, lom._LOG_MASK])   # log1p on Ms/A/K, not angle
    sess = ort.InferenceSession(str(lom.MODEL), lom._SESSION_OPTIONS)
    ylog = sess.run(None, {sess.get_inputs()[0].name: Xp})[0]
    return np.expm1(ylog)                                   # inverse log transform


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


def parity_plot(df, pred, use, cls, ang_deg, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    for i, (t, ax) in enumerate(zip(TARGETS, axes)):
        yt_all, yp_all = df[t].values, pred[:, i]
        base = use & np.isfinite(yp_all) & (yt_all > 0) & (yp_all > 0)
        allv = np.concatenate([yt_all[base], yp_all[base]]); allv = allv[allv > 0]
        lo, hi = np.percentile(allv, 1) / 1.5, allv.max() * 1.5
        ax.plot([lo, hi], [lo, hi], "--", color=MUTED, lw=1.2, zorder=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        for c in ("hard", "soft"):
            m = base & (cls == c)
            color, marker = CLASS_STYLE[c]
            ax.scatter(yt_all[m], yp_all[m], s=14, c=color, marker=marker,
                       alpha=0.5, linewidths=0, label=c, zorder=2)
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
    axes[0].legend(frameon=False, loc="lower right", fontsize=8, title="true class")
    fig.suptitle(f"Multi-angle model (V2-trained) validated on fresh V1 data "
                 f"(magnetic_materials.csv, 0 % overlap); aligned slice, "
                 f"relative angle = {ang_deg:.2f}°", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"  wrote {path}")


def main():
    df = load_mammos_csv(ROOT / "data" / "magnetic_materials.csv")
    Ms, A, K = df["Ms"].values, df["A"].values, df["K1"].values
    ang = relative_angle(df)                      # exact, derived from the data (radians)
    ang_deg = float(np.degrees(np.median(ang)))
    pred = predict_batch(Ms, A, K, ang)

    # exclude points outside the training volume (extrapolation)
    in_vol = np.asarray(lom.check_in_training_volume(Ms, A, K, ang, warn=False)).ravel()
    use = in_vol

    # descriptive true hard/soft class from Mr/Ms (>0.4 = hard); the model does not route.
    ratio = np.divide(df["Mr"].values, df["Ms"].values,
                      out=np.zeros(len(df)), where=df["Ms"].values != 0)
    cls = np.where(ratio > 0.4, "hard", "soft")

    n = len(df)
    print(f"Fresh V1 points (magnetic_materials.csv): {n}")
    print(f"  relative angle (derived)            : {ang_deg:.3f}° "
          f"(min {np.degrees(ang.min()):.3f}°, max {np.degrees(ang.max()):.3f}°) — aligned slice")
    print(f"  outside V2 training volume (excluded): {int((~in_vol).sum())}")
    print(f"  used for validation                 : {int(use.sum())}")
    print(f"  true class split (used): soft={int((use & (cls=='soft')).sum())}  "
          f"hard={int((use & (cls=='hard')).sum())}")

    rows = []
    for i, t in enumerate(TARGETS):
        s = {"target": f"{t} ({UNITS[t]})", **target_stats(df[t].values[use], pred[use, i])}
        rows.append(s)
    tbl = pd.DataFrame(rows)[["target", "N", "R2", "R2_log", "MAE", "RMSE", "MedRelErr"]]
    tbl.to_csv(OUT / "stats.csv", index=False)
    print("\n=== per-target validation statistics (fresh, in-volume V1 data; aligned slice) ===")
    with pd.option_context("display.float_format", lambda x: f"{x:.4g}"):
        print(tbl.to_string(index=False))

    parity_plot(df, pred, use, cls, ang_deg, OUT / "parity.png")


if __name__ == "__main__":
    main()
