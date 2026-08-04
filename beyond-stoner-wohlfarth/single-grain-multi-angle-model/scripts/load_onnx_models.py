"""Inference for the single-grain multi-angle (angle-dependent) forward surrogate.

Unlike the easy-axis model, this model is NOT split into hard/soft clusters: a single
random-forest regressor is trained on the whole dataset (config ``clustering.method: none``,
dataset ``_all``). The extra input is the relative angle between the external field H and the
uniaxial anisotropy axis K1.

Prediction pipeline:
    inputs [Ms, A, K1, relative_angle]  (relative_angle in RADIANS, unsigned axis-field
                                         angle arccos(|u_K . u_H|), in [0, pi/2])
      -> log1p on Ms, A, K1  (relative_angle is left untransformed; see log_exclude_cols)
      -> ONNX random forest  (the StandardScaler is baked into the ONNX graph)
      -> expm1 on the outputs
      -> [Hc, Mr, (BH)max]

The feature order MUST match the training config's
``data.input_columns = ['Ms (A/m)', 'A (J/m)', 'K (J/m^3)', 'relative_angle']``.
"""
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent

# Single unified regressor (no soft/hard split). This file is produced by
# `python3 -m scripts.train_model` once the scaler is baked into the ONNX pipeline.
MODEL = BASE_DIR / "../results/models/LogTransformation_all/random_forest.onnx"

# Processed training data the model was fitted on (same file as the training config's
# input_file). Used only to warn when a request lies outside the training volume.
TRAINING_DATA = BASE_DIR / "../data/processed/micromagnetics_angle_dependent_symmetries.csv"

# Training-data column names for the four inputs, in model order.
_INPUT_COLS = ["Ms (A/m)", "A (J/m)", "K (J/m^3)", "relative_angle"]
# Which inputs are log1p-transformed before the model (relative_angle is NOT; it matches
# the training config's log_exclude_cols=['relative_angle']).
_LOG_MASK = np.array([True, True, True, False])

_BOUNDS_CACHE = None

_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3


def _read_training_df():
    """Read the processed MaMMoS CSV (skip the '#'-prefixed metadata header)."""
    import pandas as pd
    lines = open(TRAINING_DATA).readlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Ms"))
    return pd.read_csv(TRAINING_DATA, skiprows=hdr)


def _training_bounds():
    """Per-feature (min, max) over the training data (cached), in model input order.

    Returns None if the training CSV cannot be read (then the volume check is skipped).
    """
    global _BOUNDS_CACHE
    if _BOUNDS_CACHE is not None:
        return _BOUNDS_CACHE
    try:
        df = _read_training_df()
        _BOUNDS_CACHE = [(float(df[c].min()), float(df[c].max())) for c in _INPUT_COLS]
    except Exception as exc:
        warnings.warn(f"Could not read the training volume from {TRAINING_DATA} ({exc}); "
                      f"the training-volume check will be skipped.")
        _BOUNDS_CACHE = None
    return _BOUNDS_CACHE


def _prepare_inputs(Ms, A, K, angle):
    """Prepare inputs as an ONNX-compatible array of shape (n_samples, 4)."""
    Ms_arr = np.atleast_1d(Ms).astype(np.float32)
    A_arr = np.atleast_1d(A).astype(np.float32)
    K_arr = np.atleast_1d(K).astype(np.float32)
    ang_arr = np.atleast_1d(angle).astype(np.float32)

    if not (Ms_arr.shape == A_arr.shape == K_arr.shape == ang_arr.shape):
        raise ValueError(
            f"Input arrays must have the same shape. Got Ms: {Ms_arr.shape}, "
            f"A: {A_arr.shape}, K: {K_arr.shape}, angle: {ang_arr.shape}"
        )

    original_shape = Ms_arr.shape
    is_scalar = all(np.isscalar(v) for v in (Ms, A, K, angle))

    X = np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K_arr.ravel(),
                         ang_arr.ravel()]).astype(np.float32)
    return X, original_shape, is_scalar


def check_in_training_volume(Ms, A, K, angle, warn=True):
    """Check whether each (Ms, A, K, relative_angle) input lies inside the training volume.

    Returns a boolean (array) that is True where all four inputs are within the training
    data's min/max box. Does NOT block prediction — when ``warn`` is True it emits a warning
    for out-of-volume inputs, whose predictions are extrapolations and may be unreliable.
    """
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K, angle)
    bounds = _training_bounds()
    if bounds is None:                       # could not read training data; do not block
        return True if is_scalar else np.ones(original_shape, dtype=bool)

    in_range = np.ones(X.shape[0], dtype=bool)
    per_feat_out = {}
    for j, name in enumerate(_INPUT_COLS):
        lo, hi = bounds[j]
        out = (X[:, j] < lo) | (X[:, j] > hi)
        in_range &= ~out
        if out.any():
            per_feat_out[name] = int(out.sum())
    n_out = int((~in_range).sum())
    if warn and n_out:
        detail = ", ".join(f"{k}={v}" for k, v in per_feat_out.items())
        warnings.warn(
            f"{n_out} of {X.shape[0]} input(s) fall OUTSIDE the training volume "
            f"(out-of-range counts per feature: {detail}). These predictions are "
            f"extrapolations beyond the fitted data and may be unreliable.",
            stacklevel=2,
        )
    if is_scalar:
        return bool(in_range.item())
    return in_range.reshape(original_shape)


def calculate_extrinsic_properties(Ms, A, K, angle):
    """Predict Hc, Mr and (BH)max for the angle-dependent single-grain model.

    Parameters
    ----------
    Ms, A, K : float or array
        Spontaneous magnetisation [A/m], exchange stiffness [J/m], uniaxial anisotropy
        constant K1 [J/m^3].
    angle : float or array
        Relative angle between H and the K1 axis, in RADIANS. This is the unsigned
        axis-field angle arccos(|u_K . u_H|), so the trained range is [0, pi/2]
        (0 = along the easy axis, pi/2 = perpendicular); values > pi/2 are extrapolation.
    """
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K, angle)

    # Warn (but do not block) if the request lies outside the training volume.
    check_in_training_volume(Ms, A, K, angle, warn=True)

    # Preprocess: log1p on Ms/A/K only; relative_angle is left as-is.
    X_proc = X.copy()
    X_proc[:, _LOG_MASK] = np.log1p(X_proc[:, _LOG_MASK])

    # Predict (the StandardScaler is baked into the ONNX pipeline).
    session = ort.InferenceSession(str(MODEL), _SESSION_OPTIONS)
    y_log = session.run(None, {session.get_inputs()[0].name: X_proc})[0]

    # Postprocess: invert the log1p transform on the targets.
    y = np.expm1(y_log)

    if is_scalar:
        return {"Hc": y[0, 0], "Mr": y[0, 1], "BHmax": y[0, 2]}
    return {
        "Hc": y[:, 0].reshape(original_shape),
        "Mr": y[:, 1].reshape(original_shape),
        "BHmax": y[:, 2].reshape(original_shape),
    }


if __name__ == "__main__":
    import math
    # Example: field applied along the easy axis (relative_angle = 0).
    Ms, A, K, angle = 1.0e6, 1.0e-11, 4.5e6, 0.0
    print(f"Input: Ms={Ms:.1e}, A={A:.1e}, K={K:.1e}, angle={angle:.3f} rad")
    try:
        res = calculate_extrinsic_properties(Ms, A, K, angle)
        print(f"Result: Hc={res['Hc']:.2e}, Mr={res['Mr']:.2e}, BHmax={res['BHmax']:.2e}")

        # Example: field 45 degrees off the easy axis.
        angle = math.radians(45)
        res = calculate_extrinsic_properties(Ms, A, K, angle)
        print(f"Input: angle={angle:.3f} rad -> "
              f"Hc={res['Hc']:.2e}, Mr={res['Mr']:.2e}, BHmax={res['BHmax']:.2e}")
    except Exception as e:
        print(f"Error: {e}")
