import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent

VALIDATE_INPUTS_CLASSIFIER_MODEL = BASE_DIR / "../plots/supervised_valid-invalid-inputs_clustering_pipeline.onnx"
HARDSOFT_CLASSIFIER_MODEL = BASE_DIR / "../plots/supervised_hardsoft_clustering_pipeline.onnx"
MODELS = {
    "soft": BASE_DIR / "../results/models/LogTransformation_cluster0/random_forest.onnx",
    "hard": BASE_DIR / "../results/models/LogTransformation_cluster1/random_forest.onnx",
}

# Training data the models were fitted on (same file as the training config's input_file).
# Used to warn when a prediction request lies outside the training volume (extrapolation).
TRAINING_DATA = BASE_DIR / "../data/single_grain_cube_50nm_aligned.csv"

# Fallback (min, max) design ranges for [Ms (A/m), A (J/m), K (J/m^3)], used only if the
# training CSV cannot be read. Regenerate these if the training data changes.
_FALLBACK_BOUNDS = {"Ms": (7.96e4, 3.97e6), "A": (1.0e-13, 1.0e-11), "K": (1.0e4, 9.93e6)}
_BOUNDS_CACHE = None

# Known ~1% simulation error on the target values (Hc, Mr, BHmax). It is reported as an
# irreducible aleatoric band (<target>_lo / <target>_hi) around every prediction, independent
# of which model class produced the ONNX. Combine in quadrature with any model (epistemic)
# uncertainty if that is added later.
LABEL_REL_ERR = 0.01


def _add_uncertainty_band(result, targets):
    """Attach <t>_lo / <t>_hi = prediction * (1 -/+ LABEL_REL_ERR) for each target."""
    for t in targets:
        v = result.get(t)
        result[f"{t}_lo"] = None if v is None else v * (1.0 - LABEL_REL_ERR)
        result[f"{t}_hi"] = None if v is None else v * (1.0 + LABEL_REL_ERR)
    return result


def _training_bounds():
    """Per-feature (min, max) of Ms, A, K over the training data (cached).

    Reads the min/max directly from the training CSV so the bounds stay correct if the
    data is regenerated; falls back to the documented design ranges if it cannot be read.
    """
    global _BOUNDS_CACHE
    if _BOUNDS_CACHE is not None:
        return _BOUNDS_CACHE
    try:
        import pandas as pd
        lines = open(TRAINING_DATA).readlines()
        hdr = next(i for i, l in enumerate(lines) if l.startswith("Ms,"))
        df = pd.read_csv(TRAINING_DATA, skiprows=hdr)
        _BOUNDS_CACHE = {
            "Ms": (float(df["Ms"].min()), float(df["Ms"].max())),
            "A": (float(df["A"].min()), float(df["A"].max())),
            "K": (float(df["K1"].min()), float(df["K1"].max())),
        }
    except Exception as exc:
        warnings.warn(f"Could not read the training volume from {TRAINING_DATA} ({exc}); "
                      f"using fallback design ranges.")
        _BOUNDS_CACHE = dict(_FALLBACK_BOUNDS)
    return _BOUNDS_CACHE


def check_in_training_volume(Ms, A, K, warn=True):
    """Check whether each (Ms, A, K) input lies inside the training volume.

    Returns a boolean (array) that is True where all three inputs are within the training
    data's min/max box. Does NOT block prediction — when ``warn`` is True it emits a
    warning for the out-of-volume inputs, whose predictions are extrapolations and may be
    unreliable.
    """
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K)
    bounds = _training_bounds()
    in_range = np.ones(X.shape[0], dtype=bool)
    per_feat_out = {}
    for j, name in enumerate(("Ms", "A", "K")):
        lo, hi = bounds[name]
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

_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3


def _prepare_inputs(Ms, A, K):
    """Prepare inputs as an ONNX-compatible array of shape (n_samples, 3)."""
    Ms_arr = np.atleast_1d(Ms).astype(np.float32)
    A_arr = np.atleast_1d(A).astype(np.float32)
    K_arr = np.atleast_1d(K).astype(np.float32)

    if not (Ms_arr.shape == A_arr.shape == K_arr.shape):
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Got Ms: {Ms_arr.shape}, A: {A_arr.shape}, K: {K_arr.shape}"
        )

    original_shape = Ms_arr.shape
    is_scalar = np.isscalar(Ms) and np.isscalar(A) and np.isscalar(K)

    X = np.column_stack([Ms_arr.ravel(), A_arr.ravel(), K_arr.ravel()]).astype(np.float32)
    return X, original_shape, is_scalar


def validate_input(Ms, A, K):
    """Classify the input into valid and invalid sets"""
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K)

    session = ort.InferenceSession(str(VALIDATE_INPUTS_CLASSIFIER_MODEL), _SESSION_OPTIONS)
    results = session.run(None, {session.get_inputs()[0].name: X})[0]
    labels = np.where(results == 0, "invalid", "valid")

    if is_scalar:
        return labels.item()
    return labels.reshape(original_shape)

def classify_magnetic_material(Ms, A, K):
    """Classify material as 'soft' or 'hard'."""
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K)

    session = ort.InferenceSession(str(HARDSOFT_CLASSIFIER_MODEL), _SESSION_OPTIONS)
    results = session.run(None, {session.get_inputs()[0].name: X})[0]

    labels = np.where(results == 0, "soft", "hard")

    if is_scalar:
        return labels.item()
    return labels.reshape(original_shape)


def calculate_extrinsic_properties(Ms, A, K):
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K)

    # Warn (but do not block) if the request lies outside the training volume.
    check_in_training_volume(Ms, A, K, warn=True)

    # 0. Determine whether input is valid
    print("Validating the input..\n")
    valid_input = validate_input(Ms, A, K)

    if (valid_input == "valid"):
      print("Input is valid. Starting predicitions...\n")
      # 1. Determine class
      mat_class = classify_magnetic_material(Ms, A, K)
      classes = np.atleast_1d(mat_class).ravel()

      # 2. Preprocess
      X_log = np.log1p(X)

      # 3. Predict using the correct model for each class
      y_log = np.empty((X_log.shape[0], 3), dtype=np.float32)

      for cls in ["soft", "hard"]:
          mask = classes == cls
          if np.any(mask):
              session = ort.InferenceSession(str(MODELS[cls]), _SESSION_OPTIONS)
              X_subset = X_log[mask]
              y_log[mask] = session.run(None, {session.get_inputs()[0].name: X_subset})[0]

      # 4. Postprocess
      y = np.expm1(y_log)

      if is_scalar:
          return _add_uncertainty_band({
              "Hc": y[0, 0],
              "Mr": y[0, 1],
              "BHmax": y[0, 2],
              "class": mat_class,
          }, ("Hc", "Mr", "BHmax"))

      return _add_uncertainty_band({
          "Hc": y[:, 0].reshape(original_shape),
          "Mr": y[:, 1].reshape(original_shape),
          "BHmax": y[:, 2].reshape(original_shape),
          "class": np.asarray(mat_class).reshape(original_shape),
      }, ("Hc", "Mr", "BHmax"))
    else:
      print("The input does not produce valid results. Returning None\n")
      return _add_uncertainty_band({
          "Hc": None,
          "Mr": None,
          "BHmax": None,
          "class": None,
      }, ("Hc", "Mr", "BHmax"))



if __name__ == "__main__":
    # Example: Hard Magnet
    Ms, A, K = 1.0e6, 1.0e-11, 4.5e6
    print(f"Input: Ms={Ms:.1e}, A={A:.1e}, K={K:.1e}")

    try:
        res = calculate_extrinsic_properties(Ms, A, K)
        print(
            f"Result: Class={res['class']}, "
            f"Hc={res['Hc']:.2e}, Mr={res['Mr']:.2e}, BHmax={res['BHmax']:.2e}"
        )
    except Exception as e:
        print(f"Error: {e}")
