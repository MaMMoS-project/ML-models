from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent

HARDSOFT_CLASSIFIER_MODEL = BASE_DIR / "../plots/supervised_hardsoft_clustering_pipeline_inverted.onnx"
MODELS = {
    "soft": BASE_DIR / "../results/models/LogTransformation_cluster0/random_forest.onnx",
    "hard": BASE_DIR / "../results/models/LogTransformation_cluster1/random_forest.onnx",
}

_SESSION_OPTIONS = ort.SessionOptions()
_SESSION_OPTIONS.log_severity_level = 3


def _prepare_inputs(Hc, Mr, BHmax):
    """Prepare inputs as an ONNX-compatible array of shape (n_samples, 3)."""
    Hc_arr = np.atleast_1d(Hc).astype(np.float32)
    Mr_arr = np.atleast_1d(Mr).astype(np.float32)
    BHmax_arr = np.atleast_1d(BHmax).astype(np.float32)

    if not (Hc_arr.shape == Mr_arr.shape == BHmax_arr.shape):
        raise ValueError(
            f"Input arrays must have the same shape. "
            f"Got Hc: {Hc_arr.shape}, Mr: {Mr_arr.shape}, BHmax: {BHmax_arr.shape}"
        )

    original_shape = Hc_arr.shape
    is_scalar = np.isscalar(Hc) and np.isscalar(Mr) and np.isscalar(BHmax)

    X = np.column_stack([Hc_arr.ravel(), Mr_arr.ravel(), BHmax_arr.ravel()]).astype(np.float32)
    return X, original_shape, is_scalar


def classify_magnetic_material(Hc, Mr, BHmax):
    """Classify material as 'soft' or 'hard' from extrinsic properties."""
    X, original_shape, is_scalar = _prepare_inputs(Hc, Mr, BHmax)

    session = ort.InferenceSession(str(HARDSOFT_CLASSIFIER_MODEL), _SESSION_OPTIONS)
    results = session.run(None, {session.get_inputs()[0].name: X})[0]

    labels = np.where(results == 0, "soft", "hard")

    if is_scalar:
        return labels.item()
    return labels.reshape(original_shape)


def calculate_intrinsic_properties(Hc, Mr, BHmax):
    """Predict intrinsic magnetic properties (Ms, A, K) from extrinsic properties (Hc, Mr, BHmax).

    The regression models expect log-transformed inputs and produce log-transformed outputs,
    matching the LogTransformation preprocessing used during training.

    Returns
    -------
    dict with keys 'Ms' (A/m), 'A' (J/m), 'K' (J/m^3), 'class' ('soft' or 'hard').
    """
    X, original_shape, is_scalar = _prepare_inputs(Hc, Mr, BHmax)

    # 1. Determine class
    mat_class = classify_magnetic_material(Hc, Mr, BHmax)
    classes = np.atleast_1d(mat_class).ravel()

    # 2. Preprocess: log1p matches the LogTransformation applied during training
    X_log = np.log1p(X)

    # 3. Predict using the correct model for each class
    y_log = np.empty((X_log.shape[0], 3), dtype=np.float32)

    for cls in ["soft", "hard"]:
        mask = classes == cls
        if np.any(mask):
            session = ort.InferenceSession(str(MODELS[cls]), _SESSION_OPTIONS)
            X_subset = X_log[mask]
            y_log[mask] = session.run(None, {session.get_inputs()[0].name: X_subset})[0]

    # 4. Postprocess: invert log1p
    y = np.expm1(y_log)

    if is_scalar:
        return {
            "Ms": y[0, 0],
            "A": y[0, 1],
            "K": y[0, 2],
            "class": mat_class,
        }

    return {
        "Ms": y[:, 0].reshape(original_shape),
        "A": y[:, 1].reshape(original_shape),
        "K": y[:, 2].reshape(original_shape),
        "class": np.asarray(mat_class).reshape(original_shape),
    }


if __name__ == "__main__":
    # Example: Hard Magnet (typical values for NdFeB-like material)
    Hc, Mr, BHmax = 1.5e6, 8e5, 3e5
    print(f"Input: Hc={Hc:.1e} A/m, Mr={Mr:.1e} A/m, BHmax={BHmax:.1e} J/m^3")

    try:
        res = calculate_intrinsic_properties(Hc, Mr, BHmax)
        print(
            f"Result: Class={res['class']}, "
            f"Ms={res['Ms']:.2e} A/m, A={res['A']:.2e} J/m, K={res['K']:.2e} J/m^3"
        )
    except Exception as e:
        print(f"Error: {e}")
