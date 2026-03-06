from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent

CLASSIFIER_MODEL = BASE_DIR / "../plots/supervised_clustering_pipeline.onnx"
MODELS = {
    "soft": BASE_DIR / "../results/models/LogTransformation_cluster0/random_forest.onnx",
    "hard": BASE_DIR / "../results/models/LogTransformation_cluster1/random_forest.onnx",
}

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


def classify_magnetic_material(Ms, A, K):
    """Classify material as 'soft' or 'hard'."""
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K)

    session = ort.InferenceSession(str(CLASSIFIER_MODEL), _SESSION_OPTIONS)
    results = session.run(None, {session.get_inputs()[0].name: X})[0]

    labels = np.where(results == 0, "soft", "hard")

    if is_scalar:
        return labels.item()
    return labels.reshape(original_shape)


def calculate_extrinsic_properties(Ms, A, K):
    X, original_shape, is_scalar = _prepare_inputs(Ms, A, K)

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
        return {
            "Hc": y[0, 0],
            "Mr": y[0, 1],
            "BHmax": y[0, 2],
            "class": mat_class,
        }

    return {
        "Hc": y[:, 0].reshape(original_shape),
        "Mr": y[:, 1].reshape(original_shape),
        "BHmax": y[:, 2].reshape(original_shape),
        "class": np.asarray(mat_class).reshape(original_shape),
    }


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
