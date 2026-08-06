from pathlib import Path

import numpy as np
import onnxruntime as ort

BASE_DIR = Path(__file__).resolve().parent

HARDSOFT_CLASSIFIER_MODEL = BASE_DIR / "../plots/inverse_supervised_hardsoft_clustering_pipeline.onnx"
MODELS = {
    "soft": next((BASE_DIR / "../results/best_model_cluster0").glob("*.onnx")),
    "hard": next((BASE_DIR / "../results/best_model_cluster1").glob("*.onnx")),
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


# NOTE ON UNCERTAINTY (this is the INVERSE model): the ~1% simulation error is on the
# EXTRINSIC quantities Hc, Mr, (BH)max, which here are the *inputs* — the intrinsic targets
# (Ms, A, K1) are exact. So the 1% is INPUT noise, not label noise: it is propagated through
# the pipeline by Monte-Carlo, NOT applied as a flat output band (its effect on each output
# depends on the local sensitivity of the inverse map — e.g. it is enormous for the
# ill-posed exchange stiffness A).
LABEL_REL_ERR = 0.01


def _predict_raw(X):
    """classify -> per-class regressor -> expm1 on a raw (N,3) [Hc,Mr,BHmax] array.
    Returns (y[N,3] = Ms,A,K ; classes[N])."""
    Xf = np.asarray(X, np.float32)
    cs = ort.InferenceSession(str(HARDSOFT_CLASSIFIER_MODEL), _SESSION_OPTIONS)
    classes = np.where(cs.run(None, {cs.get_inputs()[0].name: Xf})[0] == 0, "soft", "hard").ravel()
    X_log = np.log1p(Xf)
    y_log = np.empty((Xf.shape[0], 3), dtype=np.float32)
    for cls in ("soft", "hard"):
        m = classes == cls
        if m.any():
            s = ort.InferenceSession(str(MODELS[cls]), _SESSION_OPTIONS)
            y_log[m] = s.run(None, {s.get_inputs()[0].name: X_log[m]})[0]
    return np.expm1(y_log), classes


def calculate_intrinsic_properties(Hc, Mr, BHmax, propagate_input_noise=True, n_mc=64, seed=0):
    """Predict intrinsic properties (Ms, A, K) from extrinsic properties (Hc, Mr, BHmax).

    The regressors expect log-transformed inputs and produce log-transformed outputs, matching
    the LogTransformation preprocessing used during training.

    With ``propagate_input_noise`` (default True), the ~1% simulation error on the extrinsic
    INPUTS is propagated to the outputs by Monte-Carlo (``n_mc`` draws of 1% multiplicative
    Gaussian noise), and ``<t>_lo`` / ``<t>_hi`` report the 16th/84th percentiles (≈ ±1σ).

    Returns
    -------
    dict with keys 'Ms', 'A', 'K' (+ '<t>_lo'/'<t>_hi') and 'class' ('soft'/'hard').
    """
    X, original_shape, is_scalar = _prepare_inputs(Hc, Mr, BHmax)
    y, classes = _predict_raw(X)                         # point prediction from the given inputs
    N = X.shape[0]

    lo = hi = None
    if propagate_input_noise and n_mc and n_mc > 1:
        rng = np.random.default_rng(seed)
        noise = rng.normal(1.0, LABEL_REL_ERR, size=(N, n_mc, 3)).astype(np.float32)
        Xmc = (X[:, None, :] * noise).reshape(N * n_mc, 3)
        ymc, _ = _predict_raw(Xmc)
        ymc = ymc.reshape(N, n_mc, 3)
        lo = np.percentile(ymc, 16, axis=1)              # ≈ -1σ from 1% input noise
        hi = np.percentile(ymc, 84, axis=1)              # ≈ +1σ

    def shape(a):
        if a is None:
            return None
        return float(a[0]) if is_scalar else a.reshape(original_shape)

    result = {}
    for i, nm in enumerate(("Ms", "A", "K")):
        result[nm] = shape(y[:, i])
        result[f"{nm}_lo"] = shape(None if lo is None else lo[:, i])
        result[f"{nm}_hi"] = shape(None if hi is None else hi[:, i])
    result["class"] = classes[0] if is_scalar else np.asarray(classes).reshape(original_shape)
    return result


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
