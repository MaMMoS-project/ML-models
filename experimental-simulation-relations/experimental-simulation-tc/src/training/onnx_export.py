"""ONNX export for the Curie-temperature correction models.

This project's models take the input vector

    X = [ compound_embedding(200) | RE-features(7)?  | Tc_sim(1) ]

with Tc_sim ALWAYS the last column, and predict either Tc_exp directly or the
correction delta = Tc_exp - Tc_sim (when delta_learning is on). Each exported ONNX
therefore accepts that full input vector and returns the model's raw output; the
predict script (src/predict_tc.py) adds Tc_sim back when the file is a `_delta` model.

Only the RAW-200D embedding variant is exported, because:
  * the PCA-compressed variants use an OFFLINE PCA (compress_embedding_PCA.py) whose
    fitted object is NOT persisted, so a PCA model could not be served from a formula;
  * the no-embedding models take only Tc_sim and ignore the composition.
So raw_200D is the only variant a formula-based predict script can actually serve.

Model families exported: Linear, RandomForest, LightGBM, MLP. Symbolic Regression is
NOT ONNX-exportable (it is a symbolic expression, not a tensor graph) and is skipped.

Export never breaks training: every call is wrapped by the trainer in try/except, and
missing optional deps (skl2onnx / onnxmltools) simply skip the export with a message.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

# --- optional sklearn->ONNX stack -----------------------------------------
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    from sklearn.pipeline import Pipeline
    _SKL2ONNX = True
except Exception:  # pragma: no cover
    _SKL2ONNX = False

# skl2onnx has no native LightGBM converter; register onnxmltools' one so a Pipeline
# containing an LGBMRegressor converts too. If onnxmltools is absent, LightGBM export
# is skipped (training still works).
_LGBM_ONNX = False
if _SKL2ONNX:
    try:
        from skl2onnx import update_registered_converter
        from skl2onnx.common.shape_calculator import (
            calculate_linear_regressor_output_shapes,
        )
        from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
            convert_lightgbm,
        )
        from lightgbm import LGBMRegressor
        update_registered_converter(
            LGBMRegressor, "LightGbmLGBMRegressor",
            calculate_linear_regressor_output_shapes, convert_lightgbm,
        )
        _LGBM_ONNX = True
    except Exception:  # pragma: no cover
        _LGBM_ONNX = False

# The registered LightGBM converter emits ai.onnx.ml v5, which skl2onnx 1.20 does not
# support, so cap it at v3 (verified to round-trip exactly).
_ONNX_TARGET_OPSET = {"": 17, "ai.onnx.ml": 3}

ONNX_DIRNAME = "onnx_models"


# ---------------------------------------------------------------------------
# sklearn (Linear / RandomForest / LightGBM)
# ---------------------------------------------------------------------------
def export_sklearn_onnx(model, scaler, input_dim: int, out_path: Path) -> None:
    """Export a sklearn/LightGBM model (optionally preceded by a StandardScaler) to ONNX.

    The ONNX accepts the full ``input_dim``-D input vector and applies the scaler (if
    given) internally before the model, so callers feed the raw [emb|re?|Tc_sim] vector.
    """
    if not _SKL2ONNX:
        raise RuntimeError("skl2onnx not installed")
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    steps.append(("model", model))
    pipe = Pipeline(steps)
    initial_type = [("X", FloatTensorType([None, input_dim]))]
    onnx_proto = convert_sklearn(pipe, initial_types=initial_type,
                                 target_opset=_ONNX_TARGET_OPSET)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onnx_proto.SerializeToString())


# ---------------------------------------------------------------------------
# MLP (torch.onnx) — bundle the StandardScaler into the graph
# ---------------------------------------------------------------------------
def export_mlp_onnx(mlp, scaler, input_dim: int, out_path: Path) -> None:
    """Export the trained MLP (with its StandardScaler) to ONNX via torch.onnx."""
    import torch
    import torch.nn as nn

    class _MLPWithScaler(nn.Module):
        def __init__(self, scaler, mlp):
            super().__init__()
            self.register_buffer("scaler_mean",
                                 torch.tensor(scaler.mean_, dtype=torch.float32))
            self.register_buffer("scaler_scale",
                                 torch.tensor(scaler.scale_, dtype=torch.float32))
            # this project's MLP stores its layers in `.network`
            self.net = mlp.network

        def forward(self, x):
            x = (x - self.scaler_mean) / self.scaler_scale
            return self.net(x).view(-1)

    full = _MLPWithScaler(scaler, mlp).cpu().eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        full, dummy, str(out_path), dynamo=False,
        input_names=["X"], output_names=["Tc_pred"],
        dynamic_axes={"X": {0: "batch_size"}, "Tc_pred": {0: "batch_size"}},
        opset_version=17,
    )


# ---------------------------------------------------------------------------
# Orchestrator called from each trainer
# ---------------------------------------------------------------------------
def _results_root(output_dir) -> Path:
    """The project's results/ dir, found by walking up from a trainer output_dir."""
    p = Path(output_dir)
    for cand in [p, *p.parents]:
        if cand.name == "results":
            return cand
    return p.parent  # fallback: parent of the trainer's own output dir


def maybe_export_onnx(*, family: str, model, scaler, input_dim: int,
                      dataset_name: str, use_embedding: bool,
                      embedding_type: Optional[str], loader,
                      aug_label: Optional[str], output_dir) -> Optional[Path]:
    """Export one model to ONNX iff it is the deployable raw-200D embedding variant.

    Gating: only ``use_embedding and embedding_type is None`` (raw 200-D) is exported;
    PCA and no-embedding variants are skipped (see module docstring). Symbolic
    Regression must not reach here.

    File name:  <dataset>[_<aug>]_<family>[_refeats][_delta].onnx  under results/onnx_models/
    The `_delta` marker tells predict_tc to add Tc_sim back to the model output.
    Returns the written path, or None if skipped.
    """
    if not (use_embedding and embedding_type is None):
        return None  # only raw_200D embedding models are servable from a formula
    if family == "lgbm" and not _LGBM_ONNX:
        print("    ONNX export skipped (LightGBM needs onnxmltools)")
        return None

    delta = bool(getattr(loader, "delta_learning", False))
    refeats = bool(getattr(loader, "use_re_features", False))

    parts = [dataset_name]
    if aug_label:
        parts.append(str(aug_label))
    parts.append(family)
    if refeats:
        parts.append("refeats")
    if delta:
        parts.append("delta")
    fname = "_".join(parts) + ".onnx"

    onnx_dir = _results_root(output_dir) / ONNX_DIRNAME
    out_path = onnx_dir / fname

    if family == "mlp":
        export_mlp_onnx(model, scaler, input_dim, out_path)
    else:
        export_sklearn_onnx(model, scaler, input_dim, out_path)
    print(f"    ONNX -> {out_path.name}")
    return out_path
