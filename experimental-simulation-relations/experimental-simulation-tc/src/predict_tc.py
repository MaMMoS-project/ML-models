"""Predict the (corrected) experimental Curie temperature from a compound + its Tc_sim.

This is the sim->exp CORRECTION model: unlike a direct compound->Tc predictor, each ONNX
here takes the input vector

    X = [ matscholar200 compound embedding (200) | RE physics features (7)? | Tc_sim (1) ]

so you MUST supply BOTH the chemical formula AND its simulated Curie temperature Tc_sim.
The ONNX returns either Tc_exp directly, or the correction (Tc_exp - Tc_sim) for models whose
file name ends in `_delta` (trained with delta_learning); this script adds Tc_sim back in that
case. Whether the 7 RE features are needed is read from the file name (`_refeats`).

Only raw-200D embedding models are exported (see src/training/onnx_export.py), so every ONNX
here is servable directly from a formula.

Usage:
    python -m src.predict_tc --compound Nd2Fe14B --tc-sim 550
    python -m src.predict_tc --compound Fe3Pt --tc-sim 420 \
        --model results/onnx_models/RE-Free-Augm_combined_augmented_lgbm.onnx

With no --model, all RE / RE-free models matching the compound's chemistry are run and
tabulated (RE compounds -> RE-* models, else -> RE-Free-* models; All-* models always apply).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt
from pymatgen.core import Composition

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMB_FILE = PROJECT_ROOT / "data" / "embeddings" / "element" / "matscholar200.json"
ONNX_DIR = PROJECT_ROOT / "results" / "onnx_models"

# Import the SAME RE-feature module the trainer used, so a _refeats model gets an
# identical 7-D feature block in the identical order.
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
try:
    from src.re_features import compute_re_features, RE_FEATURE_NAMES
except Exception:  # pragma: no cover
    from re_features import compute_re_features, RE_FEATURE_NAMES

# Rare-earth elements for RE / RE-free routing (matches build_merged_tc.RARE_EARTHS).
RARE_EARTHS = {"La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y"}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def load_elem_features(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compound_embedding(formula: str, elem_features: dict) -> np.ndarray:
    """200-D matscholar200 embedding: composition-weighted average of element vectors."""
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    total = sum(amounts.values())
    dim = len(next(iter(elem_features.values())))
    vec = np.zeros(dim, dtype=np.float64)
    for el, amt in amounts.items():
        if el not in elem_features:
            raise ValueError(f"element {el!r} not in matscholar200 embedding")
        vec += (amt / total) * np.asarray(elem_features[el], dtype=np.float64)
    return vec


def re_feature_vector(formula: str) -> np.ndarray:
    """The 7 RE physics features in RE_FEATURE_NAMES order (zero for RE-free)."""
    feats = compute_re_features(formula)
    return np.array([feats[k] for k in RE_FEATURE_NAMES], dtype=np.float64)


def contains_re(formula: str) -> bool:
    try:
        return any(el.symbol in RARE_EARTHS for el in Composition(formula).elements)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Model discovery / routing
# ---------------------------------------------------------------------------
_DATASETS = ("RE-Free-Pairs", "RE-Free-Augm", "RE-Pairs", "RE-Augm", "All-Pairs", "All-Augm")


def _model_dataset(path: Path) -> "str | None":
    name = path.name
    for ds in _DATASETS:                       # longest (RE-Free-*) first
        if name.startswith(ds + "_") or name == ds + ".onnx":
            return ds
    return None


def discover_models(onnx_dir: Path, is_re: bool) -> "list[Path]":
    """RE compounds -> RE-* models; RE-free -> RE-Free-* models; All-* always included."""
    out = []
    for p in sorted(onnx_dir.glob("*.onnx")):
        ds = _model_dataset(p)
        if ds is None:
            continue
        if ds.startswith("All-"):
            out.append(p)
        elif ds.startswith("RE-Free-"):
            if not is_re:
                out.append(p)
        elif ds.startswith("RE-"):
            if is_re:
                out.append(p)
    return out


def predict_one(onnx_path: Path, emb: np.ndarray, tc_sim: float) -> float:
    """Run one ONNX model and return the reconstructed Tc_exp [K]."""
    name = onnx_path.name
    needs_refeats = "_refeats" in name
    is_delta = "_delta" in name

    parts = [emb]
    if needs_refeats:
        parts.append(_current_re_feats)   # set by predict() before the loop
    parts.append(np.array([tc_sim], dtype=np.float64))
    x = np.concatenate(parts).astype(np.float32)[None, :]

    sess = rt.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    exp_dim = sess.get_inputs()[0].shape[-1]
    if isinstance(exp_dim, int) and exp_dim != x.shape[1]:
        raise ValueError(f"{name}: expects input dim {exp_dim}, built {x.shape[1]} "
                         f"(refeats={needs_refeats})")
    out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0]).ravel()[0]
    return float(out) + (tc_sim if is_delta else 0.0)


_current_re_feats = None  # module-level scratch for the RE feature block of the query


# ---------------------------------------------------------------------------
def predict(compound: str, tc_sim: float, model: "Path | None") -> "list[tuple]":
    global _current_re_feats
    elem = load_elem_features(EMB_FILE)
    emb = compound_embedding(compound, elem)
    _current_re_feats = re_feature_vector(compound)
    is_re = contains_re(compound)

    if model is not None:
        models = [model]
    else:
        if not ONNX_DIR.exists():
            raise SystemExit(f"No ONNX dir at {ONNX_DIR}. Run the training pipeline first.")
        models = discover_models(ONNX_DIR, is_re)
        if not models:
            raise SystemExit(f"No matching ONNX models in {ONNX_DIR} for "
                             f"{'RE' if is_re else 'RE-free'} compound {compound!r}.")

    rows = []
    for p in models:
        try:
            tc = predict_one(p, emb, tc_sim)
            rows.append((p.name, f"{tc:8.1f}"))
        except Exception as e:
            rows.append((p.name, f"ERROR: {e}"))
    return rows, is_re


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compound", required=True, help="Chemical formula, e.g. Nd2Fe14B")
    ap.add_argument("--tc-sim", required=True, type=float, dest="tc_sim",
                    help="Simulated Curie temperature Tc_sim [K] for this compound (required).")
    ap.add_argument("--model", default=None, type=Path,
                    help="Path to a single .onnx model. Omit to run all chemistry-matching models.")
    args = ap.parse_args()

    rows, is_re = predict(args.compound, args.tc_sim, args.model)

    print(f"\nCompound : {args.compound}   ({'RE' if is_re else 'RE-free'})")
    print(f"Tc_sim   : {args.tc_sim:.1f} K   ->  predicted Tc_exp:")
    print(f"{'-'*72}")
    print(f"{'model (onnx)':<56} {'Tc_exp [K]':>12}")
    print(f"{'-'*72}")
    for name, val in rows:
        print(f"{name:<56} {val:>12}")
    print(f"{'-'*72}")
    print("Note: sim->exp correction models — predictions depend on the supplied Tc_sim.")


if __name__ == "__main__":
    main()
