#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict Curie temperature (Tc) for a new compound using a trained ONNX model.

Each ONNX model stored under results/onnx_models/ encodes the full inference
pipeline: raw 200-D matscholar200 compound embedding → (optional PCA) →
(optional scaler) → regression model → Tc in Kelvin.  The user therefore only
needs to supply a chemical formula; no manual preprocessing is required.

When using --all the script automatically detects whether a compound contains
rare-earth elements and restricts predictions to the matching dataset models
(RE or RE-Free) plus the combined All models.

Usage examples
--------------
# Predict with a specific model
python src/predict_tc.py --compound Fe3Pt --model results/onnx_models/All_pca_16_rf.onnx

# Run all available ONNX models and print a comparison table
python src/predict_tc.py --compound Fe3Pt --all

# Predict using only the best model for the detected compound type (RE or RE-Free)
python src/predict_tc.py --compound Fe3Pt --best
python src/predict_tc.py --compound Nd2Fe14B --best

# Predict a list of compounds (one formula per line)
python src/predict_tc.py --compounds-file compounds.txt --all

# List available models
python src/predict_tc.py --list
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import onnxruntime as rt
from pymatgen.core import Composition

PROJECT_ROOT = Path(__file__).parent.parent
EMB_FILE     = PROJECT_ROOT / "data" / "embeddings" / "element" / "matscholar200.json"
ONNX_DIR     = PROJECT_ROOT / "results" / "onnx_models"
BEST_CSV     = PROJECT_ROOT / "results" / "exp_tc_best_by_dataset.csv"

# RE physics features — needed for models trained with re_features:true, whose ONNX
# takes a 207-D [embedding | 7 feats] input. Import the SAME module the trainer uses so
# the feature definitions and ordering are guaranteed identical.
sys.path.insert(0, str(PROJECT_ROOT))
from src.re_features import compute_re_features, RE_FEATURE_NAMES

# Sc, Y and all lanthanides (La–Lu)
RE_ELEMENTS = frozenset({
    "Sc", "Y",
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
})


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def _load_elem_features(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def compound_embedding(formula: str, elem_features: dict) -> np.ndarray:
    """Return the 200-D matscholar200 embedding for a compound formula."""
    try:
        comp   = Composition(formula)
        el_amt = comp.get_el_amt_dict()
    except Exception as exc:
        raise ValueError(f"Cannot parse formula '{formula}': {exc}") from exc

    amounts = np.array(list(el_amt.values()), dtype=np.float64)
    weights = amounts / amounts.sum()
    dim     = len(next(iter(elem_features.values())))
    vec     = np.zeros(dim, dtype=np.float32)

    for el, w in zip(el_amt.keys(), weights):
        if el not in elem_features:
            raise ValueError(
                f"Element '{el}' is not in the matscholar200 vocabulary "
                f"and cannot be embedded."
            )
        vec += float(w) * np.array(elem_features[el], dtype=np.float32)

    return vec


def re_feature_vector(formula: str) -> np.ndarray:
    """Return the 7 rare-earth physics features for a formula, in RE_FEATURE_NAMES
    order — matching the trainer's np.hstack([embedding, RE feats]). Zero for RE-free
    compounds. Only used when the selected ONNX model expects the 207-D input.
    """
    feats = compute_re_features(formula)
    return np.array([feats[k] for k in RE_FEATURE_NAMES], dtype=np.float32)


# ---------------------------------------------------------------------------
# RE detection
# ---------------------------------------------------------------------------

def contains_re(formula: str) -> bool:
    """Return True if the formula contains at least one rare-earth element."""
    comp = Composition(formula)
    return any(str(el) in RE_ELEMENTS for el in comp.elements)


# ---------------------------------------------------------------------------
# Best-model lookup
# ---------------------------------------------------------------------------

def _load_best_model_tags() -> "dict[str, str]":
    """Return {onnx_base_name: 'best for <Dataset>'} from exp_tc_best_by_dataset.csv.

    Returns an empty dict if the CSV does not exist yet (i.e. training not done).
    """
    if not BEST_CSV.exists():
        return {}
    tags: dict = {}
    with open(BEST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            base = f"{row['Dataset']}_{row['Embedding']}_{row['Model'].lower()}"
            # Tag both the embedding-only and the RE-features (_refeats) file names, so
            # whichever variant is on disk gets the "best for <Dataset>" label in --all.
            tags[base] = f"best for {row['Dataset']}"
            tags[f"{base}_refeats"] = f"best for {row['Dataset']}"
    return tags


def _load_best_model_by_dataset() -> "dict[str, str]":
    """Return {Dataset: onnx_base_name} for the best model per dataset."""
    if not BEST_CSV.exists():
        return {}
    result: dict = {}
    with open(BEST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            base = f"{row['Dataset']}_{row['Embedding']}_{row['Model'].lower()}"
            result[row['Dataset']] = base
    return result


def _resolve_group(base: str, groups: "dict[str, list[Path]]") -> "str | None":
    """Map a best_by_dataset base name to an actually-present ONNX group.

    The comparison CSV records a suffix-less base (e.g. ``RE_raw_200D_lgbm``), but a
    ``re_features:true`` run writes the model with a ``_refeats`` marker
    (``RE_raw_200D_lgbm_refeats``). Prefer an exact match; otherwise fall back to the
    ``_refeats`` variant. Returns None if neither is present.
    """
    if base in groups:
        return base
    if f"{base}_refeats" in groups:
        return f"{base}_refeats"
    return None


# ---------------------------------------------------------------------------
# Ensemble grouping and RE filtering
# ---------------------------------------------------------------------------

def group_onnx_models(onnx_dir: Path) -> "dict[str, list[Path]]":
    """Group ONNX files by logical model name.

    Files named <base>_e<N>.onnx are treated as ensemble members of <base>.
    Files without the suffix are single-member groups.
    Returns an ordered dict {base_name: [sorted list of member paths]}.
    """
    groups: dict = {}
    for p in sorted(onnx_dir.glob("*.onnx")):
        base = re.sub(r"_e\d+$", "", p.stem)
        groups.setdefault(base, []).append(p)
    return dict(sorted(groups.items()))


def _model_dataset(onnx_path: Path) -> "str | None":
    """Infer which dataset a model file was trained on from its name prefix.

    ONNX files are named ``<Dataset>_<embedding>_<model>...onnx`` with Dataset one of
    All / RE / RE-Free. Returns "All", "RE", "RE-Free", or None if unrecognised.
    ("RE-Free_" is checked before "RE_" since the latter is a prefix of the former.)
    """
    name = onnx_path.name
    if name.startswith("RE-Free_"):
        return "RE-Free"
    if name.startswith("RE_"):
        return "RE"
    if name.startswith("All_"):
        return "All"
    return None


def _filter_groups(groups: "dict[str, list[Path]]", is_re: bool) -> "dict[str, list[Path]]":
    """Keep only models relevant for the compound type.

    RE compound    → RE_* and All_* models
    RE-free compound → RE-Free_* and All_* models
    """
    result = {}
    for base, members in groups.items():
        if is_re and base.startswith("RE-Free_"):
            continue
        if not is_re and base.startswith("RE_"):
            continue
        result[base] = members
    return result


# ---------------------------------------------------------------------------
# ONNX inference
# ---------------------------------------------------------------------------

def _ort_session(onnx_path: Path) -> rt.InferenceSession:
    """Create an ONNX Runtime session with explicit thread count.

    Avoids pthread_setaffinity_np errors on HPC systems with restricted CPU sets.
    """
    opts = rt.SessionOptions()
    n_threads = max(1, os.cpu_count() or 1)
    opts.intra_op_num_threads = n_threads
    opts.inter_op_num_threads = 1
    return rt.InferenceSession(str(onnx_path), sess_options=opts)


def predict_with_model(onnx_path: Path, emb: np.ndarray,
                       re_feats: "np.ndarray | None" = None) -> float:
    """Run ONNX inference and return scalar Tc prediction (K).

    Models trained with re_features:true expect a 207-D input
    ``[embedding | 7 RE feats]``; embedding-only models expect 200-D. The correct input
    is chosen from the ONNX graph's declared input width, so a directory mixing both
    kinds of model is served transparently.
    """
    sess        = _ort_session(onnx_path)
    input_name  = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name
    n_in        = sess.get_inputs()[0].shape[1]
    if isinstance(n_in, int) and n_in == emb.shape[0] + len(RE_FEATURE_NAMES):
        if re_feats is None:
            raise ValueError(
                "model expects RE features (207-D input) but none were supplied"
            )
        x = np.concatenate([emb, re_feats])
    else:
        x = emb
    X           = x.reshape(1, -1).astype(np.float32)
    result      = sess.run([output_name], {input_name: X})[0]
    return float(np.squeeze(result))


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def _print_table(
    emb: np.ndarray,
    groups: "dict[str, list[Path]]",
    best_tags: "dict[str, str]",
    re_feats: "np.ndarray | None" = None,
) -> None:
    """Run inference for all groups and print one summary row per logical model."""
    rows = []
    max_ensemble = 0
    for base, members in groups.items():
        member_preds = []
        for p in members:
            try:
                member_preds.append(predict_with_model(p, emb, re_feats))
            except Exception as exc:
                print(f"  {p.name}  ERROR: {exc}")
        if member_preds:
            max_ensemble = max(max_ensemble, len(member_preds))
            rows.append((base, member_preds))

    if not rows:
        print("  (no predictions)")
        return

    tc_headers = "".join(f"  {'Tc' + str(i+1) + ' (K)':>9}" for i in range(max_ensemble))
    print(f"{'Model':<55}  {'mean Tc (K)':>11}  {'Std (K)':>8}{tc_headers}")
    print("-" * (78 + 11 * max_ensemble))
    for base, member_preds in rows:
        n = len(member_preds)
        mean_tc = float(np.mean(member_preds))
        std_tc  = float(np.std(member_preds)) if n > 1 else float("nan")
        tag     = f"  [{best_tags[base]}]" if base in best_tags else ""
        label   = f"{base}{tag}"
        tc_vals = "".join(f"  {tc:>9.1f}" for tc in member_preds)
        tc_vals += "".join(f"  {'':>9}" for _ in range(max_ensemble - n))
        std_str = f"{std_tc:>8.1f}" if n > 1 else f"{'':>8}"
        print(f"{label:<55}  {mean_tc:>11.1f}  {std_str}{tc_vals}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict Tc from a compound formula using a trained ONNX model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--compound", "-c",
        help="Chemical formula to predict, e.g. Fe3Pt or Nd2Fe14B",
    )
    parser.add_argument(
        "--compounds-file", "-f",
        help="Path to a text file with one formula per line.",
    )
    parser.add_argument(
        "--model", "-m",
        help="Path to a single .onnx model file.",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Run all available ONNX models (filtered by RE content) and print a table.",
    )
    parser.add_argument(
        "--best", "-b",
        action="store_true",
        help="Run only the best model for the detected compound type (RE or RE-Free).",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available ONNX models and exit.",
    )
    args = parser.parse_args()

    # --list: show available models
    if args.list:
        groups = group_onnx_models(ONNX_DIR)
        if not groups:
            print(f"No ONNX models found in {ONNX_DIR}")
            print("Run one of the train_exp_tc*.py scripts to generate them.")
        else:
            print(f"Available models in {ONNX_DIR}:")
            for base, members in groups.items():
                n = len(members)
                tag = f"  [ensemble × {n}]" if n > 1 else ""
                print(f"  {base}{tag}")
        return

    # Build compound list
    compounds = []
    if args.compound:
        compounds.append(args.compound.strip())
    if args.compounds_file:
        compounds_path = Path(args.compounds_file)
        if not compounds_path.exists():
            print(f"ERROR: compounds file not found: {compounds_path}", file=sys.stderr)
            sys.exit(1)
        with open(compounds_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    compounds.append(line)

    if not compounds:
        parser.error("Provide --compound and/or --compounds-file.")

    if not args.model and not args.all and not args.best:
        parser.error("Specify --model <path>, --all, or --best.")

    # Load element embeddings once
    if not EMB_FILE.exists():
        print(f"ERROR: element embedding file not found:\n  {EMB_FILE}", file=sys.stderr)
        sys.exit(1)
    elem_features = _load_elem_features(EMB_FILE)

    # For --all / --best: load groups and best-model tags once
    if args.all or args.best:
        all_groups = group_onnx_models(ONNX_DIR)
        if not all_groups:
            print(f"No ONNX models found in {ONNX_DIR}")
            print("Run one of the train_exp_tc*.py scripts to generate them.")
            sys.exit(1)
        best_tags = _load_best_model_tags()
    if args.best:
        best_by_dataset = _load_best_model_by_dataset()
        if not best_by_dataset:
            print(f"ERROR: {BEST_CSV} not found. Run training first.", file=sys.stderr)
            sys.exit(1)

    # For --model: resolve path once
    if args.model:
        onnx_path = Path(args.model)
        if not onnx_path.exists():
            print(f"ERROR: model file not found: {onnx_path}", file=sys.stderr)
            sys.exit(1)

    sep = "=" * 70

    for formula in compounds:
        print(f"\n{sep}")

        # Compute embedding (and the RE features, for any re_features:true models)
        try:
            emb = compound_embedding(formula, elem_features)
            re_feats = re_feature_vector(formula)
        except ValueError as exc:
            print(f"Compound : {formula}")
            print(f"ERROR: {exc}")
            continue

        if args.all:
            is_re    = contains_re(formula)
            re_label = "contains RE" if is_re else "RE-free"
            print(f"Compound : {formula}  ({re_label})")
            print(sep)
            filtered = _filter_groups(all_groups, is_re)
            _print_table(emb, filtered, best_tags, re_feats)
        elif args.best:
            is_re        = contains_re(formula)
            re_label     = "contains RE" if is_re else "RE-free"
            dataset_key  = "RE" if is_re else "RE-Free"
            print(f"Compound : {formula}  ({re_label})")
            print(sep)
            best_base = best_by_dataset.get(dataset_key)
            if best_base is None:
                print(f"No best model found for dataset '{dataset_key}'. Run training first.")
                continue
            resolved = _resolve_group(best_base, all_groups)
            if resolved is None:
                print(f"Best model '{best_base}' not found in {ONNX_DIR}.")
                continue
            _print_table(emb, {resolved: all_groups[resolved]}, best_tags, re_feats)
        else:
            is_re    = contains_re(formula)
            model_ds = _model_dataset(onnx_path)
            re_label = "contains RE" if is_re else "RE-free"
            print(f"Compound : {formula}  ({re_label})")
            print(f"Model    : {onnx_path.name}")
            # Refuse to predict with a model trained on the wrong chemistry — the RE and
            # RE-Free models do not extrapolate across the RE boundary (e.g. Fe through
            # the RE model gives a badly wrong Tc). The All model is valid for both.
            if model_ds == "RE" and not is_re:
                print("ERROR: this is a RE-free compound but the model is the RE-only "
                      "model — refusing to predict. Use an All_* or RE-Free_* model.")
                continue
            if model_ds == "RE-Free" and is_re:
                print("ERROR: this compound contains rare-earth element(s) but the model "
                      "is the RE-Free model — refusing to predict. Use an All_* or RE_* "
                      "model.")
                continue
            tc = predict_with_model(onnx_path, emb, re_feats)
            print(f"Tc pred  : {tc:.1f} K")


if __name__ == "__main__":
    main()
