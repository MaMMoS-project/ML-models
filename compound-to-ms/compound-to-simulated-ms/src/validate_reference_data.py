#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the trained simulated-Ms models against an external reference set.

For every compound in ``data/validation_reference.csv`` this script predicts Ms using ONLY
the BEST model for that compound's chemistry:

    * a rare-earth compound      -> the best RE       model (from sim_ms_best_by_dataset.csv)
    * a rare-earth-free compound  -> the best RE-Free  model

NOTE (simulated model): the reference Ms values are EXPERIMENTAL; this model predicts a
SIMULATED (DFT) Ms, which can differ systematically. Read the errors as bundling that
simulation-vs-experiment offset with model error, not as pure model accuracy.

The best model per dataset is an ENSEMBLE of ONNX members (``<base>_e<N>.onnx``), so the
prediction is reported as the ensemble mean +/- standard deviation — the honest spread over
the ensemble, not a best-of-N cherry-pick.

The heavy lifting (embedding, RE detection, best-model lookup, ``_refeats`` resolution, ONNX
inference AND the log1p->A/m expm1 inversion) is imported from ``predict_ms.py`` so this
script exercises the exact deployed prediction path and reports physical Ms in A/m.

Because Ms spans orders of magnitude, the summary reports BOTH a mean absolute error (A/m)
and a median absolute RELATIVE error (%), the latter being the more meaningful measure.
Non-magnetic entries (reference Ms ~ 0) are shown for sanity but excluded from the stats:
the models are trained on Ms > threshold and have no "zero" regime.

Output:
    * a table on stdout: compound | RE? | reference | prediction +/- std | rel.err
    * the same written to results/validation_reference_predictions.csv (--out to change)

Usage:
    python src/validate_reference_data.py
    python src/validate_reference_data.py --ref data/validation_reference.csv --out table.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the deployed prediction path verbatim (predict_with_model applies expm1 -> A/m).
from src.predict_ms import (
    EMB_FILE,
    ONNX_DIR,
    BEST_CSV,
    _load_elem_features,
    compound_embedding,
    re_feature_vector,
    contains_re,
    group_onnx_models,
    _resolve_group,
    _load_best_model_by_dataset,
    predict_with_model,
)

REF_CSV = PROJECT_ROOT / "data" / "validation_reference.csv"
OUT_CSV = PROJECT_ROOT / "results" / "validation_reference_predictions.csv"


def _ensemble_predict(members, emb, re_feats):
    """Return (mean, std, n) in A/m over all ONNX members of the best model group."""
    preds = []
    for p in members:
        try:
            preds.append(predict_with_model(p, emb, re_feats))
        except Exception as exc:  # noqa: BLE001
            print(f"    {p.name}  ERROR: {exc}", file=sys.stderr)
    if not preds:
        return None, None, 0
    n = len(preds)
    return float(np.mean(preds)), (float(np.std(preds)) if n > 1 else float("nan")), n


def _read_reference(ref_path: Path):
    """Yield dicts from the reference CSV. Reference Ms may be blank (non-magnetic)."""
    with open(ref_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("reference_Ms_A_per_m") or "").strip()
            try:
                ref = float(raw) if raw else None
            except ValueError:
                ref = None
            yield {
                "formula": row["formula"].strip(),
                "ref": ref,
                "magnetic_type": (row.get("magnetic_type") or "").strip(),
                "is_fm": (row.get("is_ferro_or_ferri") or "").strip().lower() == "yes",
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ref", default=str(REF_CSV), help="Reference CSV (default: data/validation_reference.csv).")
    parser.add_argument("--out", default=str(OUT_CSV), help="Output CSV for the results table.")
    args = parser.parse_args()

    ref_path = Path(args.ref)
    if not ref_path.exists():
        print(f"ERROR: reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(1)
    if not EMB_FILE.exists():
        print(f"ERROR: element embedding file not found:\n  {EMB_FILE}", file=sys.stderr)
        sys.exit(1)

    groups = group_onnx_models(ONNX_DIR)
    if not groups:
        print(f"No ONNX models found in {ONNX_DIR}. Run training first.", file=sys.stderr)
        sys.exit(1)
    best_by_dataset = _load_best_model_by_dataset()
    if not best_by_dataset:
        print(f"ERROR: {BEST_CSV} not found. Run training first.", file=sys.stderr)
        sys.exit(1)

    elem_features = _load_elem_features(EMB_FILE)

    results = []
    for entry in _read_reference(ref_path):
        formula = entry["formula"]
        try:
            is_re = contains_re(formula)
            emb = compound_embedding(formula, elem_features)
            re_feats = re_feature_vector(formula)
        except Exception as exc:  # noqa: BLE001 — parse / vocabulary errors
            results.append({**entry, "dataset": None, "model": None,
                            "pred": None, "std": None, "n": 0, "error": str(exc)})
            continue

        dataset_key = "RE" if is_re else "RE-Free"
        best_base = best_by_dataset.get(dataset_key)
        resolved = _resolve_group(best_base, groups) if best_base else None
        if resolved is None:
            results.append({**entry, "dataset": dataset_key, "model": None,
                            "pred": None, "std": None, "n": 0,
                            "error": f"no best model for {dataset_key}"})
            continue

        mean, std, n = _ensemble_predict(groups[resolved], emb, re_feats)
        results.append({**entry, "dataset": dataset_key, "model": resolved,
                        "pred": mean, "std": std, "n": n, "error": None})

    _print_table(results)
    _write_csv(Path(args.out), results)
    _print_summary(results)


def _fmt(x):
    return f"{x:.4g}" if isinstance(x, (int, float)) and x == x else "-"


def _relerr(pred, ref):
    return (pred - ref) / ref if (pred is not None and ref) else None


def _print_table(results) -> None:
    print(f"\n{'Compound':<12}  {'RE':<7}  {'Ref (A/m)':>11}  {'Pred (A/m)':>11}  "
          f"{'Std (A/m)':>11}  {'Rel.err':>8}  Best model")
    print("-" * 104)
    for r in results:
        if r["error"] and r["pred"] is None:
            print(f"{r['formula']:<12}  {'-':<7}  {_fmt(r['ref']):>11}  {'ERROR':>11}  "
                  f"{'-':>11}  {'-':>8}  {r['error']}")
            continue
        re_lbl = "RE" if r["dataset"] == "RE" else "RE-free"
        rel = _relerr(r["pred"], r["ref"])
        rel_s = f"{100*rel:+.0f}%" if rel is not None else "-"
        model_short = (r["model"] or "").replace("_raw_200D", "").replace("_refeats", "")
        print(f"{r['formula']:<12}  {re_lbl:<7}  {_fmt(r['ref']):>11}  {_fmt(r['pred']):>11}  "
              f"{_fmt(r['std']):>11}  {rel_s:>8}  {model_short} (x{r['n']})")


def _write_csv(out_path: Path, results) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound", "dataset", "best_model", "n_ensemble", "magnetic_type",
                    "is_ferro_or_ferri", "reference_A_per_m", "prediction_A_per_m",
                    "std_A_per_m", "rel_error"])
        for r in results:
            rel = _relerr(r["pred"], r["ref"])
            w.writerow([
                r["formula"], r["dataset"] or "", r["model"] or "", r["n"],
                r["magnetic_type"], "yes" if r["is_fm"] else "no",
                "" if r["ref"] is None else r["ref"],
                "" if r["pred"] is None else round(r["pred"], 1),
                "" if (r["std"] is None or r["std"] != r["std"]) else round(r["std"], 1),
                "" if rel is None else round(rel, 4),
            ])
    print(f"\nWrote {out_path}")


def _print_summary(results) -> None:
    """Stats only over ferro/ferrimagnets with a reference (the model's actual target).

    Non-magnetic entries are excluded — the models are trained on Ms > threshold and have no
    zero regime, so they cannot reproduce Ms ~ 0.
    """
    fair = [r for r in results
            if r["is_fm"] and r["ref"] is not None and r["pred"] is not None]
    if not fair:
        return
    abs_err = [abs(r["pred"] - r["ref"]) for r in fair]
    rel_err = [abs((r["pred"] - r["ref"]) / r["ref"]) for r in fair]
    print(f"\nOver {len(fair)} ferro/ferrimagnetic reference compounds:")
    print(f"  mean |err|        = {np.mean(abs_err):.3g} A/m")
    print(f"  median |rel.err|  = {100*np.median(rel_err):.1f} %   (more meaningful; Ms spans decades)")
    print("(Non-magnetic entries excluded — models are trained on Ms > threshold, no zero regime.)")


if __name__ == "__main__":
    main()
