#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the trained SIMULATED-Tc models against an external reference set.

For every compound in ``data/validation_reference.csv`` this script predicts Tc using
ONLY the BEST model for that compound's chemistry:

    * a rare-earth compound  -> the best  RE       model (from sim_tc_best_by_dataset.csv)
    * a rare-earth-free compound -> the best RE-Free model

The best model per dataset is an ENSEMBLE of ONNX members (``<base>_e<N>.onnx``), so the
prediction is reported as the ensemble mean +/- standard deviation — the honest spread over
the ensemble, not a best-of-N cherry-pick.

The heavy lifting (embedding, RE detection, best-model lookup, ``_refeats`` resolution,
ONNX inference) is imported from ``predict_tc.py`` so this script stays a thin wrapper and
cannot drift from the deployed prediction path.

NOTE (simulated model): the reference values are EXPERIMENTAL Curie/Neel temperatures,
while this model predicts a SIMULATED (DFT/spin-dynamics) Tc. Read the error column as
bundling that simulation-vs-experiment offset with model error, not as pure model accuracy.

Output:
    * a table printed to stdout: compound | reference | prediction +/- std
    * the same table written to results/validation_reference_predictions.csv (--out to change)

Usage:
    python src/validate_reference_data.py
    python src/validate_reference_data.py --ref data/validation_reference.csv --out mytable.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse the deployed prediction path verbatim.
from src.predict_tc import (
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
    """Return (mean, std, n) over all ONNX members of the best model group."""
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
    """Yield dicts from the reference CSV. Reference value may be blank (nonmagnetic)."""
    with open(ref_path, newline="") as f:
        for row in csv.DictReader(f):
            raw = (row.get("reference_Tc_or_TN_K") or "").strip()
            try:
                ref = float(raw) if raw else None
            except ValueError:
                ref = None
            yield {
                "formula": row["formula"].strip(),
                "ref": ref,
                "magnetic_type": (row.get("magnetic_type") or "").strip(),
                "is_curie": (row.get("is_curie_temperature") or "").strip().lower() == "yes",
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


def _fmt(x, prec=1):
    return f"{x:.{prec}f}" if isinstance(x, (int, float)) and x == x else "-"


def _print_table(results) -> None:
    print(f"\n{'Compound':<12}  {'RE':<7}  {'Ref (K)':>8}  {'Pred (K)':>9}  {'Std (K)':>8}  "
          f"{'Err (K)':>8}  Best model")
    print("-" * 100)
    for r in results:
        if r["error"] and r["pred"] is None:
            print(f"{r['formula']:<12}  {'-':<7}  {_fmt(r['ref']):>8}  {'ERROR':>9}  "
                  f"{'-':>8}  {'-':>8}  {r['error']}")
            continue
        re_lbl = "RE" if r["dataset"] == "RE" else "RE-free"
        err = (r["pred"] - r["ref"]) if (r["pred"] is not None and r["ref"] is not None) else None
        model_short = (r["model"] or "").replace("_raw_200D", "").replace("_refeats", "")
        print(f"{r['formula']:<12}  {re_lbl:<7}  {_fmt(r['ref']):>8}  {_fmt(r['pred']):>9}  "
              f"{_fmt(r['std']):>8}  {_fmt(err):>8}  {model_short} (x{r['n']})")


def _write_csv(out_path: Path, results) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["compound", "dataset", "best_model", "n_ensemble", "magnetic_type",
                    "is_curie_temperature", "reference_K", "prediction_K", "std_K", "error_K"])
        for r in results:
            err = (r["pred"] - r["ref"]) if (r["pred"] is not None and r["ref"] is not None) else None
            w.writerow([
                r["formula"], r["dataset"] or "", r["model"] or "", r["n"],
                r["magnetic_type"], "yes" if r["is_curie"] else "no",
                "" if r["ref"] is None else r["ref"],
                "" if r["pred"] is None else round(r["pred"], 2),
                "" if (r["std"] is None or r["std"] != r["std"]) else round(r["std"], 2),
                "" if err is None else round(err, 2),
            ])
    print(f"\nWrote {out_path}")


def _print_summary(results) -> None:
    """MAE only over true Curie temperatures with a reference (the model's actual target).

    Antiferromagnets (Neel T) and nonmagnetic entries are excluded from the error stat —
    the model predicts a ferro/ferri Tc and was never trained to reproduce those.
    """
    fair = [r for r in results
            if r["is_curie"] and r["ref"] is not None and r["pred"] is not None]
    if not fair:
        return
    abs_err = [abs(r["pred"] - r["ref"]) for r in fair]
    print(f"\nMAE over {len(fair)} true-Curie reference compounds: {np.mean(abs_err):.1f} K "
          f"(median |err| {np.median(abs_err):.1f} K)")
    print("(Antiferromagnets = Neel T and nonmagnetic entries excluded — the models "
          "predict a ferro/ferri Curie temperature only.)")


if __name__ == "__main__":
    main()
