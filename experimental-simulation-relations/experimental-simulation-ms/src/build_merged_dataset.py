"""Create the merged Ms pairs dataset from the preprocessed per-compound Ms tables.

This is the FIRST step of the pipeline. It builds ``data/merged_df_python.csv``, which
every downstream step consumes:

    src/build_merged_dataset.py  ->  data/merged_df_python.csv
        -> src/augment_data.py       (outputs/Augm_*.csv, Pairs_*.csv)
        -> src/create_embeddings.py  (compound embeddings)
        -> src/training_*            (model training)

NEW DATA SOURCING (2026-11-09)
------------------------------
Instead of re-reading the raw databases, this step now consumes the *preprocessed*
per-compound Ms tables produced by the two sibling projects and copied into
``preprocessed_data/``:

    preprocessed_data/Experimental_Ms_all.csv  (composition, Ms[A/m]) -- genuinely
        MEASURED Ms: literature_values + Bhandari I/XIII + NEMAD, Materials Project
        EXCLUDED (from compound-to-experimental-ms).
    preprocessed_data/Simulated_Ms_all.csv      (composition, Ms[A/m]) -- DFT Ms:
        OQMD + Bhandari XII + energy-filtered Materials Project
        (from compound-to-simulated-ms).

The raw provenance of these two tables lives in ``data/raw/`` (literature_values.csv,
Bhandari_I_exp.csv, Bhandari_XIII_exp.csv, Nemad_extracted_Ms.csv, MP_energy_filtered.csv,
oqmd_stable.csv, Bhandari_XII_sim.csv) but is NOT read here -- the two projects' own
preprocessing (50,000 A/m threshold, ferrimagnet removal, MP energy-filter / MP exclusion,
reduced-formula median dedup) is already baked into the copied CSVs.

Output schema / units (UNCHANGED, so augment_data / training_* run as before):
    composition                 pymatgen REDUCED chemical formula (str)
    Ms (ampere/meter)_e         experimental Ms [A/m]  (NaN if no measured value)
    Ms (ampere/meter)_s         simulated (DFT) Ms [A/m]  (NaN if no DFT value)
    material_id                 placeholder id (see note below)
    has_rare_earth              bool: reduced formula contains a rare-earth element

Reduced-formula join & deduplication:
    Both preprocessed tables are already reduced-formula median-deduplicated inside their
    own project, but with DIFFERENT string conventions, so a measured compound and its
    simulated counterpart may be written differently. Here every composition on BOTH sides
    is RE-canonicalised to its pymatgen ``reduced_formula`` (reduced atom counts + canonical
    element ordering); all Ms values sharing a reduced formula are POOLED and reduced with a
    SINGLE median (one ``groupby(reduced_formula).median()`` per side, no median-of-medians).
    The two sides are then OUTER-merged on the reduced formula, so a composition present on
    only one side survives (needed by augment_data's sim-only / exp-only phases); rows with
    BOTH _e and _s are the exp/sim pairs. Compositions that pymatgen cannot parse are dropped.

material_id note:
    material_id is a non-meaningful placeholder: a random int in [1, 1000] (seeded), so it
    is NOT unique across the rows. Pass --unique-ids for a real unique id.

Usage:
    python -m src.build_merged_dataset                 # writes data/merged_df_python.csv
    python -m src.build_merged_dataset --preprocessed-dir preprocessed_data --out data/merged_df_python.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

# ---------------------------------------------------------------------------
# Constants and schema
# ---------------------------------------------------------------------------
KEY = "composition"
EXP_COL = "Ms (ampere/meter)_e"
SIM_COL = "Ms (ampere/meter)_s"

# Rare earths: Sc, Y and the lanthanides La..Lu (matches the original definition).
RARE_EARTHS = {
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}


def reduced_formula(formula) -> "str | None":
    """Return the pymatgen reduced formula (reduced atom counts + canonical element
    ordering) of *formula*, or None if it cannot be parsed."""
    try:
        return Composition(str(formula)).reduced_formula
    except Exception:
        return None


def has_rare_earth(formula) -> "bool | None":
    """Return True/False if *formula* contains a rare-earth element, or None if the
    formula cannot be parsed. Uses pymatgen (the project's standard, robust parser)
    rather than a hand-rolled formula parser."""
    try:
        elements = {str(e) for e in Composition(str(formula)).elements}
    except Exception:
        return None
    return bool(elements & RARE_EARTHS)


# ---------------------------------------------------------------------------
# Per-side loader: preprocessed (composition, Ms) -> reduced-formula median
# ---------------------------------------------------------------------------
def _load_side(path: Path, out_col: str) -> "tuple[pd.DataFrame, int, int]":
    """Read a preprocessed ``composition, Ms`` table, re-canonicalise the composition to
    its pymatgen reduced formula, and take ONE pooled median Ms per reduced formula.

    Returns (DataFrame[composition, out_col], n_unparsable, n_rows_kept).
    """
    df = pd.read_csv(path)
    if "composition" not in df.columns or "Ms" not in df.columns:
        raise ValueError(f"{path} must have columns 'composition','Ms'; got {list(df.columns)}")
    df = df[["composition", "Ms"]].copy()
    df["__key"] = df["composition"].map(reduced_formula)
    n_unparsable = int(df["__key"].isna().sum())
    df = df.dropna(subset=["__key"]).copy()
    df[out_col] = pd.to_numeric(df["Ms"], errors="coerce")
    agg = df.groupby("__key", as_index=False)[out_col].median().rename(columns={"__key": KEY})
    return agg, n_unparsable, int(len(df))


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_merged_dataset(preprocessed_dir: Path, out_path: Path,
                         seed: int = 0, unique_ids: bool = False) -> pd.DataFrame:
    preprocessed_dir = Path(preprocessed_dir)
    exp_path = preprocessed_dir / "Experimental_Ms_all.csv"
    sim_path = preprocessed_dir / "Simulated_Ms_all.csv"

    exp, exp_unp, exp_kept = _load_side(exp_path, EXP_COL)
    sim, sim_unp, sim_kept = _load_side(sim_path, SIM_COL)

    # Outer-merge on the reduced formula: keep exp-only and sim-only rows too.
    merged = pd.merge(exp, sim, on=KEY, how="outer")

    # Rare-earth flag; drop any formula pymatgen cannot parse (should be none post-reduce).
    merged["has_rare_earth"] = merged[KEY].map(has_rare_earth)
    n_before = len(merged)
    merged = merged[merged["has_rare_earth"].notna()].reset_index(drop=True)
    merged["has_rare_earth"] = merged["has_rare_earth"].astype(bool)

    # material_id placeholder
    if unique_ids:
        merged["material_id"] = np.arange(1, len(merged) + 1)
    else:
        merged["material_id"] = np.random.default_rng(seed).integers(1, 1001, size=len(merged))

    merged = merged[[KEY, EXP_COL, SIM_COL, "material_id", "has_rare_earth"]]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    both = merged[EXP_COL].notna() & merged[SIM_COL].notna()
    print(f"Experimental source: {exp_path}  "
          f"(kept {exp_kept} rows -> {len(exp)} reduced formulas; {exp_unp} unparsable)")
    print(f"Simulated    source: {sim_path}  "
          f"(kept {sim_kept} rows -> {len(sim)} reduced formulas; {sim_unp} unparsable)")
    print(f"Saved {out_path}  ({len(merged)} rows; dropped {n_before - len(merged)} unparsable)")
    print(f"  experimental values : {int(merged[EXP_COL].notna().sum())}")
    print(f"  simulated values    : {int(merged[SIM_COL].notna().sum())}")
    print(f"  pairs (both e & s)  : {int(both.sum())}")
    print(f"  pairs RE / RE-free  : {int((both & merged['has_rare_earth']).sum())} / "
          f"{int((both & ~merged['has_rare_earth']).sum())}")
    print(f"  RE / RE-free (all)  : {int(merged['has_rare_earth'].sum())} / "
          f"{int((~merged['has_rare_earth']).sum())}")
    return merged


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preprocessed-dir", default=str(root / "preprocessed_data"),
                   help="Directory with Experimental_Ms_all.csv and Simulated_Ms_all.csv "
                        "(default: <project>/preprocessed_data).")
    p.add_argument("--out", default=str(root / "data" / "merged_df_python.csv"),
                   help="Output CSV path (default: <project>/data/merged_df_python.csv).")
    p.add_argument("--seed", type=int, default=0, help="Seed for the placeholder material_id.")
    p.add_argument("--unique-ids", action="store_true",
                   help="Use unique sequential material_id instead of the random placeholder.")
    args = p.parse_args()
    build_merged_dataset(args.preprocessed_dir, args.out, seed=args.seed, unique_ids=args.unique_ids)


if __name__ == "__main__":
    main()
