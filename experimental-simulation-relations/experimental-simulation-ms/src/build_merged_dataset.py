"""Create the merged Ms training dataset from the raw data sources.

This is the FIRST step of the pipeline. It aggregates experimental and simulated
saturation-magnetisation (Ms) values for magnetic compounds from several raw databases
in ``data/`` into a single table, ``data/merged_df_python.csv``, which every downstream
step consumes:

    src/build_merged_dataset.py  ->  data/merged_df_python.csv
        -> src/augment_data.py       (outputs/Augm_*.csv, Pairs_*.csv)
        -> src/create_embeddings.py  (compound embeddings)
        -> src/training_*            (model training)

Run this once before training whenever ``data/merged_df_python.csv`` does not yet exist
(or when the raw sources change).

Output schema / units:
    composition                 chemical formula (str)
    Ms (ampere/meter)_e         experimental saturation magnetisation [A/m]
    Ms (ampere/meter)_s         simulated (DFT) saturation magnetisation [A/m]
    material_id                 placeholder id (see note below)
    has_rare_earth              bool: composition contains a rare-earth element

Raw sources (in ``data/``) and their unit conversions to A/m:
    experimental (-> _e):
        literature_values.csv        'mu0Ms (T)'                 / MU_0
        Bhandari_I_exp.csv           'Ms_exp (MA/m)'             * 1e6
        Bhandari_XIII_exp.csv        'Ms (MA/m)'                 * 1e6
        mp_fm_dedup_exp_data.csv     'total_magnetization_normalized_vol' [mu_B/A^3]
                                                                 * MU_B / ANGSTROM**3
    simulated (-> _s):
        oqmd_stable.csv              'Ms' [T, = mu0*M]           / MU_0
        Bhandari_XII_sim.csv         'Ms (A/m)'                  (already A/m)
        mp_fm_dedup_sim_data.csv     'total_magnetization_normalized_vol' [mu_B/A^3]
                                                                 * MU_B / ANGSTROM**3

Deduplication of duplicate compositions:
    All experimental (resp. simulated) Ms values for a composition -- from every source
    AND every duplicate row -- are POOLED and reduced with a SINGLE median (one
    ``groupby(composition).median()``, NOT a median-of-medians / no per-source
    pre-averaging). Experimental and simulated columns are deduplicated independently,
    then outer-merged on composition, so a composition with only one of the two
    survives. Compositions whose formula cannot be parsed are dropped.

Formula parsing:
    Compositions are parsed with pymatgen. Entries that are not fixed chemical formulas
    -- parametric or placeholder strings such as 'SrLaFexCoxAlxO0.2' or '0.2xy15*' --
    cannot be parsed and are dropped (these are experimental-only and never form
    exp/sim pairs).

material_id note:
    material_id is a non-meaningful placeholder: a random int in [1, 1000] (seeded), so
    it is NOT unique across the ~160k rows. Pass --unique-ids for a real unique id.

Usage:
    python -m src.build_merged_dataset                         # writes data/merged_df_python.csv
    python -m src.build_merged_dataset --data-dir data --out data/merged_df_python.csv
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
MU_0 = 4e-7 * np.pi          # vacuum permeability [T*m/A]
MU_B = 9.2740100657e-24      # Bohr magneton [J/T]
ANGSTROM = 1e-10             # [m]

KEY = "composition"
EXP_COL = "Ms (ampere/meter)_e"
SIM_COL = "Ms (ampere/meter)_s"

# Rare earths: Sc, Y and the lanthanides La..Lu (matches the original definition).
RARE_EARTHS = {
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}


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
# Per-source loaders -> DataFrame[composition, <EXP_COL or SIM_COL>]
# ---------------------------------------------------------------------------
def _frame(comp, values, col) -> pd.DataFrame:
    out = pd.DataFrame({KEY: comp.astype(str).str.strip(),
                        col: pd.to_numeric(values, errors="coerce")})
    return out


def load_oqmd(d: Path) -> pd.DataFrame:            # simulated
    df = pd.read_csv(d / "oqmd_stable.csv")        # real header -> real 'Ms' (Tesla)
    return _frame(df["composition"], df["Ms"] / MU_0, SIM_COL)


def load_literature(d: Path) -> pd.DataFrame:      # experimental
    df = pd.read_csv(d / "literature_values.csv", sep=";")
    return _frame(df["Compound"], df["mu0Ms (T)"] / MU_0, EXP_COL)


def load_bhandari_i(d: Path) -> pd.DataFrame:      # experimental (MA/m)
    df = pd.read_csv(d / "Bhandari_I_exp.csv", sep="|")
    return _frame(df["Material"], df["Ms_exp (MA/m)"] * 1e6, EXP_COL)


def load_bhandari_xiii(d: Path) -> pd.DataFrame:   # experimental (MA/m)
    df = pd.read_csv(d / "Bhandari_XIII_exp.csv", sep="|")
    return _frame(df["Material"], df["Ms (MA/m)"] * 1e6, EXP_COL)


def load_bhandari_xii(d: Path) -> pd.DataFrame:    # simulated (already A/m)
    df = pd.read_csv(d / "Bhandari_XII_sim.csv", sep=";")
    return _frame(df["material"], df["Ms (A/m)"], SIM_COL)


def _mp(d: Path, fname: str, col: str) -> pd.DataFrame:
    # total_magnetization_normalized_vol is in mu_B / Angstrom^3 -> A/m
    df = pd.read_csv(d / fname,
                     usecols=["formula_pretty", "total_magnetization_normalized_vol"])
    ms = df["total_magnetization_normalized_vol"] * MU_B / (ANGSTROM ** 3)
    return _frame(df["formula_pretty"], ms, col)


def load_mp_exp(d: Path) -> pd.DataFrame:
    return _mp(d, "mp_fm_dedup_exp_data.csv", EXP_COL)


def load_mp_sim(d: Path) -> pd.DataFrame:
    return _mp(d, "mp_fm_dedup_sim_data.csv", SIM_COL)


EXP_LOADERS = [load_literature, load_bhandari_i, load_bhandari_xiii, load_mp_exp]
SIM_LOADERS = [load_oqmd, load_bhandari_xii, load_mp_sim]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
def build_merged_dataset(data_dir: Path, out_path: Path,
                         seed: int = 0, unique_ids: bool = False) -> pd.DataFrame:
    data_dir = Path(data_dir)

    exp = pd.concat([ld(data_dir) for ld in EXP_LOADERS], ignore_index=True)
    sim = pd.concat([ld(data_dir) for ld in SIM_LOADERS], ignore_index=True)

    # Single pooled median per composition (skips NaN); exp and sim done independently.
    exp = exp.groupby(KEY, as_index=False)[EXP_COL].median()
    sim = sim.groupby(KEY, as_index=False)[SIM_COL].median()

    merged = pd.merge(exp, sim, on=KEY, how="outer")

    # Rare-earth flag; drop unparsable compositions.
    merged["has_rare_earth"] = merged[KEY].map(has_rare_earth)
    n_before = len(merged)
    merged = merged[merged["has_rare_earth"].notna()].reset_index(drop=True)
    merged["has_rare_earth"] = merged["has_rare_earth"].astype(bool)

    # material_id
    if unique_ids:
        merged["material_id"] = np.arange(1, len(merged) + 1)
    else:
        merged["material_id"] = np.random.default_rng(seed).integers(1, 1001, size=len(merged))

    merged = merged[[KEY, EXP_COL, SIM_COL, "material_id", "has_rare_earth"]]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    both = merged[EXP_COL].notna() & merged[SIM_COL].notna()
    print(f"Saved {out_path}  ({len(merged)} rows; dropped {n_before - len(merged)} unparsable)")
    print(f"  experimental values : {int(merged[EXP_COL].notna().sum())}")
    print(f"  simulated values    : {int(merged[SIM_COL].notna().sum())}")
    print(f"  pairs (both e & s)  : {int(both.sum())}")
    print(f"  RE / RE-free        : {int(merged['has_rare_earth'].sum())} / "
          f"{int((~merged['has_rare_earth']).sum())}")
    return merged


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(root / "data"),
                   help="Directory with the raw source CSVs (default: <project>/data).")
    p.add_argument("--out", default=None,
                   help="Output CSV path (default: <data-dir>/merged_df_python.csv).")
    p.add_argument("--seed", type=int, default=0, help="Seed for the placeholder material_id.")
    p.add_argument("--unique-ids", action="store_true",
                   help="Use unique sequential material_id instead of the random placeholder.")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    out = Path(args.out) if args.out else data_dir / "merged_df_python.csv"
    build_merged_dataset(data_dir, out, seed=args.seed, unique_ids=args.unique_ids)


if __name__ == "__main__":
    main()
