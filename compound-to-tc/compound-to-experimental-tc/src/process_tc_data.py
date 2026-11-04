#!/usr/bin/env python
# coding: utf-8
"""Build the SIMULATED-Tc preprocessed_data from the CURATED simulation-only sources.

This project (compound-to-simulated-tc) now produces Simulation_Tc*.csv ONLY, and from the
curated split files in ``data/curated/`` rather than from the raw sources. Those curated
files come from the Curated-Tc-data curation:
  * step 03 split every raw source into ``<source>_experimental.csv`` / ``<source>_simulated.csv``
    using each source's own exp/sim flag, overridden for RE-free compounds by partner UU's
    Sim-TC-HH-NH-AV.xlsx (prefer measured; theory -> simulated), and DROPPED the entries UU
    flagged as wrong/unusable;
  * step 04 regenerated the preprocessed data from those splits.
This script reproduces step 04's SIMULATION path exactly: pool every
``data/curated/*_simulated.csv`` and run the same clean / reduced-formula / median-dedup /
RE-split as the original pipeline. All "which compounds are kept vs dropped" decisions are
already baked into the curated ``*_simulated.csv`` files, so this script just pools and
processes them (no further dropping here).

Outputs (preprocessed_data/):
    Simulation_Tc_all.csv       composition, Tc_sim
    Simulation_Tc_RE.csv        composition, Tc_sim   (rare-earth)
    Simulation_Tc_RE-Free.csv   composition, Tc_sim   (rare-earth free)
    Simulated_Tc.csv            copy of Simulation_Tc_all.csv

The original raw-source generator (produced both experimental and simulated) is kept at
``src/process_tc_data.py.bak``.

Run from the project root (pinned env, pymatgen==2025.6.14):
    python src/process_tc_data.py
"""

import glob
import os

import pandas as pd
from pymatgen.core import Composition

# ---------------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------------

RARE_EARTH_ELEMENTS = [
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Sc", "Y",
]

CURATED_DIR = "./data/curated"
OUT_DIR = "./preprocessed_data"


# ---------------------------------------------------------------------------
# Helpers (identical to the original generator / Curated-Tc-data step 04)
# ---------------------------------------------------------------------------

def clean_tc_column(series: pd.Series) -> pd.Series:
    """Strip common unit/annotation noise from a Tc string column and cast to float."""
    return (
        series.astype(str)
        .str.replace(r"( )?K", "", regex=True)
        .str.replace(r"Tc( )?=", "", regex=True)
        .str.replace(r">", "", regex=False)
        .str.replace(r"±[0-9]*", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
    )


def contains_rare_earth(composition: str) -> bool:
    return any(elem in composition for elem in RARE_EARTH_ELEMENTS)


def reduced_formula(formula):
    """Canonical reduced formula for deduplication (CoFe2O4 -> Fe2CoO4, H4O2 -> H2O), or
    None if pymatgen cannot parse the string."""
    try:
        return Composition(str(formula).strip()).reduced_formula
    except Exception:
        return None


def process(df: pd.DataFrame, tc_col_out: str) -> pd.DataFrame:
    """Clean Tc strings, drop non-numeric/negative rows, reduce formula, median-deduplicate."""
    df = df.copy()
    df["Tc"] = clean_tc_column(df["Tc"])

    # Drop rows that are still non-numeric after cleaning
    df = df[df["Tc"].notna()]

    # Drop physically impossible negative Curie temperatures before deduplication.
    df = df[df["Tc"] >= 0]

    # Canonicalise to the reduced formula BEFORE deduplication so different spellings /
    # element orderings / multiples of the same compound are pooled together.
    df["composition"] = df["composition"].apply(reduced_formula)
    df = df[df["composition"].notna()]

    # Deduplicate: take median Tc per (reduced) composition
    df = df.groupby("composition", as_index=False)["Tc"].median()
    df = df.rename(columns={"Tc": tc_col_out})

    # Annotate rare-earth membership
    df["contains_re"] = df["composition"].apply(contains_rare_earth)

    return df


def split_re(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (RE-containing, RE-free) DataFrames, both without the flag column."""
    re_df   = df[df["contains_re"]].drop(columns="contains_re").reset_index(drop=True)
    re_free = df[~df["contains_re"]].drop(columns="contains_re").reset_index(drop=True)
    return re_df, re_free


# ---------------------------------------------------------------------------
# Load curated simulation-only sources
# ---------------------------------------------------------------------------

def load_curated_simulated() -> tuple[pd.DataFrame, list]:
    """Pool (composition, Tc) from every curated ``*_simulated.csv`` file in data/curated/.
    Auxiliary files starting with '_' (e.g. _dropped_flagged_wrong.csv) are ignored."""
    files = sorted(
        f for f in glob.glob(os.path.join(CURATED_DIR, "*_simulated.csv"))
        if not os.path.basename(f).startswith("_")
    )
    if not files:
        raise SystemExit(f"No '*_simulated.csv' files found in {CURATED_DIR}")
    frames = [pd.read_csv(f)[["composition", "Tc"]] for f in files]
    return pd.concat(frames, ignore_index=True), [os.path.basename(f) for f in files]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    pool, files = load_curated_simulated()
    print(f"Simulated (curated) — pooled {len(pool)} rows from {len(files)} files:")
    for f in files:
        print(f"    {f}")

    df_sim = process(pool, tc_col_out="Tc_sim")
    print(f"Simulated — total: {len(df_sim)}, "
          f"duplicates: {df_sim['composition'].duplicated().sum()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_all = df_sim.drop(columns="contains_re")
    out_all.to_csv(os.path.join(OUT_DIR, "Simulation_Tc_all.csv"), index=False)
    out_all.to_csv(os.path.join(OUT_DIR, "Simulated_Tc.csv"), index=False)  # alias copy

    df_sim_re, df_sim_re_free = split_re(df_sim)
    df_sim_re.to_csv(os.path.join(OUT_DIR, "Simulation_Tc_RE.csv"), index=False)
    df_sim_re_free.to_csv(os.path.join(OUT_DIR, "Simulation_Tc_RE-Free.csv"), index=False)

    print(f"Wrote Simulation_Tc_all / _RE / _RE-Free (+ Simulated_Tc) to {OUT_DIR}")


if __name__ == "__main__":
    main()
