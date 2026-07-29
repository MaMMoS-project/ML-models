#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pre-process the raw sources into the SIMULATED saturation-magnetisation dataset.

Stage-1 of the compound-to-simulated-ms pipeline (analogous to
``compound-to-simulated-tc/src/process_tc_data.py``, but for Ms and specialised to the
simulated target only — this project trains a model that predicts the simulated Ms directly
from a compound formula).

It reads the raw SIMULATED Ms sources from ``data/``, converts every source to a common unit
(ampere/metre, A/m), pools all values for a composition and reduces them with a SINGLE median
(matching my_ms/src/build_merged_dataset.py and the compound-to-tc projects — no mean, no
median-of-medians, no per-source pre-averaging), flags rare-earth membership with pymatgen,
drops compositions whose formula pymatgen cannot parse, and writes:

    preprocessed_data/Simulated_Ms.csv          composition, Ms, contains_re
    preprocessed_data/Simulated_Ms_all.csv      composition, Ms
    preprocessed_data/Simulated_Ms_RE.csv       composition, Ms   (rare-earth only)
    preprocessed_data/Simulated_Ms_RE-Free.csv  composition, Ms   (rare-earth free)

Ms is the simulated (DFT) saturation magnetisation in A/m.

Raw SIMULATED sources (in data/) and their conversion to A/m:
    oqmd_stable.csv           csv      'composition'  'Ms' [T, = mu0*M]     / MU_0
    Bhandari_XII_sim.csv      sep=';'  'material'     'Ms (A/m)'            (already A/m)
    mp_fm_dedup_sim_data.csv  csv      'formula_pretty'
                              'total_magnetization_normalized_vol' [mu_B/A^3] * MU_B / A^3

Usage:
    python -m src.preprocess_ms_data
    python src/preprocess_ms_data.py --data-dir data --out-dir preprocessed_data
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymatgen.core import Composition

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MU_0 = 4e-7 * np.pi          # vacuum permeability [T*m/A]
MU_B = 9.2740100657e-24      # Bohr magneton [J/T]
ANGSTROM = 1e-10             # [m]

KEY = "composition"
MS_COL = "Ms"                # simulated saturation magnetisation [A/m]

# Rare earths: Sc, Y and the lanthanides La..Lu.
RARE_EARTHS = {
    "Sc", "Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd",
    "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
}


def has_rare_earth(formula) -> "bool | None":
    """True/False if *formula* contains a rare-earth element, or None if pymatgen cannot
    parse it (used both to flag RE and to drop non-fixed-formula placeholder strings)."""
    try:
        elements = {str(e) for e in Composition(str(formula)).elements}
    except Exception:
        return None
    return bool(elements & RARE_EARTHS)


def reduced_formula(formula) -> "str | None":
    """Canonical reduced formula for deduplication (e.g. CoFe2O4 -> Fe2CoO4), or None if the
    string cannot be parsed. Used so different spellings of the same compound are pooled
    together instead of surviving as separate rows with conflicting Ms values."""
    try:
        return Composition(str(formula).strip()).reduced_formula
    except Exception:
        return None


# Ferrimagnet families that are essentially always ferrimagnetic and identifiable from the
# formula. Their DFT/collinear "Ms" is the UNCOMPENSATED sublattice sum, several-fold larger
# than the true net Ms, so by default they are dropped from the training set (see README /
# further_improvements.txt #2). IMPORTANT: composition CANNOT identify ferrimagnetism in
# general (it is a magnetic-structure property), so this is a high-confidence CURATED subset
# only -- to be extended later; the dataset may still contain UNKNOWN ferrimagnets.
_GARNET_A = {"Y", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er",
             "Tm", "Yb", "Lu", "Bi"}                 # iron-garnet A-site cations
_SPINEL_M = {"Mg", "Mn", "Co", "Ni", "Cu", "Zn", "Cd", "Li"}   # classic spinel-ferrite A-site


def is_known_ferrimagnet(formula) -> bool:
    """True for the high-confidence ferrimagnet families: magnetite Fe3O4, classic spinel
    ferrites MFe2O4, iron garnets R3Fe5O12, hexaferrites MFe12O19. Curated, NOT exhaustive."""
    try:
        d = Composition(str(formula)).get_el_amt_dict()
    except Exception:
        return False
    O = d.get("O", 0.0)
    Fe = d.get("Fe", 0.0)
    if O <= 0 or Fe <= 0:
        return False
    cats = sum(v for k, v in d.items() if k != "O")
    noFe = {k: v for k, v in d.items() if k not in ("O", "Fe")}
    # iron garnet R3Fe5O12 (Fe:O = 5:12; all non-Fe cations are garnet A-site)
    if abs(Fe / O - 5 / 12) < 1e-9 and abs(cats / O - 8 / 12) < 1e-9 \
            and noFe and all(k in _GARNET_A for k in noFe):
        return True
    # hexaferrite MFe12O19 (Fe:O = 12:19; M in Ba/Sr/Pb)
    if abs(Fe / O - 12 / 19) < 1e-9 and (set(noFe) & {"Ba", "Sr", "Pb"}):
        return True
    # spinel: magnetite Fe3O4, or classic MFe2O4 (single textbook divalent cation)
    if abs(cats / O - 3 / 4) < 1e-9:
        if abs(Fe - 3) < 1e-9 and not noFe:
            return True
        if abs(Fe - 2) < 1e-9 and len(noFe) == 1 and set(noFe) <= _SPINEL_M:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-source loaders -> DataFrame[composition, Ms]
# ---------------------------------------------------------------------------
def _frame(comp, values) -> pd.DataFrame:
    out = pd.DataFrame({KEY: comp.astype(str).str.strip(),
                        MS_COL: pd.to_numeric(values, errors="coerce")})
    return out.dropna(subset=[MS_COL])


def load_oqmd(d: Path) -> pd.DataFrame:            # Tesla (mu0*M) -> A/m
    df = pd.read_csv(d / "oqmd_stable.csv")
    return _frame(df["composition"], df["Ms"] / MU_0)


def load_bhandari_xii(d: Path) -> pd.DataFrame:    # already A/m
    df = pd.read_csv(d / "Bhandari_XII_sim.csv", sep=";")
    return _frame(df["material"], df["Ms (A/m)"])


def load_mp_sim(d: Path) -> pd.DataFrame:          # mu_B / Angstrom^3 -> A/m
    df = pd.read_csv(d / "mp_fm_dedup_sim_data.csv",
                     usecols=["formula_pretty", "total_magnetization_normalized_vol"])
    ms = df["total_magnetization_normalized_vol"] * MU_B / (ANGSTROM ** 3)
    return _frame(df["formula_pretty"], ms)


SIM_LOADERS = [load_oqmd, load_bhandari_xii, load_mp_sim]


# ---------------------------------------------------------------------------
# Aggregate, flag, split
# ---------------------------------------------------------------------------
def process(data_dir: Path, ms_threshold: float = 50_000.0,
            include_ferrimagnets: bool = False) -> pd.DataFrame:
    """Pool all simulated sources, canonicalise each formula to its reduced formula so
    variants of the same compound merge (e.g. CoFe2O4 / Fe2CoO4 -> Fe2CoO4), take a single
    median per reduced composition, drop compounds with Ms <= ms_threshold (as my_ms does;
    pass 0 to disable), and by default DROP known ferrimagnets (whose collinear DFT Ms is not
    the net Ms; pass include_ferrimagnets=True to keep them). Returns
    [composition, Ms, contains_re]."""
    pooled = pd.concat([ld(data_dir) for ld in SIM_LOADERS], ignore_index=True)

    # Canonicalise to reduced formula BEFORE deduplication, so different spellings of the
    # same compound are pooled together. Unparsable strings become NaN and are dropped.
    pooled[KEY] = pooled[KEY].map(reduced_formula)
    n_unparsable = int(pooled[KEY].isna().sum())
    pooled = pooled.dropna(subset=[KEY])

    # single pooled median per (reduced) composition
    df = pooled.groupby(KEY, as_index=False)[MS_COL].median()

    # Discard low-Ms compounds (near-zero / non-magnetic; DFT Ms unreliable there).
    if ms_threshold and ms_threshold > 0:
        n0 = len(df)
        df = df[df[MS_COL] > ms_threshold].reset_index(drop=True)
        df.attrs["thr_dropped"] = n0 - len(df)
    else:
        df.attrs["thr_dropped"] = 0

    # Known ferrimagnets: their DFT Ms is the uncompensated collinear moment, not the net Ms.
    ferri = df[KEY].map(is_known_ferrimagnet)
    df.attrs["n_ferri"] = int(ferri.sum())
    df.attrs["ferri_included"] = bool(include_ferrimagnets)
    if not include_ferrimagnets:
        df = df[~ferri].reset_index(drop=True)

    # reduced formulae are all parseable, so this is never None here.
    df["contains_re"] = df[KEY].map(has_rare_earth).astype(bool)
    df.attrs["dropped"] = n_unparsable
    return df


def split_re(df: pd.DataFrame):
    re_df   = df[df["contains_re"]].drop(columns="contains_re").reset_index(drop=True)
    re_free = df[~df["contains_re"]].drop(columns="contains_re").reset_index(drop=True)
    return re_df, re_free


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=str(root / "data"),
                   help="Directory with the raw source files (default: <project>/data).")
    p.add_argument("--out-dir", default=str(root / "preprocessed_data"),
                   help="Output directory (default: <project>/preprocessed_data).")
    p.add_argument("--ms-threshold", type=float, default=50_000.0,
                   help="Drop compounds with Ms <= this (A/m); 0 disables. "
                        "Default 50000, as in my_ms.")
    p.add_argument("--include-ferrimagnets", action="store_true",
                   help="Keep known ferrimagnets (default: drop them). Their DFT/collinear Ms "
                        "is the uncompensated sublattice sum, not the net saturation "
                        "magnetisation.")
    args = p.parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = process(data_dir, ms_threshold=args.ms_threshold,
                 include_ferrimagnets=args.include_ferrimagnets)
    re_df, re_free = split_re(df)

    df.drop(columns="contains_re").to_csv(out_dir / "Simulated_Ms_all.csv", index=False)
    df.to_csv(out_dir / "Simulated_Ms.csv", index=False)
    re_df.to_csv(out_dir / "Simulated_Ms_RE.csv", index=False)
    re_free.to_csv(out_dir / "Simulated_Ms_RE-Free.csv", index=False)

    print(f"Simulated Ms  — {len(df)} compositions "
          f"(Ms>{args.ms_threshold:.0f} A/m dropped {df.attrs['thr_dropped']}; "
          f"dropped {df.attrs['dropped']} unparsable)")
    print(f"  RE / RE-free   : {len(re_df)} / {len(re_free)}")
    print(f"  Ms [A/m] range : {df[MS_COL].min():.3e} .. {df[MS_COL].max():.3e} "
          f"(median {df[MS_COL].median():.3e})")
    n_ferri = df.attrs.get("n_ferri", 0)
    if args.include_ferrimagnets:
        print(f"  !! CAREFUL: {n_ferri} KNOWN ferrimagnets ARE INCLUDED — their Ms is the "
              f"collinear (uncompensated) moment, not the net magnetisation.")
    else:
        print(f"  Dropped {n_ferri} KNOWN ferrimagnets (magnetite / spinel ferrites / "
              f"garnets / hexaferrites).")
        print(f"  !! CAREFUL: the dataset MAY STILL CONTAIN UNKNOWN ferrimagnets — "
              f"composition cannot identify ferrimagnetism in general.")
    print(f"  wrote 4 files to {out_dir}/")


if __name__ == "__main__":
    main()
