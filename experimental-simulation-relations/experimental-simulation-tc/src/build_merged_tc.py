"""Create the merged Curie-temperature training dataset from the raw data sources.

Stage-0 of the my_tc pipeline (analogous to my_ms/src/build_merged_dataset.py). It
aggregates experimental and simulated Curie temperatures for magnetic compounds from six
raw sources into a single lean table, ``data/merged_curie_temp.csv``, which
``src/augment_data.py`` consumes.

Output schema (plain CSV):
    composition          chemical formula (str)
    Tc_sim               simulated Curie temperature [K]
    Tc_exp               experimental Curie temperature [K]
    contains_rare_earth  bool: composition contains a rare-earth element (Sc, Y, La..Lu)
    use_for_emb          bool: composition is a clean fixed formula (safe to embed)

Deduplication (the fix vs the old notebook):
    For each composition, ALL simulated values from EVERY sim source are pooled and
    reduced with a SINGLE median; likewise ALL experimental values from every exp source.
    One median, every source included, no per-source pre-averaging. This removes the old
    notebook's three problems: (1) mean-vs-median inconsistency (NEMAD-exp was averaged
    with mean, everything else with median), (2) median-of-median (it took a median/mean
    across columns that were themselves per-source medians), and (3) silently dropping
    the dftv2/spin/literature sources from the unified target via a priority combine.

Raw sources (in data/) and their columns:
    simulated (-> Tc_sim):
        m-tcsum_nur_new.csv    'T_C/theo [K]'                        (DFT v2)
        combinded_tables.xlsx  'T_C/theo [K]'                        (combined / RE-free)
        sd_tc_data.csv         'Tc_sim'                              (spin dynamics)
        MagneticMaterials_All.csv 'Curie' where Experimental==False  (NEMAD)
    experimental (-> Tc_exp):
        m-tcsum_nur_new.csv    'T_C_exp_[K]'
        combinded_tables.xlsx  'T_C_exp [K]'
        DS1+DS2.csv            'TC'
        literature_values_prepared.csv 'Tc'
        MagneticMaterials_All.csv 'Curie' where Experimental==True

Usage:
    python -m src.build_merged_tc          # writes data/merged_curie_temp.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent               # my-tc/src
OUT_DIR = HERE.parent / "data"                        # my-tc/data
RAW_DIR = OUT_DIR                                      # raw source files live in data/ too

KEY = "composition"
RARE_EARTHS = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb",
               "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Sc", "Y"]

# Junk / placeholder tokens: a composition containing any of these is not a clean fixed
# formula and is flagged use_for_emb=False (carried over verbatim from the notebook).
_REMOVAL_TOKENS = sorted([
    ",", "%", "x", "+", "_", ")", "□", "Gr", "ln", "with", "vacancy", "Td", "_5", "_bal",
    "-", "-x", "0.473.5", "0.7–0.3", "0.7FeTiO3–0.3Fe2O3", "1-", "1-y", "5 nm nanocrystals",
    "5Fe2O3·9H2O", "53.5–19.5–27.0", "53.5Ni–19.5Fe–27.0Ga", "54.0Ni–19.0Fe–27.0Ga",
    "55.0Ni–18.0Fe–27.0Ga", "57", "600", "79", "@", "Ba2Co2Cl2(C2O4)3·4H2O", "BaFeO3−y",
    "Benzo", "Bismuth", "Bx", "Bulk", "B2O3 · Fe2O3 · 4CoO", "B2O3 · Fe2O3 · 4CuO",
    "B2O3 · Fe2O3 · 4MgO", "B2O3 · Fe2O3 · 4NiO", "Cx", "Cy", "DAE", "Emim", "Fc", "Febal",
    "Ferrihydrite", "Fe0.6Co0.4)0.75Si0.05B0.20)94Nb4Gd2",
    "Fe0.6Co0.4)0.75Si0.05B0.20)94Nb4Y2", "Fe0.6Co0.4)0.75Si0.05B0.20)95Nb4Gd1",
    "Pr0.6Sr0.35□0.05MnO3", "Fe0.6Co0.4)0.75Si0.05B0.20)95Nb4Mo1",
    "Fe0.6Co0.4)0.75Si0.05B0.20)95Nb4Y1", "Fe0.6Co0.4)0.75Si0.05B0.20)95Nb4Zr1",
    "Fe2O3(0.15)(70Bi2O320ZnO10PbO)0.85", "Fe2O3(0.20)(70Bi2O320ZnO10PbO)0.80",
    "Fe2O3(0.25)(70Bi2O320ZnO10PbO)0.75", "Hollow", "Hx", "Iron", "L10", "L12", "Nano",
    "Ndx", "Nx", "Ny", "O4–0.7", "Orthorhombic", "PbFe0.5S1.5)1.16NbS2", "Permalloy", "Pdx",
    "Prx", "Pure", "Pu0.34Y0.66Sb (x=0.34)", "R", "Solid", "Tt", "UH3)0.85Nb0.15", "Unknown",
    "Y3Fe5O12", "YFe1․5Co2․5B", "YbxFe4Sb12", "YIG", "alnico", "annealed at 590°", "at.%",
    "bcc", "bilayer", "cementite", "chemically disordered", "cmdp", "co-doped", "cryst.",
    "defective", "diffused", "doped", "en", "ferri", "fcc", "hcp", "in", "meta",
    "nanoparticles", "oxidized", "p.", "pct", "phase", "ppm", "pz", "sarcopside", "sx",
    "under pressure", "with 10 at%", "with 8 at%", "x=0.1", "x=0.25", "≈7", "β", "βUSe2",
    "α", "δ", "η", "λ", "⋅", "•", "•2", "․", "ₓ", "ᵧ", "−x", "−y", "–", "~0.9", ")0.75)94",
    ")96", "(1-x)BiFeO3-xPb(Mg1/3Nb2/3)O3 (x=0.1)", "(0.85)YMnO3/0.15ZnFe2O4",
    "(0.90)YMnO3/0.10ZnFe2O4", "(0.95)YMnO3/0.05ZnFe2O4", "(CH3)3NH]", "(Fe1−yPdy)", "(ND4)",
    "(NH4)", "(TCNE]", "(U1−xTbx)", "(Y1−xUx)", "(x=0.005)", "/", ":", "[TCNE]", "[(CH3)3NH]",
    "·", "·  · 4", "·11.5", "·2", "·6.6",
])


def _frame(comp, values, col):
    """(composition, <col>) frame with numeric values, NaN dropped."""
    out = pd.DataFrame({KEY: comp.astype(str).str.strip(),
                        col: pd.to_numeric(values, errors="coerce")})
    return out.dropna(subset=[col])


def _nemad_numeric(series):
    """Parse the NEMAD 'Curie' column: strip ' K', extract the leading number,
    treat negatives as missing."""
    s = series.astype(str).str.replace(" K", "", regex=False)
    v = pd.to_numeric(s.str.extract(r"([-]?\d+\.?\d*)")[0], errors="coerce")
    v[v < 0] = np.nan
    return v


def load_sources(raw_dir: Path):
    """Return (list of sim frames, list of exp frames), each [composition, Tc_sim/Tc_exp]."""
    dftv2 = pd.read_csv(raw_dir / "m-tcsum_nur_new.csv")
    comb = pd.read_excel(raw_dir / "combinded_tables.xlsx")
    spin = pd.read_csv(raw_dir / "sd_tc_data.csv")
    ds12 = pd.read_csv(raw_dir / "DS1+DS2.csv")
    lit = pd.read_csv(raw_dir / "literature_values_prepared.csv")
    nem = pd.read_csv(raw_dir / "MagneticMaterials_All.csv")
    nem_tc = _nemad_numeric(nem["Curie"])
    nem_is_exp = nem["Experimental"] == True  # noqa: E712

    sim_frames = [
        _frame(dftv2["System"], dftv2["T_C/theo [K]"], "Tc_sim"),
        _frame(comb["System"], comb["T_C/theo [K]"], "Tc_sim"),
        _frame(spin[KEY], spin["Tc_sim"], "Tc_sim"),
        _frame(nem.loc[~nem_is_exp, "Material_Name"], nem_tc[~nem_is_exp], "Tc_sim"),
    ]
    exp_frames = [
        _frame(dftv2["System"], dftv2["T_C_exp_[K]"], "Tc_exp"),
        _frame(comb["System"], comb["T_C_exp [K]"], "Tc_exp"),
        _frame(ds12["Name"], ds12["TC"], "Tc_exp"),
        _frame(lit[KEY], lit["Tc"], "Tc_exp"),
        _frame(nem.loc[nem_is_exp, "Material_Name"], nem_tc[nem_is_exp], "Tc_exp"),
    ]
    return sim_frames, exp_frames


def build_merged_tc(raw_dir: Path = RAW_DIR, out_path: Path | None = None) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    out_path = Path(out_path) if out_path else OUT_DIR / "merged_curie_temp.csv"

    sim_frames, exp_frames = load_sources(raw_dir)

    # Single pooled median per composition, sim and exp independently, all sources.
    sim = (pd.concat(sim_frames, ignore_index=True)
           .groupby(KEY, as_index=False)["Tc_sim"].median())
    exp = (pd.concat(exp_frames, ignore_index=True)
           .groupby(KEY, as_index=False)["Tc_exp"].median())

    merged = pd.merge(sim, exp, on=KEY, how="outer")

    # normalise subscript digits in composition strings
    subscript_map = str.maketrans(
        {s: n for s, n in zip("₀₁₂₃₄₅₆₇₈₉", "0123456789")}
    )
    merged[KEY] = merged[KEY].str.translate(subscript_map)

    # rare-earth flag (substring match on the formula string, as in the notebook)
    merged["contains_rare_earth"] = merged[KEY].apply(
        lambda c: any(el in c for el in RARE_EARTHS)
    )
    # clean-formula flag for embedding
    pattern = "|".join(map(re.escape, _REMOVAL_TOKENS))
    merged["use_for_emb"] = ~merged[KEY].str.contains(pattern, regex=True)

    merged = merged[[KEY, "Tc_sim", "Tc_exp", "contains_rare_earth", "use_for_emb"]]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)

    both = merged["Tc_sim"].notna() & merged["Tc_exp"].notna()
    print(f"Wrote {out_path}  ({len(merged)} rows)")
    print(f"  Tc_sim values : {int(merged['Tc_sim'].notna().sum())}")
    print(f"  Tc_exp values : {int(merged['Tc_exp'].notna().sum())}")
    print(f"  pairs (both)  : {int(both.sum())}")
    print(f"  RE / RE-free  : {int(merged['contains_rare_earth'].sum())} / "
          f"{int((~merged['contains_rare_earth']).sum())}")
    print(f"  use_for_emb   : {int(merged['use_for_emb'].sum())}")
    return merged


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", default=str(RAW_DIR), help="Directory with the raw sources.")
    p.add_argument("--out", default=None, help="Output CSV (default: data/merged_curie_temp.csv).")
    args = p.parse_args()
    build_merged_tc(Path(args.raw_dir), Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
