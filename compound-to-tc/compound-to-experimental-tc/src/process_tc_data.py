#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RARE_EARTH_ELEMENTS = [
    "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Sc", "Y",
]


# ---------------------------------------------------------------------------
# Helper functions
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


def extract_tc_rows(
    source: pd.DataFrame,
    composition_col: str,
    tc_col: str,
) -> pd.DataFrame:
    """Return a tidy (composition, Tc) DataFrame from *source*, dropping NaN Tc rows."""
    mask = source[tc_col].notna()
    return pd.DataFrame({
        "composition": source.loc[mask, composition_col].values,
        "Tc": source.loc[mask, tc_col].values,
    })


# ---------------------------------------------------------------------------
# Load and aggregate data
# ---------------------------------------------------------------------------

def load_all_experimental() -> pd.DataFrame:
    frames = []

    # Uppsala data
    dft = pd.read_csv("./data/m-tcsum_nur_new.csv")
    frames.append(extract_tc_rows(dft, "System", "T_C_exp_[K]"))

    # Literature data
    lit = pd.read_csv("./data/literature_values_prepared.csv")
    frames.append(extract_tc_rows(lit, "composition", "Tc"))

    # DS1 + DS2
    ds = pd.read_csv("./data/DS1+DS2.csv")
    frames.append(extract_tc_rows(ds, "Name", "TC"))

    # Combined RE-free tables
    combined = pd.read_excel("./data/combinded_tables.xlsx")
    frames.append(extract_tc_rows(combined, "System", "T_C_exp [K]"))

    # NEMAD (experimental only)
    nemad = pd.read_csv("./data/MagneticMaterials_All.csv")
    mask_nemad = (nemad["Experimental"] == True) & nemad["Curie"].notna()
    frames.append(pd.DataFrame({
        "composition": nemad.loc[mask_nemad, "Material_Name"].values,
        "Tc": nemad.loc[mask_nemad, "Curie"].values,
    }))

    return pd.concat(frames, ignore_index=True)


def load_all_simulated() -> pd.DataFrame:
    frames = []

    # Uppsala data
    dft = pd.read_csv("./data/m-tcsum_nur_new.csv")
    frames.append(extract_tc_rows(dft, "System", "T_C/theo [K]"))

    # Combined RE-free tables
    combined = pd.read_excel("./data/combinded_tables.xlsx")
    frames.append(extract_tc_rows(combined, "System", "T_C/theo [K]"))

    # NEMAD (simulated only)
    nemad = pd.read_csv("./data/MagneticMaterials_All.csv")
    mask_nemad = (nemad["Experimental"] == False) & nemad["Curie"].notna()
    frames.append(pd.DataFrame({
        "composition": nemad.loc[mask_nemad, "Material_Name"].values,
        "Tc": nemad.loc[mask_nemad, "Curie"].values,
    }))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Clean, deduplicate, and split
# ---------------------------------------------------------------------------

def process(df: pd.DataFrame, tc_col_out: str) -> pd.DataFrame:
    """Clean Tc strings, drop non-numeric rows, deduplicate by median."""
    df = df.copy()
    df["Tc"] = clean_tc_column(df["Tc"])

    # Drop rows that are still non-numeric after cleaning
    df = df[df["Tc"].notna()]

    # Drop physically impossible negative Curie temperatures as early as
    # possible — before deduplication — so they never contribute to the
    # per-composition median below.
    df = df[df["Tc"] >= 0]

    # Deduplicate: take median Tc per composition
    df = df.groupby("composition", as_index=False)["Tc"].median()
    df = df.rename(columns={"Tc": tc_col_out})

    # Annotate rare-earth membership
    df["contains_re"] = df["composition"].apply(contains_rare_earth)

    return df


def split_re(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (RE-containing, RE-free) DataFrames, both without the flag column."""
    re_df     = df[df["contains_re"]].drop(columns="contains_re").reset_index(drop=True)
    re_free   = df[~df["contains_re"]].drop(columns="contains_re").reset_index(drop=True)
    return re_df, re_free


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Experimental ---
    df_exp = process(load_all_experimental(), tc_col_out="Tc_exp")
    print(f"Experimental  — total: {len(df_exp)}, "
          f"duplicates: {df_exp['composition'].duplicated().sum()}")

    df_exp.drop(columns="contains_re").to_csv("./preprocessed_data/Experimental_Tc_all.csv", index=False)
    df_exp.to_csv("./preprocessed_data/Experimental_Tc.csv", index=False)

    df_exp_re, df_exp_re_free = split_re(df_exp)
    df_exp_re.to_csv("./preprocessed_data/Experimental_Tc_RE.csv", index=False)
    df_exp_re_free.to_csv("./preprocessed_data/Experimental_Tc_RE-Free.csv", index=False)

    # --- Simulated ---
    df_sim = process(load_all_simulated(), tc_col_out="Tc_sim")
    print(f"Simulated     — total: {len(df_sim)}, "
          f"duplicates: {df_sim['composition'].duplicated().sum()}")

    df_sim_out = df_sim.drop(columns="contains_re")
    df_sim_out.to_csv("./preprocessed_data/Simulation_Tc_all.csv", index=False)
    df_sim_out.to_csv("./preprocessed_data/Simulated_Tc.csv", index=False)

    df_sim_re, df_sim_re_free = split_re(df_sim)
    df_sim_re.to_csv("./preprocessed_data/Simulation_Tc_RE.csv", index=False)
    df_sim_re_free.to_csv("./preprocessed_data/Simulation_Tc_RE-Free.csv", index=False)


if __name__ == "__main__":
    main()
