import pandas as pd
import numpy as np 

from src.load_data import load_data
import src.ms_aux

import pdb
import re

def merge_dataframes(dfs: dict, merge_key: str, how: str = "outer"):
    """
    Merge a dictionary of DataFrames on a shared key.
    Each DataFrame’s non-key columns are suffixed with its dict key.
    """
    merged_df = None

    for name, df in dfs.items():
        # Add dataset name as suffix to avoid collisions
        df = df.rename(columns={col: f"{col}_{name}" for col in df.columns if col != merge_key})

        if merged_df is None:
            merged_df = df.copy()
        else:
            merged_df = pd.merge(merged_df, df, on=merge_key, how=how)

    return merged_df

def combine_dataframes(dfs: dict[pd.DataFrame],
                       merge_key: str,
                       save_dir: str):
    """
    Combine loaded dataframes (both experimental and simulation) into one.
    Separately process exp (_e) and sim (_s) columns per dataframe, deduplicate, 
    filter invalid compositions, and add rare-earth info. Rows with unparsable 
    compositions are dropped.
    """

    if not dfs:
        return pd.DataFrame()

    exp_list = []
    sim_list = []

    for df in dfs.values():
        # Detect exp and sim columns for this dataframe
        df_exp_cols = [c for c in df.columns if "Ms (ampere/meter)" in c and "_e" in c]
        df_sim_cols = [c for c in df.columns if "Ms (ampere/meter)" in c and "_s" in c]

        # Experimental
        if df_exp_cols:
            df_exp = df[[merge_key] + df_exp_cols].copy()
            if len(df_exp_cols) > 1:
                df_exp["Ms (ampere/meter)_e"] = df_exp[df_exp_cols].median(axis=1, skipna=True)
            else:
                df_exp["Ms (ampere/meter)_e"] = df_exp[df_exp_cols[0]]
            exp_list.append(df_exp[[merge_key, "Ms (ampere/meter)_e"]])

        # Simulation
        if df_sim_cols:
            df_sim = df[[merge_key] + df_sim_cols].copy()
            if len(df_sim_cols) > 1:
                df_sim["Ms (ampere/meter)_s"] = df_sim[df_sim_cols].median(axis=1, skipna=True)
            else:
                df_sim["Ms (ampere/meter)_s"] = df_sim[df_sim_cols[0]]
            sim_list.append(df_sim[[merge_key, "Ms (ampere/meter)_s"]])

    # Concatenate all exp and sim dfs
    final_exp = pd.concat(exp_list, ignore_index=True) if exp_list else None
    final_sim = pd.concat(sim_list, ignore_index=True) if sim_list else None

    # Deduplicate by composition
    if final_exp is not None:
        final_exp = final_exp.groupby(merge_key, as_index=False).median(numeric_only=True)
    if final_sim is not None:
        final_sim = final_sim.groupby(merge_key, as_index=False).median(numeric_only=True)

    # Merge experimental and simulation
    if final_exp is not None and final_sim is not None:
        final_df = pd.merge(final_exp, final_sim, on=merge_key, how="outer")
    elif final_exp is not None:
        final_df = final_exp
        final_df["Ms (ampere/meter)_s"] = np.nan
    elif final_sim is not None:
        final_df = final_sim
        final_df["Ms (ampere/meter)_e"] = np.nan
    else:
        return pd.DataFrame()

    # Count rows where both exp and sim exist
    both_mask = final_df["Ms (ampere/meter)_e"].notna() & final_df["Ms (ampere/meter)_s"].notna()
    print("Number of rows with both experimental and simulation values:", both_mask.sum())

    # Generate fake material IDs
    final_df["material_id"] = np.random.randint(1, 1001, size=len(final_df))
    
    # Add rare-earth info (drop rows with unparsable compositions)
    def safe_has_rare_earth(comp):
        try:
            return ms_aux.has_rare_earth(comp)
        except Exception:
            return None  # mark unparseable compositions as None

    final_df["has_rare_earth"] = final_df[merge_key].apply(safe_has_rare_earth)

    # Drop rows where rare-earth info could not be computed
    final_df = final_df[final_df["has_rare_earth"].notna()].reset_index(drop=True)

    # Save CSV
    final_df.to_csv(save_dir + 'merged_df_python.csv', index=False)
    print('Combined DF saved to:', save_dir + 'merged_df_python.csv')
    print('Nr of simulated values:', final_df["Ms (ampere/meter)_s"].notna().sum())
    print('Nr of experimental values:', final_df["Ms (ampere/meter)_e"].notna().sum())
    
    print('Combined DF RE Materials:', final_df[final_df['has_rare_earth']].shape[0])
    print('Combined DF RE-Free Materials:', final_df[final_df['has_rare_earth']==False].shape[0])

    return final_df