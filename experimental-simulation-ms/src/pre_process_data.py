import pandas as pd
import numpy as np
import pdb

from typing import List, Tuple
from tqdm import tqdm

from pymatgen.core import Composition
from composition_data import CompositionData

import torch

import pandas as pd
import matplotlib.pyplot as plt

import re

def create_pairs_and_remove_small_ms_values(df, sim_col, exp_col, threshold):
    df_pairs_rm_ms_small = df[((df[exp_col] > threshold) & (df[sim_col] > threshold))]
    return df_pairs_rm_ms_small 

def remove_small_ms_values(df, sim_col, exp_col, threshold):
    df_rm = df[
                ((df[exp_col].isna()) | (df[exp_col] > threshold)) &
                ((df[sim_col].isna()) | (df[sim_col] > threshold))
            ]
    return df_rm

def create_pairwise_dataset(df, sim_col, exp_col):
    df_pairs = df[df[sim_col].notna() & df[exp_col].notna()].copy()
    return df_pairs

def create_to_augment_exp_val_dataset(df, sim_col, exp_col):
    return df[(df[sim_col].notna() & df[exp_col].isna())].copy()

def is_valid_composition(comp):
    """Return True if composition looks like a valid chemical formula."""
    
    invalids_comps = []
    if not isinstance(comp, str):
        return False
    comp = comp.strip()
    if comp == "":
        return False

    # # Original check (invalid characters including *)
    # if re.search(r"[xy\*\∗,()]", comp):  
    #     return False
    
    # New check: similar but allow '*' 
    if re.search(r"∗", comp):  # '*' removed
        return False

    # if re.search(r"[\-\−_]", comp):      
    #     return False
    # if re.match(r"^\d", comp):           
    #     return False
    # if "Alnico" in comp or "alloy" in comp or "del" in comp:
    #     return False
    # if not re.search(r"[A-Z]", comp):    
    #     return False
    return True

def add_mat200_embeddings_to_df(df: pd.DataFrame,
                                emb_col: str,
                                embedding_path: str,
                                task_dict: dict):

    CD = CompositionData(df=df,
                         task_dict=task_dict,
                         elem_embedding=embedding_path,
                         inputs="composition",
                         identifiers=("composition", "composition")) 
    
    # NOTE: we write composition instead of material_id as we do not have one 

    embedding_dict = {}
    failure_count = 0
    success_count = 0

    for idx in tqdm(range(len(CD)), desc="Embedding compositions"):
        try:
            composition = CD[idx]
        except Exception as e:
            print(f"[Data access error] Skipping CD[{idx}] due to: {e}")
            failure_count += 1
            continue

        try:
            (elem_weights, elem_feas, _, _), _, composition_string, _ = composition
        except Exception as e:
            print(f"[Unpacking error] Skipping CD[{idx}] due to: {e}")
            failure_count += 1
            continue

        try:
            comp_dict = Composition(composition_string).get_el_amt_dict()
        except Exception as e:
            print(f"[Invalid format] Skipping CD[{idx}] composition '{composition_string}' due to: {e}")
            failure_count += 1
            continue
        
        # Weighted mean pooling of element features
        comp_embedding = torch.zeros((1, 200))
        for i in range(elem_weights.shape[0]):
            comp_embedding += elem_weights[i] * elem_feas[i,:] 

        embedding_dict[composition_string] = comp_embedding.numpy()
        success_count += 1

    # Map the computed embeddings to the dataframe
    df[emb_col] = df['composition'].map(embedding_dict)

    print(f"\n Embedding generation complete.")
    print(f"Successes: {success_count}")
    print(f"Failures: {failure_count}")
    print(f"Missing embeddings in dataframe: {(df[emb_col].isnull()).sum()}")

    return df

def scatter_plot_raw_data(df, sim_col, exp_col, 
                          mask_re, mask_re_free,
                          plot_path):
    
    # recompute nr samples
    nr_re_samples = df[sim_col][mask_re].shape[0]
    nr_re_free_samples = df[sim_col][mask_re_free].shape[0]
    
    plt.figure(figsize=(5,5))
    plt.scatter(df[sim_col][mask_re], 
                df[exp_col][mask_re],
                color='blue', alpha=0.5, label=f'RE, {nr_re_samples} samples')
    plt.scatter(df[sim_col][mask_re_free],
                df[exp_col][mask_re_free],
                color='orange', alpha=0.5, label=f'RE-Free, {nr_re_free_samples} samples')
    plt.grid(True)
    plt.legend()
    plt.xlabel('Ms_s (A/m)')
    plt.ylabel('Ms_e (A/m)')
    plt.title('All')
    plt.savefig(plot_path + f'data_all_{df.shape[0]}_samples.png')
    plt.close()

    plt.figure(figsize=(5,5))
    plt.scatter(df[sim_col][mask_re], 
                df[exp_col][mask_re],
                color='blue', label=f'RE, {df[sim_col][mask_re].shape[0]} samples')
    plt.grid(True)
    plt.xlabel('Ms_s (A/m)')
    plt.ylabel('Ms_e (A/m)')
    plt.title('RE')
    plt.legend()
    plt.savefig(plot_path + f'data_re_{df[sim_col][mask_re].shape[0]}_samples.png')
    plt.close()

    plt.figure(figsize=(5,5))
    plt.scatter(df[sim_col][mask_re_free], 
                df[exp_col][mask_re_free],
                color='orange', label=f'RE-Free, {df[sim_col][mask_re_free].shape[0]} samples')
    plt.grid(True)
    plt.xlabel('Ms_s (A/m)')
    plt.ylabel('Ms_e (A/m)')
    plt.title(f'RE-Free, {df[sim_col][mask_re_free].shape[0]} samples')
    plt.legend()
    plt.savefig(plot_path + f'data_re_free_{df[sim_col][mask_re_free].shape[0]}_samples.png')
    plt.close()

    
def scatter_plot_data(x, y, title, plot_path, color, nr_samples):
    
    plt.figure(figsize=(5,5))
    plt.scatter(x, 
                y,
                color=color, 
                alpha=0.5, 
                label=f'{title}, {nr_samples} samples')
    
    plt.grid(True)
    plt.legend()
    plt.xlabel('Ms_s (A/m)')
    plt.ylabel('Ms_e (A/m)')
    plt.title(title)
    plt.savefig(plot_path)
    plt.close()