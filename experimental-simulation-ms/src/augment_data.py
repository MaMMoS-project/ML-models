# Standard libraries
import pandas as pd
import numpy as np

import yaml
import shutil
import joblib
import os 

# Plotting / visualizing
import seaborn as sns
import matplotlib.pyplot as plt

# Distributions 
from scipy.stats import norm, laplace, rv_histogram
import statsmodels

from pre_process_data import create_pairwise_dataset, create_to_augment_exp_val_dataset

import argparse

# Custom modules
import mammos_entity as me
import mammos_units as u

import pdb

sns.set_style("whitegrid") 


def scatter_plot_augmented_data(real: pd.DataFrame, 
                                mock: pd.DataFrame, 
                                label: str,
                                save_dir: str):
    pass

def histogram_plot_delta(delta: pd.Series,
                         bins: int,
                         label: str,
                         save_dir: str):

    plt.figure(figsize=(8,6), dpi=100)
    plt.hist(delta, bins=bins, label=label)
    plt.xlabel("ΔMs (A/m)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + f'{label}_deltas.png')
    plt.close()

def bootstrap_mock_values(df_to_augment: pd.DataFrame, 
                          df_pairs: pd.DataFrame, 
                          sim_col: str,
                          delta_col: str,
                          mock_flag: str,
                          threshold: float = 0.):
    
    mask = df_to_augment[mock_flag]
    n = mask.sum()

    # print(f"Nr of exp vals to sample {mock_flag}:", n)

    mock_data = np.full(n, -np.inf)

    zeros = np.zeros(n)

    data = df_to_augment.loc[mask, sim_col].values

    while np.any(mock_data < zeros):

        # indices that still need updating
        cond = mock_data <= threshold

        # Bootstrap sample residuals for only those positions
        sampled_residuals = np.random.choice(df_pairs[delta_col].values.ravel(), 
                                             size=cond.sum(), replace=True)

        # Update only the entries in mock_all that are still < 0
        mock_data[cond] = data[cond] + sampled_residuals
    
    return mock_data

def augment_data_experimental(config):
    
    print('------ Augment experimental values ------')
     
    ARTIFACT_PATH = config['artifact_paths']
    DATA_PATH = config['DATA_DIR'] 
    TO_AUGMENT = config['TO_AUGMENT']
    SAVE_AUGMENTED_DATA = config['SAVE_AUGMENTED_DATA']
    
    sim_col = config['sim_col']
    exp_col = config['exp_col']
    
    threshold = config['threshold']

    # === Make sure directories exist ===
    os.makedirs(ARTIFACT_PATH['root'], exist_ok=True)

    # === Load dataset ===
    df = pd.read_csv(TO_AUGMENT, index_col=0)  
    
    # Filter pairs
    df_pairs = create_pairwise_dataset(df, sim_col, exp_col)
        
    # Data to augment
    df_augmented = create_to_augment_exp_val_dataset(df, sim_col, exp_col) 
    
    # calc delta
    delta_col = 'exp-minus-sim'
    df_pairs[delta_col] = df_pairs[exp_col] - df_pairs[sim_col] 
    
    ### Define RE and RE-Free Datasets
    mask_re_pairs = df_pairs['has_rare_earth']
    mask_re_free_pairs = mask_re_pairs != 1

    df_pairs_re = df_pairs[mask_re_pairs.values].copy()
    df_pairs_re_free = df_pairs[mask_re_free_pairs.values].copy()

    # Plote delta histograms
    os.makedirs(ARTIFACT_PATH['data_plots'], exist_ok=True)
    histogram_plot_delta(df_pairs_re[delta_col], 70,
                         'RE', ARTIFACT_PATH['data_plots'])
    
    histogram_plot_delta(df_pairs_re_free[delta_col], 70,
                         'RE-Free', ARTIFACT_PATH['data_plots'])
        
    mask_all_to_augment = df_augmented[sim_col].notna() & df_augmented[exp_col].isna() 
    mask_re_to_augment =  df_augmented[sim_col].notna() & df_augmented['has_rare_earth'] & df_augmented[exp_col].isna()
    mask_re_free_to_augment =  df_augmented[sim_col].notna() & (mask_re_to_augment != 1 ) & df_augmented[exp_col].isna()

    # Add label for mock data
    df_augmented['is_mock_all'] = mask_all_to_augment
    df_augmented['is_mock_re'] = mask_re_to_augment
    df_augmented['is_mock_re_free'] = mask_re_free_to_augment 

    print('Pairs All Materials:', df_pairs.shape[0])
    print('Pair RE Materials', df_pairs_re.shape[0])
    print('Pair RE-Free Materials', df_pairs_re_free.shape[0])  
    print('-------------------------------------------------------------------------')  
    
    print('Before mock data generation, exp-val NaNs in ALL:', df_augmented.loc[mask_all_to_augment, 
                                                                'Ms (ampere/meter)_e'].isna().sum())

    print('Before mock data generation, exp-val NaNs in RE:', df_augmented.loc[mask_re_to_augment, 
                                                              'Ms (ampere/meter)_e'].isna().sum())
    
    print('Before mock data generation, exp-val NaNs in RE-Free:', df_augmented.loc[mask_re_free_to_augment, 
                                                                   'Ms (ampere/meter)_e'].isna().sum())
    print('-------------------------------------------------------------------------')  
    
    print('Number of measured experimental values before mock data generation:')
    # ALL
    print('ALL:', df[exp_col].notna().sum())

    # RE
    mask_re = df['has_rare_earth'] 
    print('RE:', df.loc[mask_re, exp_col].notna().sum())

    # RE-Free
    mask_re_free = df_augmented['has_rare_earth'] == False

    # Count non-NaN experimental values
    print('RE-Free:', df_augmented.loc[mask_re_free, exp_col].notna().sum())
    print('-------------------------------------------------------------------------')  
    
    # Augment missing experimental values per group
    # ------ All ------
    mock_all = bootstrap_mock_values(df_augmented, df_pairs, 
                                     sim_col, delta_col, 
                                     'is_mock_all', 
                                     threshold=threshold)
    
    df_augmented.loc[mask_all_to_augment, 'Ms (ampere/meter)_e_mock_all'] = mock_all
    

    # ------ RE ------
    mock_re = bootstrap_mock_values(df_augmented, df_pairs, sim_col, delta_col, 
                                    'is_mock_re', threshold=threshold)

    df_augmented.loc[df_augmented['is_mock_re'], 'Ms (ampere/meter)_e_mock_re'] = mock_re
    

    # ------ RE-Free ------
    mock_re_free = bootstrap_mock_values(df_augmented, df_pairs, 
                                         sim_col, delta_col, 
                                         'is_mock_re_free', threshold=threshold)
    
    df_augmented.loc[df_augmented['is_mock_re_free'], 'Ms (ampere/meter)_e_mock_re_free'] = mock_re_free

    # Store info string on augmented data in csv TODO
    # info_row = pd.DataFrame({
    # 'info': [
    #     '# Info: Use df[df["contains_rare_earth"]] for RE data, '
    #     'and ~[...] for RE-free data. '
    #     'The columns Ms_delta_all, Ms_delta_re, and Ms_delta_re_free contain deltas '
    #     'calculated on the augmented data.'
    # ]})

    # pd.concat([info_row, df_to_augment], ignore_index=True).to_csv(
    #     SAVE_DATA_DIR,
    #     index=False
    # )

    df_combined = pd.concat([df_augmented, df_pairs], ignore_index=True)
    
   # Boolean mask of duplicate rows
    duplicates_mask = df_combined.duplicated()

    # Number of duplicate rows
    num_duplicates = duplicates_mask.sum()
    print(f"Number of duplicate rows: {num_duplicates}")
    df_combined['Ms (ampere/meter)_e_all_augmented'] = df_combined['Ms (ampere/meter)_e_mock_all'].combine_first(
        df_combined['Ms (ampere/meter)_e']
    )

    df_combined['Ms (ampere/meter)_e_re_augmented'] = df_combined['Ms (ampere/meter)_e_mock_re'].combine_first(df_combined['Ms (ampere/meter)_e'])

    df_combined['Ms (ampere/meter)_e_re_free_augmented'] = df_combined['Ms (ampere/meter)_e_mock_re_free'].combine_first(
        df_combined['Ms (ampere/meter)_e']
    )

    df_combined_re = df_combined[df_combined['has_rare_earth']].copy()
    df_combined_re['exp-minus-sim'] = df_combined['Ms (ampere/meter)_e_re_augmented'] - df_combined_re['Ms (ampere/meter)_s']

    df_combined_re_free = df_combined[df_combined['has_rare_earth']==False].copy()
    df_combined_re_free['exp-minus-sim'] = df_combined_re_free['Ms (ampere/meter)_e_re_free_augmented'] - df_combined_re_free['Ms (ampere/meter)_s']
  
    print("Shape df_augmented:", df_combined.shape[0])
    print("RE-Materials after augmentation:", df_combined_re.shape[0])
    print("RE-Free Materials after augmentation:", df_combined_re_free.shape[0])
    
    df_combined.to_csv(SAVE_AUGMENTED_DATA)
    df_combined_re = df_combined_re[['composition', 'material_id', 'Ms (ampere/meter)_s', 'has_rare_earth',
                                     'Ms (ampere/meter)_e_re_augmented', 'exp-minus-sim']].copy()
    
    df_combined_re.to_csv('data/re_augmented_data.csv')

    df_combined_re_free = df_combined_re_free[['composition', 'material_id', 'Ms (ampere/meter)_s', 'has_rare_earth',
                                               'Ms (ampere/meter)_e_re_free_augmented', 'exp-minus-sim']].copy()
    df_combined_re_free.to_csv('data/re_free_augmented_data.csv')
    print(df_combined.head)
