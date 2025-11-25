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

from pre_process_data import filter_pairs

import argparse

# Custom modules
import mammos_entity as me
import mammos_units as u

import pdb

sns.set_style("whitegrid") 


def plot_histograms_delta(df: pd.DataFrame,
                          save_dir: str):

    def optimal_bins(data):
        data = np.asarray(data)
        q75, q25 = np.percentile(data, [75 ,25])
        iqr = q75 - q25
        n = len(data)
        bin_width = 2 * iqr / (n ** (1/3))
        if bin_width == 0:  # fallback if IQR = 0
            return int(np.sqrt(n))
        bins = int(np.ceil((data.max() - data.min()) / bin_width))
        return bins

    delta_col = 'exp-minus-sim'
    mask_re = df['has_rare_earth']
    mask_re_free = mask_re != 1

    ### Define RE and RE-Free Datasets
    df_pairs_re = df[mask_re.values]
    df_pairs_re_free = df[mask_re_free.values]

    # combine all delta values to compute a common bin size
    all_data = np.concatenate([
        df[delta_col].values,
        df_pairs_re[delta_col].values,
        df_pairs_re_free[delta_col].values
    ])

    bins = optimal_bins(all_data)

    # Plotting
    plt.figure(figsize=(8,6), dpi=100)
    plt.hist(df[delta_col].values.flatten(), bins=bins, alpha=0.8, label=f"All - {df[delta_col].shape[0]} samples")
    plt.hist(df_pairs_re[delta_col].values.flatten(), bins=bins, color='orange', alpha=0.8, label=f"RE - {df_pairs_re[delta_col].shape[0]} samples")
    plt.hist(df_pairs_re_free[delta_col].values.flatten(), bins=bins, color='green', alpha=0.5, label=f"RE-free - {df_pairs_re_free[delta_col].shape[0]} samples")

    plt.xlim(-0.25e6, 1.5e6)  
    plt.xlabel("Δ Ms (K)") 
    plt.ylabel("Frequency")
    plt.title("Distribution of Δ Ms (A/m)")  
    plt.legend()
    plt.grid(True)
    plt.savefig(save_dir + 'deltas_histogram_plot.png')
    plt.close()

def scatter_plot_augmented_data(real: pd.DataFrame, 
                                mock: pd.DataFrame, 
                                label: str,
                                save_dir: str):
    pass

def histogram_plot_augmented_data(real: pd.DataFrame, 
                                  mock: pd.DataFrame, 
                                  label: str,
                                  save_dir: str):

    pass

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
     
    ARTIFACT_PATH = config['artifact_paths']
    DATA_PATH = config['data_path'] 
    SAVE_DATA_DIR = config['save_data_dir']

    sim_col = config['sim_col']
    exp_col = config['exp_col']
    
    threshold = config['threshold']

    # === Make sure directories exist ===
    os.makedirs(ARTIFACT_PATH['root'], exist_ok=True)

    # === Load dataset ===
    df = pd.read_csv(DATA_PATH, index_col=0)  
    
    # Filter pairs
    df_pairs = filter_pairs(df, sim_col, exp_col)
        
    # Data to augment
    df_to_augment = df[df[sim_col].notna() & df[exp_col].isna()].copy()

    # calc delta
    delta_col = 'exp-minus-sim'
    df_pairs[delta_col] = df_pairs[exp_col] - df_pairs[sim_col] 

    plot_histograms_delta(df_pairs, ARTIFACT_PATH['data_plots'])

    ### Define RE and RE-Free Datasets
    mask_re_pairs = df_pairs['has_rare_earth']
    mask_re_free_pairs = mask_re_pairs != 1

    df_pairs_re = df_pairs[mask_re_pairs.values].copy()
    df_pairs_re_free = df_pairs[mask_re_free_pairs.values].copy()
    
    mask_all_to_augment = df_to_augment[sim_col].notna() & df_to_augment[exp_col].isna() 
    mask_re_to_augment =  df_to_augment[sim_col].notna() & df_to_augment['has_rare_earth'] & df_to_augment[exp_col].isna()
    mask_re_free_to_augment =  df_to_augment[sim_col].notna() & (mask_re_to_augment != 1 ) & df_to_augment[exp_col].isna()

    # Add label for mock data
    df_to_augment['is_mock_all'] = mask_all_to_augment
    df_to_augment['is_mock_re'] = mask_re_to_augment
    df_to_augment['is_mock_re_free'] = mask_re_free_to_augment 

    print('Pairs All Materials:', df_pairs.shape[0])
    print('Pair RE Materials', df_pairs_re.shape[0])
    print('Pair RE-Free Materials', df_pairs_re_free.shape[0])    
    
    print('Before mock data generation, exp-val NaNs in ALL:', df_to_augment.loc[mask_all_to_augment, 
                                                                'Ms (ampere/meter)_e'].isna().sum())

    print('Before mock data generation, exp-val NaNs in RE:', df_to_augment.loc[mask_re_to_augment, 
                                                              'Ms (ampere/meter)_e'].isna().sum())
    
    print('Before mock data generation, exp-val NaNs in RE-Free:', df_to_augment.loc[mask_re_free_to_augment, 
                                                                   'Ms (ampere/meter)_e'].isna().sum())
    
    print('Number of non-NaN experimental values before mock:')
    # ALL
    print('ALL:', df[exp_col].notna().sum())

    # RE
    mask_re = df['has_rare_earth'] 
    print('RE:', df.loc[mask_re, exp_col].notna().sum())

    # RE-Free
    mask_re_free = df_to_augment['has_rare_earth'] == False

    # Count non-NaN experimental values
    print('RE-Free:', df_to_augment.loc[mask_re_free, exp_col].notna().sum())

    # Augment missing experimental values per group
    # ------ All ------
    mock_all = bootstrap_mock_values(df_to_augment, df_pairs, 
                                     sim_col, delta_col, 'is_mock_all', threshold=threshold)
    df_to_augment.loc[mask_all_to_augment, 'Ms (ampere/meter)_e_mock_all'] = mock_all
    
    df_to_augment['Ms (ampere/meter)_e_all_w_mock'] = df_to_augment['Ms (ampere/meter)_e'].fillna(
    df_to_augment['Ms (ampere/meter)_e_mock_all']
)
        
    # add delta column
    df_to_augment['Ms_delta_all'] = df_to_augment['Ms (ampere/meter)_e_all_w_mock'] - df_to_augment[sim_col]

    # ------ RE ------
    mock_re = bootstrap_mock_values(df_to_augment, df_pairs, sim_col, delta_col, 
                                    'is_mock_re', threshold=threshold)
    
    df_to_augment.loc[df_to_augment['is_mock_re'], 'Ms (ampere/meter)_e_mock_re'] = mock_re
    
    # Fill NaNs in the original column only for RE rows
    df_to_augment['Ms (ampere/meter)_e_re_w_mock'] = df_to_augment['Ms (ampere/meter)_e']
    df_to_augment.loc[df_to_augment['is_mock_re'], 'Ms (ampere/meter)_e_re_w_mock'] = \
        df_to_augment.loc[df_to_augment['is_mock_re'], 'Ms (ampere/meter)_e'].fillna(
            df_to_augment.loc[df_to_augment['is_mock_re'], 'Ms (ampere/meter)_e_mock_re']
        )
    
    # add delta column
    df_to_augment['Ms_delta_re'] = df_to_augment['Ms (ampere/meter)_e_re_w_mock'] - df_to_augment[sim_col]
    
    # ------ RE-Free ------
    mock_re_free = bootstrap_mock_values(df_to_augment, df_pairs, 
                                         sim_col, delta_col, 
                                         'is_mock_re_free', threshold=threshold)
    
    df_to_augment.loc[df_to_augment['is_mock_re_free'], 'Ms (ampere/meter)_e_mock_re_free'] = mock_re_free
    df_to_augment['Ms (ampere/meter)_e_re_free_w_mock'] = df_to_augment['Ms (ampere/meter)_e']
    df_to_augment.loc[df_to_augment['is_mock_re_free'], 'Ms (ampere/meter)_e_re_free_w_mock'] = \
        df_to_augment.loc[df_to_augment['is_mock_re_free'], 'Ms (ampere/meter)_e'].fillna(
            df_to_augment.loc[df_to_augment['is_mock_re_free'], 'Ms (ampere/meter)_e_mock_re_free']
        )
    
    # add delta column
    df_to_augment['Ms_delta_re_free'] = df_to_augment['Ms (ampere/meter)_e_re_free_w_mock'] - df_to_augment[sim_col]
    
    print("Nr of samples after augmentation ALL:", df_to_augment['Ms_delta_all'].notna().sum())
    print("Nr of samples after augmentation RE:", df_to_augment['Ms_delta_re'].notna().sum()) 
    print("Nr of samples after augmentation RE-Free:", df_to_augment['Ms_delta_re_free'].notna().sum())
    
    # Store augmented data in csv
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
    # Rows that have duplicate compositions
    duplicates_pairs = df_pairs[df_pairs.duplicated(subset='composition', keep=False)]

    print("Duplicate compositions in df_pairs:")
    print("Number of duplicate rows:", duplicates_pairs.shape[0])
    
    df_combined = pd.concat([df_to_augment, df_pairs], ignore_index=True)
    
    print("Shape df_to_augment:", df_to_augment.shape[0])
    print("Shape df_pairs:", df_pairs.shape[0])
    print("Shape df_combined:", df_combined.shape[0])

    df_combined.to_csv(SAVE_DATA_DIR)
    
    ## PLOTTING CODE BELOW ###
    
#     sns.set_style("whitegrid") 

#     plt.figure(figsize=(8,5), dpi=100)

#     # Plot histograms with assigned colors and labels
#     # All
#     plt.figure(figsize=(8, 5), dpi=100)
#     plt.hist(df_to_augment[exp_col], bins=50, 
#              label=f'Real: {df_to_augment[exp_col].notna().sum()} samples', alpha=0.5)
#     plt.hist(mock_all, 
#              bins=100, label=f'Mock: {mock_all.shape[0]} samples', alpha=0.5)
#     plt.xlabel('Ms (A/m)')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 'all_histogram_real_vs_mock.png')
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 'all_histogram_real_vs_mock.pdf')
#     plt.close()
    
#     # RE
#     plt.figure(figsize=(8, 5), dpi=100)

#     # Real values
#     plt.hist(
#         df_to_augment[df_to_augment['has_rare_earth']]['Ms (ampere/meter)_e'],
#         bins=50,
#         label=f"Real: {df_to_augment[df_to_augment['has_rare_earth']]['Ms (ampere/meter)_e'].notna().sum()} samples",
#         alpha=0.5
#     )

#     # Mock values
#     plt.hist(
#         mock_re_free,
#         bins=100,
#         label=f"Mock: {mock_re.shape[0]} samples",
#         alpha=0.5
#     )

#     plt.xlabel('Ms (A/m)')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()

#     # Save plots
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 're_histogram_real_vs_mock.png', dpi=300)
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 're_histogram_real_vs_mock.pdf')
#     plt.close()
      
#     # RE-Free
#     plt.figure(figsize=(8, 5), dpi=100)
#     plt.hist(df_to_augment[df_to_augment["has_rare_earth"] == False]["Ms (ampere/meter)_e"], 
#              bins=50, 
#              label=f"Real: {df_to_augment[df_to_augment['has_rare_earth'] == False]['Ms (ampere/meter)_e'].notna().sum()} samples", 
#              alpha=0.5)
#     plt.hist(mock_re_free, 
#              bins=100, label=f'Mock: {mock_re_free.shape[0]} samples', alpha=0.5)
#     plt.xlabel('Ms (A/m)')
#     plt.ylabel('Frequency')
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 're_free_histogram_real_vs_mock.png')
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 're_free_histogram_real_vs_mock.pdf')
#     plt.close()
#     # Plot mock vs real experimental data
#     plt.figure(figsize=(5,5))
#     plt.scatter(df_to_augment[sim_col], df_to_augment[exp_col], alpha=0.5, label='real', color='blue')
#     plt.scatter(df_to_augment[sim_col], df_to_augment['Ms (ampere/meter)_e_mock_all'], 
#                 color='orange', label='mock', alpha=0.5)
#     plt.grid(True)
#     plt.xlabel('Ms_s (A/m)')
#     plt.ylabel('Ms_e (A/m)')
#     plt.legend()
#     plt.title('All')
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 'all_real_vs_mock.png')
#     # plt.show()
#     plt.close()
    
#     # Plot mock vs real experimental data
#     plt.figure(figsize=(5,5))
#     plt.scatter(df_to_augment[sim_col], df_to_augment[exp_col], alpha=0.5, label='real', color='blue')
#     plt.scatter(df_to_augment[sim_col], df_to_augment['Ms (ampere/meter)_e_mock_re'], 
#                 color='orange', label='mock', alpha=0.5)
#     plt.grid(True)
#     plt.xlabel('Ms_s (A/m)')
#     plt.ylabel('Ms_e (A/m)')
#     plt.legend()
#     plt.title('RE')
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 're_real_vs_mock.png')
#     # plt.show()
#     plt.close()

#     # Plot mock vs real experimental data
#     plt.figure(figsize=(5,5))
#     plt.scatter(df_to_augment[sim_col], df_to_augment[exp_col], alpha=0.5, label='real', color='blue')
#     plt.scatter(df_to_augment[sim_col], df_to_augment['Ms (ampere/meter)_e_mock_re_free'], 
#                 label='mock', alpha=0.5, color='orange')
#     plt.grid(True)
#     plt.xlabel('Ms_s (A/m)')
#     plt.ylabel('Ms_e (A/m)')
#     plt.legend()
#     plt.title('RE-Free')
#     plt.savefig(ARTIFACT_PATH['data_plots'] + 're_free_real_vs_mock.png')
#     # plt.show()
#     plt.close()
    
    # scatter_plot_augmented_data(df_to_augment, ARTIFACT_PATH['plot_dir'])
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Path to experiment YML config file.")
    parser.add_argument(
        "--configdir",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )

    # python src/augment_data.py --config "configs/augment_data_ms.yml"

    args = parser.parse_args()

    with open(args.configdir, "r") as f:
        config = yaml.safe_load(f)

    augment_data_experimental(config)