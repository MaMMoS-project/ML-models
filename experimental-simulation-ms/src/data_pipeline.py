from load_data import load_data
from combine_dataframes import combine_dataframes
from augment_data import augment_data_experimental
from pre_process_data import create_pairwise_dataset, remove_small_ms_values

# Plotting / visualizing
import seaborn as sns
import matplotlib.pyplot as plt

import argparse
import pdb
import yaml

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Path to experiment YML config file.")
    
    # Parse config yml file
    parser.add_argument(
        "--configdir",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    
    args = parser.parse_args()

    with open(args.configdir, "r") as f:
        config = yaml.safe_load(f)   
        
    DATA_DIR = config['DATA_DIR']
    THRESHOLD_MS = 50000
    
    query = ['oqmd', 'literature',
             # 'mtc', 'mtc_nur',  not used atm, idk how to convert Tesla to A/m.
             'bhandari_i',
             'bhandari_xii', 'bhandari_xiii',
             'magnetic_materials_exp', 'magnetic_materials_sim'
            ]
    
    # ------------ Load dataframes ------------  
    df_list = load_data(*query, data_path=DATA_DIR)
    
    print('Loading done!')
    
    # Combine dataframes into one file
    df_combined = combine_dataframes(dfs=df_list,
                                     merge_key='composition',
                                     save_dir=DATA_DIR)
    
    print('Shape combined dataframe from different data sources:', df_combined.shape[0])

    # ------------ Throw away small Ms values ------------
    print('------------ Remove small Ms values ------------')
    df_small_ms_rm = remove_small_ms_values(df_combined, 
                                             sim_col='Ms (ampere/meter)_s', 
                                             exp_col='Ms (ampere/meter)_e', 
                                             threshold=THRESHOLD_MS)
    
    df_small_ms_rm.to_csv(DATA_DIR + f'merged_df_no_Ms_leq_{THRESHOLD_MS}.csv')
    print('Shape DF after rm small Ms values:', df_small_ms_rm.shape[0])
    print('Nr RE Materials after rm small Ms values:', df_small_ms_rm[df_small_ms_rm['has_rare_earth']].shape[0])
    print('Nr RE-Free Materials after rm small Ms values:', df_small_ms_rm[df_small_ms_rm['has_rare_earth']==False].shape[0]) 
    
    # ------------ Create Pairwise Dataset ------------
    print('------------ Create Pairwise Dataset without small Ms values ------------')
    df_pairs = create_pairwise_dataset(df_small_ms_rm,
                                       sim_col='Ms (ampere/meter)_s', 
                                       exp_col='Ms (ampere/meter)_e')
    
    df_pairs.to_csv(DATA_DIR + f'pairwise_df_no_Ms_leq_{THRESHOLD_MS}.csv')
    print('Shape Pairwise DF:', df_pairs.shape[0])  
    
    # python src/data_pipeline.py --configdir "configs/data_pipeline_ms.yml"

    augment_data_experimental(config)