import pandas as pd
import sys
import os

import mammos_entity as me
import mammos_units as u
import numpy as np


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_preprocessing import preprocess_data, analyze_dataframe, compute_and_plot_Mr_over_Ms, make_reduced_dataset
from src.utils.plot_utils import plot_3d_parameter_space
from src.utils.log_to_file import log_output
#from src.utils.clustering_hardsoft import threshold_clustering, kmeans_clustering
#from src.utils.supervised_clustering import supervised_clustering
# Create log directory 

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
    
@log_output('logs/pre_processing.txt')
def analyze_magnetic_data(data_path=None):
    
    # Create plots directory 
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Read the input file with mammos reader. BUT do NOT (yet)use the entities for simplicity
    if data_path != None:
        content_minidrive = me.io.entities_from_file(f"{data_path}mumax3_mindrive_cube_all_params.csv")
        content_relaxdriver = me.io.entities_from_file(f"{data_path}mumax3_relaxdriver_cube_all_params.csv")       
    
    else:
        content_minidrive = me.io.entities_from_file("./data/mumax3_mindrive_cube_all_params.csv")
        content_relaxdriver = me.io.entities_from_file("./data/mumax3_relaxdriver_cube_all_params.csv")
        
    # create procesed data dir
    os.makedirs('./data/processed/', exist_ok=True)

    df_minidrive = content_minidrive.to_dataframe(include_units=False)
    df_relaxdriver = content_relaxdriver.to_dataframe(include_units=False)
    df_combined = pd.concat([df_minidrive, df_relaxdriver])

    print("\nApplying symmetry operations\n")
    #apply symmetries and create copy of the dataframe
    df = make_reduced_dataset(df_combined)
    df = df.rename(columns={"Ms": "Ms (A/m)", "A": "A (J/m)", "K1": "K (J/m^3)", "Hc": "Hc (A/m)", "Mr": "Mr (A/m)", "BHmax": "BHmax (J/m^3)"})
    print("\nRemoving very small BHmax values\n")
    total = df.shape[0]
    df_low= df[np.log10(df['BHmax (J/m^3)']) < -5.0]
    print("\n Removing ",df_low.shape[0]," datapoints out of ",total, "points \n")
    df= df[np.log10(df['BHmax (J/m^3)']) >= -5.0]

    #Write preprocessed data to folder
    print("\n Writting applied symmetries data for training into ./data/processed/ \n")
    f = open('./data/processed/micromagnetics_angle_dependent_symmetries.csv', 'w')
    f.write('#mammos csv v1\n')
    f.write('#SpontaneousMagnetization,ExchangeStiffnessConstant,UniaxialAnisotropyConstant,CoercivityHcExternal,Remanence,MaximumEnergyProduct,,,\n')
    f.write('#https://w3id.org/emmo/domain/magnetic_material#EMMO_032731f8-874d-5efb-9c9d-6dafaa17ef25,https://w3id.org/emmo/domain/magnetic_material#EMMO_526ed2a5-a017-590e-8eb8-8a900f2b3b78,https://w3id.org/emmo/domain/magnetic_material#EMMO_49a882d1-9ce7-522b-91e7-3a460f25f5ac,https://w3id.org/emmo/domain/magnetic_material#EMMO_fe101d1d-f1f7-54f8-886b-fa6d6052ce98,https://w3id.org/emmo/domain/magnetic_material#EMMO_8fc78216-4859-53c2-b41e-e38062b04054,https://w3id.org/emmo/domain/magnetic_material#EMMO_e1028129-c23e-57ac-9174-2f34ddbf3926,,,\n')
    f.write('#A / m,J / m,J / m3,A / m,A / m,J / m3,rad,,\n')
    df.to_csv(f, index=False)
    f.close()

    # Basic analysis
    X, y = analyze_dataframe(df, save_path=plots_dir)

    # Compute and plot Mr/Ms ratio
    ratio_arr, r_min, r_max = compute_and_plot_Mr_over_Ms(
        df['Ms (A/m)'], 
        df['Mr (A/m)'], 
        df['A (J/m)'], 
        df['K (J/m^3)'],
        save_path=plots_dir
    )

    # Different preprocessing scenarios
    input_cols = ['Ms (A/m)', 'A (J/m)', 'K (J/m^3)', 'relative_angle']
    output_cols = ['Hc (A/m)', 'Mr (A/m)', 'BHmax (J/m^3)'] 

    # 1. Basic preprocessing
    X_processed, y_processed = preprocess_data(
        df,
        input_columns=input_cols,
        output_columns=output_cols,
        remove_negative=True,
        convert_to_tesla=False,
        apply_log_transform=False,
        save_path=plots_dir
    )

    # 2. With log transform
    X_processed_log, y_processed_log = preprocess_data(
        df,
        input_columns=input_cols,
        output_columns=output_cols,
        remove_negative=True,
        convert_to_tesla=False,
        apply_log_transform=True,
        log_exclude_cols=['relative_angle'],
        save_path=plots_dir
    )

    # Plot 3D parameter space
    plot_3d_parameter_space(df, x_col='Ms (A/m)', y_col='A (J/m)', z_col='K (J/m^3)', save_path=plots_dir)

    print("\nAnalysis complete. Please check the plots directory for visualizations.")
    
if __name__ == "__main__":
    analyze_magnetic_data()