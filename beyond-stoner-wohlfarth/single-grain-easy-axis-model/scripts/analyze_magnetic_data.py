import sys
import os

import mammos_entity as me
import mammos_units as u


# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.data_preprocessing import preprocess_data, analyze_dataframe, compute_and_plot_Mr_over_Ms
from src.utils.plot_utils import plot_3d_parameter_space
from src.utils.clustering_hardsoft import threshold_clustering, kmeans_clustering
from src.utils.supervised_clustering import supervised_clustering
from src.utils.log_to_file import log_output 

# Create log directory 
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)
    
@log_output('logs/pre_processing.txt')
def analyze_magnetic_data(data_path=None, plots_dir=None):
    
    # # Create plots directory 
    # plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    # os.makedirs(plots_dir, exist_ok=True)

    if data_path is None:
        # Read the input file with mammos reader. BUT do NOT (yet)use the entities for simplicity
        content = me.io.entities_from_file("./data/magnetic_materials.csv")
        
    else:
        content = me.io.entities_from_file(data_path)

    df = content.to_dataframe(include_units=False)
    df = df.rename(columns={"Ms": "Ms (A/m)", "A": "A (J/m)", "K1": "K (J/m^3)", "Hc": "Hc (A/m)", "Mr": "Mr (A/m)", "BHmax": "BHmax (J/m^3)"})

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
    input_cols = ['Ms (A/m)', 'A (J/m)', 'K (J/m^3)']
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
        save_path=plots_dir
    )

    # 3. Log transform excluding Ms and Mr
    X_processed_log_noM, y_processed_log_noM = preprocess_data(
        df,
        input_columns=input_cols,
        output_columns=output_cols,
        remove_negative=True,
        convert_to_tesla=False,
        apply_log_transform=True,
        log_exclude_cols=['Ms (A/m)', 'Mr (A/m)'],
        save_path=plots_dir
    )

    # 4. Log transform with Mr threshold
    X_processed_log_Mr8, y_processed_log_Mr8 = preprocess_data(
        df,
        input_columns=input_cols,
        output_columns=output_cols,
        remove_negative=True,
        convert_to_tesla=False,
        apply_log_transform=True,
        remove_smallMr=True,
        save_path=plots_dir
    )

    # 5. Log transform with Mr threshold and new features
    X_processed_log_Mr8_feat, y_processed_log_Mr8_feat = preprocess_data(
        df,
        input_columns=input_cols,
        output_columns=output_cols,
        remove_negative=True,
        convert_to_tesla=False,
        apply_log_transform=True,
        remove_smallMr=True,
        create_new_features=True,
        save_path=plots_dir
    )

    # Plot 3D parameter space
    plot_3d_parameter_space(df, x_col='Ms (A/m)', y_col='A (J/m)', z_col='K (J/m^3)', save_path=plots_dir)

    # Perform clustering analysis
    print("\n Performing clustering analysis...")

    # Threshold-based clustering
    df_threshold = threshold_clustering(df, save_path=plots_dir)

    print("\n Threshold clustering results:")
    print("Number of soft magnets: ", (df_threshold['Clusters'] == 0).sum())
    print("Number of hard magnets: ", (df_threshold['Clusters'] == 1).sum())
    
    # Supervised clustering
    df_supervised = supervised_clustering(df, save_path=plots_dir)

    print("\nSupervised clustering results:")
    print("Number of soft magnets: ", (df_supervised['Clusters_Supervised'] == 0).sum())
    print("Number of hard magnets: ", (df_supervised['Clusters_Supervised'] == 1).sum())

    # K-means clustering
    df_kmeans = kmeans_clustering(df, save_path=plots_dir)
    print("\nK-means clustering results:")
    print("Number of soft magnets: ", (df_kmeans['Clusters_KMeans'] == 0).sum())
    print("Number of hard magnets: ", (df_kmeans['Clusters_KMeans'] == 1).sum())
    print("\nAnalysis complete. Please check the plots directory for visualizations.")

if __name__ == "__main__":
    
    plots_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    analyze_magnetic_data(plots_dir=plots_dir)