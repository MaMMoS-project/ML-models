import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_dataframe(df, output_columns = ['Hc (A/m)','Mr (A/m)','BHmax (J/m^3)'], 
                      input_columns = ['Ms (A/m)', 'A (J/m)', 'K (J/m^3)'],
                      save_path=None):

    # Drop non-numeric columns
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Verify shape of the DataFrame
    print("Shape of the DataFrame:", df_numeric.shape)
    
    # Check for missing values
    if df_numeric.isnull().sum().sum() > 0:
        print("Warning: There are missing values in the DataFrame.")
        # Optional: You can handle missing values here, e.g., by filling or dropping them
        # df_numeric = df_numeric.fillna(method='ffill')  # Example: forward fill
    else:
        print("No missing values in the DataFrame.")
    
    # Display basic statistics and range of values
    stats = df_numeric.describe()
    stats.loc['range'] = stats.loc['max'] - stats.loc['min']  # Add range row
    print("Basic statistics:")
    print(stats)
    
    # Plot histograms for all numeric columns
    df_numeric.hist(figsize=(10, 10), bins=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/histograms.png", bbox_inches='tight', dpi=300)
    
    # Show correlation matrix as a heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png", bbox_inches='tight', dpi=300)

    fig, axes = plt.subplots(len(input_columns), len(output_columns), figsize=(15, 10))
    for i, inp in enumerate(input_columns):
        for j, out in enumerate(output_columns):
            sns.kdeplot(
                x=df[inp], 
                y=df[out], 
                ax=axes[i, j], 
                cmap="Reds", 
                fill=True, 
                thresh=0.05
            )
            axes[i, j].set_xlabel(inp)
            axes[i, j].set_ylabel(out)
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/kdeplots.png", bbox_inches='tight', dpi=300)
    
    # Prepare X and y for regression
    X = df[input_columns]
    y = df[output_columns]
    return X, y


def compute_and_plot_Mr_over_Ms(Ms_values, Mr_values, A_values, K_values, save_path=None):
    """
    Compute the ratio Mr/Ms, plot it in a scatter plot,
    and return min and max of the ratio.

    Parameters
    ----------
    Ms_values : array-like
        Saturation magnetization values (A/m).
    Mr_values : array-like
        Remanent magnetization values (A/m).
    A_values  : array-like
        Exchange stiffness constant
    K_values  : array-like
        Magnetocrystalline anisotropy constant

    Returns
    -------
    ratio : np.ndarray
        Array of Mr/Ms ratios.
    ratio_min : float
        Minimum of the ratio.
    ratio_max : float
        Maximum of the ratio.
    """
    # Convert inputs to numpy arrays (if they aren't already)
    Ms_arr = np.array(Ms_values, dtype=float)
    Mr_arr = np.array(Mr_values, dtype=float)
    A_arr = np.array(A_values, dtype=float)
    K_arr = np.array(K_values, dtype=float)


    # Safely compute Mr/Ms, avoiding division by zero if Ms is zero in any entries
    ratio = np.divide(Mr_arr, Ms_arr, out=np.zeros_like(Mr_arr, dtype=float), where=Ms_arr!=0)

    mu0 = 4.0 * np.pi * 1e-7
    l_K = np.sqrt(np.divide(A_arr, K_arr, out=np.zeros_like(Mr_arr, dtype=float), where=Ms_arr!=0))
    l_A = np.sqrt(np.divide(2*A_arr, (mu0*Ms_arr**2), out=np.zeros_like(Mr_arr, dtype=float), where=Ms_arr!=0))

    ratio_l_K_l_A = np.divide(l_K, l_A, out=np.zeros_like(Mr_arr, dtype=float), where=Mr_arr!=0)

    print(Mr_arr[0],Ms_arr[0],ratio[0], Mr_arr[0]/Ms_arr[0],l_K[0],l_A[0],l_K[0]/l_A[0])

    Mr_arr_soft=[Mr_arr[0]]
    Ms_arr_soft=[Ms_arr[0]]
    ratio_soft=[ratio[0]]
    ratio_l_soft=[l_K[0]/l_A[0]]

    for i in range(len(Mr_arr)):
        if (Mr_arr[i] != 0 and ratio_l_K_l_A[i] <= 1.):
            Mr_arr_soft.append(Mr_arr[i])
            Ms_arr_soft.append(Ms_arr[i])
            ratio_soft.append(ratio[i])
            ratio_l_soft.append(ratio_l_K_l_A[i])



    print("#Soft magnets:",len(ratio_soft), "out of ",len(ratio))

    # Compute min and max
    ratio_min = np.min(ratio)
    ratio_max = np.max(ratio)

    # Plot scatter of Ms vs. ratio
    plt.figure(figsize=(6, 4))
    plt.scatter(Ms_arr, ratio, c='blue', edgecolor='k', alpha=0.7, label='Mr/Ms ratio')
    plt.xlabel('Saturation Magnetization Ms (A/m)')
    plt.ylabel('Mr / Ms')
    plt.title('Scatter Plot of Mr/Ms vs. Ms')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/Mr_over_Ms.png", bbox_inches='tight', dpi=300)

    # Plot scatter of Ms vs. ratio 
    plt.figure(figsize=(6, 4))
    plt.scatter(Ms_arr, ratio, c='blue', edgecolor='k', alpha=0.7, label='Mr/Ms ratio')
    plt.scatter(Ms_arr_soft, ratio_soft, c='red', edgecolor='k', alpha=0.7, label='Mr/Ms where l_K<=l_A')
    plt.xlabel('Saturation Magnetization Ms (A/m)')
    plt.ylabel('Mr / Ms')
    plt.title('Scatter Plot of Mr/Ms vs. Ms')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/Mr_over_Ms_soft.png", bbox_inches='tight', dpi=300)
        
    return ratio, ratio_min, ratio_max


def preprocess_data(
    df,
    input_columns,
    output_columns,
    remove_negative=True,
    convert_to_tesla=False,
    apply_log_transform=False,
    log_exclude_cols=None,
    remove_smallMr=False,
    create_new_features=False,
    save_path=None
):
    """
    Preprocess magnetic materials dataset.

    Steps:
    1. Remove rows with negative values in the chosen columns (optional).
    2. Convert values from A/m to Tesla for specified columns, if desired.
    3. Apply a log transform to the specified columns, except for the ones in log_exclude_cols.
    4. Plot pairwise distributions (kdeplot) of input vs. output columns.
    5. Return X (input DataFrame) and y (output DataFrame).

    Parameters
    ----------
    df : pd.DataFrame
        The original DataFrame containing your data.
    input_columns : list of str
        Column names to be used as input features (X).
    output_columns : list of str
        Column names to be predicted (y).
    remove_negative : bool, default True
        Whether to remove rows with negative values in any of the specified columns.
    convert_to_tesla : bool, default False
        Whether to convert the values from A/m to Tesla in the specified columns.
        Assumes that the columns indeed represent magnetization (or field) in A/m.
    apply_log_transform : bool, default False
        Whether to apply a log transform to the selected columns (see below).
    log_exclude_cols : list of str, optional
        Columns that should NOT be log-transformed even if apply_log_transform=True.
        Default is None, which means log-transform all included columns.

    Returns
    -------
    X : pd.DataFrame
        Preprocessed features (input_columns).
    y : pd.DataFrame
        Preprocessed targets (output_columns).
    """
    df = df.copy()  # Doesn't modify original data

    # 1. Create additional physically motivated features IF Ms, A, K are known - for preliminary analysis
    #    The formulas assume Ms in A/m, K in J/m^3, A in J/m.
    #    If Ms was converted to Tesla, these formulas are no longer in standard SI form.
    if create_new_features:
        mu0 = 4.0 * np.pi * 1e-7  # vacuum permeability
        required_cols = {'Ms (A/m)', 'A (J/m)', 'K (J/m^3)'}

        # Check if we have Ms, A, K among the input columns
        if required_cols.issubset(set(input_columns)):
            # Avoid divide-by-zero or negative values (we already removed negatives, but check zero)
            df = df[(df['Ms (A/m)'] != 0) & (df['K (J/m^3)'] != 0)]

            # Calculate new features
            df['H_K']       = 2.0 * df['K (J/m^3)'] / (mu0 * df['Ms (A/m)'])
            df['Q']         = 2.0 * df['K (J/m^3)'] / (mu0 * df['Ms (A/m)']**2)
            df['DW_width']  = np.sqrt(df['A (J/m)'] / df['K (J/m^3)'])
            df['DW_energy'] = 4.0 * np.sqrt(df['A (J/m)'] * df['K (J/m^3)'])

            # Add newly created columns to input list
            # so they become part of X
            additional_features = ['H_K', 'Q', 'DW_width', 'DW_energy']
            for feat in additional_features:
                if feat not in input_columns:
                    input_columns.append(feat)
        else:
            # Optional: print a warning if Ms/A/K are missing
            print("Warning: Ms, A, K not all found in input_columns. Additional features not created.")

    # Combine input + output columns (after potentially adding new features)
    all_cols = input_columns + output_columns

    # 2. Remove negative values
    if remove_negative:
        for col in all_cols:
            df = df[df[col] >= 0]

    # 3. Remove small Mr values : < 8
    if remove_smallMr:
        #df = df[np.log1p(df['Mr (A/m)']) > 8]
        df = df[np.log(df['Mr (A/m)']) > 8]
        
    # 4. Convert from A/m to Tesla if requested
    if convert_to_tesla:
        mu0 = 4.0 * np.pi * 1e-7  # vacuum permeability (H/m)
        for col in all_cols:
            df[col] = df[col] * mu0

    # 3. Apply log transform if requested
    if apply_log_transform:
        # If not provided, default to empty list
        if log_exclude_cols is None:
            log_exclude_cols = []

        # Remove rows with zero only in columns that should have log-transform
        for col in all_cols:
            if col not in log_exclude_cols:
                df = df[df[col] > 0]  # remove zero before log

        for col in all_cols:
            if col not in log_exclude_cols:
                df[col] = np.log1p(df[col])  # log(1 + x)

    # 4. Plot pairwise distributions
    fig, axes = plt.subplots(len(input_columns), len(output_columns), figsize=(15, 10))

    # Handle axes shape if there's only 1 input or 1 output
    if len(input_columns) == 1 and len(output_columns) == 1:
        axes = np.array([[axes]])
    elif len(input_columns) == 1:
        axes = np.array([axes])
    elif len(output_columns) == 1:
        axes = np.array([axes]).T

    for i, inp in enumerate(input_columns):
        for j, out in enumerate(output_columns):
            sns.kdeplot(
                x=df[inp],
                y=df[out],
                ax=axes[i, j],
                cmap="Reds",
                fill=True,
                thresh=0.05
            )
            axes[i, j].set_xlabel(inp)
            axes[i, j].set_ylabel(out)

    plt.tight_layout()


    # 5. Prepare X and y
    X = df[input_columns]
    #print(output_columns)
    y = df[output_columns]
    #print("Y.shape: ",y.shape)

    return X, y

