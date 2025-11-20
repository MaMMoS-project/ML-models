import numpy as np
import pandas as pd

# Physical constant (vacuum permeability), SI
MU0 = 4.0 * np.pi * 1e-7  # H/m = N/A^2

def compute_HA(K, Ms, mu0: float = MU0):
    """
    Anisotropy field: HA = 2*K / (mu0 * Ms)
    K: J/m^3, Ms: A/m -> HA: A/m
    """
    K = np.asarray(K, dtype=float)
    Ms = np.asarray(Ms, dtype=float)
    denom = mu0 * Ms
    HA = np.divide(2.0 * K, denom, out=np.full_like(K, np.nan, dtype=float), where=denom!=0)
    return HA

def compute_Hc(Ms, K, A=None, alpha: float = 0.81, neff: float = 1/3, mu0: float = MU0):
    """
    Coercivity via Kronmüller: Hc = alpha * HA - neff * Ms
    Ms: A/m, K: J/m^3 -> Hc: A/m
    A (J/m) unused here; included for API symmetry.
    """
    Ms = np.asarray(Ms, dtype=float)
    HA = compute_HA(K, Ms, mu0)
    Hc = alpha * HA - neff * Ms
    return Hc

def compute_Mr(Ms, K, A=None, neff: float = 1/3, mu0: float = MU0):
    """
    Remanence: Mr = Ms * (1 - (neff * Ms)/HA)
    Ms: A/m, K: J/m^3 -> Mr: A/m
    """
    Ms = np.asarray(Ms, dtype=float)
    HA = compute_HA(K, Ms, mu0)
    ratio = np.divide(neff * Ms, HA, out=np.full_like(Ms, np.nan, dtype=float), where=HA!=0)
    Mr = Ms * (1.0 - ratio)
    return Mr

def compute_BHmax(Mr, neff: float = 1/3, mu0: float = MU0):
    """
    Maximum energy product (simple linear demag model):
    BHmax = mu0 * (1 - neff)^2 * Mr^2 / 4
    Mr: A/m -> BHmax: J/m^3
    """
    Mr = np.asarray(Mr, dtype=float)
    return mu0 * (1.0 - neff)**2 * (Mr**2) / 4.0

def compute_outputs_from_df(df: pd.DataFrame,
                            alpha: float = 0.81,
                            neff: float = 1/3,
                            mu0: float = MU0) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required columns.
    alpha : float, optional
        Alpha parameter for Kronmüller equation. Default is 0.81.
    neff : float, optional
        Effective demagnetization factor. Default is 1/3.
    mu0 : float, optional
        Vacuum permeability. Default is MU0.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added output columns.
    """
    Ms = df['Ms (A/m)'].to_numpy(dtype=float)
    K  = df['K (J/m^3)'].to_numpy(dtype=float)

    Hc = compute_Hc(Ms, K, alpha=alpha, neff=neff, mu0=mu0)
    Mr = compute_Mr(Ms, K, neff=neff, mu0=mu0)
    BHmax = compute_BHmax(Mr, neff=neff, mu0=mu0)

    out = df.copy()
    out['Hc (A/m)'] = Hc
    out['Mr (A/m)'] = Mr
    out['BHmax (J/m^3)'] = BHmax
    return out

def add_magnetic_properties(df: pd.DataFrame,
                           input_columns: list = ['Ms (A/m)', 'K (J/m^3)'],
                           output_columns: list = ['Hc (A/m)', 'Mr (A/m)', 'BHmax (J/m^3)'],
                           compute_differences: bool = False,
                           alpha: float = 0.81,
                           neff: float = 1/3,
                           mu0: float = MU0) -> pd.DataFrame:
    """
    Add magnetic properties to the DataFrame based on input columns.
    Can also compute differences between ground truth and computed values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required columns.
    input_columns : list, optional
        List of input column names. Default is ['Ms (A/m)', 'K (J/m^3)'].
    output_columns : list, optional
        List of output column names to be added. Default is ['Hc (A/m)', 'Mr (A/m)', 'BHmax (J/m^3)'].
    compute_differences : bool, optional
        If True, computes difference columns between ground truth and computed values.
        Default is False.
    alpha : float, optional
        Alpha parameter for Kronmüller equation. Default is 0.81.
    neff : float, optional
        Effective demagnetization factor. Default is 1/3.
    mu0 : float, optional
        Vacuum permeability. Default is MU0.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added output columns and optionally difference columns.
    """
    # Verify that required input columns are present
    required_cols = {'Ms (A/m)', 'K (J/m^3)'}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Extract required arrays
    Ms = df['Ms (A/m)'].to_numpy(dtype=float)
    K = df['K (J/m^3)'].to_numpy(dtype=float)
    
    # Store original values if they exist and we need to compute differences
    original_values = {}
    if compute_differences:
        for col in output_columns:
            if col in df.columns:
                original_values[col] = df[col].to_numpy(dtype=float)
    
    # Compute analytical properties and add them with '_analyt' suffix
    # Compute Hc (coercivity)
    Hc_analyt = compute_Hc(Ms, K, alpha=alpha, neff=neff, mu0=mu0)
    result_df['Hc_analyt (A/m)'] = Hc_analyt
    
    # Compute Mr (remanence)
    Mr_analyt = compute_Mr(Ms, K, neff=neff, mu0=mu0)
    result_df['Mr_analyt (A/m)'] = Mr_analyt
    
    # Compute BHmax (maximum energy product)
    BHmax_analyt = compute_BHmax(Mr_analyt, neff=neff, mu0=mu0)
    result_df['BHmax_analyt (J/m^3)'] = BHmax_analyt
    
    # Compute differences if requested and original values exist
    if compute_differences:
        # Map between original columns and their analytical counterparts
        column_mapping = {
            'Hc (A/m)': 'Hc_analyt (A/m)',
            'Mr (A/m)': 'Mr_analyt (A/m)',
            'BHmax (J/m^3)': 'BHmax_analyt (J/m^3)'
        }
        
        for orig_col, analyt_col in column_mapping.items():
            if orig_col in df.columns:
                # Create difference column (original - analytical)
                diff_col_name = f"{orig_col.split(' ')[0]}_diff {orig_col.split(' ')[1]}"
                result_df[diff_col_name] = df[orig_col] - result_df[analyt_col]
    
    return result_df

def add_derived_magnetic_properties(df: pd.DataFrame,
                                   input_columns: list = ['Ms (A/m)', 'A (J/m)', 'K (J/m^3)'],
                                   output_columns: list = ['H_K', 'Q', 'DW_width', 'DW_energy'],
                                   mu0: float = MU0) -> pd.DataFrame:
    """
    Add derived magnetic properties to the DataFrame based on input columns.
    These are additional physical properties that can be derived from Ms, A, and K.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the required columns.
    input_columns : list, optional
        List of input column names. Default is ['Ms (A/m)', 'A (J/m)', 'K (J/m^3)'].
    output_columns : list, optional
        List of output column names to be added. Default is ['H_K', 'Q', 'DW_width', 'DW_energy'].
    mu0 : float, optional
        Vacuum permeability. Default is MU0.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added derived properties.
    """
    # Verify that required input columns are present
    required_cols = {'Ms (A/m)', 'A (J/m)', 'K (J/m^3)'}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        raise ValueError(f"Missing required columns: {missing}")
    
    # Make a copy to avoid modifying the original
    result_df = df.copy()
    
    # Avoid divide-by-zero or negative values
    valid_mask = (df['Ms (A/m)'] != 0) & (df['K (J/m^3)'] != 0)
    
    # Calculate derived properties only for valid rows
    if 'H_K' in output_columns:
        # Anisotropy field
        result_df.loc[valid_mask, 'H_K'] = 2.0 * df.loc[valid_mask, 'K (J/m^3)'] / (mu0 * df.loc[valid_mask, 'Ms (A/m)'])
    
    if 'Q' in output_columns:
        # Quality factor
        result_df.loc[valid_mask, 'Q'] = 2.0 * df.loc[valid_mask, 'K (J/m^3)'] / (mu0 * df.loc[valid_mask, 'Ms (A/m)']**2)
    
    if 'DW_width' in output_columns:
        # Domain wall width
        result_df.loc[valid_mask, 'DW_width'] = np.sqrt(df.loc[valid_mask, 'A (J/m)'] / df.loc[valid_mask, 'K (J/m^3)'])
    
    if 'DW_energy' in output_columns:
        # Domain wall energy
        result_df.loc[valid_mask, 'DW_energy'] = 4.0 * np.sqrt(df.loc[valid_mask, 'A (J/m)'] * df.loc[valid_mask, 'K (J/m^3)'])
    
    return result_df
