import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_parameter_space(df, x_col, y_col, z_col, log_scale=False, color_by=None, title="3D Parameter Space", save_path=None):
    """
    Visualizes parameters in a 3D scatter plot with zoom and rotation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the parameters to be visualized.
    x_col : str
        Column name for the X-axis.
    y_col : str
        Column name for the Y-axis.
    z_col : str
        Column name for the Z-axis.
    log_scale : bool, default False
        Whether to apply logarithmic scale to the Y and Z axes.
    color_by : str, optional
        Column name to color the points (e.g., one of the input parameters).
    title : str, default "3D Parameter Space"
        Title for the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes3D object
        The figure and 3D plot for further customization if needed.
    """

    backend = plt.get_backend()  
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extracting data
    x_data = df[x_col].values
    y_data = df[y_col].values
    z_data = df[z_col].values
    
    # Color by another column if provided
    if color_by is not None:
        color_data = df[color_by].values
    else:
        color_data = x_data  # Default to coloring by X values
    
    # 3D Scatter Plot
    scatter = ax.scatter(x_data, y_data, z_data, 
                         c=color_data, cmap="viridis", alpha=0.7, edgecolor='k', s=50)
    
    # Axis Labels
    ax.set_xlabel(f"{x_col}")
    ax.set_ylabel(f"{y_col}")
    ax.set_zlabel(f"{z_col}")
    ax.set_title(title)

    # Optionally set log scale for better visibility of large range parameters
    if log_scale:
        ax.set_yscale('log')
        ax.set_zscale('log')

    # Colorbar for reference
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.6, aspect=10)
    cbar.set_label(f"{color_by if color_by else x_col}")
    
    # Allow interactive zooming, rotating
    if save_path:
        plt.savefig(f"{save_path}/3d_parameter_space.png", bbox_inches='tight', dpi=300)
        
    if "inline" not in backend.lower():
        plt.show()
    else:
        plt.ioff()
    plt.close()
    
    return fig, ax
