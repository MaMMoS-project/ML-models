import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV, train_test_split
from typing import Dict, Tuple, Any
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

def threshold_clustering(df, Ms_col='Ms (A/m)', Mr_col='Mr (A/m)', threshold=0.6, save_path=None):
    """
    Cluster magnetic materials into hard and soft using a simple threshold on Mr/Ms ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the magnetic data
    Ms_col : str
        Name of the column containing saturation magnetization values
    Mr_col : str
        Name of the column containing remanent magnetization values
    threshold : float
        Threshold value for Mr/Ms ratio (default: 0.6)
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Clusters' column (0: soft, 1: hard)
    """
    # Compute Mr/Ms ratio
    ratio = np.divide(df[Mr_col], df[Ms_col], 
                     out=np.zeros_like(df[Mr_col], dtype=float), 
                     where=df[Ms_col]!=0)
    
    # Create cluster array (0: soft, 1: hard)
    cluster_arr = np.zeros(ratio.size)
    cluster_arr[ratio > threshold] = 1
    
    # Create a copy of the dataframe to avoid modifying the original
    df_clustered = df.copy()
    
    # Add clusters column
    if 'Clusters' in df_clustered.columns:
        df_clustered['Clusters'] = cluster_arr
    else:
        df_clustered.insert(len(df_clustered.columns), "Clusters", cluster_arr)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(df[Ms_col][cluster_arr == 0], ratio[cluster_arr == 0], 
               c='blue', label='Soft', alpha=0.6)
    plt.scatter(df[Ms_col][cluster_arr == 1], ratio[cluster_arr == 1], 
               c='red', label='Hard', alpha=0.6)
    plt.axhline(y=threshold, color='k', linestyle='--', alpha=0.5, 
               label=f'Threshold ({threshold})')
    plt.xlabel('Ms (A/m)')
    plt.ylabel('Mr/Ms ratio')
    plt.title('Threshold-based Clustering of Magnetic Materials')
    plt.legend()
    
    if (save_path is not None):
        plt.savefig(f"{save_path}/threshold_clustering.png", bbox_inches='tight', dpi=300)
        plt.ioff()
        
    else:
        plt.show()
        
    plt.close()    
        
    return df_clustered

def get_hard_soft_clusters(df: pd.DataFrame, method: str = 'kmeans', **kwargs) -> Dict[str, pd.DataFrame]:
    """Get hard and soft magnetic material clusters.
    
    Args:
        df: Input dataframe
        method: Clustering method ('kmeans', 'threshold', or 'supervised')
        **kwargs: Additional arguments for clustering functions
        
    Returns:
        Dictionary containing:
            - all: Original dataframe with cluster labels
            - cluster0: Soft magnetic materials
            - cluster1: Hard magnetic materials
    """
    if method == 'kmeans':
        df_clustered = kmeans_clustering(df, **kwargs)
        cluster_col = 'Clusters_KMeans'
    elif method == 'supervised':
        df_clustered = supervised_clustering(df, **kwargs)
        cluster_col = 'Clusters_Supervised'
    else:
        df_clustered = threshold_clustering(df, **kwargs)
        cluster_col = 'Clusters'
    
    # Split into clusters
    cluster0 = df_clustered[df_clustered[cluster_col] == 0].copy()
    cluster1 = df_clustered[df_clustered[cluster_col] == 1].copy()
    
    return {
        'all': df_clustered,
        'cluster0': cluster0,
        'cluster1': cluster1
    }


def kmeans_clustering(df, Ms_col='Ms (A/m)', Mr_col='Mr (A/m)', save_path=None):
    """
    Cluster magnetic materials into hard and soft using K-means clustering on Ms and Mr/Ms ratio.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the magnetic data
    Ms_col : str
        Name of the column containing saturation magnetization values
    Mr_col : str
        Name of the column containing remanent magnetization values
    save_path : str, optional
        Path to save the plot
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added 'Clusters_KMeans' column (0/1 for soft/hard)
    """
    # Compute Mr/Ms ratio
    ratio = np.divide(df[Mr_col], df[Ms_col], 
                     out=np.zeros_like(df[Mr_col], dtype=float), 
                     where=df[Ms_col]!=0)
    
    # Prepare data for clustering
    X = np.column_stack([df[Ms_col], ratio])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Ensure cluster 1 corresponds to higher Mr/Ms ratio (hard magnets)
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    if cluster_centers[0][1] > cluster_centers[1][1]:
        clusters = 1 - clusters
    
    # Create a copy of the dataframe to avoid modifying the original
    df_clustered = df.copy()
    
    # Add clusters column
    if 'Clusters_KMeans' in df_clustered.columns:
        df_clustered['Clusters_KMeans'] = clusters
    else:
        df_clustered.insert(len(df_clustered.columns), "Clusters_KMeans", clusters)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.scatter(df[Ms_col][clusters == 0], ratio[clusters == 0], 
               c='blue', label='Soft', alpha=0.6)
    plt.scatter(df[Ms_col][clusters == 1], ratio[clusters == 1], 
               c='red', label='Hard', alpha=0.6)
    
    # Plot cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, 
               linewidths=3, label='Cluster Centers')
    
    plt.xlabel('Ms (A/m)')
    plt.ylabel('Mr/Ms ratio')
    plt.title('K-means Clustering of Magnetic Materials')
    plt.legend()
    
    if (save_path is not None):
        plt.savefig(f"{save_path}/kmeans_clustering.png", bbox_inches='tight', dpi=300)
    
    else:
        plt.show()
        
    plt.close()
    return df_clustered