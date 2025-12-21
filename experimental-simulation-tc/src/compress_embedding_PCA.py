#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compress Embeddings for Training Pipeline 

This script creates PCA-compressed embeddings for the paired Curie temperature dataset.
It focuses specifically on PCA components of sizes 8, 16, and 32 to ensure they are available
for the training scripts.

The script:
1. Loads datasets with embeddings from create_embeddings.py output files in src/out
   (Pairs_*_emb_w_embeddings.pkl and Augm_*_emb_w_embeddings.pkl)
2. Filters for compositions with both Tc_sim and Tc_exp
3. Creates PCA-compressed versions with 8, 16, and 32 components
4. Saves the datasets with compressed embeddings as *_PCA.pkl files

Usage:
    python src/compress_embedding_PCA.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from typing import Dict, Optional, List

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from src.composition_data import CompositionData

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class PCAEmbeddingCompressor:
    """Compress compound embeddings using PCA - focused on 8, 16, and 32 components."""
    
    def __init__(self, input_dir: Path, output_dir: Path):
        """
        Initialize the embedding compressor.
        
        Parameters
        ----------
        input_dir : Path
            Directory containing input embedding files
        output_dir : Path
            Directory for output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def reconstruction_error_score(self, estimator, X):
        """Calculate reconstruction error for PCA tuning."""
        X_transformed = estimator.transform(X)
        X_reconstructed = estimator.inverse_transform(X_transformed)
        return -mean_squared_error(X, X_reconstructed)
    
    def create_pca_components(self, embeddings: np.ndarray, n_components: int) -> tuple:
        """
        Create PCA compression for given number of components.
        
        Parameters
        ----------
        embeddings : np.ndarray
            Original embeddings (N x 200)
        n_components : int
            Number of PCA components
            
        Returns
        -------
        tuple
            (pca_model, explained_variance_ratio)
        """
        print(f"\n  Fitting PCA with {n_components} components...")
        
        pca = PCA(n_components=n_components, svd_solver='auto', random_state=RANDOM_SEED)
        pca.fit(embeddings)
        
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  ✓ Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
        
        return pca, explained_var
    
    def compress_embeddings(self, df: pd.DataFrame, component_sizes: List[int] = [8, 16, 32]) -> pd.DataFrame:
        """
        Create PCA-compressed versions of embeddings.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with compound_embedding column
        component_sizes : List[int]
            List of PCA component sizes to generate
            
        Returns
        -------
        pd.DataFrame
            Dataset with additional PCA-compressed embedding columns
        """
        print("\n" + "="*60)
        print("COMPRESSING EMBEDDINGS WITH PCA")
        print("="*60)
        
        # Verify that compound_embedding exists
        if 'compound_embedding' not in df.columns:
            raise ValueError("DataFrame must contain 'compound_embedding' column")
        
        # Stack embeddings
        #mask = df['compound_embedding'].notnull()
        #print(f"Number of null embeddings: {len(df) - len(df[mask])}")
        #embeddings = np.stack(df.loc[mask, 'compound_embedding'].values)

        embeddings = np.stack(df['compound_embedding'].values)
        print(f"Original embeddings shape: {embeddings.shape}")
        
        print("\nCreating PCA compressions:")
        pca_models = {}
        
        for n_comp in component_sizes:
            # Check if the column already exists
            col_name = f'comp_emb_pca_{n_comp}_components'
            if col_name in df.columns:
                print(f"  ✓ Column {col_name} already exists, skipping")
                continue

            # Create PCA components
            pca, explained_var = self.create_pca_components(embeddings, n_comp)
            pca_models[n_comp] = pca
            
            # Transform all embeddings
            embedding_dict = {}
            
            # Use CompositionData to maintain consistency
            task_dict = {'Tc_exp': 'regression'}
            comp_data = CompositionData(
                df=df,
                task_dict=task_dict,
                elem_embedding="matscholar200",
                inputs="composition",
                identifiers=("composition", "composition")
            )
    
            for item in comp_data:
                try:
                    (elem_weights, elem_feas, self_idx, nbr_idx), target, composition_string, _ = item
                    
                    # Create original embedding
                    comp_embedding = torch.zeros((1, elem_feas.shape[1]))
                    for w, feat in zip(elem_weights, elem_feas):
                        comp_embedding += w * feat
                    
                    # Compress with PCA
                    compressed = pca.transform(comp_embedding.numpy())
                    embedding_dict[composition_string] = compressed[0]
                except:
                    continue
            
            # Add to dataframe
            col_name = f'comp_emb_pca_{n_comp}_components'
            df[col_name] = df['composition'].map(embedding_dict)
            print(f"  ✓ Added column: {col_name}")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """
        Save compressed embeddings dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with compressed embeddings
        filename : str
            Output filename
        """
        output_path = self.output_dir / filename
        df.to_pickle(output_path)
        print(f"\n✓ Saved: {output_path}")
        print(f"  Samples: {len(df)}")
        print(f"  Columns: {len(df.columns)}")
        
        # Print embedding columns
        emb_cols = [col for col in df.columns if 'emb' in col.lower()]
        print(f"\n  Embedding columns:")
        for col in emb_cols:
            if col in df.columns and len(df) > 0 and df[col].iloc[0] is not None:
                shape_info = "not available"
                try:
                    if hasattr(df[col].iloc[0], 'shape'):
                        shape_info = df[col].iloc[0].shape
                    else:
                        shape_info = len(df[col].iloc[0])
                except:
                    pass
                print(f"    - {col}: {shape_info}")
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        print(f"\nDataset Statistics:")
        print(f"  Total samples: {len(df)}")
        print(f"  Paired samples (Tc_sim & Tc_exp): {df['pair_exists'].sum() if 'pair_exists' in df.columns else 'N/A'}")
        print(f"  With embeddings: {df['use_for_emb'].sum() if 'use_for_emb' in df.columns else len(df)}")
        
        if 'contains_rare_earth' in df.columns:
            print(f"  Rare-earth containing: {df['contains_rare_earth'].sum()}")
            print(f"  Rare-earth free: {(~df['contains_rare_earth']).sum()}")
        
        print(f"\nEmbedding Columns:")
        emb_cols = [col for col in df.columns if 'emb' in col.lower()]
        for col in sorted(emb_cols):
            non_null = df[col].notna().sum()
            print(f"  {col}: {non_null} non-null")
        
        print(f"\n" + "="*60)


def main():
    """Main execution function."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Input directory where create_embeddings.py stores its outputs
    input_dir = project_root / 'outputs'
    # Output directory (same as input but we'll use different filenames)
    output_dir = input_dir
    
    print("="*60)
    print("PCA EMBEDDING COMPRESSION PIPELINE")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create compressor
    compressor = PCAEmbeddingCompressor(input_dir, output_dir)

    # List of embedding-rich input files produced by create_embeddings.py
    input_files = {
        'All_orig': 'Pairs_all_emb_w_embeddings.pkl',
        'RE_orig': 'Pairs_RE_emb_w_embeddings.pkl',
        'RE-free_orig': 'Pairs_RE_Free_emb_w_embeddings.pkl',
        'All_aug': 'Augm_all_emb_w_embeddings.pkl',
        'RE_aug': 'Augm_RE_emb_w_embeddings.pkl',
        'RE-free_aug': 'Augm_RE_Free_emb_w_embeddings.pkl',
    }

    # Process each dataset
    component_sizes = [8, 16, 32]  # Focus on these three PCA component sizes
    
    for name, fname in input_files.items():
        in_path = input_dir / fname
        if not in_path.exists():
            print(f"Warning: Input file not found for {name}: {in_path}")
            continue
        
        print("\n" + "="*60)
        print(f"PROCESSING DATASET: {name}")
        print("="*60)
        print(f"Input file: {in_path}")
        
        # Load data
        try:
            df = pd.read_pickle(in_path)
            print(f"  Loaded {len(df)} samples")
        except Exception as e:
            print(f"Error loading {in_path}: {e}")
            continue
        
        # Make sure compound_embedding exists
        if 'compound_embedding' not in df.columns:
            print(f"  Error: 'compound_embedding' column not found in {in_path}")
            continue
            
        # Compress embeddings with specific PCA components
        print(f"  Adding PCA components: {component_sizes}")
        df_compressed = compressor.compress_embeddings(df, component_sizes=component_sizes)
        
        # Save result to a new file with _PCA.pkl suffix
        new_filename = fname.replace('.pkl', '_PCA.pkl')
        compressor.save_dataset(df_compressed, filename=new_filename)
        
        # Print summary for this dataset
        compressor.print_summary(df_compressed)
    
    print("\n✓ PCA compression complete for all available datasets!")
    print("\nYou can now run the training pipeline with:")
    print("  python src/training_original_emb.py")
    print("  python src/training_augmented_emb.py")
    print("\nNOTE: The compressed embeddings are saved in new files with _PCA.pkl suffix.")


if __name__ == '__main__':
    main()
