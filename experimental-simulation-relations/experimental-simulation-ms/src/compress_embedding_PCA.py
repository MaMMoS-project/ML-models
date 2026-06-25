#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compress Ms Embeddings with PCA

Loads pickle files produced by create_embeddings.py and adds PCA-compressed
embedding columns (8, 16, 32, 64 components).

Usage:
    python -m src.compress_embedding_PCA
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from typing import List

sys.path.append(str(Path(__file__).parent.parent))
from src.composition_data import CompositionData
from src.log_to_file import log_output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EXP_COL = 'Ms (ampere/meter)_e'

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


class PCAEmbeddingCompressor:
    """Compress compound embeddings using PCA."""

    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_pca_components(self, embeddings: np.ndarray, n_components: int):
        print(f"\n  Fitting PCA with {n_components} components...")
        pca = PCA(n_components=n_components, svd_solver='auto', random_state=RANDOM_SEED)
        pca.fit(embeddings)
        explained_var = pca.explained_variance_ratio_.sum()
        print(f"  ✓ Explained variance: {explained_var:.4f} ({explained_var*100:.2f}%)")
        return pca, explained_var

    def compress_embeddings(
        self, df: pd.DataFrame, component_sizes: List[int] = [8, 16, 32, 64]
    ) -> pd.DataFrame:
        """Add PCA-compressed embedding columns to df."""
        print("\n" + "="*60)
        print("COMPRESSING EMBEDDINGS WITH PCA")
        print("="*60)

        if 'compound_embedding' not in df.columns:
            raise ValueError("DataFrame must contain 'compound_embedding' column")

        embeddings = np.stack(df['compound_embedding'].values)
        print(f"Original embeddings shape: {embeddings.shape}")

        task_dict = {EXP_COL: 'regression'}

        for n_comp in component_sizes:
            col_name = f'comp_emb_pca_{n_comp}_components'
            if col_name in df.columns:
                print(f"  ✓ {col_name} already exists, skipping")
                continue

            pca, _ = self.create_pca_components(embeddings, n_comp)

            try:
                comp_data = CompositionData(
                    df=df,
                    task_dict=task_dict,
                    elem_embedding="matscholar200",
                    inputs="composition",
                    identifiers=("composition", "composition"),
                )
                embedding_dict = {}
                for item in comp_data:
                    try:
                        (elem_weights, elem_feas, self_idx, nbr_idx), target, comp_str, _ = item
                        comp_embedding = torch.zeros((1, elem_feas.shape[1]))
                        for w, feat in zip(elem_weights, elem_feas):
                            comp_embedding += w * feat
                        compressed = pca.transform(comp_embedding.numpy())
                        embedding_dict[comp_str] = compressed[0]
                    except Exception:
                        continue
            except Exception:
                # Fallback: compress the stored compound_embedding directly
                embedding_dict = {}
                for _, row in df.iterrows():
                    comp_str = row['composition']
                    emb = row.get('compound_embedding')
                    if emb is not None:
                        embedding_dict[comp_str] = pca.transform(emb.reshape(1, -1))[0]

            df[col_name] = df['composition'].map(embedding_dict)
            print(f"  ✓ Added column: {col_name}")

        return df

    def save_dataset(self, df: pd.DataFrame, filename: str):
        output_path = self.output_dir / filename
        df.to_pickle(output_path)
        print(f"\n✓ Saved: {output_path}")
        print(f"  Samples: {len(df)}")
        emb_cols = [c for c in df.columns if 'emb' in c.lower()]
        for col in emb_cols:
            non_null = df[col].notna().sum()
            print(f"    {col}: {non_null} non-null")

    def print_summary(self, df: pd.DataFrame):
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total samples: {len(df)}")
        if 'has_rare_earth' in df.columns:
            print(f"  RE:      {df['has_rare_earth'].sum()}")
            print(f"  RE-free: {(~df['has_rare_earth']).sum()}")
        emb_cols = [c for c in df.columns if 'emb' in c.lower()]
        for col in sorted(emb_cols):
            print(f"  {col}: {df[col].notna().sum()} non-null")
        print("="*60)


@log_output('logs/compress_embeddings_PCA.txt')
def compress_embeddings_PCA():
    """Main execution function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    input_dir = project_root / 'outputs'
    output_dir = input_dir

    print("="*60)
    print("PCA EMBEDDING COMPRESSION PIPELINE")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Input/Output directory: {input_dir}")

    compressor = PCAEmbeddingCompressor(input_dir, output_dir)

    input_files = {
        'All_pairs':              'Pairs_all_w_embeddings.pkl',
        'RE_pairs':               'Pairs_RE_w_embeddings.pkl',
        'RE_free_pairs':          'Pairs_RE_Free_w_embeddings.pkl',
        'Augm_combined_all':      'Augm_combined_all_w_embeddings.pkl',
        'Augm_combined_RE':       'Augm_combined_RE_w_embeddings.pkl',
        'Augm_combined_RE_Free':  'Augm_combined_RE_Free_w_embeddings.pkl',
    }

    component_sizes = [8, 16, 32, 64]

    for name, fname in input_files.items():
        in_path = input_dir / fname
        if not in_path.exists():
            print(f"Warning: Input file not found for {name}: {in_path}")
            continue

        print("\n" + "="*60)
        print(f"PROCESSING: {name}")
        print("="*60)

        try:
            df = pd.read_pickle(in_path)
            print(f"  Loaded {len(df)} samples")
        except Exception as e:
            print(f"Error loading {in_path}: {e}")
            continue

        if 'compound_embedding' not in df.columns:
            print(f"  Error: 'compound_embedding' not found in {in_path}")
            continue

        df_compressed = compressor.compress_embeddings(df, component_sizes=component_sizes)
        new_filename = fname.replace('.pkl', '_PCA.pkl')
        compressor.save_dataset(df_compressed, filename=new_filename)
        compressor.print_summary(df_compressed)

    print("\n✓ PCA compression complete!")
    print("\nYou can now run:")
    print("  python -m src.training_pairs_emb")


if __name__ == '__main__':
    compress_embeddings_PCA()
