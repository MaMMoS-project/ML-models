# -*- coding: utf-8 -*-
"""Create Compound Embeddings for Curie Temperature Datasets

This script creates compound embeddings from Matscholar200 element embeddings
using an element-abundance weighted sum approach. For example:
    H2O embedding = 2 × [H embedding] + 1 × [O embedding]

The script:
1. Loads paired and augmented datasets from data augmentation step (Pairs_*.csv and Augm_*.csv)
2. Creates 200-dimensional compound embeddings
3. Filters out compositions that cannot be parsed
4. Generates t-SNE visualizations colored by majority element
5. Saves datasets with embeddings (_w_embeddings suffix)

File naming convention:
- Input files: Pairs_*.csv (original data), Augm_*.csv (augmented data)
- Output files: *_w_embeddings.pkl (with embedding vectors) 

Reference:
    Matscholar200 embeddings provide 200-dimensional vectors for each chemical element.
    The compound embedding preserves the neighborhood principle: similar compounds
    cluster together in the embedding space.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple
from pymatgen.core import Composition
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

# Add parent directory to path to import composition_data
sys.path.append(str(Path(__file__).parent.parent))
from src.composition_data import CompositionData

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


class EmbeddingCreator:
    """Create compound embeddings from element embeddings."""
    
    def __init__(self, embedding_path: str, output_dir: str):
        """
        Initialize the embedding creator.
        
        Parameters
        ----------
        embedding_path : str
            Path to the Matscholar200 element embeddings JSON file
        output_dir : str
            Directory for output files
        """
        self.embedding_path = Path(embedding_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.artifacts_dir = self.output_dir / 'embeddings_tsne_plots'
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load element embeddings
        print(f"Loading element embeddings from {self.embedding_path}")
        with open(self.embedding_path, 'r') as f:
            self.elem_features = json.load(f)
        
        self.embedding_dim = len(next(iter(self.elem_features.values())))
        print(f"✓ Loaded {len(self.elem_features)} element embeddings ({self.embedding_dim}D)")
        
    def create_compound_embedding(self, composition_string: str) -> Optional[np.ndarray]:
        """
        Create compound embedding from composition string using weighted sum.
        
        Parameters
        ----------
        composition_string : str
            Chemical formula (e.g., "Fe2O3", "BaTiO3")
            
        Returns
        -------
        np.ndarray or None
            200-dimensional embedding vector, or None if composition cannot be parsed
        """
        try:
            # Parse composition
            comp = Composition(composition_string)
            comp_dict = comp.get_el_amt_dict()
            
            # Get elements and their amounts
            elements = list(comp_dict.keys())
            amounts = np.array(list(comp_dict.values()))
            
            # Normalize amounts (weights)
            weights = amounts / np.sum(amounts)
            
            # Create weighted sum of element embeddings
            compound_embedding = np.zeros(self.embedding_dim)
            for element, weight in zip(elements, weights):
                if element not in self.elem_features:
                    print(f"  Warning: Element '{element}' not in embedding dictionary")
                    return None
                compound_embedding += weight * np.array(self.elem_features[element])
            
            return compound_embedding
            
        except Exception as e:
            # Cannot parse composition (e.g., contains variables, doping notation)
            return None
    
    def get_majority_element(self, composition_string: str) -> Optional[str]:
        """
        Get the element with highest abundance in composition.
        
        Parameters
        ----------
        composition_string : str
            Chemical formula
            
        Returns
        -------
        str or None
            Element symbol with highest abundance
        """
        try:
            comp = Composition(composition_string)
            el_amt_dict = comp.get_el_amt_dict()
            return max(el_amt_dict, key=el_amt_dict.get)
        except:
            return None
    
    def process_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Add compound embeddings to dataset using CompositionData class.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with 'composition' column
        dataset_name : str
            Name of dataset (for logging)
            
        Returns
        -------
        pd.DataFrame
            Dataset with added 'compound_embedding' and 'majority_element' columns
        """
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")
        # Filter out rows with missing compositions
        before_len = len(df)
        df = df[df['composition'].notna()].copy()
        if len(df) != before_len:
            print(f"Filtered out {before_len - len(df)} rows with NaN composition")

        # Filter out rows with invalid composition strings
        def _is_valid_composition(comp: str) -> bool:
            try:
                Composition(comp)
                return True
            except Exception:
                return False

        before_len_valid = len(df)
        df = df[df['composition'].apply(_is_valid_composition)].copy()
        if len(df) != before_len_valid:
            print(f"Filtered out {before_len_valid - len(df)} rows with invalid composition strings")

        # Filter out rows containing elements without embeddings
        embedding_elements = set(self.elem_features.keys())

        def _has_only_known_elements(comp: str) -> bool:
            comp_dict = Composition(comp).get_el_amt_dict()
            return all(el in embedding_elements for el in comp_dict.keys())

        mask_known = df['composition'].apply(_has_only_known_elements)
        if not mask_known.all():
            bad_rows = df[~mask_known]
            bad_elements = set()
            for comp in bad_rows['composition']:
                comp_dict = Composition(comp).get_el_amt_dict()
                for el in comp_dict.keys():
                    if el not in embedding_elements:
                        bad_elements.add(el)
            print(
                f"Filtered out {len(bad_rows)} rows with elements missing from embeddings: "
                f"{sorted(bad_elements)}"
            )
            df = df[mask_known].copy()
        
        # Create task dict (dummy for embedding purposes)
        task_dict = {'Tc_exp': 'regression'}
        
        # Initialize CompositionData to parse compositions
        try:
            comp_data = CompositionData(
                df=df,
                task_dict=task_dict,
                elem_embedding="matscholar200",
                inputs="composition",
                identifiers=("composition", "composition")
            )
            print(f"✓ CompositionData initialized")
        except Exception as e:
            print(f"✗ Error initializing CompositionData: {e}")
            # Fallback to simple approach
            return self._process_dataset_fallback(df, dataset_name)
        
        # Create embeddings using weighted sum approach from notebook
        embedding_dict = {}
        successful = 0
        failed = 0
        
        print(f"Creating embeddings using element-abundance weighted sum...")
        
        for idx, item in enumerate(comp_data):
            try:
                (elem_weights, elem_feas, self_idx, nbr_idx), target, composition_string, _ = item
                
                # Weighted mean pooling of element vectors (as in notebook)
                comp_embedding = torch.zeros((1, elem_feas.shape[1]))
                for w, feat in zip(elem_weights, elem_feas):
                    comp_embedding += w * feat
                
                # Store embedding
                embedding_dict[composition_string] = comp_embedding.numpy()[0]
                successful += 1
                
                if (idx + 1) % 100 == 0:
                    print(f"  Progress: {idx+1}/{len(comp_data)}", end='\r')
                    
            except Exception as e:
                # Composition cannot be parsed
                failed += 1
                continue
        
        print(f"\n✓ Successfully created embeddings: {successful}/{len(df)}")
        if failed > 0:
            print(f"✗ Failed to create embeddings: {failed}/{len(df)}")
            print(f"  (Compositions with non-standard notation)")
        
        # Map embeddings back to dataframe
        df['compound_embedding'] = df['composition'].map(embedding_dict)
        df['use_for_emb'] = df['compound_embedding'].notnull()
        
        # Add majority element
        df['majority_element'] = df['composition'].apply(self.get_majority_element)
        
        return df
    
    def _process_dataset_fallback(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Fallback method using simple approach."""
        print("Using fallback embedding method...")
        
        embeddings = []
        majority_elements = []
        successful = 0
        failed = 0
        
        for idx, composition in enumerate(df['composition']):
            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx+1}/{len(df)}", end='\r')
            
            embedding = self.create_compound_embedding(composition)
            majority_elem = self.get_majority_element(composition)
            
            if embedding is not None:
                embeddings.append(embedding)
                majority_elements.append(majority_elem)
                successful += 1
            else:
                embeddings.append(None)
                majority_elements.append(None)
                failed += 1
        
        print(f"\n✓ Successfully created embeddings: {successful}/{len(df)}")
        if failed > 0:
            print(f"✗ Failed to create embeddings: {failed}/{len(df)}")
        
        df['compound_embedding'] = embeddings
        df['majority_element'] = majority_elements
        df['use_for_emb'] = df['compound_embedding'].notnull()
        
        return df
    
    def create_tsne_visualization(self, df: pd.DataFrame, dataset_name: str):
        """
        Create t-SNE visualization colored by majority element.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset with compound_embedding column
        dataset_name : str
            Name of dataset (for title and filename)
        """
        print(f"\n{'='*60}")
        print(f"Creating t-SNE visualization for {dataset_name}")
        print(f"{'='*60}")
        
        # Filter to samples with embeddings
        df_emb = df[df['use_for_emb'] == True].copy()
        print(f"Samples with valid embeddings: {len(df_emb)}")
        
        if len(df_emb) < 10:
            print("⚠ Not enough samples for t-SNE visualization")
            return
        
        # Stack embeddings
        embeddings = np.stack(df_emb['compound_embedding'].values)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Get majority elements for coloring
        majority_elements = df_emb['majority_element'].values
        unique_elements = sorted(set(majority_elements))
        element_to_color = {el: i for i, el in enumerate(unique_elements)}
        color_indices = np.array([element_to_color[el] for el in majority_elements])
        
        print(f"Unique majority elements: {len(unique_elements)}")
        print(f"Running t-SNE (perplexity=30, random_state={RANDOM_SEED})...")
        
        # Adjust perplexity if needed
        perplexity = min(30, len(df_emb) // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        print("✓ t-SNE complete")
        
        # Create visualization
        palette = sns.color_palette("husl", len(unique_elements))
        
        plt.figure(figsize=(10, 8), dpi=100)
        scatter = plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=color_indices, cmap=ListedColormap(palette),
            s=30, alpha=0.7, edgecolors='w', linewidth=0.5
        )
        
        # Create legend (show max 45 elements)
        max_legend_items = 45
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=el,
                      markerfacecolor=palette[i], markersize=8)
            for i, el in enumerate(unique_elements[:max_legend_items])
        ]
        
        if len(unique_elements) > max_legend_items:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                     label=f'... and {len(unique_elements) - max_legend_items} more',
                                     markerfacecolor='gray', markersize=8))
        
        plt.legend(handles=handles, title="Majority Element",
                  bbox_to_anchor=(1.05, 1), loc='upper left',
                  ncol=1 if len(unique_elements) <= 10 else 2,
                  fontsize=9)
        
        plt.title(f't-SNE Projection of Compound Embeddings\n'
                 f'Colored by Majority Element',
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        filename = f"tsne_{dataset_name.lower().replace(' ', '_')}_majority_element.png"
        output_path = self.artifacts_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved visualization: {output_path}")
        plt.close()
        
        # Also create a version colored by Tc_delta if available
        if 'Tc_delta' in df_emb.columns:
            self._create_tsne_tc_delta(embeddings_2d, df_emb, dataset_name)
    
    def _create_tsne_tc_delta(self, embeddings_2d: np.ndarray,
                             df_emb: pd.DataFrame, dataset_name: str):
        """Create t-SNE visualization colored by Tc_delta."""
        # Get non-NaN Tc_delta values
        valid_mask = df_emb['Tc_delta'].notna()
        if valid_mask.sum() < 10:
            return
        
        embeddings_2d_valid = embeddings_2d[valid_mask]
        tc_delta_valid = df_emb[valid_mask]['Tc_delta'].values
        
        plt.figure(figsize=(10, 8), dpi=100)
        scatter = plt.scatter(
            embeddings_2d_valid[:, 0], embeddings_2d_valid[:, 1],
            c=tc_delta_valid, cmap='coolwarm',
            s=30, alpha=0.7, edgecolors='w', linewidth=0.5
        )
        plt.colorbar(scatter, label='Tc_delta (K)', pad=0.02)
        
        plt.title(f't-SNE Projection of {dataset_name} Compound Embeddings\n'
                 f'Colored by Tc_delta (n={valid_mask.sum()})',
                 fontsize=14, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filename = f"tsne_{dataset_name.lower().replace(' ', '_')}_tc_delta.png"
        output_path = self.artifacts_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved Tc_delta visualization: {output_path}")
        plt.close()
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """
        Save dataset with embeddings.
        
        Parameters
        ----------
        df : pd.DataFrame
            Dataset to save
        filename : str
            Output filename
        """
        output_path = self.output_dir / filename
        
        # Save as pickle (preserves numpy arrays)
        pkl_path = output_path.with_suffix('.pkl')
        df.to_pickle(pkl_path)
        print(f"✓ Saved pickle: {pkl_path}")
        
    def print_summary(self, datasets: Dict[str, pd.DataFrame]):
        """Print summary statistics."""
        print(f"\n{'='*60}")
        print("EMBEDDING CREATION SUMMARY")
        print(f"{'='*60}\n")
        
        print(f"{'Dataset':<30} {'Total':>10} {'With Emb':>10} {'% Valid':>10}")
        print("-" * 60)
        
        for name, df in datasets.items():
            total = len(df)
            with_emb = df['use_for_emb'].sum()
            pct = (with_emb / total * 100) if total > 0 else 0
            print(f"{name:<30} {total:>10} {with_emb:>10} {pct:>9.1f}%")
        
        print(f"\n{'='*60}")
        print("OUTPUTS")
        print(f"{'='*60}")
        print(f"Embeddings (PKL):   {self.output_dir}")
        print(f"Visualizations:     {self.artifacts_dir}")
        print(f"{'='*60}\n")


def main():
    """Main execution function."""
    # Determine paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    print('script_dir', script_dir)
    print('project_root', project_root )
    # Input paths
    augmented_dir = project_root / 'outputs'  
    embedding_file = project_root / 'data' / 'embeddings' / 'element' / 'matscholar200.json'
    
    # Output directory - standardized to project-level 'outputs'
    output_dir = project_root / 'outputs'
    
    print("="*60)
    print("COMPOUND EMBEDDING CREATION")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Element embeddings: {embedding_file}")
    print(f"Data directory: {augmented_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Check if embedding file exists
    if not embedding_file.exists():
        print(f"Error: Embedding file not found at {embedding_file}")
        sys.exit(1)
    
    # Check if embedding-filtered input data exists (original + augmented)
    required_files = [
        'Pairs_all_emb.csv',
        'Pairs_RE_emb.csv',
        'Pairs_RE_Free_emb.csv',
        'Augm_all_emb.csv',
        'Augm_RE_emb.csv',
        'Augm_RE_Free_emb.csv',
    ]
    
    for filename in required_files:
        filepath = augmented_dir / filename
        if not filepath.exists():
            print(f"Error: Required file not found: {filepath}")
            print("Please run augment_data.py first.")
            sys.exit(1)
    
    # Create embedding creator
    creator = EmbeddingCreator(str(embedding_file), str(output_dir))
    
    # Load embedding-ready datasets (already filtered by use_for_emb == True)
    print("\n" + "="*60)
    print("LOADING EMBEDDING-READY DATASETS")
    print("="*60)
    
    datasets = {}
    
    # Helper to load MammoS-style CSVs via mammos_entity (if available)
    def _load_mammos_csv(path: Path) -> pd.DataFrame:
        try:
            import mammos_entity as me  # type: ignore[import]
            return me.io.entities_from_csv(str(path)).to_dataframe(include_units=False)
        except ImportError:
            return pd.read_csv(path, skiprows=4)
    
    # Map input filenames to dataset keys
    file_to_name = {
        'Pairs_all_emb.csv': 'All_orig',
        'Pairs_RE_emb.csv': 'RE_orig',
        'Pairs_RE_Free_emb.csv': 'RE-free_orig',
        'Augm_all_emb.csv': 'All_aug',
        'Augm_RE_emb.csv': 'RE_aug',
        'Augm_RE_Free_emb.csv': 'RE-free_aug',
    }

    for fname in required_files:
        dataset_name = file_to_name[fname]
        print(f"Loading {dataset_name} dataset from {fname}...")
        df = _load_mammos_csv(augmented_dir / fname)

        # Some augmented files may still carry an 'info' column; drop it if present
        if 'info' in df.columns:
            df = df[df['info'] != 'info'].copy()
            df = df.drop(columns=['info'])

        before_len = len(df)
        df = df[df['composition'].notna()].copy()
        if len(df) != before_len:
            print(f"  Filtered out {before_len - len(df)} rows with NaN composition ({dataset_name})")

        datasets[dataset_name] = df
        print(f"✓ Loaded {dataset_name}: {len(df)} samples")
    
    # Process each dataset
    processed_datasets = {}
    for name, df in datasets.items():
        df_processed = creator.process_dataset(df, name)
        processed_datasets[name] = df_processed
        
        # Create t-SNE visualization
        creator.create_tsne_visualization(df_processed, name)
        
        # Save dataset
        filename_map = {
            'All_orig': 'Pairs_all_emb_w_embeddings',
            'RE_orig': 'Pairs_RE_emb_w_embeddings',
            'RE-free_orig': 'Pairs_RE_Free_emb_w_embeddings',
            'All_aug': 'Augm_all_emb_w_embeddings',
            'RE_aug': 'Augm_RE_emb_w_embeddings',
            'RE-free_aug': 'Augm_RE_Free_emb_w_embeddings',
        }
        creator.save_dataset(df_processed, filename_map[name])
    
    # Print summary
    creator.print_summary(processed_datasets)
    
    print("✓ Embedding creation complete!")


if __name__ == '__main__':
    main()
