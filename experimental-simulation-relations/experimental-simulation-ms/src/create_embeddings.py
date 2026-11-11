# -*- coding: utf-8 -*-
"""Create Compound Embeddings for Ms Datasets

Loads pairs from data/merged_df_python.csv, splits into All/RE/RE-free,
and creates 200-dimensional Matscholar200 compound embeddings for each split.

Usage:
    python -m src.create_embeddings
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional
from pymatgen.core import Composition
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap

sys.path.append(str(Path(__file__).parent.parent))
try:
    import torch
    from src.composition_data import CompositionData
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    CompositionData = None
    _TORCH_AVAILABLE = False
from src.log_to_file import log_output

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

SIM_COL = 'Ms (ampere/meter)_s'
EXP_COL = 'Ms (ampere/meter)_e'

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)


class EmbeddingCreator:
    """Create compound embeddings from Matscholar200 element embeddings."""

    def __init__(self, embedding_path: str, output_dir: str):
        self.embedding_path = Path(embedding_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = self.output_dir / 'embeddings_tsne_plots'
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading element embeddings from {self.embedding_path}")
        with open(self.embedding_path, 'r') as f:
            self.elem_features = json.load(f)
        self.embedding_dim = len(next(iter(self.elem_features.values())))
        print(f"✓ Loaded {len(self.elem_features)} element embeddings ({self.embedding_dim}D)")

    def create_compound_embedding(self, composition_string: str) -> Optional[np.ndarray]:
        """Create compound embedding from composition string using weighted sum."""
        try:
            comp = Composition(composition_string)
            comp_dict = comp.get_el_amt_dict()
            elements = list(comp_dict.keys())
            amounts = np.array(list(comp_dict.values()))
            weights = amounts / np.sum(amounts)
            compound_embedding = np.zeros(self.embedding_dim)
            for element, weight in zip(elements, weights):
                if element not in self.elem_features:
                    return None
                compound_embedding += weight * np.array(self.elem_features[element])
            return compound_embedding
        except Exception:
            return None

    def get_majority_element(self, composition_string: str) -> Optional[str]:
        try:
            comp = Composition(composition_string)
            el_amt_dict = comp.get_el_amt_dict()
            return max(el_amt_dict, key=el_amt_dict.get)
        except Exception:
            return None

    def process_dataset(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Add compound_embedding and majority_element columns to dataset."""
        print(f"\n{'='*60}")
        print(f"Processing {dataset_name}")
        print(f"{'='*60}")
        print(f"Total samples: {len(df)}")

        before_len = len(df)
        df = df[df['composition'].notna()].copy()
        if len(df) != before_len:
            print(f"Filtered out {before_len - len(df)} rows with NaN composition")

        def _is_valid(comp: str) -> bool:
            try:
                Composition(comp)
                return True
            except Exception:
                return False

        before_valid = len(df)
        df = df[df['composition'].apply(_is_valid)].copy()
        if len(df) != before_valid:
            print(f"Filtered out {before_valid - len(df)} rows with invalid compositions")

        embedding_elements = set(self.elem_features.keys())

        def _has_known_elements(comp: str) -> bool:
            return all(el in embedding_elements for el in Composition(comp).get_el_amt_dict())

        mask_known = df['composition'].apply(_has_known_elements)
        if not mask_known.all():
            bad_elements = set()
            for comp in df[~mask_known]['composition']:
                for el in Composition(comp).get_el_amt_dict():
                    if el not in embedding_elements:
                        bad_elements.add(el)
            print(
                f"Filtered out {(~mask_known).sum()} rows with unknown elements: "
                f"{sorted(bad_elements)}"
            )
            df = df[mask_known].copy()

        task_dict = {EXP_COL: 'regression'}
        if not _TORCH_AVAILABLE:
            print("torch not available — using fallback embedding method")
            return self._process_fallback(df, dataset_name)
        try:
            comp_data = CompositionData(
                df=df,
                task_dict=task_dict,
                elem_embedding="matscholar200",
                inputs="composition",
                identifiers=("composition", "composition"),
            )
            print(f"✓ CompositionData initialized")
        except Exception as e:
            print(f"✗ Error initializing CompositionData: {e}")
            return self._process_fallback(df, dataset_name)

        embedding_dict = {}
        successful = 0
        failed = 0

        print("Creating embeddings using element-abundance weighted sum...")
        for idx, item in enumerate(comp_data):
            try:
                (elem_weights, elem_feas, self_idx, nbr_idx), target, composition_string, _ = item
                comp_embedding = torch.zeros((1, elem_feas.shape[1]))
                for w, feat in zip(elem_weights, elem_feas):
                    comp_embedding += w * feat
                embedding_dict[composition_string] = comp_embedding.numpy()[0]
                successful += 1
                if (idx + 1) % 100 == 0:
                    print(f"  Progress: {idx+1}/{len(comp_data)}", end='\r')
            except Exception:
                failed += 1
                continue

        print(f"\n✓ Successfully created embeddings: {successful}/{len(df)}")
        if failed > 0:
            print(f"✗ Failed: {failed}/{len(df)}")

        df['compound_embedding'] = df['composition'].map(embedding_dict)
        df['use_for_emb'] = df['compound_embedding'].notnull()
        df['majority_element'] = df['composition'].apply(self.get_majority_element)
        return df

    def _process_fallback(self, df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Fallback: compute embeddings without CompositionData."""
        print("Using fallback embedding method...")
        embeddings = []
        majority_elements = []
        successful = 0
        failed = 0
        for idx, composition in enumerate(df['composition']):
            if (idx + 1) % 100 == 0:
                print(f"  Progress: {idx+1}/{len(df)}", end='\r')
            emb = self.create_compound_embedding(composition)
            maj = self.get_majority_element(composition)
            embeddings.append(emb)
            majority_elements.append(maj)
            if emb is not None:
                successful += 1
            else:
                failed += 1
        print(f"\n✓ Successful: {successful}/{len(df)}")
        df['compound_embedding'] = embeddings
        df['majority_element'] = majority_elements
        df['use_for_emb'] = df['compound_embedding'].notnull()
        return df

    def create_tsne_visualization(self, df: pd.DataFrame, dataset_name: str):
        """Create t-SNE visualization colored by majority element."""
        print(f"\nCreating t-SNE for {dataset_name}")
        df_emb = df[df['use_for_emb'] == True].copy()
        if len(df_emb) < 10:
            print("⚠ Not enough samples for t-SNE")
            return

        embeddings = np.stack(df_emb['compound_embedding'].values)
        majority_elements = df_emb['majority_element'].values
        unique_elements = sorted(set(majority_elements))
        element_to_color = {el: i for i, el in enumerate(unique_elements)}
        color_indices = np.array([element_to_color[el] for el in majority_elements])

        perplexity = min(30, len(df_emb) // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=RANDOM_SEED)
        embeddings_2d = tsne.fit_transform(embeddings)

        palette = sns.color_palette("husl", len(unique_elements))
        plt.figure(figsize=(10, 8), dpi=100)
        plt.scatter(
            embeddings_2d[:, 0], embeddings_2d[:, 1],
            c=color_indices, cmap=ListedColormap(palette),
            s=30, alpha=0.7, edgecolors='w', linewidth=0.5,
        )
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=el,
                      markerfacecolor=palette[i], markersize=8)
            for i, el in enumerate(unique_elements[:45])
        ]
        plt.legend(handles=handles, title="Majority Element",
                  bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.title(f't-SNE: {dataset_name}\nColored by Majority Element', fontsize=14)
        plt.xlabel('t-SNE Dim 1', fontsize=12)
        plt.ylabel('t-SNE Dim 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        fname = f"tsne_{dataset_name.lower().replace(' ', '_')}_majority_element.png"
        plt.savefig(self.artifacts_dir / fname, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {self.artifacts_dir / fname}")
        plt.close()

        # Also visualize by Ms_delta if available
        delta_col = 'Ms_delta'
        if delta_col not in df_emb.columns and SIM_COL in df_emb.columns and EXP_COL in df_emb.columns:
            valid = df_emb[SIM_COL].notna() & df_emb[EXP_COL].notna()
            if valid.sum() >= 10:
                delta_values = (df_emb.loc[valid, EXP_COL] - df_emb.loc[valid, SIM_COL]).values
                emb_valid = embeddings_2d[valid.values]
                plt.figure(figsize=(10, 8), dpi=100)
                sc = plt.scatter(
                    emb_valid[:, 0], emb_valid[:, 1],
                    c=delta_values, cmap='coolwarm',
                    s=30, alpha=0.7, edgecolors='w', linewidth=0.5,
                )
                plt.colorbar(sc, label='Ms_exp - Ms_sim (A/m)', pad=0.02)
                plt.title(f't-SNE: {dataset_name}\nColored by Ms_delta', fontsize=14)
                plt.xlabel('t-SNE Dim 1', fontsize=12)
                plt.ylabel('t-SNE Dim 2', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                fname2 = f"tsne_{dataset_name.lower().replace(' ', '_')}_ms_delta.png"
                plt.savefig(self.artifacts_dir / fname2, dpi=150, bbox_inches='tight')
                print(f"✓ Saved: {self.artifacts_dir / fname2}")
                plt.close()

    def save_dataset(self, df: pd.DataFrame, filename: str):
        """Save dataset with embeddings as pickle."""
        pkl_path = (self.output_dir / filename).with_suffix('.pkl')
        df.to_pickle(pkl_path)
        print(f"✓ Saved pickle: {pkl_path}")

    def print_summary(self, datasets: Dict[str, pd.DataFrame]):
        print(f"\n{'='*60}")
        print("EMBEDDING CREATION SUMMARY")
        print(f"{'='*60}")
        print(f"{'Dataset':<30} {'Total':>10} {'With Emb':>10} {'% Valid':>10}")
        print("-" * 60)
        for name, df in datasets.items():
            total = len(df)
            with_emb = df['use_for_emb'].sum()
            pct = (with_emb / total * 100) if total > 0 else 0
            print(f"{name:<30} {total:>10} {with_emb:>10} {pct:>9.1f}%")
        print(f"\nOutputs: {self.output_dir}")
        print(f"{'='*60}\n")


@log_output('logs/create_embeddings.txt')
def create_embeddings():
    """Main execution function."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    embedding_file = project_root / 'data' / 'embeddings' / 'element' / 'matscholar200.json'
    output_dir = project_root / 'outputs'
    csv_path = project_root / 'data' / 'merged_df_python.csv'

    print("="*60)
    print("COMPOUND EMBEDDING CREATION")
    print("="*60)
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Element embeddings: {embedding_file}")
    print(f"Data CSV: {csv_path}")
    print(f"Output directory: {output_dir}")

    if not embedding_file.exists():
        print(f"Error: Embedding file not found at {embedding_file}")
        sys.exit(1)

    if not csv_path.exists():
        print(f"Error: Data CSV not found at {csv_path}")
        sys.exit(1)

    # Load and filter pairs
    df_raw = pd.read_csv(csv_path)
    df_pairs_all = df_raw[df_raw[SIM_COL].notna() & df_raw[EXP_COL].notna()].copy()
    df_pairs_re = df_pairs_all[df_pairs_all['has_rare_earth'] == True].copy()
    df_pairs_re_free = df_pairs_all[df_pairs_all['has_rare_earth'] == False].copy()

    print(f"\nPairs available:")
    print(f"  All:     {len(df_pairs_all)}")
    print(f"  RE:      {len(df_pairs_re)}")
    print(f"  RE-free: {len(df_pairs_re_free)}")

    creator = EmbeddingCreator(str(embedding_file), str(output_dir))

    # Original pairs (always processed)
    datasets = {
        'All_pairs':     (df_pairs_all,     'Pairs_all_w_embeddings'),
        'RE_pairs':      (df_pairs_re,      'Pairs_RE_w_embeddings'),
        'RE_free_pairs': (df_pairs_re_free, 'Pairs_RE_Free_w_embeddings'),
    }

    # Augmented datasets (processed if augment_data.py has been run)
    augmented_file_map = {
        'Augm_combined_all':      ('Augm_combined_all.csv',      'Augm_combined_all_w_embeddings'),
        'Augm_combined_RE':       ('Augm_combined_RE.csv',       'Augm_combined_RE_w_embeddings'),
        'Augm_combined_RE_Free':  ('Augm_combined_RE_Free.csv',  'Augm_combined_RE_Free_w_embeddings'),
    }
    for name, (csv_file, out_stem) in augmented_file_map.items():
        csv_path_aug = output_dir / csv_file
        if csv_path_aug.exists():
            df_aug = pd.read_csv(csv_path_aug)
            df_aug = df_aug[df_aug[SIM_COL].notna() & df_aug[EXP_COL].notna()].copy()
            datasets[name] = (df_aug, out_stem)
            print(f"Found augmented file: {csv_file} ({len(df_aug)} rows)")
        else:
            print(f"Skipping {csv_file} (not found — run augment_data.py to generate)")

    processed = {}
    for name, (df, out_stem) in datasets.items():
        df_proc = creator.process_dataset(df, name)
        processed[name] = df_proc
        creator.create_tsne_visualization(df_proc, name)
        creator.save_dataset(df_proc, out_stem)

    creator.print_summary(processed)
    print("✓ Embedding creation complete!")


if __name__ == '__main__':
    create_embeddings()
