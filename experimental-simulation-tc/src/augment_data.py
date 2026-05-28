# -*- coding: utf-8 -*-
"""Data Augmentation for Curie Temperature Dataset

This script implements the data augmentation procedure for creating augmented
datasets from the original data EC_curie_temp.csv. It uses bootstrap sampling from the delta
temperature distribution (Tc_exp - Tc_sim) to generate mock experimental values
for materials that only have simulated Curie temperatures.

The script creates three augmented datasets for All materials, RE-containing materials and RE-free materials, separately..
The script also creates the three corresponding embedding-compatible datasets.

The script also creates the corresponding six datasets from the original dat (without augmentation)

Output location:
- CSV files: src/out/
- Plots: src/out/distributions_plots/

File naming convention:
- Original paired data:         Pairs_all.csv, Pairs_RE.csv, Pairs_RE_Free.csv
- Phase 1 (Tc_sim-only augm.): Augm_sim_all.csv, Augm_sim_RE.csv, Augm_sim_RE_Free.csv
- Phase 2 (Tc_exp-only augm.): Augm_exp_all.csv, Augm_exp_RE.csv, Augm_exp_RE_Free.csv
- Phase 3 (combined augm.):    Augm_combined_all.csv, Augm_combined_RE.csv, Augm_combined_RE_Free.csv
- Embedding-compatible files have '_emb' suffix

Column simplification:
- Files contain only essential columns: composition, Tc_sim, Tc_exp, Tc_delta
- Plus necessary metadata columns: contains_rare_earth, pair_exists, use_for_emb

Expected output dimensions:
- All: 839 → 2474 samples
- All (embedding compatible): 764 → 2013 samples
- RE: 547 → 1261 samples
- RE (embedding compatible): 507 → 1083 samples
- RE-free: 292 → 1213 samples
- RE-free (embedding compatible): 257 → 930 samples
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple

import sys
import os

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Configure plotting style
sns.set_style("whitegrid")

# Create log directory 
from src.log_to_file import log_output
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

class CurieTempAugmenter:
    """
    Handle data augmentation for Curie temperature datasets.
    """
    
    def __init__(self, data_path: str, output_dir: str = None):
        """
        Initialize the augmenter.
        
        Parameters
        ----------
        data_path : str
            Path to the EC_curie_temp.csv file
        output_dir : str, optional
            Directory for output files. Defaults to ../data relative to data_path
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir) if output_dir else self.data_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data containers
        self.df_curie_temp = None
        self.df_tc_pairs = None
        self.df_tc_pairs_re = None
        self.df_tc_pairs_re_free = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the Curie temperature data from CSV.
        
        Returns
        -------
        pd.DataFrame
            Loaded dataframe
        """
        print(f"Loading data from {self.data_path}")
        
        try:
            # Try loading with mammos_entity if available
            import mammos_entity as me
            #self.df_curie_temp = me.io.entities_from_csv(
            self.df_curie_temp = me.from_csv(
                str(self.data_path)
            ).to_dataframe(include_units=False)
            print(f"Loaded {self.df_curie_temp.shape[0]} entries using mammos_entity")
        except ImportError:
            # Fallback to pandas
            print("mammos_entity not available, using pandas directly")
            # Skip the first 4 lines (MammoS format header)
            self.df_curie_temp = pd.read_csv(self.data_path, skiprows=4)
            print(f"Loaded {self.df_curie_temp.shape[0]} entries using pandas")
        
        return self.df_curie_temp
    
    def filter_paired_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Filter data to only include pairs with both sim and exp values.
        
        Returns
        -------
        tuple of pd.DataFrame
            (all_pairs, re_pairs, re_free_pairs)
        """
        print("\n" + "="*60)
        print("FILTERING PAIRED DATA")
        print("="*60)
        
        # Filter for pairs that exist (have both Tc_sim and Tc_exp)
        self.df_tc_pairs = self.df_curie_temp[
            self.df_curie_temp['pair_exists']
        ].copy()
        print(f"All paired data: {self.df_tc_pairs.shape[0]} samples")
        
        # Filter for RE-containing compounds
        self.df_tc_pairs_re = self.df_tc_pairs[
            self.df_tc_pairs['contains_rare_earth']
        ].copy()
        print(f"RE paired data: {self.df_tc_pairs_re.shape[0]} samples")
        
        # Filter for RE-free compounds
        self.df_tc_pairs_re_free = self.df_tc_pairs[
            ~self.df_tc_pairs['contains_rare_earth']
        ].copy()
        print(f"RE-free paired data: {self.df_tc_pairs_re_free.shape[0]} samples")
        
        return self.df_tc_pairs, self.df_tc_pairs_re, self.df_tc_pairs_re_free
    
    def bootstrap_augment(self, 
                         mask: pd.Series, 
                         delta_distribution: np.ndarray,
                         column_name: str) -> np.ndarray:
        """
        Generate mock experimental values using bootstrap sampling.
        
        This method samples from the delta distribution and adds it to Tc_sim
        values. If the result is negative, it resamples until all values are positive.
        
        Parameters
        ----------
        mask : pd.Series
            Boolean mask indicating which rows to augment
        delta_distribution : np.ndarray
            Distribution of Tc_delta values to sample from
        column_name : str
            Name of the augmentation (for logging)
            
        Returns
        -------
        np.ndarray
            Array of mock experimental values (all positive)
        """
        n = mask.sum()
        print(f"\n  Generating {n} mock Tc_exp values for {column_name}...")
        
        # Check if we have any rows to process
        if n == 0:
            print("  Warning: No rows match the mask! Returning empty array.")
            return np.array([])
        
        # Initialize with impossible values
        mock_values = np.full(n, -np.inf)
        zeros = np.zeros(n)
        
        # Get the simulated values for these positions
        data = self.df_curie_temp.loc[mask, 'Tc_sim'].values
        
        iterations = 0
        max_iterations = 100  # Prevent infinite loops
        
        # Keep sampling until all values are positive
        while np.any(mock_values < zeros) and iterations < max_iterations:
            # Find positions that still need updating
            cond = mock_values < 0
            
            # Bootstrap sample residuals for only those positions
            sampled_residuals = np.random.choice(
                delta_distribution, 
                size=cond.sum(), 
                replace=True
            )
            
            # Update only the entries that are still < 0
            mock_values[cond] = data[cond] + sampled_residuals
            iterations += 1
        
        print(f"  Completed after {iterations} iteration(s)")
        print(f"  Mean: {mock_values.mean():.2f} K, Std: {mock_values.std():.2f} K")
        print(f"  Min: {mock_values.min():.2f} K, Max: {mock_values.max():.2f} K")

        return mock_values

    def bootstrap_augment_sim(self,
                              mask: pd.Series,
                              delta_distribution: np.ndarray,
                              column_name: str) -> np.ndarray:
        """
        Generate mock simulated values using bootstrap sampling.

        Mirror of bootstrap_augment: samples from the delta distribution and
        subtracts it from Tc_exp to produce mock Tc_sim = Tc_exp - sampled_delta.
        Resamples until all returned values are positive.
        """
        n = mask.sum()
        print(f"\n  Generating {n} mock Tc_sim values for {column_name}...")

        if n == 0:
            print("  Warning: No rows match the mask! Returning empty array.")
            return np.array([])

        mock_values = np.full(n, -np.inf)
        zeros = np.zeros(n)
        data = self.df_curie_temp.loc[mask, 'Tc_exp'].values

        iterations = 0
        max_iterations = 100

        while np.any(mock_values < zeros) and iterations < max_iterations:
            cond = mock_values < 0
            sampled_residuals = np.random.choice(
                delta_distribution,
                size=cond.sum(),
                replace=True
            )
            mock_values[cond] = data[cond] - sampled_residuals
            iterations += 1

        print(f"  Completed after {iterations} iteration(s)")
        print(f"  Mean: {mock_values.mean():.2f} K, Std: {mock_values.std():.2f} K")
        print(f"  Min: {mock_values.min():.2f} K, Max: {mock_values.max():.2f} K")

        return mock_values

    def create_augmented_datasets(self):
        """
        Create augmented datasets for All, RE, and RE-free materials.
        """
        print("\n" + "="*60)
        print("AUGMENTING Tc_sim-ONLY MATERIALS  [Phase 1: Tc_sim known, Tc_exp missing  →  mock Tc_exp]")
        print("="*60)

        # Identify materials with sim but no exp data
        df_sim_only = self.df_curie_temp[
            self.df_curie_temp['Tc_sim'].notna() &
            self.df_curie_temp['Tc_exp'].isna()
        ].copy()
        print(f"\nMaterials with Tc_sim but no Tc_exp: {df_sim_only.shape[0]}")

        # Create labels for mock data
        print("\nLabeling Tc_sim-only materials for Phase 1 augmentation")
        
        # All materials
        self.df_curie_temp['is_mock_all'] = self.df_curie_temp['composition'].isin(
            df_sim_only['composition']
        )
        n_mock_all = self.df_curie_temp['is_mock_all'].sum()
        
        # RE materials
        self.df_curie_temp['is_mock_re'] = (
            self.df_curie_temp['composition'].isin(df_sim_only['composition']) & 
            self.df_curie_temp['contains_rare_earth']
        )
        n_mock_re = self.df_curie_temp['is_mock_re'].sum()
        
        # RE-free materials
        self.df_curie_temp['is_mock_re_free'] = (
            self.df_curie_temp['composition'].isin(df_sim_only['composition']) & 
            ~self.df_curie_temp['contains_rare_earth']
        )
        n_mock_re_free = self.df_curie_temp['is_mock_re_free'].sum()
        
        # Generate mock data for each dataset
        print("\nGenerating mock Tc_exp values (bootstrap from paired delta distribution):")

        # All materials
        print("\n1. ALL MATERIALS  (using all-pairs delta distribution)")
        mask_all = self.df_curie_temp['is_mock_all']
        mask_all = mask_all & self.df_curie_temp['Tc_sim'].notna()
        mock_all = self.bootstrap_augment(
            mask_all,
            self.df_tc_pairs['Tc_delta'].values,
            'All materials'
        )
        self.df_curie_temp.loc[mask_all, 'Tc_exp_mock_all_bootstrap'] = mock_all

        # RE materials
        print("\n2. RE MATERIALS  (using RE-pairs delta distribution)")
        mask_re = self.df_curie_temp['is_mock_re']
        mask_re = mask_re & self.df_curie_temp['Tc_sim'].notna()
        mock_re = self.bootstrap_augment(
            mask_re,
            self.df_tc_pairs_re['Tc_delta'].values,
            'RE materials'
        )
        self.df_curie_temp.loc[mask_re, 'Tc_exp_mock_re_bootstrap'] = mock_re

        # RE-free materials
        print("\n3. RE-FREE MATERIALS  (using RE-free-pairs delta distribution)")
        mask_re_free = self.df_curie_temp['is_mock_re_free']
        mask_re_free = mask_re_free & self.df_curie_temp['Tc_sim'].notna()
        mock_re_free = self.bootstrap_augment(
            mask_re_free,
            self.df_tc_pairs_re_free['Tc_delta'].values,
            'RE-free materials'
        )
        self.df_curie_temp.loc[mask_re_free, 'Tc_exp_mock_re_free_bootstrap'] = mock_re_free

    def create_exp_only_augmented_datasets(self):
        """
        Create augmented datasets for exp-only materials (Tc_exp known, Tc_sim missing).

        Generates mock Tc_sim = Tc_exp - sampled_delta, sampling from the same paired
        delta distributions used in Phase 1, separately for All, RE, and RE-free splits.
        """
        print("\n" + "="*60)
        print("AUGMENTING Tc_exp-ONLY MATERIALS  [Phase 2: Tc_exp known, Tc_sim missing  →  mock Tc_sim]")
        print("="*60)

        df_exp_only = self.df_curie_temp[
            self.df_curie_temp['Tc_exp'].notna() &
            self.df_curie_temp['Tc_sim'].isna()
        ].copy()
        print(f"\nMaterials with Tc_exp but no Tc_sim: {df_exp_only.shape[0]}")

        print("\nLabeling Tc_exp-only materials for Phase 2 augmentation")

        self.df_curie_temp['is_mock_exp_all'] = self.df_curie_temp['composition'].isin(
            df_exp_only['composition']
        )
        self.df_curie_temp['is_mock_exp_re'] = (
            self.df_curie_temp['composition'].isin(df_exp_only['composition']) &
            self.df_curie_temp['contains_rare_earth']
        )
        self.df_curie_temp['is_mock_exp_re_free'] = (
            self.df_curie_temp['composition'].isin(df_exp_only['composition']) &
            ~self.df_curie_temp['contains_rare_earth']
        )

        print("\nGenerating mock Tc_sim values (bootstrap from paired delta distribution):")

        print("\n1. ALL MATERIALS  (using all-pairs delta distribution)")
        mask_all = self.df_curie_temp['is_mock_exp_all'] & self.df_curie_temp['Tc_exp'].notna()
        mock_all = self.bootstrap_augment_sim(
            mask_all,
            self.df_tc_pairs['Tc_delta'].values,
            'All materials (Tc_exp-only)'
        )
        self.df_curie_temp.loc[mask_all, 'Tc_sim_mock_all_bootstrap'] = mock_all

        print("\n2. RE MATERIALS  (using RE-pairs delta distribution)")
        mask_re = self.df_curie_temp['is_mock_exp_re'] & self.df_curie_temp['Tc_exp'].notna()
        mock_re = self.bootstrap_augment_sim(
            mask_re,
            self.df_tc_pairs_re['Tc_delta'].values,
            'RE materials (Tc_exp-only)'
        )
        self.df_curie_temp.loc[mask_re, 'Tc_sim_mock_re_bootstrap'] = mock_re

        print("\n3. RE-FREE MATERIALS  (using RE-free-pairs delta distribution)")
        mask_re_free = self.df_curie_temp['is_mock_exp_re_free'] & self.df_curie_temp['Tc_exp'].notna()
        mock_re_free = self.bootstrap_augment_sim(
            mask_re_free,
            self.df_tc_pairs_re_free['Tc_delta'].values,
            'RE-free materials (Tc_exp-only)'
        )
        self.df_curie_temp.loc[mask_re_free, 'Tc_sim_mock_re_free_bootstrap'] = mock_re_free

    def combine_real_and_mock(self):
        """
        Combine real experimental values with mock values.
        """
        print("\n" + "="*60)
        print("PHASE 1 — FILLING MISSING Tc_exp  (Tc_sim-only rows)")
        print("="*60)
        
        # First, keep track of original Tc_exp values 
        original_tc_exp = self.df_curie_temp['Tc_exp'].copy()
        original_tc_exp_count = original_tc_exp.notna().sum()
        #print(f"Original Tc_exp values: {original_tc_exp_count}")
        
        # Combine for All dataset (keep for compatibility)
        self.df_curie_temp['Tc_exp_all_w_mock'] = self.df_curie_temp['Tc_exp'].combine_first(
            self.df_curie_temp['Tc_exp_mock_all_bootstrap']
        )
        n_all = self.df_curie_temp['Tc_exp_all_w_mock'].notna().sum()
        #print(f"All dataset: {n_all} total samples")
        
        # Combine for RE dataset (keep for compatibility)
        self.df_curie_temp['Tc_exp_re_w_mock'] = self.df_curie_temp['Tc_exp'].combine_first(
            self.df_curie_temp['Tc_exp_mock_re_bootstrap']
        )
        n_re = self.df_curie_temp['Tc_exp_re_w_mock'].notna().sum()
        #print(f"RE dataset: {n_re} total samples")
        
        # Combine for RE-free dataset (keep for compatibility)
        self.df_curie_temp['Tc_exp_re_free_w_mock'] = self.df_curie_temp['Tc_exp'].combine_first(
            self.df_curie_temp['Tc_exp_mock_re_free_bootstrap']
        )
        n_re_free = self.df_curie_temp['Tc_exp_re_free_w_mock'].notna().sum()
        #print(f"RE-free dataset: {n_re_free} total samples")
        
        # Update the actual Tc_exp column with mock values where it's null
        print("\nInserting mock Tc_exp into Tc_sim-only rows:")
        
        # Create mask for missing Tc_exp values
        missing_exp_mask = self.df_curie_temp['Tc_exp'].isna()
        missing_count = missing_exp_mask.sum()
        
        # For compounds without rare earth, use RE-free mocks
        re_free_mask = missing_exp_mask & ~self.df_curie_temp['contains_rare_earth']
        self.df_curie_temp.loc[re_free_mask, 'Tc_exp'] = \
            self.df_curie_temp.loc[re_free_mask, 'Tc_exp_mock_re_free_bootstrap']
        print(f"  Updated RE-free compounds")
        
        # For compounds with rare earth, use RE mocks
        re_mask = missing_exp_mask & self.df_curie_temp['contains_rare_earth']
        self.df_curie_temp.loc[re_mask, 'Tc_exp'] = \
            self.df_curie_temp.loc[re_mask, 'Tc_exp_mock_re_bootstrap']
        print(f"  Updated RE compounds")
        
        # Verify no original values were lost
        changed_original = (self.df_curie_temp['Tc_exp'] != original_tc_exp) & original_tc_exp.notna()
        if changed_original.any():
            print(f"WARNING: {changed_original.sum()} original Tc_exp values were modified!")
        else:
            print("✓ All original Tc_exp values preserved correctly")
        
        # Verify all missing values were filled
        #still_missing = self.df_curie_temp['Tc_exp'].isna()
        #if still_missing.any():
        #    print(f"WARNING: {still_missing.sum()} Tc_exp values still missing")
        #else:
        #    print("✓ All missing Tc_exp values have been filled")
            
        print(f"Total samples with valid Tc_exp: {self.df_curie_temp['Tc_exp'].notna().sum()}")
        
    def calculate_deltas(self):
        """
        Calculate temperature deltas for augmented datasets.
        """
        print("\n" + "="*60)
        print("CALCULATING Tc_delta = Tc_exp − Tc_sim")
        print("="*60)

        # Tc_exp now includes mock values for Tc_sim-only rows (filled in combine_real_and_mock)
        self.df_curie_temp['Tc_delta'] = self.df_curie_temp['Tc_exp'] - self.df_curie_temp['Tc_sim']
        print("Tc_delta computed  (Tc_exp includes mock values for Tc_sim-only rows)")
        
        # Calculate delta for All dataset (keep for compatibility)
        self.df_curie_temp.loc[:, 'Tc_delta_all'] = (
            self.df_curie_temp['Tc_exp_all_w_mock'] - self.df_curie_temp['Tc_sim']
        )
        
        '''
        # Calculate delta for RE dataset (keep for compatibility)
        re_mask = self.df_curie_temp['contains_rare_earth']
        self.df_curie_temp.loc[re_mask, 'Tc_delta_re'] = (
            self.df_curie_temp.loc[re_mask, 'Tc_exp_re_w_mock'] - 
            self.df_curie_temp.loc[re_mask, 'Tc_sim']
        )
        
        # Calculate delta for RE-free dataset (keep for compatibility)
        re_free_mask = ~self.df_curie_temp['contains_rare_earth']
        self.df_curie_temp.loc[re_free_mask, 'Tc_delta_re_free'] = (
            self.df_curie_temp.loc[re_free_mask, 'Tc_exp_re_free_w_mock'] - 
            self.df_curie_temp.loc[re_free_mask, 'Tc_sim']
        )
        
        # Verify delta calculations
        delta_all_match = (self.df_curie_temp['Tc_delta'].fillna(0) == 
                          self.df_curie_temp['Tc_delta_all'].fillna(0)).mean()
        print(f"Delta matches delta_all for {delta_all_match:.1%} of compounds")
        '''
        
    @staticmethod
    def subsample_kolmogorov_smirnov_test(
        data1: np.ndarray,
        data2: np.ndarray,
        k: int = 500,
        B: int = 200,
        alpha: float = 0.05,
        aggregation: str = "median",
        random_state: int | None = None,
    ) -> dict:
        """
        Two-sample KS test robust to large sample sizes via subsampling.

        Parameters
        ----------
        data1, data2 : array-like
            The two large samples to compare.
        k : int
            Subsample size drawn from each sample per repetition.
            Rule of thumb: sqrt(min(n, m)) up to ~1000.
            Default 500.
        B : int
            Number of subsampling repetitions (200-500 recommended).
            Default 200
        alpha : float
            Significance level (default 0.05)
        aggregation : {"median", "fisher", "tippett"}
            How to combine the B p-values.
            - "median"  : robust, recommended default.
            - "fisher"  : powerful when effect is consistent.
            - "tippett" : min p-value (conservative).
        random_state : int | None
            Seed for reproducibility.

        Returns
        -------
        dict with keys:
            d_full      – KS statistic on full samples (effect size)
            p_full      – naive p-value (meaningless for large n)
            p_values    – array of B subsample p-values
            p_combined  – aggregated p-value
            reject      – bool, whether H0 is rejected at alpha
        """
        from scipy import stats as _stats
        rng = np.random.default_rng(random_state)
        s1 = np.asarray(data1)
        s2 = np.asarray(data2)

        # Full-sample D as effect size (ignore p_full for large n)
        d_full, p_full = _stats.ks_2samp(s1, s2)

        # Subsampling loop
        p_values = np.empty(B)
        for i in range(B):
            sub1 = rng.choice(s1, size=k, replace=False)
            sub2 = rng.choice(s2, size=k, replace=False)
            _, p_values[i] = _stats.ks_2samp(sub1, sub2)

        # Aggregate p-values
        if aggregation == "median":
            p_combined = float(np.median(p_values))
        elif aggregation == "fisher":
            stat = -2 * np.sum(np.log(np.clip(p_values, 1e-300, 1)))
            p_combined = float(_stats.chi2.sf(stat, df=2 * B))
        elif aggregation == "tippett":
            p_combined = float(np.min(p_values))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation!r}")

        return {
            "d_full":     d_full,
            "p_full":     p_full,
            "p_values":   p_values,
            "p_combined": p_combined,
            "reject":     p_combined < alpha,
        }

    def perform_distribution_tests(self, test_type="re_vs_re_free"):
        """
        Perform Kolmogorov-Smirnov tests to compare distributions.
        
        Parameters
        ----------
        test_type : str
            Type of test to perform: 're_vs_re_free' compares RE and RE-free original datasets,
            'original_vs_augmented' compares original and augmented datasets for validation.
        """
        from scipy import stats
        
        if test_type == "re_vs_re_free":
            print("\n" + "="*60)
            print("STATISTICAL DISTRIBUTION TESTS: RE VS RE-FREE")
            print("="*60)
            
            # Get the Tc_delta values for RE and RE-free datasets
            re_deltas = self.df_tc_pairs_re['Tc_delta'].values
            re_free_deltas = self.df_tc_pairs_re_free['Tc_delta'].values
            
            print("\nPerforming Kolmogorov-Smirnov test on RE vs RE-free Tc_delta distributions")
            print("Null hypothesis: The two distributions are the same")
            print("Alternative hypothesis: The two distributions are different")
            
            # Perform the two-sample Kolmogorov-Smirnov test
            ks_stat, p_value = stats.ks_2samp(re_deltas, re_free_deltas)
            
            print(f"\nKolmogorov-Smirnov test results:")
            print(f"  KS statistic: {ks_stat:.4f}")
            print(f"  p-value: {p_value:.8f}")
            
            # Interpret the results
            alpha = 0.05
            if p_value < alpha:
                print(f"  Result: Reject the null hypothesis (p < {alpha})")
                print("  The distributions of Tc_delta in RE and RE-free datasets are significantly different.")
                print("  This suggests that developing separate ML models for each dataset is justified.")
            else:
                print(f"  Result: Fail to reject the null hypothesis (p >= {alpha})")
                print("  We cannot conclude that the distributions of Tc_delta in RE and RE-free datasets are different.")
                print("  Developing separate ML models may not be necessary based on this test alone.")
            
            # Calculate basic statistics for both distributions
            print("\nDistribution statistics:")
            print(f"  RE materials:      mean = {np.mean(re_deltas):.2f} K, std = {np.std(re_deltas):.2f} K, n = {len(re_deltas)}")
            print(f"  RE-free materials: mean = {np.mean(re_free_deltas):.2f} K, std = {np.std(re_free_deltas):.2f} K, n = {len(re_free_deltas)}")
        
        elif test_type == "original_vs_augmented":
            print("\n" + "="*60)
            print("STATISTICAL DISTRIBUTION TESTS: ORIGINAL VS AUGMENTED")
            print("="*60)
            print("Performing Kolmogorov-Smirnov tests to verify augmentation quality")
            print("Null hypothesis: The original and augmented distributions are the same")
            print("Alternative hypothesis: The distributions are different")
            
            # Load the files to get Tc_delta distributions
            def load_deltas_from_csv(file_path):
                """Helper function to load Tc_delta values from a CSV file"""
                try:
                    # Skip MammoS header
                    df = pd.read_csv(file_path, skiprows=4)
                    # Remove info rows if any
                    if 'info' in df.columns:
                        df = df[df['info'].isna()]
                    # Return non-NaN Tc_delta values
                    return df['Tc_delta'].dropna().values
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
                    return np.array([])
            
            # RE dataset comparison
            print("\n1. RE DATASET COMPARISON")
            original_re_path = self.output_dir / 'Pairs_RE.csv'
            augmented_re_path = self.output_dir / 'Augm_sim_RE.csv'
            
            if original_re_path.exists() and augmented_re_path.exists():
                # Load delta values
                original_re_deltas = load_deltas_from_csv(original_re_path)
                augmented_re_deltas = load_deltas_from_csv(augmented_re_path)
                
                if len(original_re_deltas) > 0 and len(augmented_re_deltas) > 0:
                    # Perform KS test
                    ks_stat_re, p_value_re = stats.ks_2samp(original_re_deltas, augmented_re_deltas)
                    
                    # Print results
                    print(f"  Original RE samples: {len(original_re_deltas)}")
                    print(f"  Augmented RE samples: {len(augmented_re_deltas)}")
                    print(f"  KS statistic: {ks_stat_re:.4f}")
                    print(f"  p-value: {p_value_re:.8f}")
                    
                    # Interpret results
                    alpha = 0.05
                    if p_value_re < alpha:
                        print(f"  Result: Reject null hypothesis (p < {alpha})")
                        print("  The augmented RE delta distribution is significantly different from the original.")
                        print("  This suggests potential issues with the augmentation process for RE materials.")
                    else:
                        print(f"  Result: Fail to reject null hypothesis (p >= {alpha})")
                        print("  The augmented RE delta distribution is not significantly different from the original.")
                        print("  This suggests the augmentation maintained the statistical properties of the RE dataset.")
                    
                    # Compare means and standard deviations
                    print(f"  Original RE: mean = {np.mean(original_re_deltas):.2f} K, std = {np.std(original_re_deltas):.2f} K")
                    print(f"  Augmented RE: mean = {np.mean(augmented_re_deltas):.2f} K, std = {np.std(augmented_re_deltas):.2f} K")
                else:
                    print("  Could not compare distributions: insufficient data")
            else:
                print("  Could not compare distributions: missing files")
            
            # RE-free dataset comparison
            print("\n2. RE-FREE DATASET COMPARISON")
            original_re_free_path = self.output_dir / 'Pairs_RE_Free.csv'
            augmented_re_free_path = self.output_dir / 'Augm_sim_RE_Free.csv'
            
            if original_re_free_path.exists() and augmented_re_free_path.exists():
                # Load delta values
                original_re_free_deltas = load_deltas_from_csv(original_re_free_path)
                augmented_re_free_deltas = load_deltas_from_csv(augmented_re_free_path)
                
                if len(original_re_free_deltas) > 0 and len(augmented_re_free_deltas) > 0:
                    # Perform KS test
                    ks_stat_re_free, p_value_re_free = stats.ks_2samp(original_re_free_deltas, augmented_re_free_deltas)
                    
                    # Print results
                    print(f"  Original RE-free samples: {len(original_re_free_deltas)}")
                    print(f"  Augmented RE-free samples: {len(augmented_re_free_deltas)}")
                    print(f"  KS statistic: {ks_stat_re_free:.4f}")
                    print(f"  p-value: {p_value_re_free:.8f}")
                    
                    # Interpret results
                    alpha = 0.05
                    if p_value_re_free < alpha:
                        print(f"  Result: Reject null hypothesis (p < {alpha})")
                        print("  The augmented RE-free delta distribution is significantly different from the original.")
                        print("  This suggests potential issues with the augmentation process for RE-free materials.")
                    else:
                        print(f"  Result: Fail to reject null hypothesis (p >= {alpha})")
                        print("  The augmented RE-free delta distribution is not significantly different from the original.")
                        print("  This suggests the augmentation maintained the statistical properties of the RE-free dataset.")
                    
                    # Compare means and standard deviations
                    print(f"  Original RE-free: mean = {np.mean(original_re_free_deltas):.2f} K, std = {np.std(original_re_free_deltas):.2f} K")
                    print(f"  Augmented RE-free: mean = {np.mean(augmented_re_free_deltas):.2f} K, std = {np.std(augmented_re_free_deltas):.2f} K")
                else:
                    print("  Could not compare distributions: insufficient data")
            else:
                print("  Could not compare distributions: missing files")
        else:
            print(f"Unknown test_type: {test_type}. Valid options are 're_vs_re_free' or 'original_vs_augmented'.")

    def perform_large_sample_ks_tests(self):
        """
        Perform subsampling-based KS tests for all three splits (All, RE, RE-free).

        Complements perform_distribution_tests by using a subsampling approach that
        remains valid when sample sizes are large enough to make the naive KS p-value
        meaningless.
        """
        print("\n" + "="*60)
        print("LARGE-SAMPLE KS TESTS (SUBSAMPLING): ORIGINAL VS AUGMENTED")
        print("="*60)
        print("Using subsampling-based KS test robust to large sample sizes")
        print("Null hypothesis: The original and augmented distributions are the same")

        def load_deltas_from_csv(file_path):
            try:
                df = pd.read_csv(file_path, skiprows=4)
                if 'info' in df.columns:
                    df = df[df['info'].isna()]
                return df['Tc_delta'].dropna().values
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return np.array([])

        splits = [
            ("ALL",     self.output_dir / 'Pairs_all.csv',     self.output_dir / 'Augm_sim_all.csv'),
            ("RE",      self.output_dir / 'Pairs_RE.csv',      self.output_dir / 'Augm_sim_RE.csv'),
            ("RE-FREE", self.output_dir / 'Pairs_RE_Free.csv', self.output_dir / 'Augm_sim_RE_Free.csv'),
        ]

        for i, (name, orig_path, augm_path) in enumerate(splits, 1):
            print(f"\n{i}. {name} DATASET")
            if not orig_path.exists() or not augm_path.exists():
                print("  Skipping: missing files")
                continue

            orig = load_deltas_from_csv(orig_path)
            augm = load_deltas_from_csv(augm_path)

            if len(orig) == 0 or len(augm) == 0:
                print("  Skipping: insufficient data")
                continue

            k = min(500, len(orig), len(augm))
            result = self.subsample_kolmogorov_smirnov_test(
                orig, augm, k=k, B=200, random_state=RANDOM_SEED
            )

            print(f"  Original samples: {len(orig)}, Augmented samples: {len(augm)}")
            print(f"  Subsample size k={k}, B=200 repetitions")
            print(f"  KS D (effect size):    {result['d_full']:.4f}")
            print(f"  Naive p-value:         {result['p_full']:.2e}  (unreliable for large n)")
            print(f"  Median subsample p:    {result['p_combined']:.4f}")
            alpha = 0.05
            if result['reject']:
                print(f"  Result: Reject H0 (p_combined < {alpha})")
                print("  The augmented distribution differs significantly from the original.")
            else:
                print(f"  Result: Fail to reject H0 (p_combined >= {alpha})")
                print("  The augmented distribution is statistically consistent with the original.")

    @staticmethod
    def _load_deltas_from_csv(file_path) -> np.ndarray:
        """Load Tc_delta values from a MammoS CSV file."""
        try:
            df = pd.read_csv(file_path, skiprows=4)
            if 'info' in df.columns:
                df = df[df['info'].isna()]
            return df['Tc_delta'].dropna().values
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return np.array([])

    def _run_phase_ks_tests(self, phase_title: str, splits: list) -> None:
        """
        Run both standard and subsampling KS tests for a set of dataset splits.

        Parameters
        ----------
        phase_title : str
            Label printed in the section headers.
        splits : list of (name, pairs_path, augm_path)
            Each triple gives the split label and the two CSV files to compare.
        """
        from scipy import stats

        # --- Standard KS test ---
        print("\n" + "="*60)
        print(f"STANDARD KS TEST — {phase_title}")
        print("="*60)
        print("Comparing Tc_delta distribution:  paired data (Pairs_*.csv)  vs.  augmented dataset")

        for i, (name, orig_path, augm_path) in enumerate(splits, 1):
            print(f"\n{i}. {name} DATASET")
            if not Path(orig_path).exists() or not Path(augm_path).exists():
                print("  Skipping: missing files")
                continue
            orig = self._load_deltas_from_csv(orig_path)
            augm = self._load_deltas_from_csv(augm_path)
            if len(orig) == 0 or len(augm) == 0:
                print("  Skipping: insufficient data")
                continue
            ks_stat, p_value = stats.ks_2samp(orig, augm)
            alpha = 0.05
            print(f"  Original: {len(orig)} samples  |  Augmented: {len(augm)} samples")
            print(f"  KS statistic: {ks_stat:.4f}  |  p-value: {p_value:.8f}")
            print(f"  Original:  mean={np.mean(orig):.2f} K, std={np.std(orig):.2f} K")
            print(f"  Augmented: mean={np.mean(augm):.2f} K, std={np.std(augm):.2f} K")
            if p_value < alpha:
                print(f"  Result: Reject H0 (p < {alpha}) — distributions differ significantly.")
            else:
                print(f"  Result: Fail to reject H0 (p >= {alpha}) — distributions are statistically consistent.")

        # --- Subsampling KS test ---
        print("\n" + "="*60)
        print(f"LARGE-SAMPLE KS TEST (SUBSAMPLING) — {phase_title}")
        print("="*60)
        print("Subsampling-based test robust to large n — same comparison as standard test above")

        for i, (name, orig_path, augm_path) in enumerate(splits, 1):
            print(f"\n{i}. {name} DATASET")
            if not Path(orig_path).exists() or not Path(augm_path).exists():
                print("  Skipping: missing files")
                continue
            orig = self._load_deltas_from_csv(orig_path)
            augm = self._load_deltas_from_csv(augm_path)
            if len(orig) == 0 or len(augm) == 0:
                print("  Skipping: insufficient data")
                continue
            k = min(500, len(orig), len(augm))
            result = self.subsample_kolmogorov_smirnov_test(
                orig, augm, k=k, B=200, random_state=RANDOM_SEED
            )
            alpha = 0.05
            print(f"  Original: {len(orig)} samples  |  Augmented: {len(augm)} samples")
            print(f"  Subsample size k={k}, B=200 repetitions")
            print(f"  KS D (effect size): {result['d_full']:.4f}")
            print(f"  Naive p-value:      {result['p_full']:.2e}  (unreliable for large n)")
            print(f"  Median subsample p: {result['p_combined']:.4f}")
            if result['reject']:
                print(f"  Result: Reject H0 (p_combined < {alpha}) — distributions differ significantly.")
            else:
                print(f"  Result: Fail to reject H0 (p_combined >= {alpha}) — distributions are statistically consistent.")

    def plot_delta_distributions(self):
        """
        Generate plots of the delta distributions.
        """
        # Create plots directory
        save_dir = self.output_dir / 'distributions_plots'
        save_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*60)
        print("GENERATING DELTA DISTRIBUTION PLOTS  (Phase 1 paired-data distributions)")
        print("="*60)
        
        # Calculate optimal bins using Freedman-Diaconis rule
        def optimal_bins(data):
            data = np.asarray(data)
            q75, q25 = np.percentile(data, [75, 25])
            iqr = q75 - q25
            n = len(data)
            bin_width = 2 * iqr / (n ** (1/3))
            if bin_width == 0:
                return int(np.sqrt(n))
            bins = int(np.ceil((data.max() - data.min()) / bin_width))
            return bins
        
        # Combine all delta values for common bin calculation
        all_deltas = np.concatenate([
            self.df_tc_pairs['Tc_delta'].values,
            self.df_tc_pairs_re['Tc_delta'].values,
            self.df_tc_pairs_re_free['Tc_delta'].values
        ])
        bins = optimal_bins(all_deltas)
        print(f"Using {bins} bins for histograms")
        
        # Plot 1: Combined delta distribution
        plt.figure(figsize=(8, 6), dpi=100)
        plt.hist(self.df_tc_pairs['Tc_delta'], bins=bins, alpha=0.5, label="All")
        plt.hist(self.df_tc_pairs_re['Tc_delta'], bins=bins, color='green', 
                alpha=0.5, label="RE")
        plt.hist(self.df_tc_pairs_re_free['Tc_delta'], bins=bins, color='purple', 
                alpha=0.5, label="RE-free")
        plt.xlim(-100, 100)
        plt.xlabel("ΔTc (K)")
        plt.ylabel("Frequency")
        plt.title("Distribution of ΔTc (K)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / 'deltas_combined.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: deltas_combined.png")
        
        # Plot 2: RE-free and RE delta distributions
        fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
        
        # RE-free plot (left)
        axes[0].hist(self.df_tc_pairs_re_free['Tc_delta'], bins=80, label="RE-free", color='blue')
        axes[0].set_xlim(-400, 400)
        axes[0].set_xlabel("ΔTc (K)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("RE-free Materials")
        axes[0].legend()
        axes[0].grid(True)
        
        # RE plot (right)
        axes[1].hist(self.df_tc_pairs_re['Tc_delta'], bins=50, label="RE", color='blue')
        axes[1].set_xlim(-400, 400)
        axes[1].set_xlabel("ΔTc (K)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("RE Materials")
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'deltas_re_free_and_re.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: deltas_re_free_and_re.png")
        
    def _write_mammos_csv(self, path: Path, df: pd.DataFrame) -> None:
        """Write dataframe to a MammoS-compatible CSV file with simplified columns.

        The format mirrors EC_curie_temp.csv: a 4-line Mammos header followed by
        a standard CSV header and data rows.
        """
        # Keep only essential columns if they exist in the dataframe
        essential_columns = ['composition', 'Tc_sim', 'Tc_exp', 'Tc_delta']
        
        # For compatibility, keep other necessary columns if they exist
        possible_extra_columns = ['contains_rare_earth', 'pair_exists', 'use_for_emb']
        
        # Determine which columns to keep
        columns_to_keep = [col for col in essential_columns if col in df.columns]
        for col in possible_extra_columns:
            if col in df.columns:
                columns_to_keep.append(col)
                
        # Exclude 'info' column even if it exists
        # We no longer use info columns as requested
        
        # Create a simplified dataframe with only the necessary columns
        simplified_df = df[columns_to_keep].copy()
        
        # Write the Mammos header
        header_lines = [
            "#mammos csv v1\n",
            "#,,,,,,,,,,,,,,,,,\n",
            "#,,,,,,,,,,,,,,,,,\n",
            "#,,,,,,,,,,,,,,,,,\n",
        ]
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(header_lines)
        
        # Write the simplified dataframe - only write the columns we kept
        simplified_df.to_csv(path, mode="a", index=False)

    def save_augmented_data(self):
        """
        Save the augmented datasets to CSV files.
        """
        print("\n" + "="*60)
        print("PHASE 1 — SAVING Tc_sim-ONLY AUGMENTED DATASETS")
        print("  (paired rows  +  Tc_sim-only rows with mock Tc_exp)")
        print("="*60)
        
        # ------------------------------------------------------------------
        # 1) ORIGINAL DATA SPLITS (All / RE / RE-free) - FILTERED for pair_exists
        # ------------------------------------------------------------------
        # Only include entries that have both Tc_sim and Tc_exp (pair_exists = True)
        df_all_original = self.df_curie_temp[self.df_curie_temp['pair_exists']].copy()
        output_file_all_orig = self.output_dir / 'Pairs_all.csv'
        self._write_mammos_csv(output_file_all_orig, df_all_original)
        print(f"  Saved filtered original All dataset: {output_file_all_orig}")
        print(f"    Samples: {df_all_original.shape[0]}")

        df_re_original = df_all_original[df_all_original['contains_rare_earth']].copy()
        output_file_re_orig = self.output_dir / 'Pairs_RE.csv'
        self._write_mammos_csv(output_file_re_orig, df_re_original)
        print(f"  Saved filtered original RE dataset: {output_file_re_orig}")
        print(f"    Samples: {df_re_original.shape[0]}")

        df_re_free_original = df_all_original[~df_all_original['contains_rare_earth']].copy()
        output_file_re_free_orig = self.output_dir / 'Pairs_RE_Free.csv'
        self._write_mammos_csv(output_file_re_free_orig, df_re_free_original)
        print(f"  Saved filtered original RE-free dataset: {output_file_re_free_orig}")
        print(f"    Samples: {df_re_free_original.shape[0]}")

        # ------------------------------------------------------------------
        # 2) AUGMENTED DATA SPLITS (All / RE / RE-free)
        # ------------------------------------------------------------------
        # Filter to only include samples with valid Tc_delta_all
        # TODO: use Tc_delta_all as Tc_delta
        self.df_all_augmented = self.df_curie_temp[
            self.df_curie_temp['Tc_delta_all'].notna()
        ].copy()
        
        # Save All augmented dataset
        output_file = self.output_dir / 'Augm_sim_all.csv'
        # No longer adding info row
        self._write_mammos_csv(output_file, self.df_all_augmented)
        print(f"  Saved All dataset: {output_file}")
        print(f"    Samples: {self.df_all_augmented.shape[0]}")
        
        # Save RE augmented dataset
        self.df_re_augmented = self.df_all_augmented[
            self.df_all_augmented['contains_rare_earth']
        ].copy()
        self.df_re_augmented.loc[:, 'Tc_delta'] = (
            self.df_re_augmented['Tc_exp_re_w_mock'] - self.df_re_augmented['Tc_sim']
        )
        self.df_re_augmented = self.df_re_augmented[self.df_re_augmented['Tc_delta'].notna()].copy()

        output_file_re = self.output_dir / 'Augm_sim_RE.csv'
        self._write_mammos_csv(output_file_re, self.df_re_augmented)
        print(f"  Saved RE dataset: {output_file_re}")
        print(f"    Samples: {self.df_re_augmented.shape[0]}")

        # Save RE-free augmented dataset
        self.df_re_free_augmented = self.df_all_augmented[
            ~self.df_all_augmented['contains_rare_earth']
        ].copy()
        self.df_re_free_augmented.loc[:, 'Tc_delta'] = (
            self.df_re_free_augmented['Tc_exp_re_free_w_mock'] -
            self.df_re_free_augmented['Tc_sim']
        )
        self.df_re_free_augmented = self.df_re_free_augmented[
            self.df_re_free_augmented['Tc_delta'].notna()
        ].copy()

        output_file_re_free = self.output_dir / 'Augm_sim_RE_Free.csv'
        self._write_mammos_csv(output_file_re_free, self.df_re_free_augmented)
        print(f"  Saved RE-free dataset: {output_file_re_free}")
        print(f"    Samples: {self.df_re_free_augmented.shape[0]}")
        
        # ------------------------------------------------------------------
        # 3) ORIGINAL DATA SPLITS WITH use_for_emb == True
        # 4) AUGMENTED DATA SPLITS WITH use_for_emb == True
        # ------------------------------------------------------------------
        if 'use_for_emb' in self.df_curie_temp.columns:
            print("\nCreating embedding-ready datasets based on use_for_emb == True")

            # Original data, embedding-compatible - FILTERED
            df_all_orig_emb = self.df_curie_temp[
                self.df_curie_temp['use_for_emb'] & self.df_curie_temp['pair_exists']
            ].copy()
            print(f"  Original filtered embedding-compatible samples (All): {df_all_orig_emb.shape[0]}")

            output_file_all_orig_emb = self.output_dir / 'Pairs_all_emb.csv'
            self._write_mammos_csv(output_file_all_orig_emb, df_all_orig_emb)
            print(f"  Saved original filtered All embedding dataset: {output_file_all_orig_emb}")
            print(f"    Samples: {df_all_orig_emb.shape[0]}")

            df_re_orig_emb = df_all_orig_emb[df_all_orig_emb['contains_rare_earth']].copy()
            output_file_re_orig_emb = self.output_dir / 'Pairs_RE_emb.csv'
            self._write_mammos_csv(output_file_re_orig_emb, df_re_orig_emb)
            print(f"  Saved original filtered RE embedding dataset: {output_file_re_orig_emb}")
            print(f"    Samples: {df_re_orig_emb.shape[0]}")

            df_re_free_orig_emb = df_all_orig_emb[~df_all_orig_emb['contains_rare_earth']].copy()
            output_file_re_free_orig_emb = self.output_dir / 'Pairs_RE_Free_emb.csv'
            self._write_mammos_csv(output_file_re_free_orig_emb, df_re_free_orig_emb)
            print(f"  Saved original filtered RE-free embedding dataset: {output_file_re_free_orig_emb}")
            print(f"    Samples: {df_re_free_orig_emb.shape[0]}")

            # Augmented data, embedding-compatible
            if 'use_for_emb' in self.df_all_augmented.columns:
                df_all_aug_emb = self.df_all_augmented[self.df_all_augmented['use_for_emb']].copy()
                print(f"  Augmented embedding-compatible samples (All): {df_all_aug_emb.shape[0]}")

                output_file_all_aug_emb = self.output_dir / 'Augm_sim_all_emb.csv'
                self._write_mammos_csv(output_file_all_aug_emb, df_all_aug_emb)
                print(f"  Saved augmented All embedding dataset: {output_file_all_aug_emb}")
                print(f"    Samples: {df_all_aug_emb.shape[0]}")

                df_re_aug_emb = df_all_aug_emb[df_all_aug_emb['contains_rare_earth']].copy()
                output_file_re_aug_emb = self.output_dir / 'Augm_sim_RE_emb.csv'
                self._write_mammos_csv(output_file_re_aug_emb, df_re_aug_emb)
                print(f"  Saved augmented RE embedding dataset: {output_file_re_aug_emb}")
                print(f"    Samples: {df_re_aug_emb.shape[0]}")

                df_re_free_aug_emb = df_all_aug_emb[~df_all_aug_emb['contains_rare_earth']].copy()
                output_file_re_free_aug_emb = self.output_dir / 'Augm_sim_RE_Free_emb.csv'
                self._write_mammos_csv(output_file_re_free_aug_emb, df_re_free_aug_emb)
                print(f"  Saved augmented RE-free embedding dataset: {output_file_re_free_aug_emb}")
                print(f"    Samples: {df_re_free_aug_emb.shape[0]}")
            else:
                print("  Column 'use_for_emb' not found in augmented data; skipping augmented embedding-ready splits.")
        else:
            print("\nColumn 'use_for_emb' not found in original dataset; skipping embedding-ready CSV creation.")

    def save_phase2_data(self):
        """
        Save exp-only augmented datasets (Phase 2) to CSV files.

        Builds each split by concatenating the original paired rows with exp-only rows
        where Tc_sim has been replaced by the bootstrap mock value. Stores the
        exp-only DataFrames as instance variables for use in save_phase3_data.

        Output files: Augm_exp_all.csv, Augm_exp_RE.csv, Augm_exp_RE_Free.csv
        (plus _emb variants).
        """
        print("\n" + "="*60)
        print("PHASE 2 — SAVING Tc_exp-ONLY AUGMENTED DATASETS")
        print("  (paired rows  +  Tc_exp-only rows with mock Tc_sim)")
        print("="*60)

        df_paired = self.df_curie_temp[self.df_curie_temp['pair_exists']].copy()

        # --- All split ---
        mask_all = (
            self.df_curie_temp['is_mock_exp_all'] &
            self.df_curie_temp['Tc_sim_mock_all_bootstrap'].notna()
        )
        self.df_exp_only_all = self.df_curie_temp[mask_all].copy()
        self.df_exp_only_all['Tc_sim'] = self.df_exp_only_all['Tc_sim_mock_all_bootstrap']
        self.df_exp_only_all['Tc_delta'] = (
            self.df_exp_only_all['Tc_exp'] - self.df_exp_only_all['Tc_sim']
        )
        self.df_exp_only_all = self.df_exp_only_all[
            self.df_exp_only_all['Tc_delta'].notna()
        ].copy()

        df_phase2_all = pd.concat([df_paired, self.df_exp_only_all], ignore_index=True)
        output_file = self.output_dir / 'Augm_exp_all.csv'
        self._write_mammos_csv(output_file, df_phase2_all)
        print(f"  Saved exp-only All dataset: {output_file}")
        print(f"    Samples: {df_phase2_all.shape[0]}")

        # --- RE split ---
        mask_re = (
            self.df_curie_temp['is_mock_exp_re'] &
            self.df_curie_temp['Tc_sim_mock_re_bootstrap'].notna()
        )
        self.df_exp_only_re = self.df_curie_temp[mask_re].copy()
        self.df_exp_only_re['Tc_sim'] = self.df_exp_only_re['Tc_sim_mock_re_bootstrap']
        self.df_exp_only_re['Tc_delta'] = (
            self.df_exp_only_re['Tc_exp'] - self.df_exp_only_re['Tc_sim']
        )
        self.df_exp_only_re = self.df_exp_only_re[
            self.df_exp_only_re['Tc_delta'].notna()
        ].copy()

        df_phase2_re = pd.concat(
            [df_paired[df_paired['contains_rare_earth']], self.df_exp_only_re],
            ignore_index=True
        )
        output_file_re = self.output_dir / 'Augm_exp_RE.csv'
        self._write_mammos_csv(output_file_re, df_phase2_re)
        print(f"  Saved exp-only RE dataset: {output_file_re}")
        print(f"    Samples: {df_phase2_re.shape[0]}")

        # --- RE-free split ---
        mask_re_free = (
            self.df_curie_temp['is_mock_exp_re_free'] &
            self.df_curie_temp['Tc_sim_mock_re_free_bootstrap'].notna()
        )
        self.df_exp_only_re_free = self.df_curie_temp[mask_re_free].copy()
        self.df_exp_only_re_free['Tc_sim'] = self.df_exp_only_re_free['Tc_sim_mock_re_free_bootstrap']
        self.df_exp_only_re_free['Tc_delta'] = (
            self.df_exp_only_re_free['Tc_exp'] - self.df_exp_only_re_free['Tc_sim']
        )
        self.df_exp_only_re_free = self.df_exp_only_re_free[
            self.df_exp_only_re_free['Tc_delta'].notna()
        ].copy()

        df_phase2_re_free = pd.concat(
            [df_paired[~df_paired['contains_rare_earth']], self.df_exp_only_re_free],
            ignore_index=True
        )
        output_file_re_free = self.output_dir / 'Augm_exp_RE_Free.csv'
        self._write_mammos_csv(output_file_re_free, df_phase2_re_free)
        print(f"  Saved exp-only RE-free dataset: {output_file_re_free}")
        print(f"    Samples: {df_phase2_re_free.shape[0]}")

        # --- Embedding-compatible versions ---
        if 'use_for_emb' in self.df_curie_temp.columns:
            print("\nCreating embedding-ready exp-only datasets")

            for df, name in [
                (df_phase2_all,      'Augm_exp_all_emb.csv'),
                (df_phase2_re,       'Augm_exp_RE_emb.csv'),
                (df_phase2_re_free,  'Augm_exp_RE_Free_emb.csv'),
            ]:
                df_emb = df[df['use_for_emb'].eq(True)].copy()
                self._write_mammos_csv(self.output_dir / name, df_emb)
                print(f"  Saved {name}: {df_emb.shape[0]} samples")

    def save_phase3_data(self):
        """
        Save combined (sim-only + exp-only) augmented datasets (Phase 3) to CSV files.

        Concatenates the Phase 1 augmented rows (paired + sim-only with mock Tc_exp)
        with the Phase 2 exp-only rows (mock Tc_sim). The two sets are disjoint so
        there is no double-counting of paired rows.

        Output files: Augm_combined_all.csv, Augm_combined_RE.csv,
        Augm_combined_RE_Free.csv (plus _emb variants).
        """
        print("\n" + "="*60)
        print("PHASE 3 — SAVING COMBINED AUGMENTED DATASETS")
        print("  (paired rows  +  Tc_sim-only with mock Tc_exp  +  Tc_exp-only with mock Tc_sim)")
        print("="*60)

        # --- All split ---
        df_combined_all = pd.concat(
            [self.df_all_augmented, self.df_exp_only_all],
            ignore_index=True
        )
        output_file = self.output_dir / 'Augm_combined_all.csv'
        self._write_mammos_csv(output_file, df_combined_all)
        print(f"  Saved combined All dataset: {output_file}")
        print(f"    Samples: {df_combined_all.shape[0]}")

        # --- RE split ---
        df_combined_re = pd.concat(
            [self.df_re_augmented, self.df_exp_only_re],
            ignore_index=True
        )
        output_file_re = self.output_dir / 'Augm_combined_RE.csv'
        self._write_mammos_csv(output_file_re, df_combined_re)
        print(f"  Saved combined RE dataset: {output_file_re}")
        print(f"    Samples: {df_combined_re.shape[0]}")

        # --- RE-free split ---
        df_combined_re_free = pd.concat(
            [self.df_re_free_augmented, self.df_exp_only_re_free],
            ignore_index=True
        )
        output_file_re_free = self.output_dir / 'Augm_combined_RE_Free.csv'
        self._write_mammos_csv(output_file_re_free, df_combined_re_free)
        print(f"  Saved combined RE-free dataset: {output_file_re_free}")
        print(f"    Samples: {df_combined_re_free.shape[0]}")

        # --- Embedding-compatible versions ---
        if 'use_for_emb' in self.df_curie_temp.columns:
            print("\nCreating embedding-ready combined datasets")

            for df, name in [
                (df_combined_all,      'Augm_combined_all_emb.csv'),
                (df_combined_re,       'Augm_combined_RE_emb.csv'),
                (df_combined_re_free,  'Augm_combined_RE_Free_emb.csv'),
            ]:
                df_emb = df[df['use_for_emb'].eq(True)].copy()
                self._write_mammos_csv(self.output_dir / name, df_emb)
                print(f"  Saved {name}: {df_emb.shape[0]} samples")

    def print_summary(self):
        """
        Print a summary of the augmentation results by reading output files directly.
        """
        print("\n" + "="*60)
        print("AUGMENTATION SUMMARY (FROM OUTPUT FILES)")
        print("="*60)
        
        def read_csv_shape(path):
            """Helper function to read a CSV file and get its shape"""
            try:
                # Skip the MammoS header (first 4 lines)
                df = pd.read_csv(path, skiprows=4)
                
                # We no longer include info columns, but check for backward compatibility
                if 'info' in df.columns:
                    # Remove rows with info text if they exist
                    df = df[df['info'].isna() | df['info'].str.startswith('# Info:').fillna(False)]
                return df.shape[0]
            except Exception as e:
                return f"Error: {e}"
        
        # Define all the output file paths
        original_files = {
            'All': self.output_dir / 'Pairs_all.csv',
            'RE': self.output_dir / 'Pairs_RE.csv',
            'RE-free': self.output_dir / 'Pairs_RE_Free.csv',
            'All (emb)': self.output_dir / 'Pairs_all_emb.csv',
            'RE (emb)': self.output_dir / 'Pairs_RE_emb.csv',
            'RE-free (emb)': self.output_dir / 'Pairs_RE_Free_emb.csv',
        }
        
        phase1_files = {
            'All':          self.output_dir / 'Augm_sim_all.csv',
            'RE':           self.output_dir / 'Augm_sim_RE.csv',
            'RE-free':      self.output_dir / 'Augm_sim_RE_Free.csv',
            'All (emb)':    self.output_dir / 'Augm_sim_all_emb.csv',
            'RE (emb)':     self.output_dir / 'Augm_sim_RE_emb.csv',
            'RE-free (emb)':self.output_dir / 'Augm_sim_RE_Free_emb.csv',
        }
        phase2_files = {
            'All':          self.output_dir / 'Augm_exp_all.csv',
            'RE':           self.output_dir / 'Augm_exp_RE.csv',
            'RE-free':      self.output_dir / 'Augm_exp_RE_Free.csv',
            'All (emb)':    self.output_dir / 'Augm_exp_all_emb.csv',
            'RE (emb)':     self.output_dir / 'Augm_exp_RE_emb.csv',
            'RE-free (emb)':self.output_dir / 'Augm_exp_RE_Free_emb.csv',
        }
        phase3_files = {
            'All':          self.output_dir / 'Augm_combined_all.csv',
            'RE':           self.output_dir / 'Augm_combined_RE.csv',
            'RE-free':      self.output_dir / 'Augm_combined_RE_Free.csv',
            'All (emb)':    self.output_dir / 'Augm_combined_all_emb.csv',
            'RE (emb)':     self.output_dir / 'Augm_combined_RE_emb.csv',
            'RE-free (emb)':self.output_dir / 'Augm_combined_RE_Free_emb.csv',
        }

        def read_sizes(label, file_dict):
            print(f"\nReading {label} dataset files...")
            sizes = {}
            for name, path in file_dict.items():
                if path.exists():
                    sizes[name] = read_csv_shape(path)
                    print(f"  {name}: {sizes[name]} samples")
                else:
                    sizes[name] = 'N/A'
                    print(f"  {name}: File not found")
            return sizes

        original_sizes = read_sizes("original (Pairs)", original_files)
        phase1_sizes   = read_sizes("Phase 1 (sim-only)", phase1_files)
        phase2_sizes   = read_sizes("Phase 2 (exp-only)", phase2_files)
        phase3_sizes   = read_sizes("Phase 3 (combined)", phase3_files)

        # Print formatted summary table
        print("\n" + "="*60)
        print("DATASET SIZE SUMMARY")
        print("="*60)

        col = 14
        header = f"\n{'Dataset':<20} {'Original':>{col}} {'Phase1(sim)':>{col}} {'Phase2(exp)':>{col}} {'Phase3(comb)':>{col}}"
        print(header)
        print("-" * (20 + 4 * col + 3))
        for name in ['All', 'RE', 'RE-free']:
            print(
                f"{name:<20}"
                f" {original_sizes[name]!s:>{col}}"
                f" {phase1_sizes[name]!s:>{col}}"
                f" {phase2_sizes[name]!s:>{col}}"
                f" {phase3_sizes[name]!s:>{col}}"
            )

        print(f"\n{'Embedding Dataset':<20} {'Original':>{col}} {'Phase1(sim)':>{col}} {'Phase2(exp)':>{col}} {'Phase3(comb)':>{col}}")
        print("-" * (20 + 4 * col + 3))
        for name in ['All (emb)', 'RE (emb)', 'RE-free (emb)']:
            print(
                f"{name:<20}"
                f" {original_sizes.get(name, 'N/A')!s:>{col}}"
                f" {phase1_sizes.get(name, 'N/A')!s:>{col}}"
                f" {phase2_sizes.get(name, 'N/A')!s:>{col}}"
                f" {phase3_sizes.get(name, 'N/A')!s:>{col}}"
            )
        
        print("\n" + "="*60)
        print("AUGMENTATION COMPLETE!")
        print("="*60)
        
    def compare_original_vs_augmented_distributions(self):
        """
        Perform Kolmogorov-Smirnov tests to compare original and augmented delta distributions.
        This helps verify that the augmentation process maintained the same statistical properties.
        """
        from scipy import stats
        import numpy as np
        import pandas as pd
        
        print("\n" + "="*60)
        print("COMPARING ORIGINAL VS AUGMENTED DISTRIBUTIONS")
        print("="*60)
        print("Performing Kolmogorov-Smirnov tests to verify augmentation quality")
        print("Null hypothesis: The original and augmented distributions are the same")
        print("Alternative hypothesis: The distributions are different")
        
        # Load the files to get Tc_delta distributions
        def load_deltas_from_csv(file_path):
            """Helper function to load Tc_delta values from a CSV file"""
            try:
                # Skip MammoS header
                df = pd.read_csv(file_path, skiprows=4)
                # Remove info rows if any
                if 'info' in df.columns:
                    df = df[df['info'].isna()]
                # Return non-NaN Tc_delta values
                return df['Tc_delta'].dropna().values
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return np.array([])
        
        # RE dataset comparison
        print("\n1. RE DATASET COMPARISON")
        original_re_path = self.output_dir / 'Pairs_RE.csv'
        augmented_re_path = self.output_dir / 'Augm_sim_RE.csv'
        
        if original_re_path.exists() and augmented_re_path.exists():
            # Load delta values
            original_re_deltas = load_deltas_from_csv(original_re_path)
            augmented_re_deltas = load_deltas_from_csv(augmented_re_path)
            
            if len(original_re_deltas) > 0 and len(augmented_re_deltas) > 0:
                # Perform KS test
                ks_stat_re, p_value_re = stats.ks_2samp(original_re_deltas, augmented_re_deltas)
                
                # Print results
                print(f"  Original RE samples: {len(original_re_deltas)}")
                print(f"  Augmented RE samples: {len(augmented_re_deltas)}")
                print(f"  KS statistic: {ks_stat_re:.4f}")
                print(f"  p-value: {p_value_re:.8f}")
                
                # Interpret results
                alpha = 0.05
                if p_value_re < alpha:
                    print(f"  Result: Reject null hypothesis (p < {alpha})")
                    print("  The augmented RE delta distribution is significantly different from the original.")
                    print("  This suggests potential issues with the augmentation process for RE materials.")
                else:
                    print(f"  Result: Fail to reject null hypothesis (p >= {alpha})")
                    print("  The augmented RE delta distribution is not significantly different from the original.")
                    print("  This suggests the augmentation maintained the statistical properties of the RE dataset.")
                
                # Compare means and standard deviations
                print(f"  Original RE: mean = {np.mean(original_re_deltas):.2f} K, std = {np.std(original_re_deltas):.2f} K")
                print(f"  Augmented RE: mean = {np.mean(augmented_re_deltas):.2f} K, std = {np.std(augmented_re_deltas):.2f} K")
            else:
                print("  Could not compare distributions: insufficient data")
        else:
            print("  Could not compare distributions: missing files")
        
        # RE-free dataset comparison
        print("\n2. RE-FREE DATASET COMPARISON")
        original_re_free_path = self.output_dir / 'Pairs_RE_Free.csv'
        augmented_re_free_path = self.output_dir / 'Augm_sim_RE_Free.csv'
        
        if original_re_free_path.exists() and augmented_re_free_path.exists():
            # Load delta values
            original_re_free_deltas = load_deltas_from_csv(original_re_free_path)
            augmented_re_free_deltas = load_deltas_from_csv(augmented_re_free_path)
            
            if len(original_re_free_deltas) > 0 and len(augmented_re_free_deltas) > 0:
                # Perform KS test
                ks_stat_re_free, p_value_re_free = stats.ks_2samp(original_re_free_deltas, augmented_re_free_deltas)
                
                # Print results
                print(f"  Original RE-free samples: {len(original_re_free_deltas)}")
                print(f"  Augmented RE-free samples: {len(augmented_re_free_deltas)}")
                print(f"  KS statistic: {ks_stat_re_free:.4f}")
                print(f"  p-value: {p_value_re_free:.8f}")
                
                # Interpret results
                alpha = 0.05
                if p_value_re_free < alpha:
                    print(f"  Result: Reject null hypothesis (p < {alpha})")
                    print("  The augmented RE-free delta distribution is significantly different from the original.")
                    print("  This suggests potential issues with the augmentation process for RE-free materials.")
                else:
                    print(f"  Result: Fail to reject null hypothesis (p >= {alpha})")
                    print("  The augmented RE-free delta distribution is not significantly different from the original.")
                    print("  This suggests the augmentation maintained the statistical properties of the RE-free dataset.")
                
                # Compare means and standard deviations
                print(f"  Original RE-free: mean = {np.mean(original_re_free_deltas):.2f} K, std = {np.std(original_re_free_deltas):.2f} K")
                print(f"  Augmented RE-free: mean = {np.mean(augmented_re_free_deltas):.2f} K, std = {np.std(augmented_re_free_deltas):.2f} K")
            else:
                print("  Could not compare distributions: insufficient data")
        else:
            print("  Could not compare distributions: missing files")
       
    def run(self):
        """
        Execute the complete three-phase augmentation pipeline.

        Phase 1 — sim-only: generate mock Tc_exp for materials that have only Tc_sim.
        Phase 2 — exp-only: generate mock Tc_sim for materials that have only Tc_exp.
        Phase 3 — combined: merge Phase 1 and Phase 2 augmented rows into one dataset.

        Both a standard KS test and a subsampling-based KS test are run after each
        phase's CSV files are written.
        """
        print("="*60)
        print("CURIE TEMPERATURE DATA AUGMENTATION")
        print("="*60)
        print(f"Random seed: {RANDOM_SEED}")

        self.load_data()
        self.filter_paired_data()
        self.perform_distribution_tests(test_type="re_vs_re_free")

        # ── Phase 1: Sim-only augmentation ────────────────────────────────
        print("\n" + "="*60)
        print("PHASE 1 — Tc_sim-ONLY MATERIALS")
        print("  Tc_sim known, Tc_exp missing  →  bootstrap mock Tc_exp")
        print("="*60)

        self.create_augmented_datasets()
        self.combine_real_and_mock()
        self.calculate_deltas()
        self.plot_delta_distributions()
        self.save_augmented_data()

        self._run_phase_ks_tests(
            "PHASE 1: SIM-ONLY AUGMENTATION",
            [
                ("ALL",     self.output_dir / 'Pairs_all.csv',     self.output_dir / 'Augm_sim_all.csv'),
                ("RE",      self.output_dir / 'Pairs_RE.csv',      self.output_dir / 'Augm_sim_RE.csv'),
                ("RE-FREE", self.output_dir / 'Pairs_RE_Free.csv', self.output_dir / 'Augm_sim_RE_Free.csv'),
            ]
        )

        # ── Phase 2: Exp-only augmentation ────────────────────────────────
        print("\n" + "="*60)
        print("PHASE 2 — Tc_exp-ONLY MATERIALS")
        print("  Tc_exp known, Tc_sim missing  →  bootstrap mock Tc_sim")
        print("="*60)

        self.create_exp_only_augmented_datasets()
        self.save_phase2_data()

        self._run_phase_ks_tests(
            "PHASE 2: EXP-ONLY AUGMENTATION",
            [
                ("ALL",     self.output_dir / 'Pairs_all.csv',     self.output_dir / 'Augm_exp_all.csv'),
                ("RE",      self.output_dir / 'Pairs_RE.csv',      self.output_dir / 'Augm_exp_RE.csv'),
                ("RE-FREE", self.output_dir / 'Pairs_RE_Free.csv', self.output_dir / 'Augm_exp_RE_Free.csv'),
            ]
        )

        # ── Phase 3: Combined augmentation ────────────────────────────────
        print("\n" + "="*60)
        print("PHASE 3 — COMBINED AUGMENTATION")
        print("  Phase 1 (Tc_sim-only with mock Tc_exp)  +  Phase 2 (Tc_exp-only with mock Tc_sim)")
        print("="*60)

        self.save_phase3_data()

        self._run_phase_ks_tests(
            "PHASE 3: COMBINED AUGMENTATION",
            [
                ("ALL",     self.output_dir / 'Pairs_all.csv',     self.output_dir / 'Augm_combined_all.csv'),
                ("RE",      self.output_dir / 'Pairs_RE.csv',      self.output_dir / 'Augm_combined_RE.csv'),
                ("RE-FREE", self.output_dir / 'Pairs_RE_Free.csv', self.output_dir / 'Augm_combined_RE_Free.csv'),
            ]
        )

        # Summary table (sizes only — KS tests already printed inline above)
        self.print_summary()

@log_output('logs/augment_data.txt')    
def augment_data():
    """Main execution function."""
    # Determine project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # Set up paths - output to project-level 'outputs' directory
    data_path = project_root / 'data' / 'EC_curie_temp.csv'
    output_dir = project_root / 'outputs'  # Standardized output directory
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if input file exists
    if not data_path.exists():
        print(f"Error: Input file not found at {data_path}")
        print("Please ensure EC_curie_temp.csv exists in the data directory.")
        sys.exit(1)
    
    print(f"Input: {data_path}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create augmenter and run
    augmenter = CurieTempAugmenter(str(data_path), str(output_dir))
    augmenter.run()


if __name__ == '__main__':
    augment_data()
