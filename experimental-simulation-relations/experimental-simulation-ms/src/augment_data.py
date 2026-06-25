# -*- coding: utf-8 -*-
"""Data Augmentation for Ms Dataset

Implements the same three-phase bootstrap augmentation as the TC project,
adapted for Ms values. Because Ms values span many orders of magnitude,
bootstrap residuals are computed in log1p-space:

    log_delta = log1p(Ms_exp) - log1p(Ms_sim)

Phase 1 — sim-only:  rows with Ms_sim but no Ms_exp → mock Ms_exp
Phase 2 — exp-only:  rows with Ms_exp but no Ms_sim → mock Ms_sim
Phase 3 — combined:  Phase 1 rows  +  Phase 2 exp-only rows

Output (plain CSV, all/RE/RE-free splits, three phases):
    outputs/Pairs_*.csv          original pairs
    outputs/Augm_sim_*.csv       Phase 1
    outputs/Augm_exp_*.csv       Phase 2
    outputs/Augm_combined_*.csv  Phase 3

Usage:
    python -m src.augment_data
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

sns.set_style("whitegrid")

from src.log_to_file import log_output

log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(log_dir, exist_ok=True)

SIM_COL = 'Ms (ampere/meter)_s'
EXP_COL = 'Ms (ampere/meter)_e'
RE_COL  = 'has_rare_earth'

# Threshold separating the "poor-DFT" regime (Ms_sim < threshold, log_delta ~ 4–8)
# from the "normal" regime (Ms_sim >= threshold, log_delta ~ 0).  Each regime draws
# bootstrap residuals only from same-regime pairs (Option A).
REGIME_THRESHOLD = 10_000  # A/m


class MsAugmenter:
    """Bootstrap augmentation for the Ms sim↔exp dataset."""

    def __init__(self, data_path: str, output_dir: str, ms_threshold: float = 50_000):
        self.data_path    = Path(data_path)
        self.output_dir   = Path(output_dir)
        self.ms_threshold = ms_threshold if ms_threshold > 0 else None
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df          = None   # full dataset
        self.df_pairs    = None   # rows with both sim and exp
        self.df_pairs_re = None
        self.df_pairs_re_free = None

        # Filled by create_phase1 / create_phase2:
        self.df_phase1_all      = None   # pairs + sim-only with mock exp
        self.df_phase1_re       = None
        self.df_phase1_re_free  = None
        self.df_exp_only_all    = None   # exp-only rows with mock sim (Phase 2)
        self.df_exp_only_re     = None
        self.df_exp_only_re_free= None

    # ── Data loading ──────────────────────────────────────────────────────

    def load_data(self) -> pd.DataFrame:
        print(f"Loading data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        print(f"  Total rows: {len(self.df)}")
        print(f"  Rows with Ms_sim:  {self.df[SIM_COL].notna().sum()}")
        print(f"  Rows with Ms_exp:  {self.df[EXP_COL].notna().sum()}")
        print(f"  Pairs (both):      {(self.df[SIM_COL].notna() & self.df[EXP_COL].notna()).sum()}")
        print(f"  Sim-only:          {(self.df[SIM_COL].notna() & self.df[EXP_COL].isna()).sum()}")
        print(f"  Exp-only:          {(self.df[SIM_COL].isna()  & self.df[EXP_COL].notna()).sum()}")
        return self.df

    def filter_paired_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("\n" + "="*60)
        print("FILTERING PAIRED DATA")
        print("="*60)

        self.df_pairs = self.df[
            self.df[SIM_COL].notna() & self.df[EXP_COL].notna()
        ].copy()

        if self.ms_threshold is not None:
            before = len(self.df_pairs)
            self.df_pairs = self.df_pairs[
                (self.df_pairs[SIM_COL] > self.ms_threshold) &
                (self.df_pairs[EXP_COL] > self.ms_threshold)
            ].copy()
            print(f"Ms threshold {self.ms_threshold:.0f} A/m: "
                  f"dropped {before - len(self.df_pairs)} pairs "
                  f"({before} → {len(self.df_pairs)} kept)")

        # Log-space delta for bootstrap sampling
        self.df_pairs['Ms_log_delta'] = (
            np.log1p(self.df_pairs[EXP_COL]) - np.log1p(self.df_pairs[SIM_COL])
        )
        self.df_pairs['Ms_delta'] = self.df_pairs[EXP_COL] - self.df_pairs[SIM_COL]

        self.df_pairs_re      = self.df_pairs[self.df_pairs[RE_COL] == True].copy()
        self.df_pairs_re_free = self.df_pairs[self.df_pairs[RE_COL] == False].copy()

        # Regime-stratified subsets for Option A bootstrap (by Ms_sim magnitude)
        def _split_regime(df):
            poor = df[df[SIM_COL] < REGIME_THRESHOLD].copy()
            norm = df[df[SIM_COL] >= REGIME_THRESHOLD].copy()
            return poor, norm

        self.df_pairs_poor,    self.df_pairs_norm    = _split_regime(self.df_pairs)
        self.df_pairs_re_poor, self.df_pairs_re_norm = _split_regime(self.df_pairs_re)
        self.df_pairs_rf_poor, self.df_pairs_rf_norm = _split_regime(self.df_pairs_re_free)

        print(f"All pairs:     {len(self.df_pairs)}")
        print(f"  poor-DFT (Ms_sim < {REGIME_THRESHOLD:.0e}): {len(self.df_pairs_poor)}"
              f"  normal: {len(self.df_pairs_norm)}")
        print(f"RE pairs:      {len(self.df_pairs_re)}")
        print(f"  poor-DFT: {len(self.df_pairs_re_poor)}  normal: {len(self.df_pairs_re_norm)}")
        print(f"RE-free pairs: {len(self.df_pairs_re_free)}")
        print(f"  poor-DFT: {len(self.df_pairs_rf_poor)}  normal: {len(self.df_pairs_rf_norm)}")
        return self.df_pairs, self.df_pairs_re, self.df_pairs_re_free

    # ── Bootstrap helpers ─────────────────────────────────────────────────

    def _bootstrap_exp(
        self, df_to_aug: pd.DataFrame, log_delta_dist: np.ndarray, label: str
    ) -> np.ndarray:
        """Generate mock Ms_exp via log-space bootstrap (Phase 1)."""
        n = len(df_to_aug)
        if n == 0:
            print(f"  {label}: no rows — skipping")
            return np.array([])

        log_sim = np.log1p(df_to_aug[SIM_COL].values)
        mock_log = np.full(n, -np.inf)
        iterations = 0

        while np.any(mock_log <= 0) and iterations < 200:
            cond = mock_log <= 0
            sampled = np.random.choice(log_delta_dist, size=cond.sum(), replace=True)
            mock_log[cond] = log_sim[cond] + sampled
            iterations += 1

        mock_exp = np.expm1(mock_log)
        print(f"  {label}: {n} mock Ms_exp values, {iterations} iter — "
              f"mean={mock_exp.mean():.2e}, min={mock_exp.min():.2e}")
        return mock_exp

    def _bootstrap_sim(
        self, df_to_aug: pd.DataFrame, log_delta_dist: np.ndarray, label: str
    ) -> np.ndarray:
        """Generate mock Ms_sim via log-space bootstrap (Phase 2)."""
        n = len(df_to_aug)
        if n == 0:
            print(f"  {label}: no rows — skipping")
            return np.array([])

        log_exp = np.log1p(df_to_aug[EXP_COL].values)
        mock_log = np.full(n, -np.inf)
        iterations = 0

        while np.any(mock_log <= 0) and iterations < 200:
            cond = mock_log <= 0
            sampled = np.random.choice(log_delta_dist, size=cond.sum(), replace=True)
            mock_log[cond] = log_exp[cond] - sampled
            iterations += 1

        mock_sim = np.expm1(mock_log)
        print(f"  {label}: {n} mock Ms_sim values, {iterations} iter — "
              f"mean={mock_sim.mean():.2e}, min={mock_sim.min():.2e}")
        return mock_sim

    def _bootstrap_exp_stratified(
        self,
        df_to_aug: pd.DataFrame,
        pairs_poor: pd.DataFrame,
        pairs_norm: pd.DataFrame,
        label: str,
    ) -> np.ndarray:
        """Regime-stratified bootstrap for Phase 1 (Option A).

        Rows with Ms_sim < REGIME_THRESHOLD draw residuals from same-regime pairs;
        rows with Ms_sim >= REGIME_THRESHOLD draw from the normal-regime pairs.
        This prevents zero-sim residuals (~log_delta 7–8) from being applied to
        rows where DFT worked normally, eliminating the upward bias seen with the
        pooled bootstrap.
        """
        n = len(df_to_aug)
        if n == 0:
            print(f"  {label}: no rows — skipping")
            return np.array([])

        mask_poor = df_to_aug[SIM_COL].values < REGIME_THRESHOLD
        mask_norm = ~mask_poor
        log_sim   = np.log1p(df_to_aug[SIM_COL].values)
        mock_log  = np.full(n, np.nan)

        def _fill(mask, pool_df, regime_name):
            if mask.sum() == 0:
                return
            if len(pool_df) == 0:
                # Fallback: use the combined pairs if the regime pool is empty
                pool_df = pd.concat([pairs_poor, pairs_norm])
                print(f"    WARNING: {regime_name} pool empty — using combined pairs")
            delta_pool = pool_df['Ms_log_delta'].values
            buf = np.full(mask.sum(), -np.inf)
            data = log_sim[mask]
            it = 0
            while np.any(buf <= 0) and it < 200:
                cond = buf <= 0
                buf[cond] = data[cond] + np.random.choice(delta_pool, size=cond.sum(), replace=True)
                it += 1
            mock_log[mask] = buf
            print(f"    {regime_name} regime: {mask.sum()} rows, {it} iter — "
                  f"mock mean={np.expm1(buf).mean():.2e}")

        _fill(mask_poor, pairs_poor, 'poor-DFT')
        _fill(mask_norm, pairs_norm, 'normal')

        mock_exp = np.expm1(mock_log)
        print(f"  {label}: {n} total mock Ms_exp — overall mean={mock_exp.mean():.2e}")
        return mock_exp

    # ── Phase 1 ───────────────────────────────────────────────────────────

    def create_phase1_datasets(self):
        """Sim-only rows → mock Ms_exp (stratified bootstrap); append to pairs."""
        print("\n" + "="*60)
        print("PHASE 1 — Sim-only rows  (Ms_sim known, Ms_exp missing  →  mock Ms_exp)")
        print(f"  Using regime-stratified bootstrap (threshold = {REGIME_THRESHOLD:.0e} A/m)")
        print("="*60)

        df_sim_only = self.df[self.df[SIM_COL].notna() & self.df[EXP_COL].isna()].copy()
        if self.ms_threshold is not None:
            before = len(df_sim_only)
            df_sim_only = df_sim_only[df_sim_only[SIM_COL] > self.ms_threshold].copy()
            print(f"Ms threshold {self.ms_threshold:.0f} A/m (sim-only): "
                  f"dropped {before - len(df_sim_only)} rows "
                  f"({before} → {len(df_sim_only)} kept)")

        df_sim_only_re      = df_sim_only[df_sim_only[RE_COL] == True].copy()
        df_sim_only_re_free = df_sim_only[df_sim_only[RE_COL] == False].copy()

        print(f"Sim-only rows: {len(df_sim_only)} "
              f"(RE: {len(df_sim_only_re)}, RE-free: {len(df_sim_only_re_free)})")

        # All — stratified by regime using all-pairs pools
        mock_all = self._bootstrap_exp_stratified(
            df_sim_only, self.df_pairs_poor, self.df_pairs_norm, 'All'
        )
        df_sim_only_aug_all = df_sim_only.copy()
        df_sim_only_aug_all[EXP_COL] = mock_all
        df_sim_only_aug_all['Ms_delta'] = (
            df_sim_only_aug_all[EXP_COL] - df_sim_only_aug_all[SIM_COL]
        )
        self.df_phase1_all = pd.concat([self.df_pairs, df_sim_only_aug_all], ignore_index=True)

        # RE — stratified by regime using RE-specific pools
        mock_re = self._bootstrap_exp_stratified(
            df_sim_only_re, self.df_pairs_re_poor, self.df_pairs_re_norm, 'RE'
        )
        df_sim_only_aug_re = df_sim_only_re.copy()
        df_sim_only_aug_re[EXP_COL] = mock_re
        df_sim_only_aug_re['Ms_delta'] = (
            df_sim_only_aug_re[EXP_COL] - df_sim_only_aug_re[SIM_COL]
        )
        self.df_phase1_re = pd.concat([self.df_pairs_re, df_sim_only_aug_re], ignore_index=True)

        # RE-free — stratified by regime using RE-free-specific pools
        mock_re_free = self._bootstrap_exp_stratified(
            df_sim_only_re_free, self.df_pairs_rf_poor, self.df_pairs_rf_norm, 'RE-free'
        )
        df_sim_only_aug_re_free = df_sim_only_re_free.copy()
        df_sim_only_aug_re_free[EXP_COL] = mock_re_free
        df_sim_only_aug_re_free['Ms_delta'] = (
            df_sim_only_aug_re_free[EXP_COL] - df_sim_only_aug_re_free[SIM_COL]
        )
        self.df_phase1_re_free = pd.concat(
            [self.df_pairs_re_free, df_sim_only_aug_re_free], ignore_index=True
        )

        print(f"\nPhase 1 sizes — All: {len(self.df_phase1_all)}, "
              f"RE: {len(self.df_phase1_re)}, RE-free: {len(self.df_phase1_re_free)}")

    # ── Phase 2 ───────────────────────────────────────────────────────────

    def create_phase2_datasets(self):
        """Exp-only rows → mock Ms_sim; store exp-only DataFrames for Phase 3."""
        print("\n" + "="*60)
        print("PHASE 2 — Exp-only rows  (Ms_exp known, Ms_sim missing  →  mock Ms_sim)")
        print("="*60)

        df_exp_only = self.df[self.df[SIM_COL].isna() & self.df[EXP_COL].notna()].copy()
        if self.ms_threshold is not None:
            before = len(df_exp_only)
            df_exp_only = df_exp_only[df_exp_only[EXP_COL] > self.ms_threshold].copy()
            print(f"Ms threshold {self.ms_threshold:.0f} A/m (exp-only): "
                  f"dropped {before - len(df_exp_only)} rows "
                  f"({before} → {len(df_exp_only)} kept)")

        df_exp_only_re      = df_exp_only[df_exp_only[RE_COL] == True].copy()
        df_exp_only_re_free = df_exp_only[df_exp_only[RE_COL] == False].copy()

        print(f"Exp-only rows: {len(df_exp_only)} "
              f"(RE: {len(df_exp_only_re)}, RE-free: {len(df_exp_only_re_free)})")

        # All
        mock_sim_all = self._bootstrap_sim(
            df_exp_only, self.df_pairs['Ms_log_delta'].values, 'All'
        )
        self.df_exp_only_all = df_exp_only.copy()
        self.df_exp_only_all[SIM_COL] = mock_sim_all
        self.df_exp_only_all['Ms_delta'] = (
            self.df_exp_only_all[EXP_COL] - self.df_exp_only_all[SIM_COL]
        )

        # RE
        mock_sim_re = self._bootstrap_sim(
            df_exp_only_re, self.df_pairs_re['Ms_log_delta'].values, 'RE'
        )
        self.df_exp_only_re = df_exp_only_re.copy()
        self.df_exp_only_re[SIM_COL] = mock_sim_re
        self.df_exp_only_re['Ms_delta'] = (
            self.df_exp_only_re[EXP_COL] - self.df_exp_only_re[SIM_COL]
        )

        # RE-free
        mock_sim_re_free = self._bootstrap_sim(
            df_exp_only_re_free, self.df_pairs_re_free['Ms_log_delta'].values, 'RE-free'
        )
        self.df_exp_only_re_free = df_exp_only_re_free.copy()
        self.df_exp_only_re_free[SIM_COL] = mock_sim_re_free
        self.df_exp_only_re_free['Ms_delta'] = (
            self.df_exp_only_re_free[EXP_COL] - self.df_exp_only_re_free[SIM_COL]
        )

    # ── Saving ────────────────────────────────────────────────────────────

    def _save_csv(self, df: pd.DataFrame, path: Path):
        cols = [c for c in ['composition', SIM_COL, EXP_COL, 'Ms_delta', RE_COL]
                if c in df.columns]
        df[cols].to_csv(path, index=False)
        print(f"  Saved: {path}  ({len(df)} rows)")

    def save_all_phases(self):
        print("\n" + "="*60)
        print("SAVING OUTPUT FILES")
        print("="*60)

        # Original pairs
        print("\n--- Original pairs ---")
        for df, name in [
            (self.df_pairs,          'Pairs_all.csv'),
            (self.df_pairs_re,       'Pairs_RE.csv'),
            (self.df_pairs_re_free,  'Pairs_RE_Free.csv'),
        ]:
            self._save_csv(df, self.output_dir / name)

        # Phase 1 (sim-only augmented)
        print("\n--- Phase 1 (sim-only → mock exp) ---")
        for df, name in [
            (self.df_phase1_all,      'Augm_sim_all.csv'),
            (self.df_phase1_re,       'Augm_sim_RE.csv'),
            (self.df_phase1_re_free,  'Augm_sim_RE_Free.csv'),
        ]:
            self._save_csv(df, self.output_dir / name)

        # Phase 2 (exp-only augmented)
        print("\n--- Phase 2 (exp-only → mock sim) ---")
        for pairs_df, exp_df, name in [
            (self.df_pairs,         self.df_exp_only_all,     'Augm_exp_all.csv'),
            (self.df_pairs_re,      self.df_exp_only_re,      'Augm_exp_RE.csv'),
            (self.df_pairs_re_free, self.df_exp_only_re_free, 'Augm_exp_RE_Free.csv'),
        ]:
            df_p2 = pd.concat([pairs_df, exp_df], ignore_index=True)
            self._save_csv(df_p2, self.output_dir / name)

        # Phase 3 (combined = Phase 1 + exp-only rows)
        print("\n--- Phase 3 (combined) ---")
        for p1_df, exp_df, name in [
            (self.df_phase1_all,      self.df_exp_only_all,     'Augm_combined_all.csv'),
            (self.df_phase1_re,       self.df_exp_only_re,      'Augm_combined_RE.csv'),
            (self.df_phase1_re_free,  self.df_exp_only_re_free, 'Augm_combined_RE_Free.csv'),
        ]:
            df_p3 = pd.concat([p1_df, exp_df], ignore_index=True)
            self._save_csv(df_p3, self.output_dir / name)

    # ── Plots ─────────────────────────────────────────────────────────────

    def plot_delta_distributions(self):
        """Histogram of Ms_log_delta for pairs (RE, RE-free, all)."""
        save_dir = self.output_dir / 'distributions_plots'
        save_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating delta distribution plots...")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=100)

        axes[0].hist(
            self.df_pairs_re['Ms_log_delta'], bins=50, alpha=0.7,
            color='blue', label=f'RE (n={len(self.df_pairs_re)})'
        )
        axes[0].set_xlabel('log1p(Ms_exp) − log1p(Ms_sim)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('RE pairs — log-space delta')
        axes[0].legend()
        axes[0].grid(True)

        axes[1].hist(
            self.df_pairs_re_free['Ms_log_delta'], bins=50, alpha=0.7,
            color='orange', label=f'RE-free (n={len(self.df_pairs_re_free)})'
        )
        axes[1].set_xlabel('log1p(Ms_exp) − log1p(Ms_sim)')
        axes[1].set_ylabel('Count')
        axes[1].set_title('RE-free pairs — log-space delta')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_dir / 'log_delta_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: log_delta_distributions.png")

        # Combined
        plt.figure(figsize=(7, 5), dpi=100)
        for data, label, color in [
            (self.df_pairs['Ms_log_delta'],          'All',     'gray'),
            (self.df_pairs_re['Ms_log_delta'],        'RE',      'blue'),
            (self.df_pairs_re_free['Ms_log_delta'],   'RE-free', 'orange'),
        ]:
            plt.hist(data, bins=50, alpha=0.5, color=color, label=label)
        plt.xlabel('log1p(Ms_exp) − log1p(Ms_sim)')
        plt.ylabel('Count')
        plt.title('log-space Ms delta distribution (pairs)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_dir / 'log_delta_combined.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: log_delta_combined.png")

    # ── KS tests ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_log_deltas(path: Path) -> np.ndarray:
        """Load log-space deltas from a saved CSV (computed on the fly)."""
        try:
            df = pd.read_csv(path)
            mask = df[SIM_COL].notna() & df[EXP_COL].notna()
            sub = df[mask]
            return (np.log1p(sub[EXP_COL]) - np.log1p(sub[SIM_COL])).values
        except Exception as e:
            print(f"  Error loading {path}: {e}")
            return np.array([])

    @staticmethod
    def _ks_test(orig: np.ndarray, augm: np.ndarray, name: str):
        from scipy import stats
        if len(orig) == 0 or len(augm) == 0:
            print(f"  {name}: insufficient data — skipping")
            return
        ks_stat, p_value = stats.ks_2samp(orig, augm)
        alpha = 0.05
        verdict = "Reject H0" if p_value < alpha else "Fail to reject H0"
        print(f"  {name}: n_orig={len(orig)}, n_augm={len(augm)}, "
              f"KS={ks_stat:.4f}, p={p_value:.4f} → {verdict}")

    def perform_distribution_tests(self):
        """KS test: RE vs RE-free log-delta distributions (justifies separate models)."""
        from scipy import stats
        print("\n" + "="*60)
        print("KS TEST: RE vs RE-free log-delta distributions")
        print("="*60)
        re      = self.df_pairs_re['Ms_log_delta'].values
        re_free = self.df_pairs_re_free['Ms_log_delta'].values
        ks_stat, p_value = stats.ks_2samp(re, re_free)
        alpha = 0.05
        print(f"RE:      mean={re.mean():.4f}, std={re.std():.4f}, n={len(re)}")
        print(f"RE-free: mean={re_free.mean():.4f}, std={re_free.std():.4f}, n={len(re_free)}")
        print(f"KS statistic: {ks_stat:.4f},  p-value: {p_value:.8f}")
        if p_value < alpha:
            print("→ Distributions differ significantly: separate RE/RE-free models are justified.")
        else:
            print("→ Distributions are not significantly different.")

    def validate_augmentation(self):
        """KS tests comparing original pairs vs augmented datasets for each phase."""
        print("\n" + "="*60)
        print("AUGMENTATION VALIDATION (KS tests on log-space deltas)")
        print("="*60)
        for phase, files in [
            ("Phase 1 (sim→exp)", [
                ("All",     'Pairs_all.csv',      'Augm_sim_all.csv'),
                ("RE",      'Pairs_RE.csv',       'Augm_sim_RE.csv'),
                ("RE-free", 'Pairs_RE_Free.csv',  'Augm_sim_RE_Free.csv'),
            ]),
            ("Phase 3 (combined)", [
                ("All",     'Pairs_all.csv',      'Augm_combined_all.csv'),
                ("RE",      'Pairs_RE.csv',       'Augm_combined_RE.csv'),
                ("RE-free", 'Pairs_RE_Free.csv',  'Augm_combined_RE_Free.csv'),
            ]),
        ]:
            print(f"\n{phase}:")
            for name, orig_file, augm_file in files:
                orig = self._load_log_deltas(self.output_dir / orig_file)
                augm = self._load_log_deltas(self.output_dir / augm_file)
                self._ks_test(orig, augm, name)

    # ── Summary ───────────────────────────────────────────────────────────

    def print_summary(self):
        print("\n" + "="*60)
        print("AUGMENTATION SUMMARY")
        print("="*60)
        files = {
            'Pairs_all.csv':          'Original All',
            'Pairs_RE.csv':           'Original RE',
            'Pairs_RE_Free.csv':      'Original RE-free',
            'Augm_sim_all.csv':       'Phase1 All',
            'Augm_sim_RE.csv':        'Phase1 RE',
            'Augm_sim_RE_Free.csv':   'Phase1 RE-free',
            'Augm_exp_all.csv':       'Phase2 All',
            'Augm_exp_RE.csv':        'Phase2 RE',
            'Augm_exp_RE_Free.csv':   'Phase2 RE-free',
            'Augm_combined_all.csv':  'Phase3 All',
            'Augm_combined_RE.csv':   'Phase3 RE',
            'Augm_combined_RE_Free.csv': 'Phase3 RE-free',
        }
        for fname, label in files.items():
            path = self.output_dir / fname
            if path.exists():
                n = len(pd.read_csv(path))
                print(f"  {label:<25}: {n:>6} rows")
            else:
                print(f"  {label:<25}: not found")

    # ── Main entry point ──────────────────────────────────────────────────

    def run(self):
        print("="*60)
        print("Ms DATA AUGMENTATION (log-space bootstrap)")
        print("="*60)
        print(f"Random seed: {RANDOM_SEED}")

        self.load_data()
        self.filter_paired_data()
        self.perform_distribution_tests()
        self.plot_delta_distributions()
        self.create_phase1_datasets()
        self.create_phase2_datasets()
        self.save_all_phases()
        self.validate_augmentation()
        self.print_summary()


@log_output('logs/augment_data.txt')
def augment_data():
    """Main execution function."""
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--ms-threshold', type=float, default=50_000,
                        help='Drop rows with Ms_sim or Ms_exp <= this value (A/m). '
                             'Default: %(default)s. Set to 0 to disable.')
    args, _ = parser.parse_known_args()
    ms_threshold = args.ms_threshold

    script_dir   = Path(__file__).parent
    project_root = script_dir.parent
    data_path    = project_root / 'data' / 'merged_df_python.csv'
    output_dir   = project_root / 'outputs'

    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        sys.exit(1)

    print(f"Input:  {data_path}")
    print(f"Output: {output_dir}")
    print(f"Ms threshold: {ms_threshold:.0f} A/m" if ms_threshold else "Ms threshold: disabled")

    augmenter = MsAugmenter(str(data_path), str(output_dir), ms_threshold=ms_threshold)
    augmenter.run()


if __name__ == '__main__':
    augment_data()
