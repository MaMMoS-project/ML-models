# -*- coding: utf-8 -*-
"""Residual diagnostic for the rare-earth (RE) split.

Question: after the best available model, is the remaining RE error STRUCTURED by
rare-earth element (-> targeted features/models will help) or essentially WHITE
NOISE (-> we are near the experimental-measurement floor)?

Method: out-of-fold (cross-validated) predictions on all RE pairs, with compound
embeddings + log1p(Ms_sim) and the delta-learning target (predict the correction
log1p(Ms_exp) - log1p(Ms_sim), then add the baseline back). Residuals are grouped
by primary RE element and tested for clustering via:
  - eta^2  = fraction of residual variance explained by element identity
  - Kruskal-Wallis H-test (nonparametric) across element groups

Runs with sklearn models (Ridge, RandomForest) so it does not require torch.

Usage:
    python3 -m src.residual_diagnostic
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from re_features import primary_re_element, RE_PROPS

SIM_COL = 'Ms (ampere/meter)_s'
EXP_COL = 'Ms (ampere/meter)_e'
MS_THRESHOLD = 50_000.0
MIN_COUNT = 5  # minimum samples per element to include in the per-element table


def load_re_pairs() -> pd.DataFrame:
    """Load RE pairs with raw 200D embeddings, filtered to the training regime."""
    root = Path(__file__).parent.parent
    path = root / "outputs" / "Pairs_RE_w_embeddings.pkl"
    df = pd.read_pickle(path)
    df = df[df[SIM_COL].notna() & df[EXP_COL].notna()].copy()
    df = df[(df[SIM_COL] > MS_THRESHOLD) & (df[EXP_COL] > MS_THRESHOLD)].copy()
    if 'has_rare_earth' in df.columns:
        df = df[df['has_rare_earth'] == True].copy()
    print(f"Loaded {len(df)} RE pairs (threshold {MS_THRESHOLD:.0f} A/m)")
    return df


def build_xy(df: pd.DataFrame):
    """X = [embedding, log1p(Ms_sim)]; delta target = log1p(exp) - log1p(sim)."""
    emb = np.vstack(df['compound_embedding'].values)
    log_sim = np.log1p(df[SIM_COL].values)
    log_exp = np.log1p(df[EXP_COL].values)
    X = np.hstack([emb, log_sim.reshape(-1, 1)])
    y_delta = log_exp - log_sim
    return X, y_delta, log_sim, log_exp


def eta_squared(residuals: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of residual variance explained by group (element) identity."""
    grand = residuals.mean()
    ss_total = np.sum((residuals - grand) ** 2)
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        r = residuals[groups == g]
        ss_between += len(r) * (r.mean() - grand) ** 2
    return ss_between / ss_total


def analyse(name, residuals, elements, out_dir):
    print(f"\n{'='*64}\n{name}: residual structure by RE element\n{'='*64}")
    overall_std = residuals.std()
    print(f"Overall residual: mean={residuals.mean():+.4f}  std={overall_std:.4f}  "
          f"(log1p space, n={len(residuals)})")

    # Per-element table
    rows = []
    for el in np.unique(elements):
        mask = elements == el
        r = residuals[mask]
        if len(r) < MIN_COUNT:
            continue
        rows.append({
            'element': el, 'n': len(r), 'nf': int(RE_PROPS.get(el, {}).get('n_f', -1)),
            'mean_resid': r.mean(), 'std_resid': r.std(), 'MAE': np.abs(r).mean(),
        })
    tbl = pd.DataFrame(rows).sort_values('mean_resid')
    print(f"\nPer-element residuals (>= {MIN_COUNT} samples), sorted by mean:")
    print(tbl.to_string(index=False,
          formatters={'mean_resid': '{:+.4f}'.format, 'std_resid': '{:.4f}'.format,
                      'MAE': '{:.4f}'.format}))

    # Clustering quantification (use only elements meeting MIN_COUNT)
    keep = np.isin(elements, tbl['element'].values)
    r_k, e_k = residuals[keep], elements[keep]
    eta2 = eta_squared(r_k, e_k)
    groups = [r_k[e_k == el] for el in np.unique(e_k)]
    if len(groups) >= 2:
        H, p = stats.kruskal(*groups)
    else:
        H, p = float('nan'), float('nan')

    spread_of_means = tbl['mean_resid'].std()
    print(f"\nClustering metrics:")
    print(f"  eta^2 (variance explained by element) = {eta2:.3f}")
    print(f"  std of per-element mean residuals      = {spread_of_means:.4f} "
          f"(vs overall std {overall_std:.4f})")
    print(f"  Kruskal-Wallis H={H:.2f}, p={p:.2e}")
    verdict = ("STRUCTURED by element (p<0.05 and non-trivial eta^2) -> RE features/"
               "models should help" if (p < 0.05 and eta2 > 0.05)
               else "consistent with NOISE floor (weak/!=signif element structure)")
    print(f"  -> {verdict}")

    # Plot
    order = tbl.sort_values('nf')['element'].tolist()
    data = [residuals[elements == el] for el in order]
    fig, ax = plt.subplots(figsize=(max(7, 1.1 * len(order)), 5))
    ax.axhline(0, color='k', lw=1, ls='--', alpha=0.6)
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.uniform(-0.12, 0.12, len(d)),
                   d, s=10, alpha=0.4)
    ax.set_xlabel("primary RE element (ordered by 4f filling)")
    ax.set_ylabel("residual  log1p(Ms_exp) - pred")
    ax.set_title(f"RE residuals by element — {name}  (eta^2={eta2:.2f}, p={p:.1e})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"re_residuals_by_element_{name}.png"
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved plot: {fp}")
    return {'model': name, 'eta2': eta2, 'kruskal_p': p, 'overall_std': overall_std}


def main():
    out_dir = Path(__file__).parent.parent / "results" / "diagnostics"
    df = load_re_pairs()
    X, y_delta, log_sim, log_exp = build_xy(df)
    elements = np.array([primary_re_element(c) for c in df['composition'].values])

    n_known = np.sum(elements != None)  # noqa: E711
    print(f"Primary RE element resolved for {n_known}/{len(elements)} rows")

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        'Ridge': Ridge(alpha=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=42),
    }

    summary = []
    for name, model in models.items():
        # out-of-fold predictions of the delta, then reconstruct log1p(Ms_exp)
        delta_pred = cross_val_predict(model, X, y_delta, cv=cv, n_jobs=-1)
        pred_log_exp = log_sim + delta_pred
        residuals = log_exp - pred_log_exp
        # overall fit quality for context
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((log_exp - log_exp.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        print(f"\n[{name}] OOF R2 (log1p space) = {r2:.4f}")
        valid = elements != None  # noqa: E711
        summary.append(analyse(name, residuals[valid], elements[valid], out_dir))

    print(f"\n{'='*64}\nSUMMARY\n{'='*64}")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
