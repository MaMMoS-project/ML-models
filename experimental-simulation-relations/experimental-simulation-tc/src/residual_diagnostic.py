# -*- coding: utf-8 -*-
"""Residual diagnostic for the rare-earth (RE) split — Curie temperature (Tc).

Two questions:

  (1) STRUCTURE: after the best embedding-based model, is the remaining RE error
      structured by rare-earth element (-> targeted RE features/models will help)
      or essentially white noise (-> near the measurement floor)?

  (2) Eu VALENCE: is modelling Eu as Eu2+ (4f7 S-state, magnetic; de Gennes ~ Gd)
      better for Tc than the default Eu3+ (4f6, J=0, nonmagnetic)? This ports the
      Ms-derived choice in re_features.py to a Tc-specific check.

Method (Tc-specific differences vs the Ms diagnostic):
  - LINEAR space (no log transform): delta target = Tc_exp - Tc_sim (kelvin).
  - Columns Tc_sim / Tc_exp; RE flag contains_rare_earth.
  - No Ms-style regime threshold (Tc has no direct analogue) — keep all valid rows.
  - Out-of-fold (cross-validated) predictions on RE pairs, embedding + Tc_sim,
    delta-learning target (predict the correction, add Tc_sim back). Residuals are
    grouped by primary RE element and tested for clustering via eta^2 and
    Kruskal-Wallis. For (2) the SAME pipeline is re-run with RE physics features
    appended, computed once with Eu=Eu2+ and once with Eu=Eu3+.

Usage:
    python3 -m src.residual_diagnostic
"""
import copy
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
import re_features
from re_features import primary_re_element, compute_re_features, RE_PROPS, RE_FEATURE_NAMES

SIM_COL = 'Tc_sim'
EXP_COL = 'Tc_exp'
MIN_COUNT = 5  # minimum samples per element to include in the per-element table


def load_re_pairs() -> pd.DataFrame:
    """Load RE pairs with raw 200D embeddings (linear Tc space, no threshold)."""
    root = Path(__file__).parent.parent
    path = root / "outputs" / "Pairs_RE_emb_w_embeddings.pkl"
    df = pd.read_pickle(path)
    df = df[df[SIM_COL].notna() & df[EXP_COL].notna()].copy()
    if 'contains_rare_earth' in df.columns:
        df = df[df['contains_rare_earth'] == True].copy()
    print(f"Loaded {len(df)} RE pairs (linear Tc space, no regime threshold)")
    return df


def build_xy(df: pd.DataFrame):
    """X = [embedding, Tc_sim]; delta target = Tc_exp - Tc_sim (kelvin)."""
    emb = np.vstack(df['compound_embedding'].values)
    sim = df[SIM_COL].values
    exp = df[EXP_COL].values
    X = np.hstack([emb, sim.reshape(-1, 1)])
    y_delta = exp - sim
    return X, y_delta, sim, exp


def re_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Compute the 7 RE physics features for each row (using current RE_PROPS)."""
    return np.array([
        [compute_re_features(c)[k] for k in RE_FEATURE_NAMES]
        for c in df['composition'].values
    ])


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


def oof_residuals(X, y_delta, sim, exp, model):
    """Out-of-fold delta predictions -> residuals in Tc space (kelvin)."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    delta_pred = cross_val_predict(model, X, y_delta, cv=cv, n_jobs=-1)
    pred_exp = sim + delta_pred
    residuals = exp - pred_exp
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((exp - exp.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return residuals, r2


def analyse(name, residuals, elements, out_dir):
    print(f"\n{'='*64}\n{name}: residual structure by RE element\n{'='*64}")
    overall_std = residuals.std()
    print(f"Overall residual: mean={residuals.mean():+.2f} K  std={overall_std:.2f} K  "
          f"(n={len(residuals)})")

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
    print(f"\nPer-element residuals (>= {MIN_COUNT} samples), sorted by mean (K):")
    print(tbl.to_string(index=False,
          formatters={'mean_resid': '{:+.1f}'.format, 'std_resid': '{:.1f}'.format,
                      'MAE': '{:.1f}'.format}))

    keep = np.isin(elements, tbl['element'].values)
    r_k, e_k = residuals[keep], elements[keep]
    eta2 = eta_squared(r_k, e_k)
    groups = [r_k[e_k == el] for el in np.unique(e_k)]
    H, p = stats.kruskal(*groups) if len(groups) >= 2 else (float('nan'), float('nan'))

    print(f"\nClustering metrics:")
    print(f"  eta^2 (variance explained by element) = {eta2:.3f}")
    print(f"  Kruskal-Wallis H={H:.2f}, p={p:.2e}")
    verdict = ("STRUCTURED by element -> RE features/models should help"
               if (p < 0.05 and eta2 > 0.05)
               else "consistent with NOISE floor")
    print(f"  -> {verdict}")

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
    ax.set_ylabel("residual  Tc_exp - pred  (K)")
    ax.set_title(f"RE Tc residuals by element — {name}  (eta^2={eta2:.2f}, p={p:.1e})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fp = out_dir / f"re_tc_residuals_by_element_{name}.png"
    fig.savefig(fp, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  saved plot: {fp}")
    return {'model': name, 'eta2': eta2, 'kruskal_p': p, 'overall_std': overall_std}


def eu_valence_ab(df, X_base, y_delta, sim, exp, elements, out_dir):
    """A/B: does Eu=Eu2+ (S-state) beat Eu=Eu3+ (J=0) for predicting Tc deltas?"""
    print(f"\n{'#'*64}\nEu VALENCE A/B TEST (Eu2+ S-state  vs  Eu3+ J=0)\n{'#'*64}")

    eu_mask = elements == 'Eu'
    n_eu = int(eu_mask.sum())
    print(f"Eu-containing rows (primary RE = Eu): {n_eu}")
    print(f"Raw delta (Tc_exp - Tc_sim) on Eu rows: mean={y_delta[eu_mask].mean():+.1f} K, "
          f"median={np.median(y_delta[eu_mask]):+.1f} K")
    print(f"Raw delta on all RE rows:               mean={y_delta.mean():+.1f} K")

    print("\nFree-ion values assigned to Eu under each valence:")
    eu2 = RE_PROPS['Eu']
    print(f"  Eu2+ (4f7 S-state): mu_sat={eu2['mu_sat']:.2f}  de_gennes={eu2['de_gennes']:.2f}  s_state={eu2['s_state']:.0f}")
    print(f"  Eu3+ (4f6, J=0):    mu_sat=0.00  de_gennes=0.00  s_state=0  (nonmagnetic)")

    models = {
        'Ridge': Ridge(alpha=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=42),
    }

    results = []
    for valence in ('Eu2+', 'Eu3+'):
        saved = copy.deepcopy(re_features.RE_PROPS['Eu'])
        if valence == 'Eu3+':
            # Eu3+ : 4f6, ground J=0 -> nonmagnetic (all features zero)
            re_features.RE_PROPS['Eu'] = re_features._entry(6, 0.0, 0.0, False)
        try:
            re_mat = re_feature_matrix(df)
        finally:
            re_features.RE_PROPS['Eu'] = saved
        # X = [embedding, RE features, Tc_sim]  (Tc_sim last, as in training)
        emb = X_base[:, :-1]
        X = np.hstack([emb, re_mat, sim.reshape(-1, 1)])

        for mname, model in models.items():
            resid, r2 = oof_residuals(X, y_delta, sim, exp, model)
            results.append({
                'valence': valence, 'model': mname,
                'MAE_Eu': np.abs(resid[eu_mask]).mean(),
                'MAE_all': np.abs(resid).mean(),
                'meanResid_Eu': resid[eu_mask].mean(),
                'R2_all': r2,
            })

    res = pd.DataFrame(results)
    print("\nOOF results with RE features (Eu MAE is the decisive number):")
    print(res.to_string(index=False,
          formatters={'MAE_Eu': '{:.1f}'.format, 'MAE_all': '{:.1f}'.format,
                      'meanResid_Eu': '{:+.1f}'.format, 'R2_all': '{:.4f}'.format}))

    print("\nVerdict per model (lower Eu MAE = better valence choice):")
    for mname in ('Ridge', 'RandomForest'):
        sub = res[res['model'] == mname].set_index('valence')
        d = sub.loc['Eu3+', 'MAE_Eu'] - sub.loc['Eu2+', 'MAE_Eu']
        better = 'Eu2+' if d > 0 else 'Eu3+'
        print(f"  {mname:13s}: Eu MAE  Eu2+={sub.loc['Eu2+','MAE_Eu']:.1f} K  "
              f"Eu3+={sub.loc['Eu3+','MAE_Eu']:.1f} K  -> {better} better "
              f"(by {abs(d):.1f} K)")
    return res


def main():
    out_dir = Path(__file__).parent.parent / "results" / "diagnostics"
    df = load_re_pairs()
    X, y_delta, sim, exp = build_xy(df)
    elements = np.array([primary_re_element(c) for c in df['composition'].values])
    n_known = np.sum(elements != None)  # noqa: E711
    print(f"Primary RE element resolved for {n_known}/{len(elements)} rows")

    models = {
        'Ridge': Ridge(alpha=0.01),
        'RandomForest': RandomForestRegressor(n_estimators=300, n_jobs=1, random_state=42),
    }

    print(f"\n{'#'*64}\nPART 1 — RESIDUAL STRUCTURE (embedding + Tc_sim, NO RE features)\n{'#'*64}")
    summary = []
    valid = elements != None  # noqa: E711
    for name, model in models.items():
        residuals, r2 = oof_residuals(X, y_delta, sim, exp, model)
        print(f"\n[{name}] OOF R2 (Tc space) = {r2:.4f}")
        summary.append(analyse(name, residuals[valid], elements[valid], out_dir))

    print(f"\n{'='*64}\nPART 1 SUMMARY\n{'='*64}")
    print(pd.DataFrame(summary).to_string(index=False))

    print(f"\n{'#'*64}\nPART 2 — Eu VALENCE A/B\n{'#'*64}")
    eu_valence_ab(df, X, y_delta, sim, exp, elements, out_dir)


if __name__ == "__main__":
    main()
