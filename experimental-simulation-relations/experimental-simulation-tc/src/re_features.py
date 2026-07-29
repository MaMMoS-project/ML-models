# -*- coding: utf-8 -*-
"""Rare-earth (RE) physics features for Curie-temperature (Tc) error correction.

Standard collinear DFT/GGA mistreats the localized 4f shell, so for rare-earth
compounds the simulation-vs-experiment gap is dominated by the missing/mis-estimated
4f moment and, crucially for ORDERING TEMPERATURES, by the strength of the
RE(4f)-TM(3d) exchange. For Tc the de Gennes factor (g_J - 1)^2 J(J+1) is especially
relevant: within a RE series it governs the RE contribution to the ordering
temperature (T_ordering ~ de Gennes factor for the RKKY-coupled 4f sublattice).

This module provides:
  - RE3+ Hund's-rules constants (lanthanides La..Lu)
  - compute_re_features(composition): aggregate, stoichiometry-weighted features
  - primary_re_element(composition): the dominant RE element (for diagnostics)

All quantities are textbook free-ion RE3+ values (e.g. Jensen & Mackintosh).
Caveats: assumes RE3+ (exceptions: Eu2+, Yb2+/3+, Ce4+, Sm2+); crystal-field
quenching and finite temperature reduce the effective moment below gJ*J.
"""
from typing import Dict, List, Optional

try:
    from pymatgen.core import Composition
    _PYMATGEN = True
except Exception:
    Composition = None
    _PYMATGEN = False


# Per RE3+ ion: n_f (4f electrons), J, g_J, and derived quantities.
#   mu_sat   = g_J * J                       free-ion saturation moment (mu_B)
#   spin_proj= (g_J - 1) * J                 spin projection -> RKKY coupling sign/strength
#   de_gennes= (g_J - 1)^2 * J * (J + 1)     exchange strength / ordering-T scale
#   heavy    = n_f >= 7 (J = L+S; 4f spin couples antiparallel to 3d -> ferrimagnetic)
def _entry(n_f, J, gJ, heavy, s_state=False):
    spin_proj = (gJ - 1.0) * J
    de_gennes = (gJ - 1.0) ** 2 * J * (J + 1.0)
    return {
        'n_f': float(n_f),
        'J': float(J),
        'gJ': float(gJ),
        'mu_sat': gJ * J,
        'spin_proj': spin_proj,
        'de_gennes': de_gennes,
        'heavy': 1.0 if heavy else 0.0,
        # S-state ion: half-filled 4f7 with L=0 (Gd3+, Eu2+) -> large, isotropic,
        # CEF-unquenched, collinear pure-spin moment that collinear DFT badly misses.
        's_state': 1.0 if s_state else 0.0,
    }


RE_PROPS: Dict[str, Dict[str, float]] = {
    'La': _entry(0,  0.0, 0.0,        False),   # 4f0  — nonmagnetic
    'Ce': _entry(1,  2.5, 6.0 / 7.0,  False),   # 4f1
    'Pr': _entry(2,  4.0, 4.0 / 5.0,  False),   # 4f2
    'Nd': _entry(3,  4.5, 8.0 / 11.0, False),   # 4f3
    'Pm': _entry(4,  4.0, 3.0 / 5.0,  False),   # 4f4
    'Sm': _entry(5,  2.5, 2.0 / 7.0,  False),   # 4f5
    # Eu is commonly DIVALENT (Eu2+, 4f7 — same config as Gd3+, ~7 mu_B), not Eu3+
    # (4f6, ground J=0, nominally nonmagnetic). Modelling Eu as Eu2+ here.
    # TC NOTE: the Tc residual diagnostic (src/residual_diagnostic.py) finds Eu is
    # NOT a systematic Tc anomaly (raw delta ~0; Eu2+ vs Eu3+ change Eu MAE by ~0.2 K).
    # So for Tc this choice is immaterial (kept for consistency with my_ms / physics),
    # unlike for Ms where Eu2+ was the single largest residual correction.
    'Eu': _entry(7,  3.5, 2.0,        True, s_state=True),   # Eu2+ 4f7 S-state (was Eu3+ 4f6, J=0)
    'Gd': _entry(7,  3.5, 2.0,        True, s_state=True),   # 4f7 S-state — half-filled, max de Gennes
    'Tb': _entry(8,  6.0, 3.0 / 2.0,  True),    # 4f8
    'Dy': _entry(9,  7.5, 4.0 / 3.0,  True),    # 4f9
    'Ho': _entry(10, 8.0, 5.0 / 4.0,  True),    # 4f10
    'Er': _entry(11, 7.5, 6.0 / 5.0,  True),    # 4f11
    'Tm': _entry(12, 6.0, 7.0 / 6.0,  True),    # 4f12
    # Yb is mixed/intermediate valence; Yb2+ (4f14) is a full, NONMAGNETIC shell,
    # whereas Yb3+ (4f13) has gJ*J=4.0. Modelling Yb as Yb2+ here. NOTE: this affects
    # few samples and is the more uncertain of the two valence choices — revisit if
    # the dataset is Yb3+-dominated.
    'Yb': _entry(14, 0.0, 0.0,        True),    # Yb2+ 4f14 nonmagnetic (was Yb3+ 4f13)
    'Lu': _entry(14, 0.0, 0.0,        True),    # 4f14 — nonmagnetic
}

RE_ELEMENTS: List[str] = list(RE_PROPS.keys())

# Feature names produced by compute_re_features (stable order).
RE_FEATURE_NAMES: List[str] = [
    're_fraction',     # total atomic fraction of RE elements
    'mu_free',         # sum_i x_i * (gJ*J)_i   — magnitude of the expected moment correction
    'spin_proj',       # sum_i x_i * (gJ-1)J_i  — coupling sign/strength to 3d
    'de_gennes',       # sum_i x_i * deGennes_i — exchange strength / RE ordering-T scale (key for Tc)
    'nf_weighted',     # sum_i x_i * n_f_i      — mean 4f filling
    'heavy_fraction',  # atomic fraction of heavy-RE (n_f>=7) atoms
    's_state_fraction',# atomic fraction of S-state RE (Gd3+, Eu2+) — isolates the
                       # large, CEF-unquenched pure-spin moment DFT misses (Eu/Gd)
]

_ZERO = {k: 0.0 for k in RE_FEATURE_NAMES}


def _clean(composition: str) -> str:
    """Strip annotation characters pymatgen cannot parse (e.g. trailing '*'/'∗')."""
    return str(composition).replace('∗', '').replace('*', '').strip()


def compute_re_features(composition: str) -> Dict[str, float]:
    """Stoichiometry-weighted RE physics features for one composition.

    Returns all-zero features for RE-free compositions or on parse failure, so
    the features are safe to apply to any dataset (RE-free rows -> no effect).
    """
    if not _PYMATGEN:
        raise ImportError("pymatgen is required for RE features. pip install pymatgen")
    try:
        comp = Composition(_clean(composition))
    except Exception:
        return dict(_ZERO)

    total = comp.num_atoms
    if total <= 0:
        return dict(_ZERO)

    feats = dict(_ZERO)
    for el, amt in comp.get_el_amt_dict().items():
        props = RE_PROPS.get(el)
        if props is None:
            continue
        x = amt / total  # atomic fraction
        feats['re_fraction']      += x
        feats['mu_free']          += x * props['mu_sat']
        feats['spin_proj']        += x * props['spin_proj']
        feats['de_gennes']        += x * props['de_gennes']
        feats['nf_weighted']      += x * props['n_f']
        feats['heavy_fraction']   += x * props['heavy']
        feats['s_state_fraction'] += x * props['s_state']
    return feats


def primary_re_element(composition: str) -> Optional[str]:
    """Return the RE element with the largest amount, or None if RE-free."""
    if not _PYMATGEN:
        raise ImportError("pymatgen is required. pip install pymatgen")
    try:
        comp = Composition(_clean(composition))
    except Exception:
        return None
    re_amounts = {el: amt for el, amt in comp.get_el_amt_dict().items() if el in RE_PROPS}
    if not re_amounts:
        return None
    return max(re_amounts, key=re_amounts.get)
