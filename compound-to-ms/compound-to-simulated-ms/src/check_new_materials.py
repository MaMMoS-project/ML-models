"""Cross-check new_materials.txt against our data: is each candidate actually new?

For every compound listed in ../new_materials.txt this reports whether it is already
present in

    RAW    - the raw source files (before dedup / Ms-threshold / ferrimagnet drop),
    TRAIN  - preprocessed_data/*.csv (what the models actually train on),
    VALID  - data/validation_reference.csv + validation_ferrimagnetic_compounds_reference.csv,

and writes the result to ../new_materials_known.txt.

Matching is by pymatgen REDUCED FORMULA (the SAME canonicalisation preprocess_ms_data
uses for dedup), so spelling / element-ordering / stoichiometric-multiple variants match.

The raw sources have ~1e6 rows, so canonicalising all of them would be far too slow.
Since we only test a handful of candidates, we pre-filter raw strings by their element
SET (a cheap regex) and only call pymatgen on the few that could possibly match.

Reuses the project's own preprocess_ms_data (reduced_formula, the *_LOADERS list, KEY),
so the same script works in both compound-to-{simulated,experimental}-ms.

Run:  python -m src.check_new_materials
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
try:
    from src import preprocess_ms_data as pp
except Exception:  # pragma: no cover
    import preprocess_ms_data as pp

DATA = ROOT / "data"
PRE = ROOT / "preprocessed_data"
NEW = ROOT / "new_materials.txt"
OUT = ROOT / "new_materials_known.txt"
VALID_FILES = ["validation_reference.csv", "validation_ferrimagnetic_compounds_reference.csv"]

_ELEM_RE = re.compile(r"[A-Z][a-z]?")
_canon_cache: dict[str, "str | None"] = {}


def canon(f: str) -> "str | None":
    if f not in _canon_cache:
        _canon_cache[f] = pp.reduced_formula(f)
    return _canon_cache[f]


def elem_set(s: str) -> frozenset:
    return frozenset(_ELEM_RE.findall(s))


def read_new_materials(path: Path) -> list[str]:
    out = []
    for line in path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            out.append(s)
    return out


def main() -> None:
    project = ROOT.name
    new = read_new_materials(NEW)
    targets = {}                       # original -> canonical
    for f in new:
        targets[f] = canon(f)
    canon_targets = {c for c in targets.values() if c}
    target_elsets = {elem_set(c) for c in canon_targets}

    # --- RAW: reuse the project's loaders, pre-filter by element set ---
    loaders = getattr(pp, "SIM_LOADERS", None) or getattr(pp, "EXP_LOADERS", None) or []
    raw: dict[str, set] = {}
    for ld in loaders:
        try:
            df = ld(DATA)
        except Exception as e:
            print(f"  WARN: loader {ld.__name__} failed: {e}")
            continue
        tag = ld.__name__.replace("load_", "")
        for s in df[pp.KEY].astype(str).unique():
            if elem_set(s) in target_elsets:          # cheap filter
                c = canon(s)
                if c in canon_targets:
                    raw.setdefault(c, set()).add(tag)
        print(f"  scanned RAW source '{tag}' ({len(df)} rows)")

    # --- TRAIN: preprocessed_data/*.csv are ALREADY reduced formulas ---
    train: dict[str, set] = {}
    for p in sorted(PRE.glob("*.csv")):
        df = pd.read_csv(p)
        col = pp.KEY if pp.KEY in df.columns else df.columns[0]
        present = set(df[col].astype(str))
        for c in canon_targets:
            if c in present:
                train.setdefault(c, set()).add(p.name)

    # --- VALID: small, canonicalise the 'formula' column ---
    val: dict[str, set] = {}
    for name in VALID_FILES:
        p = DATA / name
        if not p.exists():
            continue
        df = pd.read_csv(p)
        col = "formula" if "formula" in df.columns else df.columns[0]
        present = {canon(x) for x in df[col].astype(str)}
        for c in canon_targets:
            if c in present:
                val.setdefault(c, set()).add(name)

    # --- report ---
    L = []
    L.append(f"NEW MATERIALS CROSS-CHECK — {project}")
    L.append("=" * (len(project) + 28))
    L.append("Is each compound in new_materials.txt actually new, or already present in our data?")
    L.append("Matching by pymatgen REDUCED FORMULA (same canonicalisation as the dedup), so")
    L.append("spelling / ordering / stoichiometric-multiple variants are matched.")
    L.append("")
    L.append("Sources: RAW = raw source files (before dedup / Ms-threshold / ferrimagnet drop);")
    L.append("         TRAIN = preprocessed_data/*.csv (what the models train on);")
    L.append("         VALID = data/validation_*reference.csv.")
    L.append("Regenerate with:  python -m src.check_new_materials")
    L.append("")

    known, notfound = [], []
    for f in new:
        c = targets[f]
        if c is None:
            notfound.append((f, "(pymatgen could not parse)"))
            continue
        where = []
        if c in raw:
            where.append("RAW[" + ",".join(sorted(raw[c])) + "]")
        if c in train:
            where.append("TRAIN[" + ",".join(sorted(train[c])) + "]")
        if c in val:
            where.append("VALID[" + ",".join(sorted(val[c])) + "]")
        (known if where else notfound).append((f, c, where) if where else (f, c))

    L.append(f"KNOWN (already present somewhere) — {len(known)}/{len(new)}:")
    if known:
        for f, c, where in known:
            L.append(f"  {f:<12} (reduced {c:<9}) -> " + "  ".join(where))
    else:
        L.append("  (none)")
    L.append("")
    L.append(f"NOT FOUND (genuinely new to our data) — {len(notfound)}/{len(new)}:")
    if notfound:
        for item in notfound:
            L.append(f"  {item[0]:<12} {item[1] if len(item) > 1 else ''}")
    else:
        L.append("  (none)")

    text = "\n".join(L) + "\n"
    OUT.write_text(text)
    print("\n" + text)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
