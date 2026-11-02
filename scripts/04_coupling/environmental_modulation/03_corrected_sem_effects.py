"""
SEM Environmental Control Audit -- Corrected SEM, Step 3
=========================================================
Direct / indirect / total standardized-effect decomposition with tower-block
bootstrap CIs, for whichever direction is identifiable per product (per Step 1's
`Corrected_SEM_Model_Comparison.csv`).

Reuses `cluster_bootstrap_effects()` from the original, audited
`03b_sem_effects_and_diagrams.py` UNCHANGED -- the audit found the bootstrap
effects math itself was not the source of the error; only the model-selection
step downstream of it (hardcoding GOME-2 to a tied model) was wrong. This
script fixes that by choosing, per product, the FINAL_MODEL_SELECTED from
Step 1's classification instead of a hardcoded BEST_MODEL dict, and by running
BOTH directions for GOME-2 (its direction is not identifiable) rather than one.

Output columns are tagged with DIRECTION_STATUS so a reader immediately sees
which rows come from a statistically resolved product-direction and which are
"reported for both directions because direction is not statistically
resolvable -- do not treat either as preferred" (GOME-2).

Author: project authors
Run date: 2026-09-08
"""

from __future__ import annotations
# Original script filename (sanitized): 03_corrected_sem_effects.py; public name: 03_corrected_sem_effects.py
import os as _os, sys as _sys
_p = _os.path.dirname(_os.path.abspath(__file__))
while _os.path.basename(_p) != 'scripts' and _os.path.dirname(_p) != _p:
    _p = _os.path.dirname(_p)
for _d, _dn, _fn in _os.walk(_p):  # sibling script folders are importable by module name
    if '__pycache__' not in _d and _d not in _sys.path:
        _sys.path.append(_d)
# Paths are configurable: set SIF_GPP_ROOT (working project/data root) and optionally SIF_GPP_DATA_ROOT
# (parent folder holding raw/third-party inputs). See docs/reproducibility.md.
PROJECT_ROOT = _os.environ.get("SIF_GPP_ROOT", ".")
DATA_ROOT = _os.environ.get("SIF_GPP_DATA_ROOT", PROJECT_ROOT)

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(PROJECT_ROOT)
ADV = ROOT / "SIF_EC_Advanced_Coupling_Analysis"
AUDIT = ROOT / "SIF_EC_SEM_Environmental_Control_Audit"
OUT_DIR = AUDIT / "05_sem_audit"

sys.path.insert(0, str(ADV / "12_scripts"))
from importlib import import_module  # noqa: E402
core = import_module("03a_sem_component_models")
eff_mod = import_module("03b_sem_effects_and_diagrams")  # for cluster_bootstrap_effects (unmodified, audited-OK)

MODEL_SPECS = core.MODEL_SPECS
PRODUCTS = core.PRODUCTS


def load_cache():
    with open(OUT_DIR / "_fitted_cache.pkl", "rb") as f:
        return pickle.load(f)


def main():
    cache = load_cache()
    fitted_cache = cache["fitted_cache"]
    class_df = cache["class_df"].set_index("SIF_PRODUCT")

    all_rows = []
    for product in PRODUCTS:
        row = class_df.loc[product]
        classification = row["FINAL_DIRECTION_CLASSIFICATION"]
        low_support = bool(row["LOW_SUPPORT_FLAG_LT30_TOWERS"])

        if classification == "DIRECTION_NOT_IDENTIFIABLE":
            # Report BOTH directions' saturated models (C and D) -- direction is a tie,
            # so we show the saturated form of each (the Wald test showed both saturated
            # extensions beat their own parsimonious null, but that says nothing about
            # C vs D against EACH OTHER, which remains structurally unresolvable -- see
            # Step 1's C/D total-AIC identity verification).
            models_to_run = [("C", "SIF_directional_TIED_NOT_PREFERRED"),
                              ("D", "GPP_directional_TIED_NOT_PREFERRED")]
            direction_status = ("DIRECTION_NOT_IDENTIFIABLE -- reported for both directions because direction "
                                 "is not statistically resolvable (|dAIC(A,B)|<2). Do not treat either as "
                                 "preferred.")
        else:
            final_model = row["FINAL_MODEL_SELECTED"]
            tag = "SIF_directional_SUPPORTED" if classification == "SIF_leads_GPP_supported" else "GPP_directional_SUPPORTED"
            models_to_run = [(final_model, tag)]
            direction_status = classification + (" (LOW_SUPPORT_INSUFFICIENT_DATA: only "
                                                  f"{int(row['N_TOWERS'])} towers)" if low_support else "")

        for model_key, model_tag in models_to_run:
            spec = MODEL_SPECS[model_key]
            data = fitted_cache[(product, model_key)]["data"]
            print(f"Bootstrapping {product} model {model_key} ({model_tag}): "
                  f"N={len(data)}, towers={data['SITE_ID'].nunique()} ...")
            rows = eff_mod.cluster_bootstrap_effects(data, spec, product, model_key)
            for r in rows:
                r["DIRECTION_TAG"] = model_tag
                r["DIRECTION_STATUS"] = direction_status
                r["LOW_SUPPORT_FLAG"] = low_support
            all_rows.extend(rows)

    out = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "Corrected_SEM_Effects.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")


if __name__ == "__main__":
    main()
