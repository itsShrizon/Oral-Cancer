"""fig06b — Cross-model LIME comparison on the SAME 4 binary samples."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "fig06_mod",
    os.path.join(os.path.dirname(__file__), "fig06_gradcam_composite.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    mod.build_grid(
        col_idx=mod.COL_LIME,
        title="LIME super-pixel boundaries on the same 4 binary samples - cross-model comparison",
        out_name="fig06b_lime_cross_model_composite",
    )


if __name__ == "__main__":
    main()
