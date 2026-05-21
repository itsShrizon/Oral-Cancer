"""
Ablation registry for the Custom EfficientNet V2 AttentionHub.

Each ablation key maps to:
  - a tuple of active branches (or () for the no-attention control,
    or None for the full hub which is the default proposed model)
  - a run-name suffix used to keep outputs separate from the main run

The "full" variant is intentionally aliased to the existing baseline-recipe
folder so we do not retrain the proposed model; it already exists at
results/custom_efficientnet_v2_baseline_recipe/ from the fair-comparison run.
"""

from typing import Optional, Tuple


# Ordered: 'none' (control) -> single branches -> pairs -> full.
ABLATIONS = {
    "none":          (),                          # donor Block-4 (no attention)
    "bam":           ("bam",),
    "triplet":       ("triplet",),
    "kan":           ("kan",),
    "bam_triplet":   ("bam", "triplet"),
    "bam_kan":       ("bam", "kan"),
    "triplet_kan":   ("triplet", "kan"),
    "full":          None,                        # all three (proposed model)
}


def branches_for(key: str):
    """Return the branches argument to pass to CustomEfficientNetV2."""
    if key not in ABLATIONS:
        raise ValueError(f"Unknown ablation key: {key!r}. "
                         f"Choices: {sorted(ABLATIONS.keys())}")
    return ABLATIONS[key]


def run_name_for(backbone: str, ablation: Optional[str], recipe: str = "baseline",
                 hub_version: str = "v1") -> str:
    """
    Construct the per-run results folder name.

    Conventions:
      - The original tuned run sits at results/custom_efficientnet_v2/.
      - The fair-recipe full-hub run sits at
        results/custom_efficientnet_v2_baseline_recipe/.
      - Ablation runs sit at results/custom_efficientnet_v2_ablation_<key>/
        and always use the baseline recipe.
      - 'full' ablation aliases to the existing baseline_recipe folder so we
        don't retrain the proposed model.
      - hub_version='v2' (sequential Triplet->EMA, LayerScale-gated) lives at
        results/custom_efficientnet_v2_hub_v2/. Always uses the baseline
        recipe for apples-to-apples comparison with v1 ablations.
    """
    # Hub v2 path (custom backbone only, baseline recipe enforced).
    if hub_version == "v2":
        if backbone != "custom_efficientnet_v2":
            raise ValueError("hub_version='v2' only applies to custom_efficientnet_v2")
        if ablation is not None:
            raise ValueError("hub_version='v2' is incompatible with --ablation "
                             "(v2 has its own fixed structure)")
        return f"{backbone}_hub_v2"

    # Non-custom backbones: ablation must be None (or 'full' as a no-op).
    if backbone != "custom_efficientnet_v2":
        if ablation not in (None, "full"):
            raise ValueError(f"--ablation only applies to custom_efficientnet_v2, "
                             f"got backbone={backbone!r}")
        return backbone

    # Custom backbone path:
    if ablation is None or ablation == "full":
        # Proposed model under fair recipe = existing baseline_recipe folder.
        if recipe == "baseline":
            return f"{backbone}_baseline_recipe"
        return backbone  # tuned recipe lives in plain folder

    return f"{backbone}_ablation_{ablation}"
