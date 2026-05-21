"""Data loaders for results/<run>/{classification,performance,training_time}.json
and per-class metrics parsed from evaluation_results.txt."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_DIR = REPO_ROOT / "results"

# Paper Table 1 baseline order
BASELINE_RUNS = [
    "resnet50",
    "densenet121",
    "convnext_tiny",
    "swin_t",
    "efficientnet_b0",
    "efficientnet_v2b2",
    "efficientnet_v2b3",
    "efficientnet_v2s",
    "inception_v3",
]
PROPOSED_V1 = "custom_efficientnet_v2_baseline_recipe"
PROPOSED_V2 = "custom_efficientnet_v2_hub_v2"
PROPOSED_RUNS = [PROPOSED_V1, PROPOSED_V2]

ABLATION_RUNS = [
    "custom_efficientnet_v2_ablation_none",
    "custom_efficientnet_v2_ablation_bam",
    "custom_efficientnet_v2_ablation_triplet",
    "custom_efficientnet_v2_ablation_kan",
    "custom_efficientnet_v2_ablation_bam_triplet",
    "custom_efficientnet_v2_ablation_bam_kan",
    "custom_efficientnet_v2_ablation_triplet_kan",
]
ABLATION_VARIANT_NAMES = {
    "custom_efficientnet_v2_ablation_none": "none",
    "custom_efficientnet_v2_ablation_bam": "bam",
    "custom_efficientnet_v2_ablation_triplet": "triplet",
    "custom_efficientnet_v2_ablation_kan": "kan",
    "custom_efficientnet_v2_ablation_bam_triplet": "bam_triplet",
    "custom_efficientnet_v2_ablation_bam_kan": "bam_kan",
    "custom_efficientnet_v2_ablation_triplet_kan": "triplet_kan",
    "custom_efficientnet_v2_baseline_recipe": "full",
}

ALL_TABLE1_RUNS = BASELINE_RUNS + PROPOSED_RUNS


def run_dir(run: str) -> Path:
    return RESULTS_DIR / run


def _safe_load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None


def load_classification(run: str) -> Optional[dict]:
    return _safe_load_json(run_dir(run) / "classification_metrics.json")


def load_performance(run: str) -> Optional[dict]:
    return _safe_load_json(run_dir(run) / "performance_metrics.json")


def load_training_time(run: str) -> Optional[dict]:
    return _safe_load_json(run_dir(run) / "training_time.json")


def parse_per_class(run: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Parse the sklearn classification_report block from evaluation_results.txt.

    Returns: { class_name: { 'precision': float, 'recall': float, 'f1': float, 'support': int } }
    """
    txt_path = run_dir(run) / "evaluation_results.txt"
    if not txt_path.exists():
        return None
    raw = txt_path.read_text(encoding="utf-8")
    # The per-class block is between 'Per-Class Report:' and the next 'accuracy' line
    m = re.search(r"Per-Class Report:\s*\n.*?\n((?:.*\n)+?)\s*accuracy", raw)
    if not m:
        return None
    block = m.group(1)
    out = {}
    for line in block.splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                support = int(parts[-1])
                f1 = float(parts[-2])
                rec = float(parts[-3])
                prec = float(parts[-4])
                cls = " ".join(parts[:-4])
                if cls in ("macro avg", "weighted avg"):
                    continue
                out[cls] = {"precision": prec, "recall": rec, "f1": f1, "support": support}
            except ValueError:
                continue
    return out or None


def collect_table1() -> List[dict]:
    """Returns list of dicts for all Table 1 entries (baselines + proposed).

    Each dict has: run, label, binary_acc, binary_f1, subtype_acc, subtype_f1,
    params_m, gflops, size_mb, gpu_peak_mb, p50_ms, p95_ms, train_time_min, epochs.
    """
    from .style import MODEL_LABELS

    rows = []
    for run in ALL_TABLE1_RUNS:
        c = load_classification(run)
        p = load_performance(run)
        t = load_training_time(run)
        if c is None or p is None:
            continue
        row = {
            "run": run,
            "label": MODEL_LABELS.get(run, run),
            "binary_acc": c["binary"]["accuracy"],
            "binary_f1": c["binary"]["f1_score"],
            "subtype_acc": c["subtype"]["accuracy"],
            "subtype_f1": c["subtype"]["f1_score"],
            "params": p.get("num_parameters", 0),
            "params_m": p.get("num_parameters", 0) / 1e6,
            "gflops": p.get("flops_gflops", 0.0),
            "size_mb": p.get("model_size_mb", 0.0),
            "gpu_peak_mb": p.get("gpu_peak_mb", 0.0),
            "cpu_rss_mb": p.get("cpu_rss_mb", 0.0),
            "p50_ms": p.get("latency_distribution", {}).get("p50_ms", p.get("inference_time_ms_mean", 0.0)),
            "p95_ms": p.get("latency_distribution", {}).get("p95_ms", 0.0),
            "p99_ms": p.get("latency_distribution", {}).get("p99_ms", 0.0),
            "lat_mean_ms": p.get("latency_distribution", {}).get("mean_ms", 0.0),
            "lat_std_ms": p.get("latency_distribution", {}).get("std_ms", 0.0),
            "lat_min_ms": p.get("latency_distribution", {}).get("min_ms", 0.0),
            "lat_max_ms": p.get("latency_distribution", {}).get("max_ms", 0.0),
        }
        if t:
            row["train_time_min"] = t.get("total_training_time_min", 0.0)
            row["epochs"] = t.get("epochs_completed", 0)
        else:
            row["train_time_min"] = 0.0
            row["epochs"] = 0
        rows.append(row)
    return rows


def collect_ablation() -> List[dict]:
    """Collect 8 ablation cells (7 ablation_* + baseline_recipe as 'full')."""
    rows = []
    runs = ABLATION_RUNS + [PROPOSED_V1]
    for run in runs:
        c = load_classification(run)
        p = load_performance(run)
        if c is None or p is None:
            continue
        rows.append({
            "run": run,
            "variant": ABLATION_VARIANT_NAMES.get(run, run),
            "binary_acc": c["binary"]["accuracy"],
            "subtype_acc": c["subtype"]["accuracy"],
            "binary_f1": c["binary"]["f1_score"],
            "subtype_f1": c["subtype"]["f1_score"],
            "params_m": p.get("num_parameters", 0) / 1e6,
            "gflops": p.get("flops_gflops", 0.0),
        })
    return rows


def collect_v2() -> Optional[dict]:
    """Return the v2 proposed model row."""
    c = load_classification(PROPOSED_V2)
    p = load_performance(PROPOSED_V2)
    if c is None or p is None:
        return None
    return {
        "run": PROPOSED_V2,
        "variant": "triplet_se",
        "binary_acc": c["binary"]["accuracy"],
        "subtype_acc": c["subtype"]["accuracy"],
        "binary_f1": c["binary"]["f1_score"],
        "subtype_f1": c["subtype"]["f1_score"],
        "params_m": p.get("num_parameters", 0) / 1e6,
        "gflops": p.get("flops_gflops", 0.0),
    }
