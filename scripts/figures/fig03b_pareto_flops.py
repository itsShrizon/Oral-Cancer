"""fig03b — Subtype accuracy vs. GFLOPs Pareto plot."""
from __future__ import annotations

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from fig03_pareto_params import pareto_figure


def main():
    pareto_figure(
        xkey="gflops", xlabel="GFLOPs (log scale)",
        ykey="subtype_acc", ylabel="Subtype accuracy (%)",
        title="Subtype accuracy vs. inference compute",
        out_name="fig03b_pareto_flops", log_x=True,
    )


if __name__ == "__main__":
    main()
