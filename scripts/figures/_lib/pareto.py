"""Pareto-frontier computation for efficiency vs. accuracy plots."""
from __future__ import annotations

from typing import List, Sequence, Tuple


def pareto_front(
    xs: Sequence[float], ys: Sequence[float], minimize_x: bool = True, maximize_y: bool = True
) -> List[int]:
    """Return indices on the Pareto frontier.

    By default, minimize x (e.g., params, FLOPs) and maximize y (e.g., accuracy).
    """
    assert len(xs) == len(ys)
    n = len(xs)
    order = sorted(range(n), key=lambda i: (xs[i] if minimize_x else -xs[i], -ys[i] if maximize_y else ys[i]))
    frontier = []
    best_y = -float("inf") if maximize_y else float("inf")
    for i in order:
        y = ys[i]
        if maximize_y:
            if y >= best_y:
                frontier.append(i)
                best_y = y
        else:
            if y <= best_y:
                frontier.append(i)
                best_y = y
    return frontier
