"""CMA-ES over a Monte-Carlo robust objective (Hansen, 2016), with a safe search window."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cma
import numpy as np
from scipy.stats import qmc

from robustprint import space
from robustprint.optimize.objectives import RobustProblem
from robustprint.physics.composition import FloatArray
from robustprint.physics.materials import DEFAULT_LIBRARY, MaterialLibrary
from robustprint.physics.simulator import Feedstock

#: Margin below the dominant polymer's degradation onset (ASSUMPTION, ADR-016).
DEGRADATION_MARGIN_C = 10.0


def safe_bounds(feedstock: Feedstock, library: MaterialLibrary = DEFAULT_LIBRARY) -> tuple[FloatArray, FloatArray]:
    """Search bounds with the nozzle window limited by the dominant polymer.

    The dominant polymer is the polymer component with the largest mean fraction. The
    nozzle temperature must stay at or above its full-melt temperature and at least
    ``DEGRADATION_MARGIN_C`` below its degradation onset.
    """
    low, high = space.bounds()
    mean = feedstock.mean_array()
    poly = library.mask("polymer").astype(bool)
    dominant = int(np.argmax(np.where(poly, mean, -1.0)))
    comp = library.components[dominant]
    low = low.copy()
    high = high.copy()
    low[0] = max(low[0], comp.t_min_C)
    high[0] = min(high[0], comp.t_deg_C - DEGRADATION_MARGIN_C)
    if low[0] >= high[0]:
        raise ValueError(f"Empty nozzle window for dominant polymer {comp.name}.")
    return low, high


@dataclass(frozen=True)
class OptimisationResult:
    """Best setting found and bookkeeping."""

    params: FloatArray
    score: float
    n_evaluations: int


def maximise(
    score: Callable[[FloatArray], FloatArray],
    low: FloatArray,
    high: FloatArray,
    budget: int = 1200,
    popsize: int = 16,
    n_init: int = 64,
    seed: int = 0,
) -> OptimisationResult:
    """Maximise a batched score over a box with a Sobol warm start followed by CMA-ES.

    Args:
        score: Maps (P, d) physical-unit settings to (P,) scores.
        low: Lower bounds.
        high: Upper bounds.
        budget: Total number of score evaluations (including the warm start).
        popsize: CMA-ES population size.
        n_init: Number of scrambled-Sobol warm-start points.
        seed: Seed for the Sobol design and CMA-ES.
    """
    span = high - low

    def to_x(u: FloatArray) -> FloatArray:
        return np.asarray(low + np.clip(u, 0.0, 1.0) * span, dtype=float)

    d = len(low)
    u0 = qmc.Sobol(d, scramble=True, seed=seed).random(n_init)
    s0 = score(to_x(u0))
    best_i = int(np.argmax(s0))
    best_u, best_s = u0[best_i].copy(), float(s0[best_i])
    evals = n_init
    opts = cma.CMAOptions()
    opts.set(
        {
            "bounds": [0.0, 1.0],
            "popsize": popsize,
            "seed": seed + 1,
            "verbose": -9,
            "maxfevals": max(budget - n_init, popsize),
            "tolfun": 1e-6,
        }
    )
    es = cma.CMAEvolutionStrategy(best_u, 0.2, opts)
    while not es.stop():
        cand = np.asarray(es.ask())
        vals = score(to_x(cand))
        evals += len(cand)
        es.tell(list(cand), list(-vals))
        i = int(np.argmax(vals))
        if vals[i] > best_s:
            best_s, best_u = float(vals[i]), cand[i].copy()
    return OptimisationResult(to_x(best_u), best_s, evals)


def recommend(
    problem: RobustProblem,
    objective: str,
    budget: int = 1200,
    seed: int = 0,
    library: MaterialLibrary = DEFAULT_LIBRARY,
) -> OptimisationResult:
    """Recommend settings that maximise ``objective`` for the problem's feedstock."""
    low, high = safe_bounds(problem.feedstock, library)
    return maximise(lambda x: problem.score(objective, x), low, high, budget=budget, seed=seed)
