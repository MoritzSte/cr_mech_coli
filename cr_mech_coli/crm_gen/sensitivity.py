"""
Sensitivity analysis for automatic parameter screening.

Implements a two-stage pipeline:

1. **Morris screening** — cheap (~180 evals for 17 params) one-at-a-time
   perturbation that eliminates clearly irrelevant parameters.
2. **Sobol analysis** — precise variance-based decomposition on the
   surviving parameters that quantifies main effects and interactions.

Both stages evaluate the same objective function used by the DE optimizer:
generate a synthetic image with ``create_synthetic_scene()`` and compare
it to the real image via the weighted loss.

Requires the ``SALib`` package (``pip install SALib``).
"""

import os
import json
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import inspect

import numpy as np
from tqdm import tqdm

from .config import DEFAULT_METRIC_WEIGHTS, DEFAULT_REGION_WEIGHTS
from .parameter_registry import (
    PARAMETER_REGISTRY,
    get_param_names,
    get_all_defaults,
    get_groups_for,
    get_off_values_for,
    get_params_by_group,
    build_full_params,
    cast_param,
)


def _salib_call(func, *args, seed=None, **kwargs):
    """Call a SALib function, passing *seed* only if the function accepts it.

    Older SALib versions (< 1.5) do not support the ``seed`` keyword on
    ``sample()`` / ``analyze()``.  This wrapper inspects the signature and
    silently drops the argument when it is not supported.
    """
    sig = inspect.signature(func)
    if "seed" in sig.parameters:
        return func(*args, seed=seed, **kwargs)
    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class MorrisResult:
    """Results from Morris elementary-effects screening.

    When ``grouped=True``, ``mu_star`` / ``sigma`` are indexed by group name
    and ``active_params`` is the union of parameters belonging to active
    groups.  Per-parameter ranking is only available in non-grouped mode.
    """

    param_names: List[str]
    mu_star: Dict[str, float]  # keyed by param (ungrouped) or group (grouped)
    sigma: Dict[str, float]
    active_params: List[str]  # params that passed the threshold
    inactive_params: List[str]  # params eliminated
    top_n: int
    num_evaluations: int
    best_params: Dict[str, float] = field(default_factory=dict)
    grouped: bool = False
    groups: List[str] = field(default_factory=list)  # per-param group assignment
    active_groups: List[str] = field(default_factory=list)
    inactive_groups: List[str] = field(default_factory=list)
    # Baseline-improvement filter outputs (empty if filter disabled)
    loss_off: Optional[float] = None
    improvement_rate: Dict[str, float] = field(default_factory=dict)
    best_improvement: Dict[str, float] = field(default_factory=dict)
    auto_dropped: List[str] = field(default_factory=list)  # group or param names


@dataclass
class SobolResult:
    """Results from Sobol variance-based sensitivity analysis."""

    param_names: List[str]
    s1: Dict[str, float]  # first-order indices
    st: Dict[str, float]  # total-order indices
    active_params: List[str]
    inactive_params: List[str]
    top_n: int
    num_evaluations: int
    best_params: Dict[str, float] = field(default_factory=dict)


@dataclass
class ScreeningResult:
    """Combined results from the full screening pipeline."""

    morris: Optional[MorrisResult] = None
    sobol: Optional[SobolResult] = None
    active_params: List[str] = field(default_factory=list)
    inactive_params: List[str] = field(default_factory=list)
    fixed_values: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Objective function for sensitivity analysis
# ---------------------------------------------------------------------------


class _ScreeningObjective:
    """Evaluates the image-generation loss for a given parameter vector.

    Used by both Morris and Sobol samplers.  Each call generates a synthetic
    image from *one* representative (image, mask) pair and returns the loss.
    """

    def __init__(
        self,
        image_pairs: List[Tuple[Path, Path]],
        param_names: List[str],
        fixed_params: Dict[str, float],
        weights: Dict[str, float],
        region_weights: Dict[str, float],
        n_vertices: int = 8,
    ):
        self.image_pairs = [(str(p[0]), str(p[1])) for p in image_pairs]
        self.param_names = param_names
        self.fixed_params = fixed_params
        self.weights = weights
        self.region_weights = region_weights
        self.n_vertices = n_vertices
        self.include_ms_ssim = weights.get("ms_ssim", 0.0) > 0
        self.include_power_spectrum = weights.get("power_spectrum", 0.0) > 0
        self._temp_dir = tempfile.mkdtemp(prefix="screen_")

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate loss for parameter vector *x*."""
        from .optimization import evaluate_single_pair

        params = build_full_params(self.param_names, x.tolist(), self.fixed_params)

        total_loss = 0.0
        for img_str, mask_str in self.image_pairs:
            try:
                worker_dir = Path(self._temp_dir) / f"w_{os.getpid()}"
                worker_dir.mkdir(exist_ok=True)

                loss = evaluate_single_pair(
                    real_img_path=img_str,
                    mask_path=mask_str,
                    params_dict=params,
                    temp_dir=worker_dir,
                    n_vertices=self.n_vertices,
                    weights=self.weights,
                    region_weights=self.region_weights,
                    include_ms_ssim=self.include_ms_ssim,
                    include_power_spectrum=self.include_power_spectrum,
                    save=False,
                )
                total_loss += loss
            except Exception as e:
                print(f"  [screening] Error evaluating sample: {e}")
                total_loss += 1e10

        return total_loss / max(len(self.image_pairs), 1)

    def cleanup(self):
        if Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Parallel evaluation helper
# ---------------------------------------------------------------------------


def _evaluate_parallel(
    obj: _ScreeningObjective,
    X: np.ndarray,
    num_evals: int,
    workers: int,
    desc: str = "Evaluations",
) -> np.ndarray:
    """Evaluate *obj* for every row of *X*, optionally in parallel.

    Args:
        obj: Callable that maps a 1-D parameter vector to a scalar loss.
        X: 2-D sample matrix (rows = samples).
        num_evals: Number of evaluations (== X.shape[0]).
        workers: 1 for sequential, -1 for all CPUs, >1 for that many.
        desc: Progress bar description.

    Returns:
        1-D numpy array of loss values.
    """
    if workers == 1:
        Y = np.zeros(num_evals)
        for i in tqdm(range(num_evals), desc=desc):
            Y[i] = obj(X[i])
        return Y

    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    # Respect SLURM / cgroups allocation when available (cpu_count returns the
    # full node size, which oversubscribes on shared clusters).
    available = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else multiprocessing.cpu_count()
    )
    n_workers = available if workers == -1 else workers
    # Build a plain list so each row is pickled independently
    rows = [X[i] for i in range(num_evals)]

    # Use "spawn" to avoid fork-related heap corruption in native extensions
    # (the Rust cr_mech_coli core + BLAS threads are not fork-safe).
    ctx = multiprocessing.get_context("spawn")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        Y = np.array(
            list(tqdm(pool.map(obj, rows, chunksize=4), total=num_evals, desc=desc))
        )
    return Y


# ---------------------------------------------------------------------------
# Morris screening
# ---------------------------------------------------------------------------


def _compute_baseline_statistics(
    obj: "_ScreeningObjective",
    X: np.ndarray,
    Y: np.ndarray,
    param_names: List[str],
    fixed_params: Dict[str, float],
    workers: int = 1,
) -> Tuple[Optional[float], Dict[str, float], Dict[str, float]]:
    """Compute loss at the 'off baseline' and per-param improvement stats.

    Returns:
        loss_off: Loss evaluated with every param at its ``off_value``
            (or ``default`` for params without one).  ``None`` if no params
            in *param_names* declare an off_value.
        improvement_rate: Fraction of samples where ``Y[j] < loss_off`` and
            param was perturbed away from its off_value.
        best_improvement: ``loss_off - min(Y[j])`` over samples where the
            param was perturbed (clamped to >= 0).
    """
    off_values = get_off_values_for(param_names)
    has_off = {n: v for n, v in off_values.items() if v is not None}
    if not has_off:
        return None, {}, {}

    defaults = get_all_defaults()
    x_off = np.array(
        [has_off[n] if n in has_off else defaults[n] for n in param_names],
        dtype=float,
    )
    # Route the single baseline eval through the same spawn pool used for the
    # sampling grid.  Running it in-process here would initialize VTK/OpenGL
    # in the parent, and any later fork-based pool (e.g. scipy's DE) would
    # inherit that broken GL state and flood stdout with framebuffer dumps.
    loss_off = float(
        _evaluate_parallel(obj, x_off[np.newaxis, :], 1, workers, desc="Baseline")[0]
    )

    improvement_rate: Dict[str, float] = {}
    best_improvement: Dict[str, float] = {}
    for i, n in enumerate(param_names):
        if n not in has_off:
            continue
        off_val = has_off[n]
        # Samples where param n is meaningfully perturbed away from off
        # Use a tolerance based on the parameter range to avoid float-equality issues
        lo, hi = PARAMETER_REGISTRY[n].bounds
        tol = 1e-6 * max(abs(hi - lo), 1.0)
        perturbed_mask = np.abs(X[:, i] - off_val) > tol
        if not perturbed_mask.any():
            improvement_rate[n] = 0.0
            best_improvement[n] = 0.0
            continue
        Y_pert = Y[perturbed_mask]
        improvement_rate[n] = float(np.mean(Y_pert < loss_off))
        best_improvement[n] = float(max(0.0, loss_off - Y_pert.min()))

    return loss_off, improvement_rate, best_improvement


def run_morris_screening(
    image_pairs: List[Tuple[Path, Path]],
    param_names: List[str] = None,
    fixed_params: Dict[str, float] = None,
    weights: Dict[str, float] = None,
    region_weights: Dict[str, float] = None,
    num_trajectories: int = 10,
    top_n: int = 10,
    n_vertices: int = 8,
    seed: int = 42,
    workers: int = 1,
    use_groups: bool = True,
    use_baseline_filter: bool = True,
    baseline_improvement_rate_threshold: float = 0.05,
    baseline_improvement_epsilon: float = 0.01,
) -> MorrisResult:
    """Run Morris elementary-effects screening.

    Args:
        image_pairs: Representative (image, mask) pairs for evaluation.
        param_names: Parameters to screen.  Defaults to the full registry.
        fixed_params: Values for parameters *not* in *param_names*.
        weights: Metric weights (histogram_distance, ssim, psnr).
        region_weights: Background/foreground weights.
        num_trajectories: Number of Morris trajectories (default 10).
        top_n: Keep the *top_n* parameters (ungrouped) or groups (grouped)
            with highest mu*.
        n_vertices: Vertices per cell for shape extraction.
        seed: Random seed.
        workers: Number of parallel workers.  1 = sequential,
            -1 = use all CPUs.
        use_groups: Treat parameters sharing a ``group`` tag as a single
            factor during screening (grouped Morris).  Reduces factor count
            and captures coupling between params that implement the same
            visual effect.
        use_baseline_filter: Before the mu*-based ranking, auto-drop params
            (or groups) whose perturbations never improve loss relative to
            the "everything off" baseline.  Uses ``off_value`` from the
            registry; params without an off_value are exempt.
        baseline_improvement_rate_threshold: Minimum fraction of perturbed
            samples that must improve over the off-baseline.
        baseline_improvement_epsilon: Minimum relative improvement
            ``(loss_off - min(Y)) / loss_off`` required to survive.

    Returns:
        MorrisResult with ranked parameters/groups and active/inactive sets.
    """
    from SALib.sample import morris as morris_sample
    from SALib.analyze import morris as morris_analyze

    if param_names is None:
        param_names = get_param_names()
    if fixed_params is None:
        fixed_params = {}
    if weights is None:
        weights = dict(DEFAULT_METRIC_WEIGHTS)
    if region_weights is None:
        region_weights = dict(DEFAULT_REGION_WEIGHTS)

    bounds = [PARAMETER_REGISTRY[n].bounds for n in param_names]
    groups_per_param = get_groups_for(param_names)

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds,
    }
    if use_groups:
        problem["groups"] = groups_per_param
        unique_groups = list(dict.fromkeys(groups_per_param))  # preserve order
        num_factors = len(unique_groups)
    else:
        unique_groups = []
        num_factors = len(param_names)

    print(f"\n{'='*60}")
    print("MORRIS SCREENING" + (" (GROUPED)" if use_groups else ""))
    print(f"{'='*60}")
    print(f"Parameters: {len(param_names)}")
    if use_groups:
        print(f"Groups: {num_factors} ({', '.join(unique_groups)})")
    print(f"Trajectories: {num_trajectories}")

    X = morris_sample.sample(problem, N=num_trajectories, seed=seed)
    num_evals = X.shape[0]
    print(f"Total evaluations: {num_evals}")
    print(f"Images used: {len(image_pairs)}")
    print(f"{'='*60}\n")

    obj = _ScreeningObjective(
        image_pairs=image_pairs,
        param_names=param_names,
        fixed_params=fixed_params,
        weights=weights,
        region_weights=region_weights,
        n_vertices=n_vertices,
    )

    try:
        Y = _evaluate_parallel(obj, X, num_evals, workers, desc="Morris evaluations")

        best_idx = int(np.argmin(Y))
        best_params = {n: cast_param(n, v) for n, v in zip(param_names, X[best_idx])}

        si = morris_analyze.analyze(problem, X, Y, seed=seed)

        # In grouped mode SALib returns one entry per group; in ungrouped
        # mode, one per parameter.  Key names come back in `si["names"]`.
        factor_names = list(si.get("names", unique_groups if use_groups else param_names))
        mu_star = {n: float(v) for n, v in zip(factor_names, si["mu_star"])}
        sigma = {n: float(v) for n, v in zip(factor_names, si["sigma"])}

        # Baseline-improvement filter (reuses X and Y; +1 evaluation for loss_off)
        loss_off: Optional[float] = None
        improvement_rate: Dict[str, float] = {}
        best_improvement: Dict[str, float] = {}
        auto_dropped_factors: List[str] = []
        if use_baseline_filter:
            print("Computing baseline-improvement statistics...")
            loss_off, improvement_rate, best_improvement = _compute_baseline_statistics(
                obj, X, Y, param_names, fixed_params, workers=workers
            )
            if loss_off is not None:
                auto_dropped_factors = _select_auto_dropped(
                    param_names=param_names,
                    groups_per_param=groups_per_param,
                    factor_names=factor_names,
                    use_groups=use_groups,
                    improvement_rate=improvement_rate,
                    best_improvement=best_improvement,
                    loss_off=loss_off,
                    rate_threshold=baseline_improvement_rate_threshold,
                    epsilon=baseline_improvement_epsilon,
                )
                if auto_dropped_factors:
                    print(
                        f"  Auto-dropped by baseline filter: "
                        f"{', '.join(auto_dropped_factors)}"
                    )
                else:
                    print("  No factors auto-dropped.")

        # Rank surviving factors by mu* and keep top_n
        survivors = [f for f in factor_names if f not in auto_dropped_factors]
        k = min(top_n, len(survivors))
        ranked_survivors = sorted(survivors, key=lambda n: mu_star[n], reverse=True)
        active_factors = ranked_survivors[:k]
        inactive_factors = ranked_survivors[k:] + auto_dropped_factors

        # Expand factor-level decisions back to parameter-level
        if use_groups:
            active_params = [
                n for n, g in zip(param_names, groups_per_param) if g in active_factors
            ]
            inactive_params = [
                n for n, g in zip(param_names, groups_per_param) if g not in active_factors
            ]
            active_groups = active_factors
            inactive_groups = inactive_factors
        else:
            active_params = active_factors
            inactive_params = inactive_factors
            active_groups = []
            inactive_groups = []

        # Pretty-print ranking
        print(f"\n{'='*72}")
        print("MORRIS RESULTS" + (" (per group)" if use_groups else " (per parameter)"))
        print(f"{'='*72}")
        header_label = "Group" if use_groups else "Parameter"
        print(
            f"{header_label:<24} {'mu*':>10} {'sigma':>10} "
            f"{'improve%':>10} {'best_imp':>10} {'Status':>10}"
        )
        print("-" * 72)
        all_ranked = sorted(factor_names, key=lambda n: mu_star[n], reverse=True)
        for n in all_ranked:
            if n in auto_dropped_factors:
                status = "auto-drop"
            elif n in active_factors:
                status = "ACTIVE"
            else:
                status = "dropped"
            # Baseline stats only exist per param; aggregate up to group if grouped
            if use_groups:
                members = [p for p, g in zip(param_names, groups_per_param) if g == n]
                rate_vals = [improvement_rate[m] for m in members if m in improvement_rate]
                imp_vals = [best_improvement[m] for m in members if m in best_improvement]
                rate = max(rate_vals) if rate_vals else float("nan")
                imp = max(imp_vals) if imp_vals else float("nan")
            else:
                rate = improvement_rate.get(n, float("nan"))
                imp = best_improvement.get(n, float("nan"))
            rate_str = f"{rate*100:9.1f}%" if not np.isnan(rate) else "       --"
            imp_str = f"{imp:10.4f}" if not np.isnan(imp) else "        --"
            print(
                f"{n:<24} {mu_star[n]:10.4f} {sigma[n]:10.4f} "
                f"{rate_str:>10} {imp_str:>10} {status:>10}"
            )
        if loss_off is not None:
            print(f"\nBaseline loss (off-state): {loss_off:.6f}")
        print(f"Kept top {k} {'groups' if use_groups else 'parameters'} by mu*")
        print(
            f"Active params: {len(active_params)}, "
            f"Inactive params: {len(inactive_params)}"
        )
        print(f"{'='*72}\n")

        return MorrisResult(
            param_names=param_names,
            mu_star=mu_star,
            sigma=sigma,
            active_params=active_params,
            inactive_params=inactive_params,
            top_n=top_n,
            num_evaluations=num_evals,
            best_params=best_params,
            grouped=use_groups,
            groups=groups_per_param,
            active_groups=active_groups,
            inactive_groups=inactive_groups,
            loss_off=loss_off,
            improvement_rate=improvement_rate,
            best_improvement=best_improvement,
            auto_dropped=auto_dropped_factors,
        )
    finally:
        obj.cleanup()


def _select_auto_dropped(
    param_names: List[str],
    groups_per_param: List[str],
    factor_names: List[str],
    use_groups: bool,
    improvement_rate: Dict[str, float],
    best_improvement: Dict[str, float],
    loss_off: float,
    rate_threshold: float,
    epsilon: float,
) -> List[str]:
    """Identify factors (groups or params) that fail the baseline test.

    A *group* is auto-droppable if at least one of its parameters has an
    ``off_value`` — that parameter acts as the group's master switch (e.g.
    ``bac_halo_intensity`` for the halo group).  When the master switch is
    at its off_value the other group members are inert, so tracking the
    master's improvement statistics is sufficient.  A group is dropped iff
    the best observed improvement across all its switchable members is
    below thresholds.

    In ungrouped mode, only parameters with an explicit ``off_value`` are
    eligible; all others are exempt.
    """
    dropped: List[str] = []
    if use_groups:
        for g in factor_names:
            members = [p for p, pg in zip(param_names, groups_per_param) if pg == g]
            switches = [m for m in members if m in improvement_rate]
            if not switches:
                continue  # no off_value anywhere in the group → exempt
            max_rate = max(improvement_rate[m] for m in switches)
            max_imp = max(best_improvement[m] for m in switches)
            if max_rate < rate_threshold and max_imp < epsilon * loss_off:
                dropped.append(g)
    else:
        for p in factor_names:
            if p not in improvement_rate:
                continue
            if (
                improvement_rate[p] < rate_threshold
                and best_improvement[p] < epsilon * loss_off
            ):
                dropped.append(p)
    return dropped


# ---------------------------------------------------------------------------
# Sobol analysis
# ---------------------------------------------------------------------------


def run_sobol_analysis(
    image_pairs: List[Tuple[Path, Path]],
    param_names: List[str],
    fixed_params: Dict[str, float] = None,
    weights: Dict[str, float] = None,
    region_weights: Dict[str, float] = None,
    n_samples: int = 128,
    top_n: int = 7,
    n_vertices: int = 8,
    seed: int = 42,
    workers: int = 1,
) -> SobolResult:
    """Run Sobol variance-based sensitivity analysis.

    Should be called on the *active* parameters from Morris screening
    (typically ~10-12 params) to keep the cost manageable.

    Args:
        image_pairs: Representative (image, mask) pairs.
        param_names: Parameters to analyse (Morris survivors).
        fixed_params: Values for all other parameters.
        weights: Metric weights.
        region_weights: Background/foreground weights.
        n_samples: Base sample size N.  Total evals = N * (2k + 2).
        top_n: Keep the *top_n* parameters with highest total-order ST.
        n_vertices: Vertices per cell.
        seed: Random seed.
        workers: Number of parallel workers.  1 = sequential,
            -1 = use all CPUs.

    Returns:
        SobolResult with S1 / ST indices and active/inactive sets.
    """
    from SALib.sample import saltelli
    from SALib.analyze import sobol as sobol_analyze

    if fixed_params is None:
        fixed_params = {}
    if weights is None:
        weights = dict(DEFAULT_METRIC_WEIGHTS)
    if region_weights is None:
        region_weights = dict(DEFAULT_REGION_WEIGHTS)

    bounds = [PARAMETER_REGISTRY[n].bounds for n in param_names]
    k = len(param_names)

    problem = {
        "num_vars": k,
        "names": param_names,
        "bounds": bounds,
    }

    print(f"\n{'='*60}")
    print("SOBOL ANALYSIS")
    print(f"{'='*60}")
    print(f"Parameters: {k}")
    print(f"Base samples (N): {n_samples}")

    X = _salib_call(saltelli.sample, problem, N=n_samples, seed=seed)
    num_evals = X.shape[0]
    print(f"Total evaluations: {num_evals}")
    print(f"Images used: {len(image_pairs)}")
    print(f"{'='*60}\n")

    obj = _ScreeningObjective(
        image_pairs=image_pairs,
        param_names=param_names,
        fixed_params=fixed_params,
        weights=weights,
        region_weights=region_weights,
        n_vertices=n_vertices,
    )

    try:
        Y = _evaluate_parallel(obj, X, num_evals, workers, desc="Sobol evaluations")

        best_idx = int(np.argmin(Y))
        best_params = {n: cast_param(n, v) for n, v in zip(param_names, X[best_idx])}

        si = _salib_call(sobol_analyze.analyze, problem, Y, seed=seed)

        s1 = {n: float(v) for n, v in zip(param_names, si["S1"])}
        st = {n: float(v) for n, v in zip(param_names, si["ST"])}

        k = min(top_n, len(param_names))
        ranked = sorted(param_names, key=lambda n: st[n], reverse=True)
        active = ranked[:k]
        inactive = ranked[k:]

        print(f"\n{'='*60}")
        print("SOBOL RESULTS")
        print(f"{'='*60}")
        print(f"{'Parameter':<30} {'S1':>10} {'ST':>10} {'Status':>10}")
        print("-" * 60)
        for n in ranked:
            status = "ACTIVE" if n in active else "dropped"
            print(f"{n:<30} {s1[n]:10.4f} {st[n]:10.4f} {status:>10}")
        print(f"\nKept top {k} parameters by ST")
        print(f"Active: {len(active)}, Dropped: {len(inactive)}")
        print(f"{'='*60}\n")

        return SobolResult(
            param_names=param_names,
            s1=s1,
            st=st,
            active_params=active,
            inactive_params=inactive,
            top_n=top_n,
            num_evaluations=num_evals,
            best_params=best_params,
        )
    finally:
        obj.cleanup()


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------


def run_full_screening(
    image_pairs: List[Tuple[Path, Path]],
    weights: Dict[str, float] = None,
    region_weights: Dict[str, float] = None,
    num_trajectories: int = 10,
    morris_top_n: int = 10,
    run_sobol: bool = True,
    sobol_n_samples: int = 128,
    sobol_top_n: int = 7,
    n_vertices: int = 8,
    seed: int = 42,
    workers: int = 1,
    use_best_values: bool = False,
    use_groups: bool = True,
    use_baseline_filter: bool = True,
    baseline_improvement_rate_threshold: float = 0.05,
    baseline_improvement_epsilon: float = 0.01,
) -> ScreeningResult:
    """Run the full Morris → Sobol screening pipeline.

    Args:
        image_pairs: Representative (image, mask) pairs (1-2 recommended).
        weights: Metric weights.
        region_weights: Background/foreground weights.
        num_trajectories: Morris trajectories.
        morris_top_n: Keep the top N groups (grouped mode) or parameters
            (ungrouped) after Morris screening.  In grouped mode every
            parameter belonging to a surviving group becomes active.
        run_sobol: Whether to run Sobol after Morris.
        sobol_n_samples: Sobol base sample size.
        sobol_top_n: Keep the top N parameters after Sobol analysis.
        n_vertices: Vertices per cell.
        seed: Random seed.
        workers: Number of parallel workers.  1 = sequential,
            -1 = use all CPUs.
        use_groups: Grouped Morris (see ``run_morris_screening``).
        use_baseline_filter: Baseline-improvement auto-drop stage.
        baseline_improvement_rate_threshold / baseline_improvement_epsilon:
            Thresholds forwarded to the baseline filter.

    Returns:
        ScreeningResult with the final active/inactive parameter sets.
    """
    all_names = get_param_names()

    # Stage 1: Morris
    morris_result = run_morris_screening(
        image_pairs=image_pairs,
        param_names=all_names,
        weights=weights,
        region_weights=region_weights,
        num_trajectories=num_trajectories,
        top_n=morris_top_n,
        n_vertices=n_vertices,
        seed=seed,
        workers=workers,
        use_groups=use_groups,
        use_baseline_filter=use_baseline_filter,
        baseline_improvement_rate_threshold=baseline_improvement_rate_threshold,
        baseline_improvement_epsilon=baseline_improvement_epsilon,
    )

    active_after_morris = morris_result.active_params
    defaults = get_all_defaults()

    # Identify params that were auto-dropped by the baseline filter so we can
    # pin them to their off_value instead of their default.  In grouped mode
    # `auto_dropped` holds group names; expand to the member params.
    if morris_result.grouped:
        auto_dropped_params = {
            n for n, g in zip(all_names, morris_result.groups)
            if g in morris_result.auto_dropped
        }
    else:
        auto_dropped_params = set(morris_result.auto_dropped)

    def _fixed_value_for(name: str) -> Any:
        """Pick the value used for a parameter held fixed after Morris."""
        if name in auto_dropped_params:
            off = PARAMETER_REGISTRY[name].off_value
            if off is not None:
                return off
        if use_best_values:
            return morris_result.best_params[name]
        return defaults[name]

    fixed_after_morris = {
        n: _fixed_value_for(n) for n in morris_result.inactive_params
    }

    sobol_result = None
    if run_sobol and len(active_after_morris) > 1:
        # Stage 2: Sobol on Morris survivors
        sobol_result = run_sobol_analysis(
            image_pairs=image_pairs,
            param_names=active_after_morris,
            fixed_params=fixed_after_morris,
            weights=weights,
            region_weights=region_weights,
            n_samples=sobol_n_samples,
            top_n=sobol_top_n,
            n_vertices=n_vertices,
            seed=seed,
            workers=workers,
        )

        final_active = sobol_result.active_params
        final_inactive = list(
            set(all_names) - set(final_active)
        )
        all_best = {**morris_result.best_params, **sobol_result.best_params}

        def _final_fixed_value_for(name: str) -> Any:
            if name in auto_dropped_params:
                off = PARAMETER_REGISTRY[name].off_value
                if off is not None:
                    return off
            if use_best_values:
                return all_best[name]
            return defaults[name]

        fixed_values = {n: _final_fixed_value_for(n) for n in final_inactive}
    else:
        final_active = active_after_morris
        final_inactive = morris_result.inactive_params
        fixed_values = fixed_after_morris

    result = ScreeningResult(
        morris=morris_result,
        sobol=sobol_result,
        active_params=final_active,
        inactive_params=final_inactive,
        fixed_values=fixed_values,
    )

    print(f"\n{'='*60}")
    print("SCREENING COMPLETE")
    print(f"{'='*60}")
    print(f"Final active parameters ({len(final_active)}):")
    for n in final_active:
        print(f"  - {n}")
    print(f"Fixed parameters ({len(final_inactive)}):")
    for n in final_inactive:
        print(f"  - {n} = {fixed_values[n]}")
    print(f"{'='*60}\n")

    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_screening_results(result: ScreeningResult, path: Path) -> None:
    """Save screening results to a JSON file."""
    data = {
        "active_params": result.active_params,
        "inactive_params": result.inactive_params,
        "fixed_values": result.fixed_values,
    }
    if result.morris is not None:
        data["morris"] = {
            "param_names": result.morris.param_names,
            "mu_star": result.morris.mu_star,
            "sigma": result.morris.sigma,
            "active_params": result.morris.active_params,
            "inactive_params": result.morris.inactive_params,
            "top_n": result.morris.top_n,
            "num_evaluations": result.morris.num_evaluations,
            "best_params": result.morris.best_params,
            "grouped": result.morris.grouped,
            "groups": result.morris.groups,
            "active_groups": result.morris.active_groups,
            "inactive_groups": result.morris.inactive_groups,
            "loss_off": result.morris.loss_off,
            "improvement_rate": result.morris.improvement_rate,
            "best_improvement": result.morris.best_improvement,
            "auto_dropped": result.morris.auto_dropped,
        }
    if result.sobol is not None:
        data["sobol"] = {
            "param_names": result.sobol.param_names,
            "s1": result.sobol.s1,
            "st": result.sobol.st,
            "active_params": result.sobol.active_params,
            "inactive_params": result.sobol.inactive_params,
            "top_n": result.sobol.top_n,
            "num_evaluations": result.sobol.num_evaluations,
            "best_params": result.sobol.best_params,
        }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Screening results saved to: {path}")


def load_screening_results(path: Path) -> ScreeningResult:
    """Load screening results from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    morris = None
    if "morris" in data:
        m = data["morris"]
        morris = MorrisResult(
            param_names=m["param_names"],
            mu_star=m["mu_star"],
            sigma=m["sigma"],
            active_params=m["active_params"],
            inactive_params=m["inactive_params"],
            top_n=m["top_n"],
            num_evaluations=m["num_evaluations"],
            best_params=m.get("best_params", {}),
            grouped=m.get("grouped", False),
            groups=m.get("groups", []),
            active_groups=m.get("active_groups", []),
            inactive_groups=m.get("inactive_groups", []),
            loss_off=m.get("loss_off"),
            improvement_rate=m.get("improvement_rate", {}),
            best_improvement=m.get("best_improvement", {}),
            auto_dropped=m.get("auto_dropped", []),
        )

    sobol = None
    if "sobol" in data:
        s = data["sobol"]
        sobol = SobolResult(
            param_names=s["param_names"],
            s1=s["s1"],
            st=s["st"],
            active_params=s["active_params"],
            inactive_params=s["inactive_params"],
            top_n=s["top_n"],
            num_evaluations=s["num_evaluations"],
            best_params=s.get("best_params", {}),
        )

    return ScreeningResult(
        morris=morris,
        sobol=sobol,
        active_params=data["active_params"],
        inactive_params=data["inactive_params"],
        fixed_values=data["fixed_values"],
    )
