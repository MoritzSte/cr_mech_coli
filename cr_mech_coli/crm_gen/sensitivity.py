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
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from .config import DEFAULT_METRIC_WEIGHTS, DEFAULT_REGION_WEIGHTS
from .parameter_registry import (
    PARAMETER_REGISTRY,
    get_param_names,
    get_all_defaults,
    build_full_params,
)


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass
class MorrisResult:
    """Results from Morris elementary-effects screening."""

    param_names: List[str]
    mu_star: Dict[str, float]  # mean absolute elementary effect
    sigma: Dict[str, float]  # standard deviation of elementary effects
    active_params: List[str]  # params that passed the threshold
    inactive_params: List[str]  # params eliminated
    threshold: float
    num_evaluations: int


@dataclass
class SobolResult:
    """Results from Sobol variance-based sensitivity analysis."""

    param_names: List[str]
    s1: Dict[str, float]  # first-order indices
    st: Dict[str, float]  # total-order indices
    active_params: List[str]
    inactive_params: List[str]
    threshold: float
    num_evaluations: int


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

    n_workers = multiprocessing.cpu_count() if workers == -1 else workers
    # Build a plain list so each row is pickled independently
    rows = [X[i] for i in range(num_evals)]

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        Y = np.array(
            list(tqdm(pool.map(obj, rows, chunksize=4), total=num_evals, desc=desc))
        )
    return Y


# ---------------------------------------------------------------------------
# Morris screening
# ---------------------------------------------------------------------------


def run_morris_screening(
    image_pairs: List[Tuple[Path, Path]],
    param_names: List[str] = None,
    fixed_params: Dict[str, float] = None,
    weights: Dict[str, float] = None,
    region_weights: Dict[str, float] = None,
    num_trajectories: int = 10,
    threshold: float = 0.05,
    n_vertices: int = 8,
    seed: int = 42,
    workers: int = 1,
) -> MorrisResult:
    """Run Morris elementary-effects screening.

    Args:
        image_pairs: Representative (image, mask) pairs for evaluation.
        param_names: Parameters to screen.  Defaults to the full registry.
        fixed_params: Values for parameters *not* in *param_names*.
        weights: Metric weights (histogram_distance, ssim, psnr).
        region_weights: Background/foreground weights.
        num_trajectories: Number of Morris trajectories (default 10).
        threshold: Importance threshold — keep params with
            mu* > threshold * max(mu*).  Default 0.05 (5 %).
        n_vertices: Vertices per cell for shape extraction.
        seed: Random seed.
        workers: Number of parallel workers.  1 = sequential,
            -1 = use all CPUs.

    Returns:
        MorrisResult with ranked parameters and active/inactive sets.
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

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds,
    }

    print(f"\n{'='*60}")
    print("MORRIS SCREENING")
    print(f"{'='*60}")
    print(f"Parameters: {len(param_names)}")
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

        si = morris_analyze.analyze(problem, X, Y, seed=seed)

        mu_star = {n: float(v) for n, v in zip(param_names, si["mu_star"])}
        sigma = {n: float(v) for n, v in zip(param_names, si["sigma"])}

        # Determine active/inactive
        max_mu = max(mu_star.values()) if mu_star else 1.0
        cutoff = threshold * max_mu
        active = [n for n in param_names if mu_star[n] >= cutoff]
        inactive = [n for n in param_names if mu_star[n] < cutoff]

        print(f"\n{'='*60}")
        print("MORRIS RESULTS")
        print(f"{'='*60}")
        print(f"{'Parameter':<30} {'mu*':>10} {'sigma':>10} {'Status':>10}")
        print("-" * 60)
        for n in sorted(param_names, key=lambda k: mu_star[k], reverse=True):
            status = "ACTIVE" if n in active else "dropped"
            print(f"{n:<30} {mu_star[n]:10.4f} {sigma[n]:10.4f} {status:>10}")
        print(f"\nThreshold: mu* > {cutoff:.4f} ({threshold*100:.0f}% of max)")
        print(f"Active: {len(active)}, Dropped: {len(inactive)}")
        print(f"{'='*60}\n")

        return MorrisResult(
            param_names=param_names,
            mu_star=mu_star,
            sigma=sigma,
            active_params=active,
            inactive_params=inactive,
            threshold=threshold,
            num_evaluations=num_evals,
        )
    finally:
        obj.cleanup()


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
    threshold: float = 0.01,
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
        threshold: Keep params with total-order ST > this value.
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

    X = saltelli.sample(problem, N=n_samples, seed=seed)
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

        si = sobol_analyze.analyze(problem, Y, seed=seed)

        s1 = {n: float(v) for n, v in zip(param_names, si["S1"])}
        st = {n: float(v) for n, v in zip(param_names, si["ST"])}

        active = [n for n in param_names if st[n] > threshold]
        inactive = [n for n in param_names if st[n] <= threshold]

        print(f"\n{'='*60}")
        print("SOBOL RESULTS")
        print(f"{'='*60}")
        print(f"{'Parameter':<30} {'S1':>10} {'ST':>10} {'Status':>10}")
        print("-" * 60)
        for n in sorted(param_names, key=lambda k_: st[k_], reverse=True):
            status = "ACTIVE" if n in active else "dropped"
            print(f"{n:<30} {s1[n]:10.4f} {st[n]:10.4f} {status:>10}")
        print(f"\nThreshold: ST > {threshold}")
        print(f"Active: {len(active)}, Dropped: {len(inactive)}")
        print(f"{'='*60}\n")

        return SobolResult(
            param_names=param_names,
            s1=s1,
            st=st,
            active_params=active,
            inactive_params=inactive,
            threshold=threshold,
            num_evaluations=num_evals,
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
    morris_threshold: float = 0.05,
    run_sobol: bool = True,
    sobol_n_samples: int = 128,
    sobol_threshold: float = 0.01,
    n_vertices: int = 8,
    seed: int = 42,
    workers: int = 1,
) -> ScreeningResult:
    """Run the full Morris → Sobol screening pipeline.

    Args:
        image_pairs: Representative (image, mask) pairs (1-2 recommended).
        weights: Metric weights.
        region_weights: Background/foreground weights.
        num_trajectories: Morris trajectories.
        morris_threshold: Morris importance threshold (fraction of max mu*).
        run_sobol: Whether to run Sobol after Morris.
        sobol_n_samples: Sobol base sample size.
        sobol_threshold: Sobol total-order index threshold.
        n_vertices: Vertices per cell.
        seed: Random seed.
        workers: Number of parallel workers.  1 = sequential,
            -1 = use all CPUs.

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
        threshold=morris_threshold,
        n_vertices=n_vertices,
        seed=seed,
        workers=workers,
    )

    active_after_morris = morris_result.active_params
    defaults = get_all_defaults()
    fixed_after_morris = {
        n: defaults[n] for n in morris_result.inactive_params
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
            threshold=sobol_threshold,
            n_vertices=n_vertices,
            seed=seed,
            workers=workers,
        )

        final_active = sobol_result.active_params
        final_inactive = list(
            set(all_names) - set(final_active)
        )
        # Fixed = Morris-dropped defaults + Sobol-dropped defaults
        fixed_values = {n: defaults[n] for n in final_inactive}
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
            "threshold": result.morris.threshold,
            "num_evaluations": result.morris.num_evaluations,
        }
    if result.sobol is not None:
        data["sobol"] = {
            "param_names": result.sobol.param_names,
            "s1": result.sobol.s1,
            "st": result.sobol.st,
            "active_params": result.sobol.active_params,
            "inactive_params": result.sobol.inactive_params,
            "threshold": result.sobol.threshold,
            "num_evaluations": result.sobol.num_evaluations,
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
            threshold=m["threshold"],
            num_evaluations=m["num_evaluations"],
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
            threshold=s["threshold"],
            num_evaluations=s["num_evaluations"],
        )

    return ScreeningResult(
        morris=morris,
        sobol=sobol,
        active_params=data["active_params"],
        inactive_params=data["inactive_params"],
        fixed_values=data["fixed_values"],
    )
