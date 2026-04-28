#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter optimization for synthetic microscope image generation.

This module optimizes the selected parameters to match
real microscope images using a weighted combination of:

- Color/Intensity Distribution (histogram L1 distance)
- SSIM (Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)

Uses scipy.optimize.differential_evolution with multi-threading for efficient
global optimization across multiple CPU cores.
"""

import os
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from scipy.optimize import differential_evolution
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import tempfile
import shutil
import csv
import time
import tifffile as tiff

from .scene import create_synthetic_scene
from .metrics import compute_all_metrics_with_weights, load_image
from .config import DEFAULT_METRIC_WEIGHTS, DEFAULT_REGION_WEIGHTS
from .parameter_registry import (
    get_param_names,
    build_full_params,
)
from .weight_maps import build_region_weight_maps

# ============================================================================
# Checkpoint Manager for Differential Evolution
# ============================================================================


class DECheckpointManager:
    """
    Manages checkpointing for differential evolution optimization.

    Saves population state after each iteration to allow resuming
    if the process is killed.
    """

    def __init__(self, checkpoint_dir: Path, run_start_time: str = None):
        """
        Initialize the checkpoint manager.

        Args:
            checkpoint_dir (Path): Directory for storing checkpoint files.
            run_start_time (str): Timestamp string for this run. If None, generated
                on first save.
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.run_start_time = run_start_time

    def save_checkpoint(
        self,
        iteration: int,
        population: np.ndarray,
        population_energies: np.ndarray,
        best_x: np.ndarray,
        best_fun: float,
        convergence: float,
        config: dict,
    ) -> Path:
        """
        Save current optimization state to a checkpoint file.

        Args:
            iteration (int): Current iteration number.
            population (np.ndarray): Current population array.
            population_energies (np.ndarray): Energy (loss) values for each member.
            best_x (np.ndarray): Best parameter vector found so far.
            best_fun (float): Best loss value found so far.
            convergence (float): Current convergence metric.
            config (dict): Optimization configuration for validation on resume.

        Returns:
            Path: Path to the saved checkpoint file.
        """
        if self.run_start_time is None:
            self.run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        filename = f"checkpoint_{self.run_start_time}_iter{iteration:04d}.npz"
        checkpoint_path = self.checkpoint_dir / filename
        temp_base = checkpoint_path.with_suffix(".tmp")

        try:
            np.savez(
                temp_base,
                iteration=iteration,
                population=population,
                population_energies=population_energies,
                best_x=best_x,
                best_fun=best_fun,
                convergence=convergence,
                run_start_time=self.run_start_time,
                config_json=json.dumps(config),
            )
            actual_temp_file = Path(str(temp_base) + ".npz")
            shutil.move(str(actual_temp_file), str(checkpoint_path))

        except Exception as e:
            actual_temp_file = Path(str(temp_base) + ".npz")
            if actual_temp_file.exists():
                actual_temp_file.unlink()
            raise e

        return checkpoint_path

    def find_latest_checkpoint(self) -> Path:
        """
        Find the most recent checkpoint file by modification time.

        Returns:
            Path: Path to the latest checkpoint, or None if no checkpoints exist.
        """
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*.npz"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return checkpoints[0] if checkpoints else None

    def load_checkpoint(self, checkpoint_path: Path = None) -> dict:
        """
        Load optimization state from a checkpoint file.

        Args:
            checkpoint_path (Path): Path to checkpoint file. If None, loads the
                latest checkpoint.

        Returns:
            dict: Checkpoint data containing population, energies, best solution,
                and config. Returns None if no checkpoint found or loading fails.
        """
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint()

        if checkpoint_path is None or not checkpoint_path.exists():
            return None

        try:
            with np.load(checkpoint_path, allow_pickle=True) as data:
                checkpoint = {
                    "iteration": int(data["iteration"]),
                    "population": data["population"],
                    "population_energies": data["population_energies"],
                    "best_x": data["best_x"],
                    "best_fun": float(data["best_fun"]),
                    "convergence": float(data["convergence"]),
                    "run_start_time": str(data["run_start_time"]),
                    "config": json.loads(str(data["config_json"])),
                    "checkpoint_path": checkpoint_path,
                }
            return checkpoint

        except Exception as e:
            print(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None

    def validate_checkpoint(self, checkpoint: dict, current_config: dict) -> bool:
        """
        Validate that a checkpoint is compatible with the current configuration.

        Args:
            checkpoint (dict): Loaded checkpoint data.
            current_config (dict): Current optimization configuration.

        Returns:
            bool: True if the checkpoint is compatible, False otherwise.
        """
        saved_config = checkpoint["config"]

        if "bounds" in saved_config and "bounds" in current_config:
            if not np.allclose(saved_config["bounds"], current_config["bounds"]):
                print("Warning: Checkpoint bounds do not match current bounds")
                return False

        if "popsize" in saved_config and "popsize" in current_config:
            if saved_config["popsize"] != current_config["popsize"]:
                print("Warning: Checkpoint popsize does not match current popsize")
                return False

        if "active_param_names" in saved_config and "active_param_names" in current_config:
            if saved_config["active_param_names"] != current_config["active_param_names"]:
                print(
                    "Warning: Checkpoint active parameters do not match current parameters\n"
                    f"  Checkpoint: {saved_config['active_param_names']}\n"
                    f"  Current:    {current_config['active_param_names']}"
                )
                return False

        return True


# ============================================================================
# Shared evaluation helper
# ============================================================================


def evaluate_single_pair(
    real_img_path,
    mask_path,
    params_dict,
    temp_dir,
    n_vertices,
    weights,
    region_weights,
    include_ms_ssim=False,
    include_gradient_ssim=False,
    include_power_spectrum=False,
    include_lpips=False,
    save=True,
):
    """Generate a synthetic image for one (image, mask) pair and return the loss.

    This is the single evaluation step shared by the DE optimizer and the
    sensitivity-analysis screening objectives.

    Uses soft foreground / background weight maps: SSIM and friends are
    computed once on the full image, then aggregated twice against the
    region weight maps to produce fg / bg scores without the fake-cliff
    artefacts of hard masking.

    Args:
        real_img_path: Path to the real microscope image.
        mask_path: Path to the segmentation mask.
        params_dict: Full parameter dict (all registry keys).
        temp_dir: Working directory for temporary files.
        n_vertices: Number of vertices for cell shape extraction.
        weights: Metric weights dict.
        region_weights: Region weights dict.  Must contain
            ``background``, ``foreground`` (mixing weights between the
            region losses) and ``foreground_sigma_px`` (pixel-scale of the
            soft falloff outside cells).
        include_ms_ssim: Compute Multi-Scale SSIM.
        include_gradient_ssim: Compute SSIM on Sobel-edge maps.
        include_power_spectrum: Compute power spectrum distance.
        include_lpips: Compute LPIPS perceptual distance.
        save: Whether to write output files to disk.

    Returns:
        float: Scalar loss for this image pair.
    """
    real_img_path = Path(real_img_path)
    mask_path = Path(mask_path)

    synthetic_img, _synthetic_mask = create_synthetic_scene(
        microscope_image_path=real_img_path,
        segmentation_mask_path=mask_path,
        output_dir=temp_dir,
        n_vertices=n_vertices,
        params=params_dict,
        save=save,
    )

    if synthetic_img.dtype == np.uint8:
        synthetic_img = synthetic_img.astype(np.float64) / 255.0

    real_img = load_image(real_img_path)
    original_mask = tiff.imread(mask_path)

    # Shape sanity: catch silent misalignment if the synthetic generator
    # ever stops mirroring the input mask geometry.  Cheap; surfaces as an
    # AssertionError rather than a confusing downstream metric failure.
    assert synthetic_img.shape == real_img.shape, (
        f"synthetic shape {synthetic_img.shape} != real shape {real_img.shape}"
    )
    assert original_mask.shape[:2] == real_img.shape[:2], (
        f"mask shape {original_mask.shape[:2]} != image spatial shape {real_img.shape[:2]}"
    )

    sigma_px = float(
        region_weights.get("foreground_sigma_px", DEFAULT_REGION_WEIGHTS["foreground_sigma_px"])
    )
    w_fg, w_bg = build_region_weight_maps(original_mask, sigma_px=sigma_px)

    metrics_regions = compute_all_metrics_with_weights(
        real_img,
        synthetic_img,
        w_fg=w_fg,
        w_bg=w_bg,
        include_ms_ssim=include_ms_ssim,
        include_gradient_ssim=include_gradient_ssim,
        include_power_spectrum=include_power_spectrum,
        include_lpips=include_lpips,
    )

    return compute_weighted_loss(metrics_regions, weights, region_weights=region_weights)


# ============================================================================
# Picklable Objective Function Class (for cluster multiprocessing)
# ============================================================================


class ObjectiveFunction:
    """Picklable objective function for differential evolution.

    Wraps synthetic image generation + metric computation into a callable
    that can be serialized for multiprocessing under spawn-based pools.

    The instance is constructed once and called many times by DE; each
    call receives a 1-D vector of values in the order of
    ``active_param_names`` and merges them with ``fixed_params`` (and
    registry defaults for anything in neither set) via
    :func:`build_full_params`.
    """

    def __init__(
        self,
        image_pairs,
        weights,
        temp_base_dir,
        n_vertices,
        region_weights,
        active_param_names: List[str] = None,
        fixed_params: Dict = None,
    ):
        """
        Initialize the objective function.

        Args:
            image_pairs: List of (image_path, mask_path) tuples.
            weights (dict): Metric weights for loss computation.
            temp_base_dir: Base directory for temporary worker files.
            n_vertices (int): Number of vertices for cell shape extraction.
            region_weights (dict): Weights for background and foreground regions.
            active_param_names: Ordered list of parameter names being optimized.
                When None, falls back to the full registry order.
            fixed_params: Dict of parameter values not being optimized.
                When None, uses registry defaults.
        """
        self.image_pairs = [(str(img), str(mask)) for img, mask in image_pairs]
        self.weights = weights
        self.temp_base_dir = str(temp_base_dir)
        self.n_vertices = n_vertices
        self.region_weights = region_weights
        self.active_param_names = active_param_names or get_param_names()
        self.fixed_params = fixed_params or {}
        self.include_ms_ssim = weights.get("ms_ssim", 0.0) > 0
        self.include_gradient_ssim = weights.get("gradient_ssim", 0.0) > 0
        self.include_power_spectrum = weights.get("power_spectrum", 0.0) > 0
        self.include_lpips = weights.get("lpips", 0.0) > 0

    def __call__(self, x):
        """
        Evaluate the objective function for a given parameter vector.

        Args:
            x: Parameter vector (same order as *active_param_names*).

        Returns:
            float: Average loss across all image pairs. Returns 1e10 on error.
        """
        try:
            worker_id = os.getpid()
            temp_dir = Path(self.temp_base_dir) / f"worker_{worker_id}"
            temp_dir.mkdir(exist_ok=True)

            # Build the full parameter dict from active values + fixed
            params_dict = build_full_params(
                self.active_param_names, list(x), self.fixed_params
            )

            total_loss = 0
            for real_img_path_str, mask_path_str in self.image_pairs:
                loss = evaluate_single_pair(
                    real_img_path=real_img_path_str,
                    mask_path=mask_path_str,
                    params_dict=params_dict,
                    temp_dir=temp_dir,
                    n_vertices=self.n_vertices,
                    weights=self.weights,
                    region_weights=self.region_weights,
                    include_ms_ssim=self.include_ms_ssim,
                    include_gradient_ssim=self.include_gradient_ssim,
                    include_power_spectrum=self.include_power_spectrum,
                    include_lpips=self.include_lpips,
                    save=False,
                )
                total_loss += loss

            avg_loss = total_loss / len(self.image_pairs)
            return avg_loss

        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10


# ============================================================================
# Core Functions
# ============================================================================


def find_real_images(input_dir: Path, limit: int = None) -> List[Tuple[Path, Path]]:
    """
    Find all real microscope images and their corresponding masks.

    Searches recursively for ``.tif`` files and matches them with their
    corresponding ``_masks.tif`` files. Skips files prefixed with ``syn_``.

    Args:
        input_dir (Path): Directory to search for image pairs.
        limit (int): Maximum number of pairs to return. If None, returns all.

    Returns:
        List[Tuple[Path, Path]]: List of (image_path, mask_path) tuples.
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    all_tif_files = list(input_dir.rglob("*.tif"))

    real_images = []
    for tif_file in all_tif_files:
        filename = tif_file.name

        if filename.startswith("syn_"):
            continue

        if filename.endswith("_masks.tif"):
            continue

        mask_filename = filename.replace(".tif", "_masks.tif")
        mask_path = tif_file.parent / mask_filename

        if mask_path.exists():
            real_images.append((tif_file, mask_path))
        else:
            print(f"Warning: No mask found for {tif_file}, skipping")

    if len(real_images) == 0:
        raise ValueError(f"No valid image pairs found in {input_dir}")

    real_images.sort(key=lambda x: str(x[0]))

    if limit is not None and limit > 0:
        real_images = real_images[:limit]

    print(f"Found {len(real_images)} image pairs")
    for img_path, mask_path in real_images:
        print(f"  - {img_path.name}")

    return real_images


def compute_weighted_loss(
    metrics_regions: Dict[str, Dict[str, float]],
    weights: Dict,
    region_weights: Optional[Dict] = None,
) -> float:
    """Compute the scalar weighted loss from region-partitioned metrics.

    Accepts the output of
    :func:`crm_gen.metrics.compute_all_metrics_with_weights` — a dict with
    ``"full"``, ``"fg"`` and ``"bg"`` entries, each a flat summary mapping
    metric-name → score.

    Region-split (SSIM / MS-SSIM / gradient-SSIM) terms are mixed per
    ``region_weights["background"]`` and ``region_weights["foreground"]``.
    Global-only terms (histogram distance, LPIPS, power spectrum) are
    taken from the ``"full"`` bucket and added without region mixing —
    they're defined over the whole image and would be double-counted via
    the region mixer.

    Note: this signature replaced the older
    ``(metrics, weights, metrics_bg=, metrics_fg=, region_weights=)`` form
    used before the soft-weight-map refactor.  External callers should
    pass the dict produced by ``compute_all_metrics_with_weights``.

    Args:
        metrics_regions: Dict with keys ``full`` / ``fg`` / ``bg``.
        weights: Metric-name → weight.
        region_weights: ``background`` + ``foreground`` mixing weights.

    Returns:
        Weighted loss (lower is better).
    """
    if region_weights is None:
        region_weights = DEFAULT_REGION_WEIGHTS

    def per_region_loss(summary: Dict[str, float]) -> float:
        loss = 0.0
        if "ssim_score" in summary:
            loss += weights.get("ssim", 0.0) * (1.0 - summary["ssim_score"])
        if "ms_ssim_score" in summary and weights.get("ms_ssim", 0.0) > 0:
            loss += weights["ms_ssim"] * (1.0 - summary["ms_ssim_score"])
        if "gradient_ssim_score" in summary and weights.get("gradient_ssim", 0.0) > 0:
            loss += weights["gradient_ssim"] * (1.0 - summary["gradient_ssim_score"])
        return loss

    fg_summary = metrics_regions["fg"]
    bg_summary = metrics_regions["bg"]
    full_summary = metrics_regions["full"]

    total = (
        region_weights["background"] * per_region_loss(bg_summary)
        + region_weights["foreground"] * per_region_loss(fg_summary)
    )

    # Global-only terms (not region-partitioned).
    if "histogram_distance" in full_summary:
        total += weights.get("histogram_distance", 0.0) * full_summary["histogram_distance"]
    if "lpips_distance" in full_summary and weights.get("lpips", 0.0) > 0:
        total += weights["lpips"] * full_summary["lpips_distance"]
    if (
        "power_spectrum_distance" in full_summary
        and weights.get("power_spectrum", 0.0) > 0
    ):
        total += weights["power_spectrum"] * full_summary["power_spectrum_distance"]

    return total


def optimize_parameters(
    image_pairs: List[Tuple[Path, Path]],
    bounds: List[Tuple[float, float]],
    weights: Dict,
    region_weights: Dict,
    maxiter: int = 50,
    popsize: int = 15,
    workers: int = -1,
    seed: int = 42,
    n_vertices: int = 8,
    resume: bool = False,
    no_checkpoint: bool = False,
    active_param_names: List[str] = None,
    fixed_params: Dict = None,
    checkpoint_dir: Path = None,
    init_seed_x: Optional[np.ndarray] = None,
) -> Dict:
    """
    Optimize synthetic image parameters using differential evolution.

    Minimizes a weighted loss combining histogram distance, SSIM, and PSNR
    across all image pairs. Supports checkpointing and resuming.

    Args:
        image_pairs (List[Tuple[Path, Path]]): List of (image_path, mask_path) tuples.
        bounds (List[Tuple[float, float]]): Parameter bounds for each dimension.
        weights (Dict): Metric weights for loss computation.
        region_weights (Dict): Weights for background and foreground regions.
        maxiter (int): Maximum number of iterations.
        popsize (int): Population size multiplier (population = popsize * num_params).
        workers (int): Number of worker processes (-1 = all CPUs, 1 = sequential).
        seed (int): Random seed for reproducibility.
        n_vertices (int): Number of vertices for cell shape extraction.
        resume (bool): If True, resume from latest checkpoint.
        no_checkpoint (bool): If True, disable checkpointing.
        active_param_names: Ordered list of parameter names to optimize.
            When None, uses the full registry.
        fixed_params: Dict of parameter values not being optimized.
            When None, uses registry defaults.

    Returns:
        Dict: Optimization results including 'parameters', 'optimization_info',
            'bounds', and 'weights'.
    """
    if active_param_names is None:
        active_param_names = get_param_names()
    if fixed_params is None:
        fixed_params = {}

    print("\n" + "=" * 80)
    print("STARTING PARAMETER OPTIMIZATION")
    print("=" * 80)
    print(f"Number of images: {len(image_pairs)}")
    print(f"Active parameters: {len(active_param_names)}")
    if fixed_params:
        print(f"Fixed parameters: {len(fixed_params)}")
    print("Parameter bounds:")
    for name, (low, high) in zip(active_param_names, bounds):
        print(f"  {name}: [{low}, {high}]")
    print("Metric weights:")
    for key, value in weights.items():
        print(f"  {key}: {value}")
    print("Differential evolution settings:")
    print(f"  maxiter: {maxiter}")
    print(f"  popsize: {popsize}")
    print(f"  workers: {workers}")
    print(f"  seed: {seed}")
    print("=" * 80 + "\n")

    temp_base_dir = tempfile.mkdtemp(prefix="fit_temp_")
    print(f"Using temporary directory: {temp_base_dir}\n")

    start_time = time.time()

    try:
        objective_fn = ObjectiveFunction(
            image_pairs=image_pairs,
            weights=weights,
            temp_base_dir=temp_base_dir,
            n_vertices=n_vertices,
            region_weights=region_weights,
            active_param_names=active_param_names,
            fixed_params=fixed_params,
        )

        if checkpoint_dir is None:
            checkpoint_dir = Path("./checkpoints")
        checkpoint_mgr = None if no_checkpoint else DECheckpointManager(checkpoint_dir)

        init_population = "latinhypercube"
        start_iter = 0
        run_start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if resume and checkpoint_mgr:
            checkpoint = checkpoint_mgr.load_checkpoint()
            if checkpoint:
                config = {
                    "bounds": bounds,
                    "popsize": popsize,
                    "active_param_names": active_param_names,
                }
                if checkpoint_mgr.validate_checkpoint(checkpoint, config):
                    init_population = checkpoint["population"]
                    start_iter = checkpoint["iteration"]
                    run_start_time = checkpoint["run_start_time"]
                    checkpoint_mgr.run_start_time = run_start_time
                    # Restore active/fixed params from checkpoint
                    saved_cfg = checkpoint["config"]
                    if "active_param_names" in saved_cfg:
                        active_param_names = saved_cfg["active_param_names"]
                    if "fixed_params" in saved_cfg:
                        fixed_params = saved_cfg["fixed_params"]
                    print(
                        f"\n*** RESUMING from checkpoint at iteration {start_iter} ***"
                    )
                    print(f"    Best loss so far: {checkpoint['best_fun']:.6f}")
                    print(f"    Active parameters: {active_param_names}")
                    print(f"    Checkpoint: {checkpoint['checkpoint_path'].name}\n")
                else:
                    print("Checkpoint validation failed, starting fresh")
            else:
                print("No checkpoint found, starting fresh")

        # Warm-start the init population with a caller-supplied seed vector.
        # Only applied when we're not resuming from a checkpoint (the checkpoint
        # already carries a valid population).
        if init_seed_x is not None and isinstance(init_population, str):
            init_seed_x = np.asarray(init_seed_x, dtype=float)
            if init_seed_x.shape != (len(bounds),):
                raise ValueError(
                    f"init_seed_x has shape {init_seed_x.shape}; expected ({len(bounds)},)"
                )
            rng = np.random.default_rng(seed)
            n_pop = popsize * len(bounds)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            # Latin-hypercube the rest of the population; row 0 holds the seed.
            rest = lo + rng.random((max(n_pop - 1, 0), len(bounds))) * (hi - lo)
            seeded = np.vstack([init_seed_x[None, :], rest])
            # Clamp seed into bounds in case it's slightly outside (e.g. due to
            # rounding in the caller's scaling).
            seeded[0] = np.clip(seeded[0], lo, hi)
            init_population = seeded
            print(
                f"Seeding DE init population with caller-supplied values for "
                f"{len(init_seed_x)} parameters."
            )

        remaining_iters = max(1, maxiter - start_iter)
        iteration_counter = [start_iter]

        def callback(intermediate_result):
            iteration_counter[0] += 1
            current_iter = iteration_counter[0]

            progress = current_iter / maxiter
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)

            print(f"\r\033[K", end="")
            print(
                f"Iter {current_iter:3d}/{maxiter} [{bar}] {progress * 100:5.1f}%",
                end="",
                flush=True,
            )

            if checkpoint_mgr:
                checkpoint_mgr.save_checkpoint(
                    iteration=current_iter,
                    population=intermediate_result.population,
                    population_energies=intermediate_result.population_energies,
                    best_x=intermediate_result.x,
                    best_fun=intermediate_result.fun,
                    convergence=intermediate_result.convergence
                    if hasattr(intermediate_result, "convergence")
                    else 0.0,
                    config={
                        "bounds": bounds,
                        "popsize": popsize,
                        "maxiter": maxiter,
                        "weights": weights,
                        "region_weights": region_weights,
                        "active_param_names": active_param_names,
                        "fixed_params": fixed_params,
                    },
                )

            xk = intermediate_result.x
            if current_iter % 5 == 0:
                print()
                parts = [
                    f"{n}={v:.4g}" for n, v in zip(active_param_names, xk)
                ]
                print(f"  └─ Params: {', '.join(parts)}")

        print("\n" + "┌" + "─" * 78 + "┐")
        print("│" + " " * 25 + "OPTIMIZATION RUNNING" + " " * 33 + "│")
        print("└" + "─" * 78 + "┘\n")
        print("This may take a while depending on the number of images and iterations.")
        print("Progress updates will appear below:\n")

        # Resolve worker count against the SLURM/cgroups affinity rather than
        # the full host CPU count, matching the screening pool.
        import multiprocessing as _mp
        from concurrent.futures import ProcessPoolExecutor as _PPE

        if workers == 1:
            n_workers = 1
        else:
            available = (
                len(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity")
                else _mp.cpu_count()
            )
            n_workers = available if workers == -1 else workers

        de_kwargs = dict(
            bounds=bounds,
            maxiter=remaining_iters,
            popsize=popsize,
            updating="deferred" if n_workers != 1 else "immediate",
            polish=False,
            init=init_population,
            strategy="best1bin",
            seed=seed if start_iter == 0 else None,
            disp=False,
            callback=callback,
            mutation=(0.5, 1.3),
        )

        if n_workers == 1:
            result = differential_evolution(objective_fn, workers=1, **de_kwargs)
        else:
            # Use a spawn pool so workers start with a clean interpreter.
            # scipy's default (fork) would inherit any OpenGL/VTK state that
            # lives in the parent — unsafe across fork and the source of the
            # framebuffer-dump spam on SLURM.  Spawn also keeps the Rust core
            # and BLAS threads happy, same reasoning as the screening pool.
            ctx = _mp.get_context("spawn")
            with _PPE(max_workers=n_workers, mp_context=ctx) as pool:
                result = differential_evolution(
                    objective_fn, workers=pool.map, **de_kwargs
                )

        end_time = time.time()
        elapsed_time = end_time - start_time

        print("\n")

        print("\n" + "=" * 80)
        print("OPTIMIZATION COMPLETE")
        print("=" * 80)
        print(f"Success: {result.success}")
        print(f"Message: {result.message}")
        print(f"Number of iterations: {result.nit}")
        print(f"Number of function evaluations: {result.nfev}")
        print(f"Final loss: {result.fun:.6f}")
        print(
            f"Optimization duration: {elapsed_time / 60:.2f} minutes ({elapsed_time:.1f} seconds)"
        )
        print("\nOptimized parameters:")
        for name, value in zip(active_param_names, result.x):
            print(f"  {name}: {value:.6f}")
        if fixed_params:
            print("Fixed parameters:")
            for name, value in fixed_params.items():
                print(f"  {name}: {value}")
        print("=" * 80 + "\n")

        # Build full parameter dict (active + fixed)
        full_params = build_full_params(
            active_param_names, list(result.x), fixed_params
        )

        optimization_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "optimization_info": {
                "num_images": len(image_pairs),
                "num_iterations": int(result.nit),
                "num_function_evaluations": int(result.nfev),
                "final_loss": float(result.fun),
                "optimization_duration_seconds": float(elapsed_time),
                "optimization_duration_minutes": float(elapsed_time / 60),
                "success": bool(result.success),
                "message": result.message,
                "active_param_names": active_param_names,
            },
            "parameters": {
                name: float(value) for name, value in full_params.items()
            },
            "bounds": {
                name: [float(low), float(high)]
                for name, (low, high) in zip(active_param_names, bounds)
            },
            "weights": weights,
        }
        if fixed_params:
            optimization_results["fixed_params"] = {
                k: float(v) for k, v in fixed_params.items()
            }

        return optimization_results

    finally:
        if Path(temp_base_dir).exists():
            shutil.rmtree(temp_base_dir)
            print(f"Cleaned up temporary directory: {temp_base_dir}")


def compute_final_metrics(
    image_pairs: List[Tuple[Path, Path]],
    params: Dict,
    output_dir: Path,
    n_vertices: int,
    weights: Dict = None,
    region_weights: Dict = None,
) -> Tuple[Dict, List[Dict], Dict]:
    """Compute final metrics for all images with optimized parameters.

    Generates the synthetic image for each pair using the optimized
    parameters and computes the per-image metrics using the same soft
    foreground/background weight-map machinery the DE inner loop uses.
    The CSV columns therefore reflect the values DE actually optimised
    against, not a different (and previously incompatible) hard-mask
    flavour.

    Args:
        image_pairs: List of ``(image_path, mask_path)`` tuples.
        params: Optimized parameter dictionary.
        output_dir: Output directory (currently passed through; not used
            for IO inside this function — synth_cache is returned for the
            caller to write).
        n_vertices: Vertices per cell for shape extraction.
        weights: Metric weights.  ``ms_ssim``, ``gradient_ssim``,
            ``power_spectrum``, ``lpips`` are included when their weight
            is > 0.
        region_weights: Region weights dict (must contain
            ``foreground_sigma_px``).  Falls back to
            ``DEFAULT_REGION_WEIGHTS`` when not supplied.

    Returns:
        ``(avg_metrics, per_image_metrics, synth_cache)``.  The CSV
        columns are: ``full_histogram_distance``, ``full_psnr_db``,
        ``full/fg/bg_ssim_score``, plus optional ``full/fg/bg_ms_ssim``,
        ``full/fg/bg_gradient_ssim_score``, ``full_power_spectrum``,
        ``full_lpips_distance``.  PSNR / histogram / power spectrum /
        LPIPS are global only — region columns for those were removed
        when the metrics path moved to the soft-weight-map model.
    """
    print("\nComputing final metrics with optimized parameters...")

    include_ms_ssim = weights.get("ms_ssim", 0.0) > 0 if weights else False
    include_gradient_ssim = weights.get("gradient_ssim", 0.0) > 0 if weights else False
    include_power_spectrum = weights.get("power_spectrum", 0.0) > 0 if weights else False
    include_lpips = weights.get("lpips", 0.0) > 0 if weights else False

    if region_weights is None:
        region_weights = DEFAULT_REGION_WEIGHTS
    sigma_px = float(region_weights.get(
        "foreground_sigma_px", DEFAULT_REGION_WEIGHTS["foreground_sigma_px"]
    ))

    per_image_metrics = []
    synth_cache = {}

    for real_img_path, mask_path in tqdm(image_pairs, desc="Computing metrics"):
        synthetic_img, synthetic_mask = create_synthetic_scene(
            microscope_image_path=real_img_path,
            segmentation_mask_path=mask_path,
            output_dir=None,
            n_vertices=n_vertices,
            params=params,
            save=False,
        )

        if synthetic_img.dtype == np.uint8:
            synthetic_img = synthetic_img.astype(np.float64) / 255.0

        synth_cache[real_img_path.name] = (synthetic_img, synthetic_mask)

        real_img = load_image(real_img_path)
        original_mask = tiff.imread(mask_path)

        w_fg, w_bg = build_region_weight_maps(original_mask, sigma_px=sigma_px)

        regions = compute_all_metrics_with_weights(
            real_img,
            synthetic_img,
            w_fg=w_fg,
            w_bg=w_bg,
            include_ms_ssim=include_ms_ssim,
            include_gradient_ssim=include_gradient_ssim,
            include_power_spectrum=include_power_spectrum,
            include_lpips=include_lpips,
        )

        full = regions["full"]
        fg = regions["fg"]
        bg = regions["bg"]

        entry = {
            "image_name": real_img_path.name,
            # Global-only columns (no fg/bg counterparts under the new model).
            "full_histogram_distance": full["histogram_distance"],
            "full_psnr_db": full["psnr_db"],
            # SSIM is region-partitioned.
            "full_ssim_score": full["ssim_score"],
            "fg_ssim_score": fg["ssim_score"],
            "bg_ssim_score": bg["ssim_score"],
        }
        if include_ms_ssim:
            entry["full_ms_ssim"] = full["ms_ssim_score"]
            entry["fg_ms_ssim"] = fg["ms_ssim_score"]
            entry["bg_ms_ssim"] = bg["ms_ssim_score"]
        if include_gradient_ssim:
            entry["full_gradient_ssim_score"] = full["gradient_ssim_score"]
            entry["fg_gradient_ssim_score"] = fg["gradient_ssim_score"]
            entry["bg_gradient_ssim_score"] = bg["gradient_ssim_score"]
        if include_power_spectrum:
            entry["full_power_spectrum"] = full["power_spectrum_distance"]
        if include_lpips:
            entry["full_lpips_distance"] = full["lpips_distance"]

        per_image_metrics.append(entry)

    # Dynamically average all numeric keys
    avg_metrics = {}
    if per_image_metrics:
        keys = [k for k in per_image_metrics[0] if k != "image_name"]
        for k in keys:
            avg_metrics[k] = float(np.mean([m[k] for m in per_image_metrics]))

    print("\nAverage metrics across all images:")
    print(
        f"  Full       - Histogram: {avg_metrics.get('full_histogram_distance', 0):.4f}, "
        f"SSIM: {avg_metrics.get('full_ssim_score', 0):.4f}, "
        f"PSNR: {avg_metrics.get('full_psnr_db', 0):.2f} dB"
    )
    print(
        f"  Foreground - SSIM: {avg_metrics.get('fg_ssim_score', 0):.4f}"
    )
    print(
        f"  Background - SSIM: {avg_metrics.get('bg_ssim_score', 0):.4f}"
    )
    if include_ms_ssim:
        print(
            f"  MS-SSIM    - Full: {avg_metrics.get('full_ms_ssim', 0):.4f}, "
            f"FG: {avg_metrics.get('fg_ms_ssim', 0):.4f}, "
            f"BG: {avg_metrics.get('bg_ms_ssim', 0):.4f}"
        )
    if include_gradient_ssim:
        print(
            f"  Grad-SSIM  - Full: {avg_metrics.get('full_gradient_ssim_score', 0):.4f}, "
            f"FG: {avg_metrics.get('fg_gradient_ssim_score', 0):.4f}, "
            f"BG: {avg_metrics.get('bg_gradient_ssim_score', 0):.4f}"
        )
    if include_power_spectrum:
        print(f"  PSD (full) : {avg_metrics.get('full_power_spectrum', 0):.4f}")
    if include_lpips:
        print(f"  LPIPS (full): {avg_metrics.get('full_lpips_distance', 0):.4f}")

    return avg_metrics, per_image_metrics, synth_cache


def save_results(
    results: Dict, avg_metrics: Dict, per_image_metrics: List[Dict], output_dir: Path
):
    """
    Save optimization results to JSON and CSV files.

    Args:
        results (Dict): Optimization results from optimize_parameters().
        avg_metrics (Dict): Average metrics from compute_final_metrics().
        per_image_metrics (List[Dict]): Per-image metrics list.
        output_dir (Path): Output directory for saving files.
    """
    print(f"\nSaving results to: {output_dir}")

    results["average_metrics"] = avg_metrics

    params_path = output_dir / "optimized_params.json"
    with open(params_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {params_path.name}")

    csv_path = output_dir / "per_image_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(per_image_metrics[0].keys()) if per_image_metrics else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_image_metrics)
    print(f"  Saved: {csv_path.name}")


def generate_all_synthetics(
    image_pairs: List[Tuple[Path, Path]],
    params: Dict,
    output_dir: Path,
    n_vertices: int,
):
    """
    Generate synthetic images for all image pairs with optimized parameters.

    Args:
        image_pairs (List[Tuple[Path, Path]]): List of (image_path, mask_path) tuples.
        params (Dict): Optimized parameter dictionary.
        output_dir (Path): Output directory for saving synthetic images.
        n_vertices (int): Number of vertices for cell shape extraction.
    """
    print(f"\nGenerating synthetic images for all {len(image_pairs)} image pairs...")

    synth_dir = output_dir / "synthetic_images"
    synth_dir.mkdir(exist_ok=True)

    for real_img_path, mask_path in tqdm(image_pairs, desc="Generating synthetics"):
        create_synthetic_scene(
            microscope_image_path=real_img_path,
            segmentation_mask_path=mask_path,
            output_dir=synth_dir,
            n_vertices=n_vertices,
            params=params,
        )

    print(f"  Saved synthetic images to: {synth_dir}")


# ============================================================================
# Coarse pre-fit for dominant parameters
# ============================================================================


def coarse_prefit(
    image_pairs: List[Tuple[Path, Path]],
    dominant_params: List[str],
    weights: Dict,
    region_weights: Dict,
    maxiter: int = 10,
    popsize: int = 5,
    workers: int = -1,
    seed: int = 0,
    n_vertices: int = 8,
    fixed_params: Dict = None,
) -> Dict[str, float]:
    """Short DE fit over ``dominant_params`` only; returns best-value dict.

    Purpose: before Morris / Sobol screening, park the dominant parameters
    (typically ``bg_base_brightness``) near their optimum so the screening
    loss isn't drowned in their variance.  Subordinate parameters can then
    have their real sensitivity measured.

    The dominant parameters rejoin the active set for the main DE fit —
    this pre-fit only provides good pinned values for screening and a warm
    start for the main fit.

    Args:
        image_pairs: (real, mask) pairs used for the fit.
        dominant_params: Parameter names to pre-fit.  Anything not in this
            list is treated as fixed at its default (or at the value in
            ``fixed_params`` if provided).
        weights, region_weights: Same metric / region weight dicts used by
            the main fit.  Keeping them consistent means the pre-fit
            optimises the same loss the main fit will.
        maxiter: Iterations for the coarse DE.  Small — the goal is
            near-optimal, not converged.
        popsize: Population multiplier.
        workers, seed, n_vertices: Forwarded to
            :func:`optimize_parameters`.
        fixed_params: Starting values for non-dominant parameters.  If
            None, registry defaults are used implicitly via
            ``build_full_params``.

    Returns:
        Dict mapping each name in ``dominant_params`` to its best-fit
        scalar value.
    """
    from .parameter_registry import PARAMETER_REGISTRY

    if not dominant_params:
        return {}

    fixed_params = dict(fixed_params or {})

    bounds = []
    for name in dominant_params:
        if name not in PARAMETER_REGISTRY:
            raise ValueError(
                f"Coarse pre-fit: unknown parameter '{name}'. "
                f"Not present in PARAMETER_REGISTRY."
            )
        bounds.append(PARAMETER_REGISTRY[name].bounds)

    print("\n" + "=" * 80)
    print("COARSE PRE-FIT (dominant parameters)")
    print("=" * 80)
    print(f"Parameters: {', '.join(dominant_params)}")
    print(f"Budget: maxiter={maxiter}, popsize={popsize}")
    print("=" * 80)

    result = optimize_parameters(
        image_pairs=image_pairs,
        bounds=bounds,
        weights=weights,
        region_weights=region_weights,
        maxiter=maxiter,
        popsize=popsize,
        workers=workers,
        seed=seed,
        n_vertices=n_vertices,
        resume=False,
        no_checkpoint=True,
        active_param_names=list(dominant_params),
        fixed_params=fixed_params,
        checkpoint_dir=None,
    )

    best_values = {}
    params_out = result.get("parameters", {})
    for name in dominant_params:
        if name in params_out:
            best_values[name] = float(params_out[name])

    print("\nCoarse pre-fit results:")
    for name, val in best_values.items():
        print(f"  {name} = {val:.6g}")
    print("=" * 80 + "\n")

    return best_values
