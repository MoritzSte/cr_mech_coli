#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for crm_gen (Synthetic Microscope Image Generation).

Provides one command with three subcommands::

    crm_gen run   [--config path/to/gen_config.toml]
    crm_gen clone img.tif mask.tif [--config path/to/gen_config.toml]
    crm_gen fit   path/to/input/dir [--config path/to/fit_config.toml]
"""

import argparse
import sys
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict

import numpy as np


def _build_run_parser(subparsers, default_gen_config):
    """
    Add the ``run`` subcommand to the argument parser.

    Args:
        subparsers: Subparser group from argparse.
        default_gen_config (Path): Default path to the generation config file.
    """
    run_parser = subparsers.add_parser(
        "run",
        help="Run the synthetic image generation pipeline",
        description="Run bacteria growth simulation and generate synthetic "
        "microscope images. All parameters come from the generation "
        "config file.",
    )
    run_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(default_gen_config) if default_gen_config.exists() else None,
        help="Path to generation TOML config file",
    )


def _build_clone_parser(subparsers, default_gen_config):
    """
    Add the ``clone`` subcommand to the argument parser.

    Args:
        subparsers: Subparser group from argparse.
        default_gen_config (Path): Default path to the generation config file.
    """
    clone_parser = subparsers.add_parser(
        "clone",
        help="Clone a real microscope image to synthetic",
        description="Create a synthetic version of a real microscope image "
        "using cell positions extracted from a segmentation mask. "
        "Imaging parameters come from the generation config file.",
    )
    clone_parser.add_argument(
        "microscope_image",
        type=str,
        help="Path to real microscope image (TIF)",
    )
    clone_parser.add_argument(
        "segmentation_mask",
        type=str,
        help="Path to segmentation mask (TIF)",
    )
    clone_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="./synthetic_output",
        help="Output directory (default: ./synthetic_output)",
    )
    clone_parser.add_argument(
        "--n-vertices",
        type=int,
        default=None,
        help="Number of vertices per cell (overrides config, default: 8)",
    )
    clone_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    clone_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(default_gen_config) if default_gen_config.exists() else None,
        help="Path to generation TOML config file",
    )


def _build_fit_parser(subparsers, default_fit_config):
    """
    Add the ``fit`` subcommand to the argument parser.

    Args:
        subparsers: Subparser group from argparse.
        default_fit_config (Path): Default path to the fit config file.
    """
    fit_parser = subparsers.add_parser(
        "fit",
        help="Optimize parameters to match real microscope images",
        description="Optimize synthetic image generation parameters to match "
        "real microscope images using differential evolution. "
        "The imaging parameters are the output of the fit. "
        "Optimization hyperparameters come from the fit config file.",
    )
    fit_parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing real microscope images (searched recursively)",
    )
    fit_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(default_fit_config) if default_fit_config.exists() else None,
        help="Path to fit TOML config file",
    )
    fit_parser.add_argument(
        "--auto-screen",
        action="store_true",
        default=False,
        help="Run Morris+Sobol screening before optimization to auto-discover "
        "relevant parameters",
    )
    fit_parser.add_argument(
        "--screening-results",
        type=str,
        default=None,
        help="Path to pre-computed screening results JSON (from 'crm_gen screen')",
    )


def _build_screen_parser(subparsers, default_fit_config):
    """
    Add the ``screen`` subcommand to the argument parser.

    Args:
        subparsers: Subparser group from argparse.
        default_fit_config (Path): Default path to the fit config file.
    """
    screen_parser = subparsers.add_parser(
        "screen",
        help="Run sensitivity analysis to discover relevant parameters",
        description="Run Morris screening (and optionally Sobol analysis) "
        "on real microscope images to automatically discover which "
        "parameters matter. Results can be loaded by 'crm_gen fit'.",
    )
    screen_parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing real microscope images (searched recursively)",
    )
    screen_parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=str(default_fit_config) if default_fit_config.exists() else None,
        help="Path to fit TOML config file (uses [screening] section)",
    )
    screen_parser.add_argument(
        "--skip-sobol",
        action="store_true",
        default=False,
        help="Run Morris screening only, skip Sobol analysis",
    )
    screen_parser.add_argument(
        "--num-images",
        type=int,
        default=None,
        help="Limit screening to N representative images (default: from config or 2)",
    )
    screen_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for screening results JSON (default: ./screening_results.json)",
    )


def _run_generate(config, config_path):
    """
    Run the synthetic image generation pipeline.

    Args:
        config (dict): Parsed TOML configuration.
        config_path (Path): Path to the config file (copied to output dir).
    """
    from .pipeline import run_pipeline

    pipeline_config = config.get("pipeline", {})
    simulation_config = config.get("simulation", {})
    rendering_config = config.get("rendering", {})
    synthetic_config = config.get("synthetic", {})
    background_config = config.get("background", {})
    halo_config = config.get("halo", {})
    brightness_config = config.get("brightness", {})

    # Extract pipeline parameters
    output_dir = pipeline_config.get("output_dir", "./outputs")
    n_simulations = pipeline_config.get("n_simulations", 1)
    n_frames = pipeline_config.get("n_frames", 10)
    image_size = tuple(pipeline_config.get("image_size", [512, 512]))
    seed = pipeline_config.get("seed", 0)
    if seed == 0:
        seed = None
    n_workers = pipeline_config.get("n_workers", 0)
    if n_workers == 0:
        n_workers = None
    skip_synthetic = pipeline_config.get("skip_synthetic", False)
    delete_generated = pipeline_config.get("delete_generated", False)

    # Extract simulation parameters
    n_bacteria = simulation_config.get("n_bacteria", [1, 10])
    if isinstance(n_bacteria, list):
        n_bacteria_range = (int(n_bacteria[0]), int(n_bacteria[1]))
    else:
        n_bacteria_range = (int(n_bacteria), int(n_bacteria))

    border_distance = simulation_config.get("border_distance", 5.0)
    n_vertices = int(simulation_config.get("n_vertices", 8))

    slt = simulation_config.get("spring_length_threshold", [6.0, 6.0])
    if isinstance(slt, list):
        max_bacteria_length = float(slt[0])
    else:
        max_bacteria_length = float(slt)

    # Copy config file to output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_path / "config.toml")

    run_pipeline(
        output_dir=output_dir,
        n_frames=n_frames,
        image_size=image_size,
        n_bacteria_range=n_bacteria_range,
        border_distance=border_distance,
        max_bacteria_length=max_bacteria_length,
        simulation_seed=seed,
        n_vertices=n_vertices,
        skip_synthetic=skip_synthetic,
        delete_generated=delete_generated,
        n_workers=n_workers,
        sim_param_ranges=simulation_config,
        rendering_config=rendering_config,
        synthetic_config=synthetic_config,
        background_config=background_config,
        halo_config=halo_config,
        brightness_config=brightness_config,
        n_simulations=n_simulations,
    )


def _run_clone(args, config):
    """
    Create a synthetic clone of a real microscope image.

    Args:
        args: Parsed CLI arguments (microscope_image, segmentation_mask,
            output, n_vertices, seed).
        config (dict): Parsed TOML configuration.
    """
    from .scene import create_synthetic_scene

    synthetic_config = config.get("synthetic", {})
    background_config = config.get("background", {})
    halo_config = config.get("halo", {})
    brightness_config = config.get("brightness", {})
    simulation_config = config.get("simulation", {})

    # CLI overrides > config > defaults
    n_vertices = args.n_vertices
    if n_vertices is None:
        n_vertices = int(simulation_config.get("n_vertices", 8))

    seed = args.seed
    if seed is None:
        seed = synthetic_config.get("seed", None)

    create_synthetic_scene(
        microscope_image_path=args.microscope_image,
        segmentation_mask_path=args.segmentation_mask,
        output_dir=args.output,
        n_vertices=n_vertices,
        seed=seed,
        bg_base_brightness=synthetic_config.get("bg_base_brightness", 0.56),
        bg_gradient_strength=synthetic_config.get("bg_gradient_strength", 0.027),
        bac_halo_intensity=synthetic_config.get("bac_halo_intensity", 0.40),
        bg_noise_scale=int(synthetic_config.get("bg_noise_scale", 20)),
        psf_sigma=synthetic_config.get("psf_sigma", 1.0),
        peak_signal=synthetic_config.get("peak_signal", 6000.0),
        gaussian_sigma=synthetic_config.get("gaussian_sigma", 0.01),
        brightness_mode=brightness_config.get("mode", "original"),
        brightness_range=tuple(brightness_config.get("brightness_range", [0.6, 0.3])),
        num_dark_spots_range=tuple(background_config.get("num_dark_spots_range", [0, 5])),
        brightness_noise_strength=brightness_config.get("noise_strength", 0.0),
        apply_psf=synthetic_config.get("apply_psf", True),
        apply_poisson=synthetic_config.get("apply_poisson", True),
        apply_gaussian=synthetic_config.get("apply_gaussian", True),
        halo_inner_width=halo_config.get("inner_width", 2.0),
        halo_outer_width=halo_config.get("outer_width", 50.0),
        halo_blur_sigma=halo_config.get("blur_sigma", 0.5),
        halo_fade_type=halo_config.get("fade_type", "exponential"),
        dark_spot_size_range=tuple(background_config.get("dark_spot_size_range", [2, 5])),
        num_light_spots_range=tuple(background_config.get("num_light_spots_range", [0, 0])),
        texture_strength=background_config.get("texture_strength", 0.02),
        texture_scale=background_config.get("texture_scale", 1.5),
        bg_blur_sigma=background_config.get("blur_sigma", 0.8),
    )


def _extract_weights(opt_config):
    """Extract metric and region weights from the [optimization] config section."""
    from .config import DEFAULT_METRIC_WEIGHTS, DEFAULT_REGION_WEIGHTS

    mw = opt_config.get("metric_weights", {})
    weights = {k: mw.get(k, v) for k, v in DEFAULT_METRIC_WEIGHTS.items()}

    rw = opt_config.get("region_weights", {})
    region_weights = {k: rw.get(k, v) for k, v in DEFAULT_REGION_WEIGHTS.items()}

    return weights, region_weights


def _run_screen(args, config, config_path):
    """
    Run sensitivity analysis to discover relevant parameters.

    Args:
        args: Parsed CLI arguments (input_dir, skip_sobol, num_images, output).
        config (dict): Parsed TOML configuration (uses [screening] section).
        config_path (Path): Path to the config file.
    """
    from .optimization import find_real_images
    from .sensitivity import (
        run_full_screening,
        save_screening_results,
    )

    screen_config = config.get("screening", {})
    opt_config = config.get("optimization", {})

    # Screening parameters
    num_trajectories = screen_config.get("num_trajectories", 10)
    morris_top_n = screen_config.get("morris_top_n", 10)
    run_sobol = not args.skip_sobol and screen_config.get("run_sobol", True)
    sobol_n_samples = screen_config.get("sobol_n_samples", 128)
    sobol_top_n = screen_config.get("sobol_top_n", 7)
    num_images = args.num_images or screen_config.get("num_screening_images", 2)
    n_vertices = opt_config.get("n_vertices", 8)
    seed = opt_config.get("seed", 42)
    workers = opt_config.get("workers", -1)
    use_groups = screen_config.get("use_groups", True)
    use_baseline_filter = screen_config.get("use_baseline_filter", True)
    baseline_rate_threshold = screen_config.get(
        "baseline_improvement_rate_threshold", 0.05
    )
    baseline_epsilon = screen_config.get("baseline_improvement_epsilon", 0.01)

    weights, region_weights = _extract_weights(opt_config)

    # Find image pairs (limited to num_images for screening)
    image_pairs = find_real_images(Path(args.input_dir), limit=num_images)

    use_best_values = screen_config.get("use_best_values", False)

    # Run screening
    result = run_full_screening(
        image_pairs=image_pairs,
        weights=weights,
        region_weights=region_weights,
        num_trajectories=num_trajectories,
        morris_top_n=morris_top_n,
        run_sobol=run_sobol,
        sobol_n_samples=sobol_n_samples,
        sobol_top_n=sobol_top_n,
        n_vertices=n_vertices,
        seed=seed,
        workers=workers,
        use_best_values=use_best_values,
        use_groups=use_groups,
        use_baseline_filter=use_baseline_filter,
        baseline_improvement_rate_threshold=baseline_rate_threshold,
        baseline_improvement_epsilon=baseline_epsilon,
    )

    # Save results
    output_path = Path(args.output) if args.output else Path("./screening_results.json")
    save_screening_results(result, output_path)

    print(f"\nUse with fit: crm_gen fit {args.input_dir} --screening-results {output_path}")


def _run_fit(args, config, config_path):
    """
    Optimize synthetic image parameters to match real microscope images.

    Args:
        args: Parsed CLI arguments (input_dir, auto_screen, screening_results).
        config (dict): Parsed fit TOML configuration ([optimization] section).
        config_path (Path): Path to the fit config file (copied to output dir).
    """
    from .optimization import (
        find_real_images,
        optimize_parameters,
        compute_final_metrics,
        save_results,
        generate_all_synthetics,
        coarse_prefit,
    )
    from .parameter_registry import get_param_names, PARAMETER_REGISTRY
    from .visualization import generate_comparison_plot, generate_detailed_plots

    opt_config = config.get("optimization", {})

    # input_dir comes from the CLI positional argument
    input_dir = args.input_dir

    # Optimization parameters
    output_dir_base = opt_config.get("output_dir", "")
    limit = opt_config.get("limit", 0) or None
    n_vertices = opt_config.get("n_vertices", 8)
    maxiter = opt_config.get("maxiter", 50)
    popsize = opt_config.get("popsize", 15)
    workers = opt_config.get("workers", -1)
    seed = opt_config.get("seed", 42)
    resume = opt_config.get("resume", False)
    resume_dir = opt_config.get("resume_dir", "")
    no_checkpoint = opt_config.get("no_checkpoint", False)
    save_all_synthetics = opt_config.get("save_all_synthetics", False)

    # Metric and region weights
    weights, region_weights = _extract_weights(opt_config)

    # --- Determine active parameters ---
    active_param_names = None
    fixed_params = None
    screening_result = None
    prefit_values: Dict[str, float] = {}

    def _rejoin_dominants(active, fixed, prefit_vals):
        """Return (active', fixed') with prefit-dominant params restored as active.

        Idempotent — if a name in ``prefit_vals`` is already in ``active``
        we leave it alone; if it's somehow in ``fixed`` we pop it out.
        Empty ``prefit_vals`` (e.g. older screening JSON) is a no-op.
        """
        active = list(active)
        fixed = dict(fixed)
        for name in prefit_vals:
            if name not in active:
                active.append(name)
            fixed.pop(name, None)
        return active, fixed

    if hasattr(args, "screening_results") and args.screening_results:
        # Load pre-computed screening results
        from .sensitivity import load_screening_results

        print(f"Loading screening results from: {args.screening_results}")
        screening_result = load_screening_results(Path(args.screening_results))
        prefit_values = dict(screening_result.prefit_values)
        active_param_names, fixed_params = _rejoin_dominants(
            screening_result.active_params,
            screening_result.fixed_values,
            prefit_values,
        )
        print(f"Active parameters: {len(active_param_names)}"
              + (f" (incl. {len(prefit_values)} pre-fit dominant(s))"
                 if prefit_values else ""))
        print(f"Fixed parameters: {len(fixed_params)}")

    elif hasattr(args, "auto_screen") and args.auto_screen:
        # Run screening inline before optimization
        from .sensitivity import run_full_screening, save_screening_results

        screen_config = config.get("screening", {})
        num_screen_images = screen_config.get("num_screening_images", 2)
        screen_pairs = find_real_images(Path(input_dir), limit=num_screen_images)

        # --- Coarse pre-fit of dominant parameters (optional) -----------
        # Pre-fits the parameters the user declares "dominant" (typically
        # bg_base_brightness) so their value no longer hijacks the
        # screening sensitivity analysis.  The dominant parameters rejoin
        # the active set for the main DE fit afterwards.
        prefit_cfg = screen_config.get("coarse_prefit", {})
        prefit_enabled = prefit_cfg.get("enabled", False)
        dominant_params = list(prefit_cfg.get("dominant_params", []))

        if prefit_enabled and dominant_params:
            prefit_values = coarse_prefit(
                image_pairs=screen_pairs,
                dominant_params=dominant_params,
                weights=weights,
                region_weights=region_weights,
                maxiter=int(prefit_cfg.get("maxiter", 10)),
                popsize=int(prefit_cfg.get("popsize", 5)),
                workers=workers,
                seed=seed,
                n_vertices=n_vertices,
            )

        use_best_values = screen_config.get("use_best_values", False)
        screening_result = run_full_screening(
            image_pairs=screen_pairs,
            weights=weights,
            region_weights=region_weights,
            num_trajectories=screen_config.get("num_trajectories", 10),
            morris_top_n=screen_config.get("morris_top_n", 10),
            run_sobol=screen_config.get("run_sobol", True),
            sobol_n_samples=screen_config.get("sobol_n_samples", 128),
            sobol_top_n=screen_config.get("sobol_top_n", 7),
            n_vertices=n_vertices,
            seed=seed,
            workers=workers,
            use_best_values=use_best_values,
            use_groups=screen_config.get("use_groups", True),
            use_baseline_filter=screen_config.get("use_baseline_filter", True),
            baseline_improvement_rate_threshold=screen_config.get(
                "baseline_improvement_rate_threshold", 0.05
            ),
            baseline_improvement_epsilon=screen_config.get(
                "baseline_improvement_epsilon", 0.01
            ),
            fixed_params={**(prefit_values or {})},
            param_names=[
                n for n in get_param_names() if n not in prefit_values
            ] if prefit_values else None,
        )
        # Persist the dominant pre-fit values on the screening result BEFORE
        # writing it to disk, so a later `--screening-results` resume can
        # find them and warm-start the main DE.
        screening_result.prefit_values = dict(prefit_values)

        # Dominant params rejoin the main DE: always active, excluded from
        # the screening cap, removed from fixed_values (they're variables
        # again), and their pre-fit values seed the DE init population.
        active_param_names, fixed_params = _rejoin_dominants(
            screening_result.active_params,
            screening_result.fixed_values,
            prefit_values,
        )

    # Default to all registry parameters if no screening was used
    if active_param_names is None:
        active_param_names = get_param_names()

    # Build bounds for active parameters
    bounds_config = opt_config.get("bounds", {})
    bounds = [
        tuple(bounds_config[name]) if name in bounds_config
        else PARAMETER_REGISTRY[name].bounds
        for name in active_param_names
    ]

    # Create or reuse output directory
    if resume and resume_dir:
        output_dir = Path(resume_dir)
        if not output_dir.exists():
            print(f"Error: resume_dir does not exist: {output_dir}")
            sys.exit(1)
        print(f"Resuming from: {output_dir}")
    else:
        if resume and not resume_dir:
            print("Error: resume=true but no resume_dir specified, cannot resume")
            sys.exit(1)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if not output_dir_base:
            output_dir = Path(f"./fit_results_{timestamp}")
        else:
            output_dir = Path(output_dir_base) / f"fit_results_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, output_dir / "config.toml")

    # Save screening results to output dir if available
    if screening_result is not None:
        from .sensitivity import save_screening_results

        save_screening_results(screening_result, output_dir / "screening_results.json")

    print(f"Output directory: {output_dir}\n")

    # Find image pairs
    image_pairs = find_real_images(Path(input_dir), limit=limit)

    # Build the DE init seed from the coarse pre-fit values (if any) so the
    # main run starts with dominant parameters near their optimum.  The
    # remaining dimensions fall back to their registry default.
    init_seed_x = None
    if prefit_values:
        from .parameter_registry import get_all_defaults
        defaults = get_all_defaults()
        seed_vec = []
        for name, (lo, hi) in zip(active_param_names, bounds):
            if name in prefit_values:
                seed_vec.append(float(prefit_values[name]))
            else:
                # Default may be outside narrowed bounds in edge cases.
                seed_vec.append(float(np.clip(defaults.get(name, (lo + hi) / 2), lo, hi)))
        init_seed_x = np.array(seed_vec, dtype=float)

    # Optimize parameters
    results = optimize_parameters(
        image_pairs=image_pairs,
        bounds=bounds,
        weights=weights,
        region_weights=region_weights,
        maxiter=maxiter,
        popsize=popsize,
        workers=workers,
        seed=seed,
        n_vertices=n_vertices,
        resume=resume,
        no_checkpoint=no_checkpoint,
        active_param_names=active_param_names,
        fixed_params=fixed_params,
        checkpoint_dir=output_dir / "checkpoints",
        init_seed_x=init_seed_x,
    )

    # Compute final metrics
    avg_metrics, per_image_metrics, synth_cache = compute_final_metrics(
        image_pairs=image_pairs,
        params=results["parameters"],
        output_dir=output_dir,
        n_vertices=n_vertices,
        weights=weights,
        region_weights=region_weights,
    )

    # Save results
    save_results(
        results=results,
        avg_metrics=avg_metrics,
        per_image_metrics=per_image_metrics,
        output_dir=output_dir,
    )

    # Generate comparison plot
    generate_comparison_plot(
        image_pairs=image_pairs,
        params=results["parameters"],
        output_dir=output_dir,
        n_vertices=n_vertices,
        synth_cache=synth_cache,
        active_param_names=active_param_names,
        weights=weights,
    )

    # Generate detailed per-image plots
    generate_detailed_plots(
        image_pairs=image_pairs,
        params=results["parameters"],
        per_image_metrics=per_image_metrics,
        output_dir=output_dir,
        n_vertices=n_vertices,
        synth_cache=synth_cache,
        active_param_names=active_param_names,
    )

    # Generate all synthetics if requested
    if save_all_synthetics:
        generate_all_synthetics(
            image_pairs=image_pairs,
            params=results["parameters"],
            output_dir=output_dir,
            n_vertices=n_vertices,
        )

    print("\n" + "=" * 80)
    print("ALL DONE!")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")


def crm_gen_main():
    """
    Main CLI entry point for crm_gen.

    Dispatches to the appropriate subcommand (``run``, ``clone``, or ``fit``).
    Each subcommand has its own optional ``--config`` argument with a sensible
    default: ``run`` and ``clone`` use ``configs/default_gen_config.toml``;
    ``fit`` uses ``configs/default_fit_config.toml``.
    """
    from .config import load_config

    configs_dir = Path(__file__).parent / "configs"
    default_gen_config = configs_dir / "default_gen_config.toml"
    default_fit_config = configs_dir / "default_fit_config.toml"

    parser = argparse.ArgumentParser(
        prog="crm_gen",
        description="Synthetic Microscope Image Generation for cr_mech_coli",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")
    _build_run_parser(subparsers, default_gen_config)
    _build_clone_parser(subparsers, default_gen_config)
    _build_fit_parser(subparsers, default_fit_config)
    _build_screen_parser(subparsers, default_fit_config)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    # Load configuration
    if args.config is None:
        print("Error: No config file specified and no default config found.")
        if args.command == "fit":
            print("Use --config to specify a fit TOML configuration file.")
        else:
            print("Use --config to specify a generation TOML configuration file.")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    # Dispatch to subcommand
    if args.command == "run":
        _run_generate(config, config_path)
    elif args.command == "clone":
        _run_clone(args, config)
    elif args.command == "fit":
        _run_fit(args, config, config_path)
    elif args.command == "screen":
        _run_screen(args, config, config_path)
