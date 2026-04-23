"""
Synthetic Microscope Image Generation.

A submodule for cr_mech_coli that provides:

- Synthetic microscope image generation pipeline
- Image cloning from real microscope images
- Parameter optimization to match real microscope images

Three CLI subcommands are available. Each has an optional ``--config``
argument (positional arguments always come first):

.. code-block:: bash

    crm_gen run   [--config path/to/gen_config.toml]
    crm_gen clone img.tif mask.tif [--config path/to/gen_config.toml]
    crm_gen fit   path/to/real/images/ [--config path/to/fit_config.toml]

``run`` and ``clone`` use a *generation config* (imaging and simulation
parameters). ``fit`` uses a separate *fit config* (optimization
hyperparameters and search bounds only); the imaging parameters are the
*output* of the fit. Default configs are in ``configs/``.
"""

# Configure offscreen rendering for headless cluster environments.
# Must be set BEFORE importing pyvista/vtk (i.e. before scene, optimization, etc.)
import os as _os

_os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
_os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")
_os.environ.setdefault("DISPLAY", "")
_os.environ.setdefault("VTK_USE_OFFSCREEN_EGL", "0")

# Core scene generation
from .scene import (
    create_synthetic_scene,
    apply_synthetic_effects,
)

# Pipeline
from .pipeline import (
    run_pipeline,
    run_simulation_image_gen,
    compute_cell_ages,
)

# Config
from .config import (
    load_config,
    PARAM_NAMES,
    DEFAULT_OPTIMIZATION_BOUNDS,
    DEFAULT_METRIC_WEIGHTS,
    DEFAULT_REGION_WEIGHTS,
)

# Parameter registry
from .parameter_registry import (
    PARAMETER_REGISTRY,
    ParameterDef,
    get_param_names,
    get_all_bounds,
    get_all_defaults,
    get_bounds_for,
    get_defaults_for,
    get_groups_for,
    get_off_values_for,
    get_params_by_group,
    build_full_params,
    cast_param,
)

# Background generation
from .background import (
    generate_phase_contrast_background,
)

# Filters and effects
from .filters import (
    apply_psf_blur,
    apply_halo_effect,
    apply_microscope_effects,
    apply_phase_contrast_pipeline,
    add_poisson_noise,
    add_gaussian_noise,
)

# Bacteria brightness
from .bacteria import (
    apply_original_brightness,
    apply_age_based_brightness,
    extract_original_brightness,
)

# Metrics
from .metrics import (
    compute_all_metrics,
    compute_ssim,
    compute_psnr,
    compute_color_distribution,
    compute_ms_ssim,
    compute_power_spectrum_distance,
    load_image,
    plot_metrics,
)

# Sensitivity analysis
from .sensitivity import (
    run_morris_screening,
    run_sobol_analysis,
    run_full_screening,
    save_screening_results,
    load_screening_results,
    ScreeningResult,
    MorrisResult,
    SobolResult,
)

# CLI entry point
from .main import crm_gen_main
