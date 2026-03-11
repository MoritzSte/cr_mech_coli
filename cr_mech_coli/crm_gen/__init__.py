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
    get_default_config,
    get_default_config_path,
    PARAM_NAMES,
    DEFAULT_BOUNDS,
    DEFAULT_WEIGHTS,
    DEFAULT_REGION_WEIGHTS,
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
    load_image,
    plot_metrics,
)

# CLI entry point
from .main import crm_gen_main
