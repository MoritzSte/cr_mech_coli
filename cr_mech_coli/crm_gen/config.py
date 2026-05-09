"""
Configuration utilities for crm_gen.

Provides functions for loading and managing TOML configuration files.
"""

from typing import Dict, Any
import tomllib

from .parameter_registry import (
    get_param_names,
    get_all_bounds,
    ImagingMode,
    VALID_IMAGING_MODES,
)

__all__ = [
    "load_config",
    "PARAM_NAMES",
    "DEFAULT_OPTIMIZATION_BOUNDS",
    "DEFAULT_METRIC_WEIGHTS",
    "DEFAULT_REGION_WEIGHTS",
    "DEFAULT_IMAGING_MODE",
    "ImagingMode",
    "VALID_IMAGING_MODES",
]


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load pipeline configuration from a TOML file.

    Args:
        config_path (str): Path to the TOML configuration file.

    Returns:
        Dict[str, Any]: Nested dictionary with configuration sections.
    """
    with open(config_path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Backward-compatible aliases — these now delegate to the registry
# ---------------------------------------------------------------------------

PARAM_NAMES = get_param_names()

DEFAULT_OPTIMIZATION_BOUNDS = get_all_bounds()

DEFAULT_METRIC_WEIGHTS = {
    "histogram_distance": 0.01,
    "ssim": 1.0,
    "ms_ssim": 1.0,
    "gradient_ssim": 1.0,
    "lpips": 1.0,
    "power_spectrum": 0.0,
}

DEFAULT_REGION_WEIGHTS = {
    # Background / foreground sum to 1 — the legacy two-region split.
    "background": 0.5,
    "foreground": 0.5,
    "foreground_sigma_px": 30.0,
    # Edge band: a thin Gaussian on the cell boundary that gives small
    # spatially-localised effects (edge fringe, halo, absorption transition)
    # dedicated signal in the loss.  Default 0.0 preserves backward compat
    # for existing TOMLs that don't mention it.  The recommended setting
    # for brightfield calibration is ``edge_band = 0.2`` with the bg / fg
    # weights rebalanced to ``0.4`` each so the three regions sum to 1.
    "edge_band": 0.0,
    "edge_sigma_px": 2.0,
}

# Default microscopy mode for fit / screening when neither the TOML
# config nor the CLI specifies one.  ``"all"`` reproduces the legacy
# unified-pool behaviour (every parameter is in scope).
DEFAULT_IMAGING_MODE: str = "all"
