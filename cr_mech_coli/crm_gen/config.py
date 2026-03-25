"""
Configuration utilities for crm_gen.

Provides functions for loading and managing TOML configuration files.
"""

from typing import Dict, Any
import tomllib

from .parameter_registry import (
    get_param_names,
    get_all_bounds,
)


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
    "psnr": 0.02,
    "ms_ssim": 0.0,
    "power_spectrum": 0.0,
}

DEFAULT_REGION_WEIGHTS = {"background": 0.5, "foreground": 0.5}
