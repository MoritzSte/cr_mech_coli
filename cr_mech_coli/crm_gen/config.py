"""
Configuration utilities for crm_gen.

Provides functions for loading and managing TOML configuration files.
"""

from pathlib import Path
from typing import Dict, Any
import tomllib


def get_default_config_path() -> Path:
    """
    Get the path to the default configuration file.

    Returns:
        Path: Absolute path to the default_config.toml file.
    """
    return Path(__file__).parent / "default_config.toml"


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


def get_default_config() -> Dict[str, Any]:
    """
    Load the default configuration.

    Returns:
        Dict[str, Any]: Default configuration dictionary.
    """
    default_path = get_default_config_path()
    if default_path.exists():
        return load_config(str(default_path))
    else:
        raise FileNotFoundError(f"Default config not found at: {default_path}")


# Default values for optimization parameters
DEFAULT_OPTIMIZATION_BOUNDS = [
    (0.2, 0.6),  # bg_base_brightness
    (0.0, 0.04),  # bg_gradient_strength
    (0.01, 0.6),  # bac_halo_intensity
    (1, 25),  # bg_noise_scale
    (0.1, 3.0),  # psf_sigma
    (1, 10000),  # peak_signal
    (0.001, 0.05),  # gaussian_sigma
]

PARAM_NAMES = [
    "bg_base_brightness",
    "bg_gradient_strength",
    "bac_halo_intensity",
    "bg_noise_scale",
    "psf_sigma",
    "peak_signal",
    "gaussian_sigma",
]

DEFAULT_METRIC_WEIGHTS = {"histogram_distance": 0.01, "ssim": 1.0, "psnr": 0.02}

DEFAULT_REGION_WEIGHTS = {"background": 0.5, "foreground": 0.5}
