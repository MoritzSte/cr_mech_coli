"""
Configuration utilities for crm_gen.

Provides functions for loading and managing TOML configuration files.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import tomllib


# Default values for optimization parameters
DEFAULT_BOUNDS = [
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

DEFAULT_WEIGHTS = {"histogram_distance": 0.01, "ssim": 1.0, "psnr": 0.02}

DEFAULT_REGION_WEIGHTS = {"background": 0.5, "foreground": 0.5}

# Parameters fixed at 0 for fluorescence mode (excluded from optimization)
FLUORESCENCE_FIXED_PARAMS = {
    "bg_gradient_strength": 0.0,
    "bac_halo_intensity": 0.0,
}


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


def get_active_param_config(
    imaging_mode: str,
) -> Tuple[List[str], List[Tuple[float, float]], Dict[str, float]]:
    """
    Return the active parameter names, bounds, and fixed values for an imaging mode.

    For fluorescence, ``bg_gradient_strength`` and ``bac_halo_intensity`` are
    excluded from optimization and fixed at 0.

    Args:
        imaging_mode (str): Imaging mode. Either ``"phase_contrast"`` or
            ``"fluorescence"``.

    Returns:
        active_names (List[str]): Parameter names to include in optimization.
        active_bounds (List[Tuple[float, float]]): Bounds for each active parameter.
        fixed_params (Dict[str, float]): Parameters held constant during optimization.

    Raises:
        ValueError: If ``imaging_mode`` is not ``"phase_contrast"`` or
            ``"fluorescence"``.
    """
    if imaging_mode == "fluorescence":
        fixed = FLUORESCENCE_FIXED_PARAMS
        active_names = [n for n in PARAM_NAMES if n not in fixed]
        active_bounds = [
            b for n, b in zip(PARAM_NAMES, DEFAULT_BOUNDS) if n not in fixed
        ]
        return active_names, active_bounds, fixed
    elif imaging_mode == "phase_contrast":
        return list(PARAM_NAMES), list(DEFAULT_BOUNDS), {}
    else:
        raise ValueError(
            f"Unknown imaging_mode '{imaging_mode}'. "
            "Must be 'phase_contrast' or 'fluorescence'."
        )


def rgb_uint8_to_gray_uint16(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB uint8 image to a grayscale uint16 image.

    Averages the three channels and scales from [0, 255] to [0, 65535].

    Args:
        image (np.ndarray): Input image (H x W x 3), uint8.

    Returns:
        np.ndarray: Grayscale image (H x W), uint16.
    """
    gray = np.mean(image.astype(np.float32), axis=2)
    return (gray / 255.0 * 65535).round().astype(np.uint16)


def rgb_mask_to_label_uint16(mask: np.ndarray) -> np.ndarray:
    """
    Convert an RGB color mask to a uint16 integer label mask.

    Each unique RGB color is mapped to a unique integer label. The background
    color (0, 0, 0) always maps to label 0.

    Args:
        mask (np.ndarray): RGB mask (H x W x 3), uint8.

    Returns:
        np.ndarray: Integer label mask (H x W), uint16.
    """
    encoded = (
        mask[:, :, 0].astype(np.uint32) * 65536
        + mask[:, :, 1].astype(np.uint32) * 256
        + mask[:, :, 2].astype(np.uint32)
    )
    unique_vals, inverse = np.unique(encoded, return_inverse=True)
    label_ids = np.arange(len(unique_vals), dtype=np.uint16)
    # Ensure background (encoded value 0) maps to label 0
    bg_idx = np.searchsorted(unique_vals, 0)
    if bg_idx < len(unique_vals) and unique_vals[bg_idx] == 0:
        label_ids[bg_idx] = 0
        label_ids[:bg_idx] = np.arange(1, bg_idx + 1, dtype=np.uint16)
        label_ids[bg_idx + 1 :] = np.arange(bg_idx + 1, len(unique_vals), dtype=np.uint16)
    return label_ids[inverse].reshape(mask.shape[:2])
